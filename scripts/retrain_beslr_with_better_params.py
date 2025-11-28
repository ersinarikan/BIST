#!/usr/bin/env python3
"""
BESLR için daha güvenilir HPO parametreleriyle yeniden eğitim

Bu script:
1. BESLR'ın HPO study'sinden low support olmayan en iyi trial'ı bulur
2. O trial'ın parametreleriyle yeniden eğitim yapar
"""

import sys
import os
import json
import optuna
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

# Ensure DATABASE_URL (PgBouncer port 6432)
# Set before importing anything that uses database
if 'DATABASE_URL' not in os.environ:
    try:
        secret_path = '/opt/bist-pattern/.secrets/db_password'
        if os.path.exists(secret_path):
            with open(secret_path, 'r') as sp:
                _pwd = sp.read().strip()
                if _pwd:
                    os.environ['DATABASE_URL'] = f'postgresql://bist_user:{_pwd}@127.0.0.1:6432/bist_pattern_db'
        if 'DATABASE_URL' not in os.environ:
            # Fallback to PgBouncer port
            os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db'
    except Exception:
        os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db'

# Import after DATABASE_URL is set
from scripts.continuous_hpo_training_pipeline import ContinuousHPOPipeline

def find_best_trial_with_support(db_file: Path, symbol: str, horizon: int, min_mask_count: int = 10, min_mask_pct: float = 5.0):
    """Find best trial with sufficient support (not low support)"""
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        symbol_key = f"{symbol}_{horizon}d"
        best_trial = None
        best_score = float('-inf')
        
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            # Get symbol_metrics
            symbol_metrics = trial.user_attrs.get('symbol_metrics')
            if not symbol_metrics or symbol_key not in symbol_metrics:
                continue
            
            metrics = symbol_metrics[symbol_key]
            split_metrics = metrics.get('split_metrics', [])
            
            # Calculate total mask_count across all splits
            total_mask_count = sum(s.get('mask_count', 0) for s in split_metrics)
            avg_mask_pct = sum(s.get('mask_pct', 0.0) for s in split_metrics if s.get('mask_pct') is not None) / len(split_metrics) if split_metrics else 0.0
            
            # Check if this trial has sufficient support
            if total_mask_count >= min_mask_count and avg_mask_pct >= min_mask_pct:
                # This trial has sufficient support, check if it's better
                trial_score = float(trial.value) if trial.value is not None else float('-inf')
                if trial_score > best_score:
                    best_score = trial_score
                    best_trial = trial
        
        return best_trial
    
    except Exception as e:
        print(f"Error finding best trial: {e}")
        return None

def get_trial_params(trial):
    """Extract parameters from trial"""
    if trial is None:
        return None
    
    params = trial.params.copy()
    
    # Add best_trial_number for seed calculation
    params['best_trial_number'] = trial.number
    
    # Get features_enabled from user_attrs if available
    features_enabled = {}
    if 'features_enabled' in trial.user_attrs:
        features_enabled = trial.user_attrs['features_enabled']
    elif 'symbol_metrics' in trial.user_attrs:
        # Try to extract from symbol_metrics (if available)
        pass
    
    params['features_enabled'] = features_enabled
    
    return params

def main():
    symbol = 'BESLR'
    horizon = 1
    
    print("=" * 100)
    print(f"🔄 BESLR {horizon}d için Daha Güvenilir Parametrelerle Yeniden Eğitim")
    print("=" * 100)
    print()
    
    # Find study database
    state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
    cycle = 2
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                content = f.read().strip()
                if content.count('{') > 1:
                    last_brace = content.rfind('}')
                    if last_brace > 0:
                        brace_count = 0
                        start_pos = last_brace
                        for i in range(last_brace, -1, -1):
                            if content[i] == '}':
                                brace_count += 1
                            elif content[i] == '{':
                                brace_count -= 1
                                if brace_count == 0:
                                    start_pos = i
                                    break
                        content = content[start_pos:last_brace+1]
                state = json.loads(content)
                cycle = state.get('cycle', 2)
        except Exception:
            pass
    
    db_file = Path(f'/opt/bist-pattern/hpo_studies/hpo_with_features_{symbol}_h{horizon}_c{cycle}.db')
    if not db_file.exists():
        db_file = Path(f'/opt/bist-pattern/hpo_studies/hpo_with_features_{symbol}_h{horizon}.db')
    
    if not db_file.exists():
        print(f"❌ Study database not found: {db_file}")
        return 1
    
    print(f"📂 Study Database: {db_file.name}")
    print()
    
    # Find best trial with sufficient support
    print("🔍 Low support olmayan en iyi trial aranıyor...")
    best_trial = find_best_trial_with_support(db_file, symbol, horizon, min_mask_count=10, min_mask_pct=5.0)
    
    if best_trial is None:
        print("❌ Yeterli support'a sahip trial bulunamadı!")
        print("   Tüm trial'lar low support olabilir.")
        print("   Mevcut best trial parametreleriyle devam ediliyor...")
        
        # Fallback to study's best trial
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        best_trial = study.best_trial
        print(f"   Fallback: Best trial #{best_trial.number} (score={best_trial.value})")
    else:
        print(f"✅ Bulundu: Trial #{best_trial.number}")
        print(f"   Score: {best_trial.value}")
        
        # Get metrics
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = best_trial.user_attrs.get('symbol_metrics', {}).get(symbol_key, {})
        avg_dirhit = symbol_metrics.get('avg_dirhit', 'N/A')
        split_metrics = symbol_metrics.get('split_metrics', [])
        total_mask_count = sum(s.get('mask_count', 0) for s in split_metrics)
        avg_mask_pct = sum(s.get('mask_pct', 0.0) for s in split_metrics if s.get('mask_pct') is not None) / len(split_metrics) if split_metrics else 0.0
        
        print(f"   DirHit: {avg_dirhit}%")
        print(f"   Mask Count: {total_mask_count}")
        print(f"   Mask PCT: {avg_mask_pct:.1f}%")
    
    print()
    
    # Get parameters
    best_params = get_trial_params(best_trial)
    if not best_params:
        print("❌ Parametreler alınamadı!")
        return 1
    
    print("📋 Parametreler hazırlandı")
    print()
    
    # Run training
    print("🎓 Eğitim başlatılıyor...")
    print()
    
    pipeline = ContinuousHPOPipeline()
    
    # Get HPO result (for feature flags, etc.)
    hpo_result = {
        'best_trial_number': best_trial.number,
        'best_value': float(best_trial.value) if best_trial.value is not None else None,
        'best_dirhit': best_trial.user_attrs.get('avg_dirhit'),
        'best_params': best_params
    }
    
    try:
        result = pipeline.run_training(symbol, horizon, best_params, hpo_result=hpo_result)
        
        if result:
            print()
            print("=" * 100)
            print("✅ Eğitim Tamamlandı!")
            print("=" * 100)
            print(f"   Training DirHit (WFV): {result.get('wfv_dirhit', 'N/A')}%")
            print(f"   Training DirHit (Online): {result.get('online', 'N/A')}%")
            print(f"   Training DirHit (Adaptive): {result.get('adaptive_dirhit', 'N/A')}%")
            print()
            print("📊 Önceki Sonuçlar:")
            print(f"   HPO DirHit (spurious): 100.0%")
            print(f"   Training DirHit (eski): 40.0%")
            print()
            return 0
        else:
            print("❌ Eğitim başarısız!")
            return 1
    
    except Exception as e:
        print(f"❌ Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

