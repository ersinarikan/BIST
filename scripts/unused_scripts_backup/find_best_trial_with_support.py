#!/usr/bin/env python3
"""
Find best trials with sufficient support for a symbol

Shows top N trials with sufficient support (not low support)
"""

import sys
import optuna
from pathlib import Path
import numpy as np

sys.path.insert(0, '/opt/bist-pattern')

def find_best_trials_with_support(db_file: Path, symbol: str, horizon: int, 
                                   min_mask_count: int = 10, min_mask_pct: float = 5.0,
                                   top_n: int = 10):
    """Find top N trials with sufficient support"""
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        symbol_key = f"{symbol}_{horizon}d"
        valid_trials = []
        
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
            avg_dirhit = metrics.get('avg_dirhit')
            
            # Check if this trial has sufficient support
            if total_mask_count >= min_mask_count and avg_mask_pct >= min_mask_pct:
                trial_score = float(trial.value) if trial.value is not None else float('-inf')
                valid_trials.append({
                    'trial_number': trial.number,
                    'score': trial_score,
                    'dirhit': avg_dirhit,
                    'mask_count': total_mask_count,
                    'mask_pct': avg_mask_pct,
                    'split_count': len(split_metrics)
                })
        
        # Sort by score (descending)
        valid_trials.sort(key=lambda x: x['score'], reverse=True)
        
        return valid_trials[:top_n], len(valid_trials)
    
    except Exception as e:
        print(f"Error: {e}")
        return [], 0

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/find_best_trial_with_support.py SYMBOL HORIZON [TOP_N]")
        print("Example: python3 scripts/find_best_trial_with_support.py BESLR 1 20")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    horizon = int(sys.argv[2])
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    # Find study database
    cycle = 2
    db_file = Path(f'/opt/bist-pattern/hpo_studies/hpo_with_features_{symbol}_h{horizon}_c{cycle}.db')
    if not db_file.exists():
        db_file = Path(f'/opt/bist-pattern/hpo_studies/hpo_with_features_{symbol}_h{horizon}.db')
    
    if not db_file.exists():
        print(f"❌ Study database not found: {db_file}")
        return 1
    
    print("=" * 100)
    print(f"🔍 {symbol} {horizon}d - Yeterli Support'a Sahip En İyi Trial'lar")
    print("=" * 100)
    print(f"📂 Study: {db_file.name}")
    print(f"📊 Minimum: mask_count >= 10, mask_pct >= 5.0%")
    print(f"🔝 Top {top_n} trial gösteriliyor")
    print()
    
    trials, total_count = find_best_trials_with_support(db_file, symbol, horizon, min_mask_count=10, min_mask_pct=5.0, top_n=top_n)
    
    if not trials:
        print("❌ Yeterli support'a sahip trial bulunamadı!")
        return 1
    
    print(f"✅ Toplam {total_count} trial bulundu (yeterli support)")
    print(f"📊 Top {min(top_n, len(trials))} trial gösteriliyor")
    print()
    print(f"{'#':<4} {'Trial':<6} {'Score':<12} {'DirHit':<10} {'Mask':<6} {'Mask%':<8} {'Splits':<7}")
    print("-" * 100)
    
    for i, trial in enumerate(trials, 1):
        dirhit_str = f"{trial['dirhit']:.2f}%" if trial['dirhit'] is not None else "N/A"
        print(f"{i:<4} {trial['trial_number']:<6} {trial['score']:<12.2f} {dirhit_str:<10} "
              f"{trial['mask_count']:<6} {trial['mask_pct']:<7.1f}% {trial['split_count']:<7}")
    
    print()
    print("=" * 100)
    print(f"🏆 En İyi Trial: #{trials[0]['trial_number']}")
    print(f"   Score: {trials[0]['score']:.2f}")
    print(f"   DirHit: {trials[0]['dirhit']:.2f}%" if trials[0]['dirhit'] is not None else "   DirHit: N/A")
    print(f"   Mask Count: {trials[0]['mask_count']}")
    print(f"   Mask PCT: {trials[0]['mask_pct']:.1f}%")
    print()
    
    # Compare with current best (trial 530)
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        current_best = study.best_trial
        print(f"📊 Mevcut Best Trial (spurious): #{current_best.number}")
        print(f"   Score: {current_best.value:.2f}" if current_best.value else "   Score: N/A")
        
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = current_best.user_attrs.get('symbol_metrics', {}).get(symbol_key, {})
        current_dirhit = symbol_metrics.get('avg_dirhit')
        split_metrics = symbol_metrics.get('split_metrics', [])
        current_mask_count = sum(s.get('mask_count', 0) for s in split_metrics)
        current_mask_pct = sum(s.get('mask_pct', 0.0) for s in split_metrics if s.get('mask_pct') is not None) / len(split_metrics) if split_metrics else 0.0
        
        print(f"   DirHit: {current_dirhit:.2f}%" if current_dirhit is not None else "   DirHit: N/A")
        print(f"   Mask Count: {current_mask_count} ⚠️ (low support!)")
        print(f"   Mask PCT: {current_mask_pct:.1f}% ⚠️ (low support!)")
        print()
        
        if trials[0]['trial_number'] != current_best.number:
            print(f"💡 Öneri: Trial #{trials[0]['trial_number']} kullanılmalı (daha güvenilir)")
            print(f"   Score farkı: {trials[0]['score'] - (current_best.value if current_best.value else 0):.2f}")
    except Exception as e:
        print(f"⚠️ Mevcut best trial bilgisi alınamadı: {e}")
    
    print("=" * 100)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

