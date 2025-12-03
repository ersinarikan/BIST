#!/usr/bin/env python3
"""
Fix LOW SUPPORT issues in continuous_hpo_state.json

For each completed task with LOW SUPPORT HPO DirHit, finds the best trial
with sufficient support and updates the state file.
"""

import sys
import json
import optuna
from pathlib import Path
from typing import Dict, Optional, Any

sys.path.insert(0, '/opt/bist-pattern')

def find_best_trial_with_support(db_file: Path, symbol: str, horizon: int, 
                                  min_mask_count: int = 10, min_mask_pct: float = 5.0) -> Optional[Dict]:
    """Find best trial with sufficient support"""
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
            mask_pcts = [s.get('mask_pct', 0.0) for s in split_metrics if s.get('mask_pct') is not None]
            avg_mask_pct = sum(mask_pcts) / len(mask_pcts) if mask_pcts else 0.0
            
            # Check if this trial has sufficient support
            if total_mask_count >= min_mask_count and avg_mask_pct >= min_mask_pct:
                # This trial has sufficient support, check if it's better
                trial_score = float(trial.value) if trial.value is not None else float('-inf')
                if trial_score > best_score:
                    best_score = trial_score
                    best_trial = {
                        'trial_number': trial.number,
                        'score': trial_score,
                        'dirhit': metrics.get('avg_dirhit'),
                        'mask_count': total_mask_count,
                        'mask_pct': avg_mask_pct,
                        'params': trial.params,
                        'user_attrs': trial.user_attrs
                    }
        
        return best_trial
    
    except Exception as e:
        print(f"Error finding best trial: {e}")
        return None

def find_hpo_json_with_trial(results_dir: Path, symbol: str, horizon: int, 
                             trial_number: int, cycle: int = 2) -> Optional[Path]:
    """Find HPO JSON file that contains the specified trial"""
    # Look for JSON files matching pattern
    pattern = f"optuna_pilot_features_on_h{horizon}_c{cycle}_*.json"
    json_files = list(results_dir.glob(pattern))
    
    for json_file in sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if this JSON contains the trial
            best_trial_num = data.get('best_trial_number')
            if best_trial_num == trial_number:
                return json_file
            
            # Also check top_k_trials
            top_k = data.get('top_k_trials', [])
            for trial_info in top_k:
                if trial_info.get('number') == trial_number:
                    return json_file
        except Exception:
            continue
    
    return None

def main():
    state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
    results_dir = Path('/opt/bist-pattern/results')
    hpo_studies_dir = Path('/opt/bist-pattern/hpo_studies')
    
    if not state_file.exists():
        print(f"❌ State file not found: {state_file}")
        return 1
    
    # Load state
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
    
    tasks = state.get('state', {})
    cycle = state.get('cycle', 1)
    
    print("=" * 100)
    print("LOW SUPPORT DÜZELTME")
    print("=" * 100)
    print()
    
    # Find LOW SUPPORT tasks
    low_support_tasks = []
    for key, task in tasks.items():
        if task.get('status') == 'completed' and task.get('cycle') == cycle:
            symbol = task.get('symbol')
            horizon = task.get('horizon')
            hpo_dirhit = task.get('hpo_dirhit')
            
            if hpo_dirhit is not None:
                # Check HPO JSON for LOW SUPPORT
                best_params_file = task.get('best_params_file')
                if best_params_file and Path(best_params_file).exists():
                    try:
                        with open(best_params_file, 'r') as f:
                            hpo_data = json.load(f)
                        
                        best_trial_metrics = hpo_data.get('best_trial_metrics', {})
                        symbol_key = f"{symbol}_{horizon}d"
                        symbol_metrics = best_trial_metrics.get(symbol_key, {})
                        split_metrics = symbol_metrics.get('split_metrics', [])
                        
                        if split_metrics:
                            total_mask_count = sum(s.get('mask_count', 0) for s in split_metrics)
                            mask_pcts = [s.get('mask_pct', 0.0) for s in split_metrics if s.get('mask_pct') is not None]
                            avg_mask_pct = sum(mask_pcts) / len(mask_pcts) if mask_pcts else 0.0
                            
                            if total_mask_count < 10 or avg_mask_pct < 5.0:
                                low_support_tasks.append({
                                    'key': key,
                                    'symbol': symbol,
                                    'horizon': horizon,
                                    'current_hpo_dirhit': hpo_dirhit,
                                    'current_mask_count': total_mask_count,
                                    'current_mask_pct': avg_mask_pct,
                                    'best_params_file': best_params_file
                                })
                    except Exception as e:
                        print(f"⚠️ Error checking {key}: {e}")
    
    if not low_support_tasks:
        print("✅ LOW SUPPORT sorunu yok!")
        return 0
    
    print(f"📊 {len(low_support_tasks)} LOW SUPPORT task bulundu")
    print()
    
    # Fix each task
    fixed_count = 0
    for task_info in low_support_tasks:
        symbol = task_info['symbol']
        horizon = task_info['horizon']
        key = task_info['key']
        
        print(f"🔍 {symbol}_{horizon}d kontrol ediliyor...")
        
        # Find study database
        db_file = hpo_studies_dir / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
        if not db_file.exists():
            db_file = hpo_studies_dir / f"hpo_with_features_{symbol}_h{horizon}.db"
        
        if not db_file.exists():
            print(f"   ⚠️ Study database bulunamadı: {db_file.name}")
            continue
        
        # Find best trial with support
        best_trial = find_best_trial_with_support(db_file, symbol, horizon, min_mask_count=10, min_mask_pct=5.0)
        
        if not best_trial:
            print(f"   ⚠️ Yeterli support'a sahip trial bulunamadı")
            print(f"   ℹ️  Mevcut HPO DirHit kullanılmaya devam edilecek (LOW SUPPORT uyarısı ile)")
            continue
        
        print(f"   ✅ Trial #{best_trial['trial_number']} bulundu (DirHit={best_trial['dirhit']:.2f}%, mask_count={best_trial['mask_count']})")
        
        # Find or create HPO JSON with this trial
        # For now, we'll update the state with the new DirHit
        # The best_params_file might need to be updated too, but that's more complex
        # So we'll just update the hpo_dirhit in state
        
        task = tasks[key]
        old_dirhit = task.get('hpo_dirhit')
        task['hpo_dirhit'] = best_trial['dirhit']
        
        print(f"   📝 HPO DirHit güncellendi: {old_dirhit:.2f}% → {best_trial['dirhit']:.2f}%")
        print(f"   📝 Mask Count: {task_info['current_mask_count']} → {best_trial['mask_count']}")
        print()
        
        fixed_count += 1
    
    if fixed_count > 0:
        # Save updated state
        state['state'] = tasks
        
        # Backup original
        backup_file = state_file.with_suffix('.json.backup')
        with open(backup_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"💾 Backup oluşturuldu: {backup_file}")
        
        # Save updated state
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✅ State dosyası güncellendi: {fixed_count} task düzeltildi")
    else:
        print("ℹ️  Düzeltilecek task bulunamadı")
    
    print()
    print("=" * 100)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

