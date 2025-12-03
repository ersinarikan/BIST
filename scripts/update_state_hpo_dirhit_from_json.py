#!/usr/bin/env python3
"""
Update hpo_dirhit in state file from JSON files (filtered best_dirhit)
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

from scripts.continuous_hpo_training_pipeline import STATE_FILE


def find_json_file(symbol: str, horizon: int, cycle: int) -> Optional[Path]:
    """Find HPO JSON file for symbol-horizon"""
    results_dir = Path('/opt/bist-pattern/results')
    pattern = f"optuna_pilot_features_on_h{horizon}_c{cycle}_*.json"
    json_files = list(results_dir.glob(pattern))
    
    if not json_files:
        return None
    
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if symbol in data.get('symbols', []):
                return json_file
        except Exception:
            continue
    
    return None


def main():
    # Load state
    if not STATE_FILE.exists():
        print("❌ State file not found")
        return
    
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    print("=" * 80)
    print("STATE DOSYASINDA HPO_DIRHIT GÜNCELLEME")
    print("=" * 80)
    print(f"\n🔄 Cycle: {current_cycle}")
    
    updated_count = 0
    not_found_count = 0
    already_correct_count = 0
    
    for key, task in tasks.items():
        if not isinstance(task, dict):
            continue
        if task.get('status') != 'completed':
            continue
        if task.get('cycle', 0) != current_cycle:
            continue
        
        symbol = task.get('symbol', '')
        horizon = task.get('horizon', 0)
        if not symbol or not horizon:
            parts = key.split('_')
            if len(parts) == 2:
                symbol = parts[0]
                try:
                    horizon = int(parts[1].replace('d', ''))
                except:
                    continue
            else:
                continue
        
        if horizon != 1:
            continue
        
        # Find JSON file
        json_file = find_json_file(symbol, horizon, current_cycle)
        if not json_file:
            not_found_count += 1
            continue
        
        # Read JSON
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
        except Exception:
            not_found_count += 1
            continue
        
        # Get filtered best_dirhit from JSON
        json_best_dirhit = json_data.get('best_dirhit')
        eval_spec = json_data.get('evaluation_spec', {})
        min_mc = eval_spec.get('min_mask_count', 0)
        min_mp = eval_spec.get('min_mask_pct', 0.0)
        
        # Only update if JSON has 10/5.0 filter
        if min_mc == 10 and abs(min_mp - 5.0) < 0.01:
            current_hpo_dirhit = task.get('hpo_dirhit')
            
            if json_best_dirhit is not None:
                if abs(current_hpo_dirhit - json_best_dirhit) > 0.01:
                    # Update
                    task['hpo_dirhit'] = float(json_best_dirhit)
                    tasks[key] = task
                    updated_count += 1
                    print(f"✅ {symbol}_{horizon}d: {current_hpo_dirhit:.2f}% → {json_best_dirhit:.2f}%")
                else:
                    already_correct_count += 1
        else:
            # JSON doesn't have 10/5.0 filter, skip
            pass
    
    # Save state
    if updated_count > 0:
        state['state'] = tasks
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\n✅ {updated_count} sembol güncellendi")
    else:
        print(f"\n✅ Güncellenecek sembol yok")
    
    print(f"📊 Özet:")
    print(f"   Güncellenen: {updated_count}")
    print(f"   Zaten doğru: {already_correct_count}")
    print(f"   JSON bulunamadı: {not_found_count}")


if __name__ == '__main__':
    main()

