#!/usr/bin/env python3
"""
Update hpo_dirhit for in-progress HPO tasks from study files (with 10/5.0 filter)
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

try:
    import optuna
except ImportError:
    venv_python = '/opt/bist-pattern/venv/bin/python3'
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        raise

from scripts.continuous_hpo_training_pipeline import STATE_FILE
from scripts.retrain_high_discrepancy_symbols import find_best_trial_with_filter_applied


def find_study_db(symbol: str, horizon: int, cycle: int) -> Optional[Path]:
    """Find study database file"""
    study_dir = Path('/opt/bist-pattern/hpo_studies')
    if not study_dir.exists():
        return None
    
    cycle_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}_c{cycle}.db"
    if cycle_file.exists():
        return cycle_file
    
    # Legacy format
    if cycle == 1:
        legacy_file = study_dir / f"hpo_with_features_{symbol}_h{horizon}.db"
        if legacy_file.exists():
            return legacy_file
    
    return None


def get_filtered_best_dirhit_from_study(db_file: Path, symbol: str, horizon: int,
                                       min_mask_count: int = 10, min_mask_pct: float = 5.0) -> Optional[float]:
    """Get best DirHit from study with filter applied"""
    try:
        filtered_trial, filtered_score = find_best_trial_with_filter_applied(
            db_file, symbol, horizon, min_mask_count, min_mask_pct
        )
        
        if not filtered_trial:
            return None
        
        # Get symbol-specific avg_dirhit
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = filtered_trial.user_attrs.get('symbol_metrics', {})
        if symbol_key in symbol_metrics:
            avg_dirhit = symbol_metrics[symbol_key].get('avg_dirhit')
            if avg_dirhit is not None:
                return float(avg_dirhit)
        
        # Fallback: use filtered_score
        return filtered_score
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
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
    print("HPO DEVAM EDEN SEMBOLLER İÇİN HPO_DIRHIT GÜNCELLEME")
    print("=" * 80)
    print(f"\n🔄 Cycle: {current_cycle}")
    print(f"🔍 Filter: 10/5.0")
    
    # Find in-progress HPO tasks
    in_progress_tasks = []
    for key, task in tasks.items():
        if not isinstance(task, dict):
            continue
        if task.get('status') == 'hpo_in_progress' and task.get('cycle', 0) == current_cycle:
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
            
            if horizon == 1:
                in_progress_tasks.append((key, symbol, horizon, task))
    
    print(f"\n📊 HPO devam eden: {len(in_progress_tasks)} sembol")
    
    if not in_progress_tasks:
        print("\n✅ HPO devam eden sembol yok")
        return
    
    updated_count = 0
    not_found_count = 0
    no_trial_count = 0
    
    for key, symbol, horizon, task in in_progress_tasks:
        print(f"\n🔍 {symbol}_{horizon}d...")
        
        # Find study DB
        db_file = find_study_db(symbol, horizon, current_cycle)
        if not db_file:
            print(f"   ⚠️ Study DB bulunamadı")
            not_found_count += 1
            continue
        
        # Get filtered best DirHit
        filtered_dirhit = get_filtered_best_dirhit_from_study(db_file, symbol, horizon, 10, 5.0)
        
        if filtered_dirhit is None:
            print(f"   ⚠️ Filter ile best trial bulunamadı")
            no_trial_count += 1
            continue
        
        # Update state
        current_hpo_dirhit = task.get('hpo_dirhit')
        if current_hpo_dirhit is None or abs(current_hpo_dirhit - filtered_dirhit) > 0.01:
            task['hpo_dirhit'] = float(filtered_dirhit)
            tasks[key] = task
            updated_count += 1
            if current_hpo_dirhit is not None:
                print(f"   ✅ {current_hpo_dirhit:.2f}% → {filtered_dirhit:.2f}%")
            else:
                print(f"   ✅ {filtered_dirhit:.2f}% (yeni)")
        else:
            print(f"   ✅ Zaten doğru: {filtered_dirhit:.2f}%")
    
    # Save state
    if updated_count > 0:
        state['state'] = tasks
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\n✅ {updated_count} sembol güncellendi")
    else:
        print(f"\n✅ Güncellenecek sembol yok")
    
    print(f"\n📊 Özet:")
    print(f"   Güncellenen: {updated_count}")
    print(f"   Study bulunamadı: {not_found_count}")
    print(f"   Best trial bulunamadı: {no_trial_count}")


if __name__ == '__main__':
    main()

