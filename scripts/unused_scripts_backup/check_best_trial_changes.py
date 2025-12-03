#!/usr/bin/env python3
"""
Check which symbols have different best trial numbers after applying 10/5.0 filter
"""

import sys
import os
import json
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

from scripts.retrain_high_discrepancy_symbols import find_best_trial_with_filter_applied


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


def get_json_best_trial(json_file: Path) -> Optional[int]:
    """Get best trial number from JSON"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('best_trial_number')
    except Exception:
        return None


def get_filtered_best_trial(db_file: Path, symbol: str, horizon: int,
                           min_mask_count: int = 10, min_mask_pct: float = 5.0) -> Optional[int]:
    """Get best trial number from study with filter applied"""
    try:
        filtered_trial, filtered_score = find_best_trial_with_filter_applied(
            db_file, symbol, horizon, min_mask_count, min_mask_pct
        )
        
        if filtered_trial:
            return filtered_trial.number
        return None
    except Exception as e:
        return None


def main():
    # Load state to get completed symbols
    state_file = Path('/opt/bist-pattern/results/continuous_hpo_state.json')
    if not state_file.exists():
        print("❌ State file not found")
        return
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    print("=" * 80)
    print("BEST TRIAL NUMARASI DEĞİŞİKLİKLERİ KONTROLÜ")
    print("=" * 80)
    print(f"\n🔄 Cycle: {current_cycle}")
    print(f"🔍 Filter: 10/5.0")
    
    changed_symbols = []
    unchanged_symbols = []
    no_json_count = 0
    no_study_count = 0
    no_filtered_trial_count = 0
    
    # Check all completed symbols
    for key, task in tasks.items():
        if not isinstance(task, dict):
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
            no_json_count += 1
            continue
        
        # Get JSON best trial number
        json_trial = get_json_best_trial(json_file)
        if json_trial is None:
            continue
        
        # Find study DB
        db_file = find_study_db(symbol, horizon, current_cycle)
        if not db_file:
            no_study_count += 1
            continue
        
        # Get filtered best trial number
        filtered_trial = get_filtered_best_trial(db_file, symbol, horizon, 10, 5.0)
        if filtered_trial is None:
            no_filtered_trial_count += 1
            continue
        
        # Compare
        if json_trial != filtered_trial:
            changed_symbols.append({
                'symbol': symbol,
                'horizon': horizon,
                'json_trial': json_trial,
                'filtered_trial': filtered_trial,
                'json_file': json_file.name
            })
        else:
            unchanged_symbols.append(symbol)
    
    # Print results
    print(f"\n📊 Özet:")
    print(f"   Best trial değişen: {len(changed_symbols)} sembol")
    print(f"   Best trial aynı: {len(unchanged_symbols)} sembol")
    print(f"   JSON bulunamadı: {no_json_count} sembol")
    print(f"   Study bulunamadı: {no_study_count} sembol")
    print(f"   Filtered trial bulunamadı: {no_filtered_trial_count} sembol")
    
    if changed_symbols:
        print(f"\n🔄 BEST TRIAL DEĞİŞEN SEMBOLLER ({len(changed_symbols)}):")
        print("=" * 80)
        for item in sorted(changed_symbols, key=lambda x: x['symbol']):
            print(f"   {item['symbol']}_{item['horizon']}d: "
                  f"JSON trial #{item['json_trial']} → "
                  f"Filtered trial #{item['filtered_trial']} "
                  f"({item['json_trial'] - item['filtered_trial']:+d})")
    
    if unchanged_symbols:
        print(f"\n✅ BEST TRIAL AYNI SEMBOLLER (ilk 20):")
        for symbol in sorted(unchanged_symbols)[:20]:
            print(f"   {symbol}_1d")
        if len(unchanged_symbols) > 20:
            print(f"   ... ve {len(unchanged_symbols) - 20} sembol daha")


if __name__ == '__main__':
    main()

