#!/usr/bin/env python3
"""
Update state file after JSON recreation - update best_params_file and hpo_dirhit
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

from scripts.continuous_hpo_training_pipeline import STATE_FILE


def find_latest_json(symbol: str, horizon: int, cycle: int) -> Optional[Path]:
    """Find latest JSON file for symbol"""
    results_dir = Path('/opt/bist-pattern/results')
    pattern = f"optuna_pilot_features_on_h{horizon}_c{cycle}_*.json"
    json_files = list(results_dir.glob(pattern))
    
    if not json_files:
        return None
    
    # Sort by modification time, newest first
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if symbol in data.get('symbols', []):
                return json_file
        except Exception:
            continue
    
    return None


def update_state_file(symbols: Optional[list] = None, dry_run: bool = False):
    """Update state file with latest JSON files and hpo_dirhit"""
    
    if not STATE_FILE.exists():
        print(f"❌ State file not found: {STATE_FILE}")
        return
    
    # Load state
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
    
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Get symbols to process
    if symbols:
        symbols_to_process = symbols
    else:
        # Get all completed symbols
        symbols_to_process = []
        for key, task in tasks.items():
            if not isinstance(task, dict):
                continue
            if task.get('status') == 'completed' and task.get('cycle', 0) == current_cycle:
                parts = key.split('_')
                if len(parts) == 2:
                    symbol = parts[0]
                    try:
                        horizon = int(parts[1].replace('d', ''))
                        if horizon == 1:
                            symbols_to_process.append((symbol, horizon))
                    except:
                        pass
    
    print("=" * 80)
    print("STATE DOSYASINI GÜNCELLEME")
    print("=" * 80)
    print(f"\n🔄 Cycle: {current_cycle}")
    print(f"📊 İşlenecek sembol sayısı: {len(symbols_to_process)}")
    
    if dry_run:
        print(f"\n⚠️ DRY-RUN MODE - State dosyası güncellenmeyecek")
    
    updated_count = 0
    failed_count = 0
    
    for symbol, horizon in sorted(symbols_to_process):
        key = f"{symbol}_{horizon}d"
        task = tasks.get(key, {})
        
        if not isinstance(task, dict):
            continue
        
        print(f"\n📊 {key}")
        
        # Find latest JSON
        json_file = find_latest_json(symbol, horizon, current_cycle)
        if not json_file:
            print(f"  ❌ JSON file not found")
            failed_count += 1
            continue
        
        print(f"  ✅ JSON: {json_file.name}")
        
        # Load JSON to get best_dirhit
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            best_dirhit = json_data.get('best_dirhit')
            best_trial_number = json_data.get('best_trial_number')
            evaluation_spec = json_data.get('evaluation_spec', {})
            filter_mc = evaluation_spec.get('min_mask_count', 0)
            filter_mp = evaluation_spec.get('min_mask_pct', 0.0)
            
            if best_dirhit is None:
                print(f"  ⚠️ best_dirhit not found in JSON")
                failed_count += 1
                continue
            
            print(f"  📊 Best DirHit: {best_dirhit:.2f}%")
            print(f"  📊 Best Trial: #{best_trial_number}")
            print(f"  📊 Filter: {filter_mc}/{filter_mp}")
            
            if not dry_run:
                # Update state
                task['best_params_file'] = str(json_file)
                task['hpo_dirhit'] = float(best_dirhit)
                task['best_trial_number'] = int(best_trial_number)
                tasks[key] = task
                
                print(f"  ✅ State updated")
                updated_count += 1
            else:
                print(f"  [DRY-RUN] Would update state:")
                print(f"    - best_params_file: {json_file}")
                print(f"    - hpo_dirhit: {best_dirhit:.2f}%")
                print(f"    - best_trial_number: {best_trial_number}")
                updated_count += 1
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_count += 1
    
    # Save state
    if not dry_run and updated_count > 0:
        state['state'] = tasks
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"\n✅ State file saved")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 ÖZET")
    print(f"{'='*80}")
    print(f"✅ Güncellenen: {updated_count}")
    print(f"❌ Başarısız: {failed_count}")
    print(f"📊 Toplam: {len(symbols_to_process)}")


def main():
    parser = argparse.ArgumentParser(description='Update state file after JSON recreation')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to process (default: all completed)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be updated')
    
    args = parser.parse_args()
    
    symbols = args.symbols if args.symbols else None
    update_state_file(symbols, args.dry_run)


if __name__ == '__main__':
    main()

