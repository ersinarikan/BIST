#!/usr/bin/env python3
"""
Update existing HPO JSON files with filtered best params from study databases

This script:
1. Finds completed symbols from state file
2. For each symbol, finds the study DB
3. Finds best params with filter applied (from study)
4. Updates the JSON file with new best params
5. Preserves all other data in JSON
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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
from scripts.retrain_high_discrepancy_symbols import (
    find_study_db,
    find_best_trial_with_filter_applied,
    get_best_params_from_study
)


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def find_json_file(symbol: str, horizon: int, cycle: Optional[int] = None) -> Optional[Path]:
    """Find HPO JSON file for symbol-horizon"""
    results_dir = Path('/opt/bist-pattern/results')
    
    if cycle is None:
        state = load_state()
        cycle = state.get('cycle', 1)
    
    # Try cycle format first
    pattern = f"optuna_pilot_features_on_h{horizon}_c{cycle}_*.json"
    json_files = list(results_dir.glob(pattern))
    
    if not json_files:
        # Try legacy format
        pattern = f"optuna_pilot_features_on_h{horizon}_*.json"
        json_files = list(results_dir.glob(pattern))
    
    if not json_files:
        return None
    
    # Find JSON file that contains this symbol
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            symbols = data.get('symbols', [])
            if symbol in symbols:
                return json_file
        except Exception:
            continue
    
    return None


def update_json_with_filtered_params(json_file: Path, symbol: str, horizon: int,
                                     best_params_data: Dict, dry_run: bool = False) -> bool:
    """Update JSON file with filtered best params
    
    Args:
        json_file: Path to JSON file
        symbol: Stock symbol
        horizon: Prediction horizon
        best_params_data: Dict with best_params, best_trial_number, features_enabled, etc.
        dry_run: If True, don't actually update, just show what would be updated
    
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        # Read existing JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check if symbol is in this JSON
        symbols = data.get('symbols', [])
        if symbol not in symbols:
            print(f"⚠️  {symbol} not found in {json_file.name}")
            return False
        
        # Backup original
        backup_file = json_file.with_suffix('.json.backup')
        if not backup_file.exists():
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Backup created: {backup_file.name}")
        
        if dry_run:
            print(f"🔍 DRY-RUN: Would update {json_file.name} for {symbol} {horizon}d")
            print(f"   Current best_trial_number: {data.get('best_trial_number')}")
            print(f"   New best_trial_number: {best_params_data.get('best_trial_number')}")
            print(f"   Current best_dirhit: {data.get('best_dirhit')}")
            print(f"   New best_value: {best_params_data.get('best_value')}")
            return True
        
        # Update best_params
        old_best_params = data.get('best_params', {})
        new_best_params = best_params_data.get('best_params', {})
        
        # Update main best_params
        data['best_params'] = new_best_params
        
        # Update best_trial_number
        if 'best_trial_number' in best_params_data:
            data['best_trial_number'] = best_params_data['best_trial_number']
        
        # Update best_trial dict
        if 'best_trial' in data:
            data['best_trial']['number'] = best_params_data.get('best_trial_number', data['best_trial'].get('number'))
            if 'best_value' in best_params_data:
                data['best_trial']['value'] = best_params_data['best_value']
        
        # Update best_dirhit if available
        if 'best_value' in best_params_data:
            data['best_dirhit'] = best_params_data['best_value']
        
        # Update features_enabled and feature_params
        if 'features_enabled' in best_params_data:
            data['features_enabled'] = best_params_data['features_enabled']
        
        if 'feature_params' in best_params_data:
            data['feature_params'] = best_params_data['feature_params']
        
        # Update feature_flags and hyperparameters
        if 'feature_flags' in best_params_data:
            data['feature_flags'] = best_params_data['feature_flags']
        
        if 'hyperparameters' in best_params_data:
            data['hyperparameters'] = best_params_data['hyperparameters']
        
        # Add update metadata
        data['_updated_at'] = datetime.now().isoformat()
        data['_updated_reason'] = 'filtered_best_params_from_study'
        data['_filter_used'] = best_params_data.get('filter_used', {})
        
        # Write updated JSON
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Updated {json_file.name} for {symbol} {horizon}d")
        print(f"   Best trial: #{best_params_data.get('best_trial_number')} (was #{data.get('best_trial_number', 'N/A')})")
        print(f"   Best DirHit: {best_params_data.get('best_value', 0):.2f}%")
        
        return True
    
    except Exception as e:
        print(f"❌ Error updating {json_file.name}: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def process_symbol(symbol: str, horizon: int, cycle: Optional[int] = None,
                   use_filtered: bool = True, dry_run: bool = False) -> bool:
    """Process a single symbol-horizon pair"""
    print(f"\n{'='*80}")
    print(f"🔄 Processing {symbol} {horizon}d")
    print(f"{'='*80}")
    
    # Find study DB
    study_db = find_study_db(symbol, horizon, cycle)
    if not study_db or not study_db.exists():
        print(f"❌ Study DB not found for {symbol} {horizon}d")
        return False
    
    print(f"✅ Found study DB: {study_db.name}")
    
    # Get best params with filter applied
    best_params_data = get_best_params_from_study(study_db, symbol, horizon, use_filtered=use_filtered, cycle=cycle)
    if not best_params_data:
        print(f"❌ Could not get best params from study for {symbol} {horizon}d")
        return False
    
    print(f"✅ Found best params: trial #{best_params_data.get('best_trial_number')}, "
          f"DirHit: {best_params_data.get('best_value', 0):.2f}%")
    
    # Find JSON file
    json_file = find_json_file(symbol, horizon, cycle)
    if not json_file:
        print(f"❌ JSON file not found for {symbol} {horizon}d")
        return False
    
    print(f"✅ Found JSON file: {json_file.name}")
    
    # Update JSON
    success = update_json_with_filtered_params(json_file, symbol, horizon, best_params_data, dry_run=dry_run)
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Update HPO JSON files with filtered best params')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (e.g., EKGYO_1d,BRSAN_3d)')
    parser.add_argument('--all-completed', action='store_true', help='Process all completed symbols from state')
    parser.add_argument('--cycle', type=int, help='HPO cycle number (default: from state)')
    parser.add_argument('--use-filtered', action='store_true', default=True, help='Use filtered best params (default: True)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run - show what would be updated')
    
    args = parser.parse_args()
    
    if not args.symbols and not args.all_completed:
        print("❌ Must specify --symbols or --all-completed")
        return 1
    
    symbols_to_process = []
    
    if args.symbols:
        # Parse symbols (format: SYMBOL_Hd)
        for sym_horizon in args.symbols.split(','):
            sym_horizon = sym_horizon.strip()
            parts = sym_horizon.rsplit('_', 1)
            if len(parts) == 2:
                symbol = parts[0]
                try:
                    horizon = int(parts[1].replace('d', ''))
                    symbols_to_process.append((symbol, horizon))
                except Exception:
                    print(f"⚠️  Invalid format: {sym_horizon}, skipping")
    
    if args.all_completed:
        # Get all completed symbols from state
        state_data = load_state()
        # State file structure: {'state': {...}, 'cycle': ..., ...}
        state = state_data.get('state', state_data)  # Fallback to full dict if no 'state' key
        for key, task in state.items():
            if isinstance(task, dict) and task.get('status') == 'completed':
                symbol = task.get('symbol')
                horizon = task.get('horizon')
                if symbol and horizon:
                    symbols_to_process.append((symbol, horizon))
    
    if not symbols_to_process:
        print("❌ No symbols to process")
        return 1
    
    print(f"📊 Processing {len(symbols_to_process)} symbol-horizon pairs")
    if args.dry_run:
        print("🔍 DRY-RUN mode - no files will be modified")
    
    success_count = 0
    fail_count = 0
    
    for symbol, horizon in symbols_to_process:
        try:
            success = process_symbol(symbol, horizon, cycle=args.cycle,
                                   use_filtered=args.use_filtered, dry_run=args.dry_run)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"❌ Error processing {symbol} {horizon}d: {e}")
            fail_count += 1
    
    print(f"\n{'='*80}")
    print(f"📊 Summary: {success_count} succeeded, {fail_count} failed")
    print(f"{'='*80}")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

