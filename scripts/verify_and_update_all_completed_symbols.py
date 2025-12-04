#!/usr/bin/env python3
"""
Verify and update all completed symbols:
1. Check study files for filter values
2. Verify JSON files match study filters
3. Update JSON files if needed
4. Retrain if best trial changed
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
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


def get_filter_from_study(db_file: Path, symbol: str, horizon: int) -> Tuple[int, float]:
    """Get filter values (min_mask_count, min_mask_pct) from study database"""
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        best_trial = study.best_trial
        
        # Try to get filter from best trial's user_attrs
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = best_trial.user_attrs.get('symbol_metrics', {})
        
        if isinstance(symbol_metrics, dict) and symbol_key in symbol_metrics:
            metrics = symbol_metrics[symbol_key]
            split_metrics = metrics.get('split_metrics', [])
            if split_metrics:
                first_split = split_metrics[0]
                min_mask_count = first_split.get('min_mask_count', 0)
                min_mask_pct = first_split.get('min_mask_pct', 0.0)
                return min_mask_count, min_mask_pct
        
        # Fallback: check all trials for filter values
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            symbol_metrics = trial.user_attrs.get('symbol_metrics', {})
            if isinstance(symbol_metrics, dict) and symbol_key in symbol_metrics:
                metrics = symbol_metrics[symbol_key]
                split_metrics = metrics.get('split_metrics', [])
                if split_metrics:
                    first_split = split_metrics[0]
                    min_mask_count = first_split.get('min_mask_count', 0)
                    min_mask_pct = first_split.get('min_mask_pct', 0.0)
                    return min_mask_count, min_mask_pct
        
        return 0, 0.0
    except Exception as e:
        print(f"  ⚠️ Error reading filter from study: {e}")
        return 0, 0.0


def check_json_filter(json_file: Path, symbol: str, horizon: int) -> Tuple[int, float]:
    """Get filter values from JSON file"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        eval_spec = data.get('evaluation_spec', {})
        min_mask_count = eval_spec.get('min_mask_count', 0)
        min_mask_pct = eval_spec.get('min_mask_pct', 0.0)
        return min_mask_count, min_mask_pct
    except Exception:
        return 0, 0.0


def update_json_with_filtered_params(json_file: Path, symbol: str, horizon: int,
                                     best_params_data: Dict, dry_run: bool = False) -> bool:
    """Update JSON file with filtered best params"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create backup
        if not dry_run:
            backup_file = json_file.with_suffix('.json.backup')
            if not backup_file.exists():
                import shutil
                shutil.copy2(json_file, backup_file)
                print(f"  ✅ Backup created: {backup_file.name}")
        
        # Update best params
        if 'best_params' in best_params_data:
            data['best_params'] = best_params_data['best_params']
        
        if 'best_trial_number' in best_params_data:
            data['best_trial_number'] = best_params_data['best_trial_number']
        
        if 'best_dirhit' in best_params_data:
            data['best_dirhit'] = best_params_data['best_dirhit']
        
        if 'features_enabled' in best_params_data:
            data['features_enabled'] = best_params_data['features_enabled']
        
        if 'feature_params' in best_params_data:
            data['feature_params'] = best_params_data['feature_params']
        
        # Update metadata
        data['_updated_at'] = datetime.now().isoformat()
        data['_updated_reason'] = 'Filtered best params from study'
        
        if not dry_run:
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  ✅ JSON updated: {json_file.name}")
        else:
            print(f"  [DRY-RUN] Would update: {json_file.name}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error updating JSON: {e}")
        return False


def verify_symbol(symbol: str, horizon: int, cycle: int, dry_run: bool = False) -> Dict:
    """Verify and update a single symbol"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'study_found': False,
        'json_found': False,
        'filter_match': False,
        'trial_changed': False,
        'updated': False,
        'error': None
    }
    
    print(f"\n{'='*80}")
    print(f"🔍 {symbol}_{horizon}d")
    print(f"{'='*80}")
    
    # Find study DB
    db_file = find_study_db(symbol, horizon, cycle)
    if not db_file:
        result['error'] = 'Study DB not found'
        print(f"  ❌ Study DB not found")
        return result
    
    result['study_found'] = True
    print(f"  ✅ Study DB: {db_file.name}")
    
    # Get filter from study
    study_min_mc, study_min_mp = get_filter_from_study(db_file, symbol, horizon)
    print(f"  📊 Study filter: min_mask_count={study_min_mc}, min_mask_pct={study_min_mp}")
    
    # Find JSON file
    json_file = find_json_file(symbol, horizon, cycle)
    if not json_file:
        result['error'] = 'JSON file not found'
        print(f"  ❌ JSON file not found")
        return result
    
    result['json_found'] = True
    print(f"  ✅ JSON file: {json_file.name}")
    
    # Get filter from JSON
    json_min_mc, json_min_mp = check_json_filter(json_file, symbol, horizon)
    print(f"  📊 JSON filter: min_mask_count={json_min_mc}, min_mask_pct={json_min_mp}")
    
    # Check if filters match
    if study_min_mc == json_min_mc and abs(study_min_mp - json_min_mp) < 0.01:
        result['filter_match'] = True
        print(f"  ✅ Filters match")
    else:
        result['filter_match'] = False
        print(f"  ⚠️ Filters don't match!")
    
        # Get best params with filter applied
        try:
            # Set environment variables for filter (get_best_params_from_study reads from env)
            os.environ['HPO_MIN_MASK_COUNT'] = str(study_min_mc)
            os.environ['HPO_MIN_MASK_PCT'] = str(study_min_mp)
            
            best_params_data = get_best_params_from_study(
                db_file, symbol, horizon,
                use_filtered=True
            )
            
            if not best_params_data:
                result['error'] = 'Could not get best params from study'
                print(f"  ❌ Could not get best params from study")
                return result
        except Exception as e:
            result['error'] = str(e)
            print(f"  ❌ Error getting best params: {e}")
            import traceback
            traceback.print_exc()
            return result
        
        try:
            # Check if trial number changed
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            old_trial = json_data.get('best_trial_number')
            new_trial = best_params_data.get('best_trial_number')
            
            if old_trial != new_trial:
                result['trial_changed'] = True
                print(f"  ⚠️ Trial changed: {old_trial} -> {new_trial}")
            else:
                print(f"  ✅ Trial unchanged: {new_trial}")
            
            # Update JSON if needed
            if not result['filter_match'] or result['trial_changed']:
                success = update_json_with_filtered_params(
                    json_file, symbol, horizon, best_params_data, dry_run
                )
                result['updated'] = success
            else:
                print(f"  ✅ No update needed")
        except Exception as e:
            result['error'] = str(e)
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Verify and update all completed symbols')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be updated without making changes')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to verify (default: all completed)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to verify (default: 1)')
    
    args = parser.parse_args()
    
    # Load state
    state = load_state()
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Get completed symbols
    completed_symbols = []
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
        
        if horizon in args.horizons:
            if args.symbols:
                if symbol in args.symbols:
                    completed_symbols.append((symbol, horizon))
            else:
                completed_symbols.append((symbol, horizon))
    
    print(f"📊 Found {len(completed_symbols)} completed symbols to verify")
    print(f"🔄 Cycle: {current_cycle}")
    print(f"🔍 Dry-run: {args.dry_run}")
    
    # Verify each symbol
    results = []
    for symbol, horizon in sorted(completed_symbols):
        result = verify_symbol(symbol, horizon, current_cycle, args.dry_run)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    
    study_found = sum(1 for r in results if r['study_found'])
    json_found = sum(1 for r in results if r['json_found'])
    filter_match = sum(1 for r in results if r['filter_match'])
    trial_changed = sum(1 for r in results if r['trial_changed'])
    updated = sum(1 for r in results if r['updated'])
    errors = sum(1 for r in results if r['error'])
    
    print(f"✅ Study found: {study_found}/{len(results)}")
    print(f"✅ JSON found: {json_found}/{len(results)}")
    print(f"✅ Filter match: {filter_match}/{len(results)}")
    print(f"⚠️ Trial changed: {trial_changed}/{len(results)}")
    print(f"🔄 Updated: {updated}/{len(results)}")
    if errors > 0:
        print(f"❌ Errors: {errors}/{len(results)}")
    
    # List symbols that need retraining
    if trial_changed > 0:
        print(f"\n⚠️ Symbols with changed trial (need retraining):")
        for r in results:
            if r['trial_changed']:
                print(f"  - {r['symbol']}_{r['horizon']}d")
    
    # List errors
    if errors > 0:
        print(f"\n❌ Symbols with errors:")
        for r in results:
            if r['error']:
                print(f"  - {r['symbol']}_{r['horizon']}d: {r['error']}")


if __name__ == '__main__':
    main()

