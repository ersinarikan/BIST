#!/usr/bin/env python3
"""
Update JSON files with best params found using 10/5.0 filter
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import shutil

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

from scripts.retrain_high_discrepancy_symbols import (
    find_study_db,
    find_best_trial_with_filter_applied
)


def find_json_file(symbol: str, horizon: int, cycle: int) -> Optional[Path]:
    """Find HPO JSON file for symbol-horizon"""
    results_dir = Path('/opt/bist-pattern/results')
    
    pattern = f"optuna_pilot_features_on_h{horizon}_c{cycle}_*.json"
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


def update_json_with_filtered_trial(json_file: Path, symbol: str, horizon: int,
                                    filtered_trial: optuna.Trial, filtered_score: float,
                                    min_mask_count: int, min_mask_pct: float,
                                    dry_run: bool = False) -> bool:
    """Update JSON file with filtered best trial"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create backup
        if not dry_run:
            backup_file = json_file.with_suffix('.json.backup_10_5_filter')
            if not backup_file.exists():
                shutil.copy2(json_file, backup_file)
                print(f"  ✅ Backup created: {backup_file.name}")
        
        # Get params from filtered trial
        best_params = filtered_trial.params.copy()
        
        # Get features from filtered trial
        features_enabled = {}
        feature_params = {}
        hyperparameters = {}
        
        if hasattr(filtered_trial, 'user_attrs'):
            features_enabled = filtered_trial.user_attrs.get('features_enabled', {})
            feature_params = filtered_trial.user_attrs.get('feature_params', {})
        
        # Feature flags (enable_*)
        feature_flags = {k: v for k, v in best_params.items() if k.startswith('enable_')}
        
        # Feature params
        feature_params_keys = [
            'external_min_days', 'external_smooth_alpha',
            'yolo_min_conf',
            'ml_loss_mse_weight', 'ml_loss_threshold', 'ml_dir_penalty',
            'n_seeds', 'meta_stacking_alpha',
        ]
        feature_params_keys += [k for k in best_params.keys() if k.startswith('ml_adaptive_k_') or k.startswith('ml_pattern_weight_scale_')]
        feature_params = {k: v for k, v in best_params.items() if k in feature_params_keys}
        
        # Hyperparameters
        hyperparameters = {k: v for k, v in best_params.items() if not k.startswith('enable_') and k not in feature_params_keys}
        
        # Update JSON
        data['best_params'] = best_params
        data['best_trial_number'] = int(filtered_trial.number)
        data['best_dirhit'] = float(filtered_score)
        data['best_value'] = float(filtered_score)  # Use filtered score as best value
        
        if 'best_trial' in data:
            data['best_trial']['number'] = int(filtered_trial.number)
            data['best_trial']['value'] = float(filtered_score)
        
        data['feature_flags'] = feature_flags
        data['feature_params'] = feature_params
        data['hyperparameters'] = hyperparameters
        
        # Update features_enabled
        data['features_enabled'] = {
            'ENABLE_EXTERNAL_FEATURES': '1' if feature_flags.get('enable_external_features', False) else '0',
            'ENABLE_FINGPT_FEATURES': '1' if feature_flags.get('enable_fingpt_features', False) else '0',
            'ENABLE_YOLO_FEATURES': '1' if feature_flags.get('enable_yolo_features', False) else '0',
            'ML_USE_DIRECTIONAL_LOSS': '1' if feature_flags.get('enable_directional_loss', False) else '0',
            'ENABLE_SEED_BAGGING': '1' if feature_flags.get('enable_seed_bagging', False) else '0',
            'ENABLE_TALIB_PATTERNS': '1' if feature_flags.get('enable_talib_patterns', False) else '0',
            'ML_USE_SMART_ENSEMBLE': '1' if feature_flags.get('enable_smart_ensemble', False) else '0',
            'ML_USE_STACKED_SHORT': '1' if feature_flags.get('enable_stacked_short', False) else '0',
            'ENABLE_META_STACKING': '1' if feature_flags.get('enable_meta_stacking', False) else '0',
            'ML_USE_REGIME_DETECTION': '1' if feature_flags.get('enable_regime_detection', False) else '0',
            'ENABLE_FINGPT': '1' if feature_flags.get('enable_fingpt', False) else '0',
            'ML_USE_ADAPTIVE_LEARNING': '0',
            'ENABLE_XGBOOST': '1' if best_params.get('model_choice') in ('xgb', 'all') else '0',
            'ENABLE_LIGHTGBM': '1' if best_params.get('model_choice') in ('lgbm', 'all') else '0',
            'ENABLE_CATBOOST': '1' if best_params.get('model_choice') in ('cat', 'all') else '0',
        }
        
        # Update evaluation_spec with filter
        data['evaluation_spec'] = {
            'horizon': int(horizon),
            'dirhit_threshold': data.get('evaluation_spec', {}).get('dirhit_threshold', 0.005),
            'min_mask_count': int(min_mask_count),
            'min_mask_pct': float(min_mask_pct),
            'best_trial_number': int(filtered_trial.number),
            'best_trial_seed': int(42 + filtered_trial.number),
        }
        
        # Add metadata
        data['_updated_at'] = datetime.now().isoformat()
        data['_updated_reason'] = f'Updated with 10/5.0 filter - best trial #{filtered_trial.number}'
        data['_filter_applied'] = {'min_mask_count': min_mask_count, 'min_mask_pct': min_mask_pct}
        
        if dry_run:
            print(f"  [DRY-RUN] Would update: {json_file.name}")
            print(f"    - Old best trial: #{data.get('best_trial_number', 'N/A')}")
            print(f"    - New best trial: #{filtered_trial.number}")
            print(f"    - New best DirHit: {filtered_score:.2f}%")
            return True
        
        # Write updated JSON
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✅ Updated: {json_file.name}")
        print(f"    - Best trial: #{filtered_trial.number} (was #{data.get('best_trial_number', 'N/A')})")
        print(f"    - Best DirHit: {filtered_score:.2f}%")
        print(f"    - Filter: min_mask_count={min_mask_count}, min_mask_pct={min_mask_pct}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error updating JSON: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_symbol(symbol: str, horizon: int, cycle: int, min_mask_count: int, min_mask_pct: float, dry_run: bool = False) -> bool:
    """Update JSON for a single symbol with filtered best trial"""
    print(f"\n{'='*80}")
    print(f"🔍 {symbol}_{horizon}d")
    print(f"{'='*80}")
    
    # Find study DB
    db_file = find_study_db(symbol, horizon, cycle)
    if not db_file:
        print(f"  ❌ Study DB not found")
        return False
    
    print(f"  ✅ Study DB: {db_file.name}")
    
    # Find best trial with filter applied
    print(f"  🔍 Finding best trial with filter (min_mask_count={min_mask_count}, min_mask_pct={min_mask_pct})...")
    filtered_trial, filtered_score = find_best_trial_with_filter_applied(
        db_file, symbol, horizon, min_mask_count, min_mask_pct
    )
    
    if not filtered_trial:
        print(f"  ❌ No valid trial found with filter")
        return False
    
    print(f"  ✅ Found filtered best trial: #{filtered_trial.number} (DirHit: {filtered_score:.2f}%)")
    
    # Find JSON file
    json_file = find_json_file(symbol, horizon, cycle)
    if not json_file:
        print(f"  ❌ JSON file not found")
        return False
    
    print(f"  ✅ JSON file: {json_file.name}")
    
    # Update JSON
    success = update_json_with_filtered_trial(
        json_file, symbol, horizon, filtered_trial, filtered_score,
        min_mask_count, min_mask_pct, dry_run
    )
    
    return success


def main():
    parser = argparse.ArgumentParser(description='Update JSON files with 10/5.0 filter')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Symbols to update (default: ALVES BESLR BIGEN BULGS)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to update (default: 1)')
    parser.add_argument('--min-mask-count', type=int, default=10,
                       help='Min mask count filter (default: 10)')
    parser.add_argument('--min-mask-pct', type=float, default=5.0,
                       help='Min mask pct filter (default: 5.0)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be updated')
    parser.add_argument('--cycle', type=int, default=None,
                       help='Cycle number (default: from state file)')
    
    args = parser.parse_args()
    
    # Get cycle
    if args.cycle is None:
        from scripts.continuous_hpo_training_pipeline import STATE_FILE
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            cycle = state.get('cycle', 1)
        else:
            cycle = 1
    else:
        cycle = args.cycle
    
    # Default symbols
    if not args.symbols:
        symbols = ['ALVES', 'BESLR', 'BIGEN', 'BULGS']
    else:
        symbols = args.symbols
    
    print(f"📊 Updating {len(symbols)} symbols")
    print(f"🔄 Cycle: {cycle}")
    print(f"🔍 Filter: min_mask_count={args.min_mask_count}, min_mask_pct={args.min_mask_pct}")
    print(f"🔍 Dry-run: {args.dry_run}")
    
    # Update each symbol
    updated = 0
    failed = 0
    
    for symbol in symbols:
        for horizon in args.horizons:
            success = update_symbol(symbol, horizon, cycle, args.min_mask_count, args.min_mask_pct, args.dry_run)
            if success:
                updated += 1
            else:
                failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Updated: {updated}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(symbols) * len(args.horizons)}")


if __name__ == '__main__':
    main()

