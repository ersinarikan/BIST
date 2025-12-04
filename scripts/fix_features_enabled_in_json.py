#!/usr/bin/env python3
"""
Fix features_enabled in JSON files based on model_choice
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict
import shutil

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

from scripts.continuous_hpo_training_pipeline import STATE_FILE  # noqa: E402


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def fix_features_enabled(json_file: Path, dry_run: bool = False) -> bool:
    """Fix features_enabled in JSON file based on model_choice"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        best_params = data.get('best_params', {})
        model_choice = best_params.get('model_choice', 'unknown')
        
        if model_choice == 'unknown':
            print("  ⚠️ model_choice unknown, skipping")
            return False
        
        # Get feature flags from best_params
        feature_flags = {k: v for k, v in best_params.items() if k.startswith('enable_')}
        
        # Create correct features_enabled
        features_enabled = {
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
            'ENABLE_XGBOOST': '1' if model_choice in ('xgb', 'all') else '0',
            'ENABLE_LIGHTGBM': '1' if model_choice in ('lgbm', 'all') else '0',
            'ENABLE_CATBOOST': '1' if model_choice in ('cat', 'all') else '0',
        }
        
        # Check if update is needed
        current_features = data.get('features_enabled', {})
        needs_update = False
        
        if not current_features:
            needs_update = True
        else:
            for key, value in features_enabled.items():
                if current_features.get(key) != value:
                    needs_update = True
                    break
        
        if not needs_update:
            return False
        
        # Create backup
        if not dry_run:
            backup_file = json_file.with_suffix('.json.backup_features_enabled')
            if not backup_file.exists():
                shutil.copy2(json_file, backup_file)
                print(f"  ✅ Backup created: {backup_file.name}")
        
        # Update features_enabled
        data['features_enabled'] = features_enabled
        
        # Add metadata
        if '_updated_at' not in data:
            from datetime import datetime
            data['_updated_at'] = datetime.now().isoformat()
        data['_updated_reason'] = data.get('_updated_reason', '') + '; features_enabled fixed'
        
        if dry_run:
            print(f"  [DRY-RUN] Would update: {json_file.name}")
            print(f"    model_choice: {model_choice}")
            print(f"    ENABLE_XGBOOST: {features_enabled['ENABLE_XGBOOST']}")
            print(f"    ENABLE_LIGHTGBM: {features_enabled['ENABLE_LIGHTGBM']}")
            print(f"    ENABLE_CATBOOST: {features_enabled['ENABLE_CATBOOST']}")
            return True
        
        # Write updated JSON
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✅ Updated: {json_file.name}")
        print(f"    model_choice: {model_choice}")
        print(f"    ENABLE_XGBOOST: {features_enabled['ENABLE_XGBOOST']}")
        print(f"    ENABLE_LIGHTGBM: {features_enabled['ENABLE_LIGHTGBM']}")
        print(f"    ENABLE_CATBOOST: {features_enabled['ENABLE_CATBOOST']}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Fix features_enabled in JSON files')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to fix (default: all completed)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to fix (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be fixed')
    parser.add_argument('--all', action='store_true',
                       help='Fix all JSON files, not just completed symbols')
    
    args = parser.parse_args()
    
    # Load state
    state = load_state()
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    results_dir = Path('/opt/bist-pattern/results')
    
    if args.all:
        # Fix all JSON files
        pattern = f"optuna_pilot_features_on_h*_c{current_cycle}_*.json"
        json_files = list(results_dir.glob(pattern))
        symbols_to_fix = None
    else:
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
                    except Exception:
                        continue
                else:
                    continue
            
            if horizon in args.horizons:
                if args.symbols:
                    if symbol in args.symbols:
                        completed_symbols.append((symbol, horizon))
                else:
                    completed_symbols.append((symbol, horizon))
        
        symbols_to_fix = {s for s, h in completed_symbols}
        
        # Find JSON files
        pattern = f"optuna_pilot_features_on_h{args.horizons[0]}_c{current_cycle}_*.json"
        json_files = list(results_dir.glob(pattern))
    
    print("📊 Fixing features_enabled in JSON files")
    print(f"🔄 Cycle: {current_cycle}")
    print(f"🔍 Dry-run: {args.dry_run}")
    if symbols_to_fix:
        print(f"📋 Symbols: {len(symbols_to_fix)}")
    print()
    
    fixed = 0
    skipped = 0
    failed = 0
    
    for json_file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            symbols = data.get('symbols', [])
            if not symbols:
                continue
            
            # Skip if not in symbols_to_fix
            if symbols_to_fix and not any(s in symbols_to_fix for s in symbols):
                continue
            
            symbol = symbols[0]
            print(f"\n🔍 {symbol}: {json_file.name}")
            
            if fix_features_enabled(json_file, args.dry_run):
                fixed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Fixed: {fixed}")
    print(f"⏭️ Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(json_files)}")


if __name__ == '__main__':
    main()
