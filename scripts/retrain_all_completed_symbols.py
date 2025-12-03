#!/usr/bin/env python3
"""
Retrain all completed symbols with updated JSON parameters
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

# Set DATABASE_URL
secret_file_path = '/opt/bist-pattern/.secrets/db_password'
if os.path.exists(secret_file_path):
    with open(secret_file_path, 'r') as f:
        db_password = f.read().strip()
    os.environ['DATABASE_URL'] = f"postgresql://bist_user:{db_password}@127.0.0.1:6432/bist_pattern_db"

from scripts.continuous_hpo_training_pipeline import ContinuousHPOPipeline, STATE_FILE


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


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
    parser = argparse.ArgumentParser(description='Retrain all completed symbols')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to retrain (default: all completed)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to retrain (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be retrained')
    
    args = parser.parse_args()
    
    # Load state
    state = load_state()
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Get completed symbols with models
    completed_symbols = []
    model_dir = Path('/opt/bist-pattern/.cache/enhanced_ml_models')
    
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
            # Check if model exists
            model_patterns = [
                f"{symbol}_h{horizon}_*.pkl",
                f"{symbol}_*_h{horizon}_*.pkl",
                f"{symbol}_1d_*.pkl",
            ]
            
            has_model = False
            for pattern in model_patterns:
                if list(model_dir.glob(pattern)):
                    has_model = True
                    break
            
            if has_model:
                if args.symbols:
                    if symbol in args.symbols:
                        completed_symbols.append((symbol, horizon))
                else:
                    completed_symbols.append((symbol, horizon))
    
    print(f"📊 {len(completed_symbols)} completed symbols with models found")
    print(f"🔄 Cycle: {current_cycle}")
    print(f"🔍 Dry-run: {args.dry_run}")
    print()
    
    if args.dry_run:
        print("Semboller:")
        for sym, h in sorted(completed_symbols):
            print(f"  {sym}_{h}d")
        return
    
    # Retrain each symbol
    pipeline = ContinuousHPOPipeline()
    
    success_count = 0
    failed_count = 0
    
    for symbol, horizon in sorted(completed_symbols):
        print(f"\n{'='*80}")
        print(f"🔄 {symbol}_{horizon}d")
        print(f"{'='*80}")
        
        # Find JSON file
        json_file = find_json_file(symbol, horizon, current_cycle)
        if not json_file:
            print(f"  ❌ JSON file not found")
            failed_count += 1
            continue
        
        print(f"  ✅ JSON file: {json_file.name}")
        
        try:
            # Load best params from JSON
            with open(json_file, 'r') as f:
                hpo_result = json.load(f)
            
            best_params = hpo_result.get('best_params', {})
            best_trial = hpo_result.get('best_trial_number')
            best_dirhit = hpo_result.get('best_dirhit')
            eval_spec = hpo_result.get('evaluation_spec', {})
            
            # ✅ Ensure evaluation_spec is in hpo_result for filter application
            if eval_spec and 'min_mask_count' not in hpo_result:
                hpo_result['evaluation_spec'] = eval_spec
            
            print(f"  ✅ Best trial: #{best_trial}")
            print(f"  ✅ Best DirHit: {best_dirhit}")
            print(f"  ✅ Filter: min_mask_count={eval_spec.get('min_mask_count')}, min_mask_pct={eval_spec.get('min_mask_pct')}")
            print(f"  ✅ model_choice: {best_params.get('model_choice')}")
            
            # ✅ Verify features_enabled is in best_params
            features_enabled = hpo_result.get('features_enabled', {})
            if features_enabled and 'features_enabled' not in best_params:
                best_params['features_enabled'] = features_enabled
                print(f"  ✅ Features enabled: {len(features_enabled)} flags")
            
            # Run training
            result = pipeline.run_training(symbol, horizon, best_params, hpo_result)
            if result:
                print(f"  ✅ Training completed")
                for h, dirhit in result.items():
                    if dirhit:
                        print(f"    - {h}d: DirHit={dirhit:.2f}%")
                success_count += 1
            else:
                print(f"  ⚠️ Training returned None")
                failed_count += 1
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Total: {len(completed_symbols)}")


if __name__ == '__main__':
    main()

