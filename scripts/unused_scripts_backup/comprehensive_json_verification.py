#!/usr/bin/env python3
"""
Comprehensive verification of JSON files:
1. Check best_params from study vs JSON
2. Check features_enabled consistency
3. Check evaluation_spec filter values
4. Check model_choice and feature flags
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
    find_best_trial_with_filter_applied
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


def verify_json_against_study(json_file: Path, symbol: str, horizon: int, cycle: int,
                             min_mask_count: int, min_mask_pct: float) -> Dict:
    """Comprehensive verification of JSON against study"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'json_file': str(json_file),
        'study_found': False,
        'json_found': True,
        'best_trial_match': False,
        'best_params_match': False,
        'features_enabled_match': False,
        'filter_match': False,
        'model_choice_match': False,
        'issues': [],
        'warnings': []
    }
    
    try:
        # Load JSON
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        json_best_trial = json_data.get('best_trial_number')
        json_best_params = json_data.get('best_params', {})
        json_model_choice = json_best_params.get('model_choice', 'unknown')
        json_features_enabled = json_data.get('features_enabled', {})
        json_eval_spec = json_data.get('evaluation_spec', {})
        json_filter_mc = json_eval_spec.get('min_mask_count', 0)
        json_filter_mp = json_eval_spec.get('min_mask_pct', 0.0)
        
        # Find study DB
        db_file = find_study_db(symbol, horizon, cycle)
        if not db_file:
            result['issues'].append('Study DB not found')
            return result
        
        result['study_found'] = True
        
        # Load study
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        # Find best trial with filter
        filtered_trial, filtered_score = find_best_trial_with_filter_applied(
            db_file, symbol, horizon, min_mask_count, min_mask_pct
        )
        
        if not filtered_trial:
            result['issues'].append('No valid trial found with filter')
            return result
        
        study_best_trial = filtered_trial.number
        study_best_params = filtered_trial.params.copy()
        study_model_choice = study_best_params.get('model_choice', 'unknown')
        
        # Check best trial match
        if json_best_trial == study_best_trial:
            result['best_trial_match'] = True
        else:
            result['issues'].append(f"Best trial mismatch: JSON={json_best_trial}, Study={study_best_trial}")
        
        # Check model_choice match
        if json_model_choice == study_model_choice:
            result['model_choice_match'] = True
        else:
            result['issues'].append(f"model_choice mismatch: JSON={json_model_choice}, Study={study_model_choice}")
        
        # Check best_params match (key parameters)
        key_params = ['model_choice'] + [k for k in study_best_params.keys() if k.startswith('enable_')]
        params_match = True
        for key in key_params:
            json_val = json_best_params.get(key)
            study_val = study_best_params.get(key)
            if json_val != study_val:
                params_match = False
                result['warnings'].append(f"Param mismatch '{key}': JSON={json_val}, Study={study_val}")
        
        if params_match:
            result['best_params_match'] = True
        
        # Check features_enabled consistency
        expected_xgb = '1' if study_model_choice in ('xgb', 'all') else '0'
        expected_lgbm = '1' if study_model_choice in ('lgbm', 'all') else '0'
        expected_cat = '1' if study_model_choice in ('cat', 'all') else '0'
        
        json_xgb = json_features_enabled.get('ENABLE_XGBOOST')
        json_lgbm = json_features_enabled.get('ENABLE_LIGHTGBM')
        json_cat = json_features_enabled.get('ENABLE_CATBOOST')
        
        if json_xgb == expected_xgb and json_lgbm == expected_lgbm and json_cat == expected_cat:
            result['features_enabled_match'] = True
        else:
            result['issues'].append(f"features_enabled mismatch: JSON XGB={json_xgb}, LGBM={json_lgbm}, CAT={json_cat}, Expected XGB={expected_xgb}, LGBM={expected_lgbm}, CAT={expected_cat}")
        
        # Check filter match
        if json_filter_mc == min_mask_count and abs(json_filter_mp - min_mask_pct) < 0.01:
            result['filter_match'] = True
        else:
            result['issues'].append(f"Filter mismatch: JSON={json_filter_mc}/{json_filter_mp}, Expected={min_mask_count}/{min_mask_pct}")
        
    except Exception as e:
        result['issues'].append(f"Error: {e}")
        import traceback
        result['issues'].append(traceback.format_exc())
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Comprehensive JSON verification')
    parser.add_argument('--min-mask-count', type=int, default=10,
                       help='Min mask count filter (default: 10)')
    parser.add_argument('--min-mask-pct', type=float, default=5.0,
                       help='Min mask pct filter (default: 5.0)')
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
    
    print(f"📊 Verifying {len(completed_symbols)} completed symbols")
    print(f"🔄 Cycle: {current_cycle}")
    print(f"🔍 Filter: min_mask_count={args.min_mask_count}, min_mask_pct={args.min_mask_pct}")
    print()
    
    # Verify each symbol
    results = []
    for symbol, horizon in sorted(completed_symbols):
        json_file = find_json_file(symbol, horizon, current_cycle)
        if not json_file:
            results.append({
                'symbol': symbol,
                'horizon': horizon,
                'json_found': False,
                'issues': ['JSON file not found']
            })
            continue
        
        result = verify_json_against_study(
            json_file, symbol, horizon, current_cycle,
            args.min_mask_count, args.min_mask_pct
        )
        results.append(result)
    
    # Summary
    print(f"{'='*80}")
    print("📊 VERIFICATION SUMMARY")
    print(f"{'='*80}\n")
    
    all_ok = []
    has_issues = []
    has_warnings = []
    
    for r in results:
        if r.get('issues'):
            has_issues.append(r)
        elif r.get('warnings'):
            has_warnings.append(r)
        else:
            all_ok.append(r)
    
    print(f"✅ All OK: {len(all_ok)}")
    print(f"⚠️ Has warnings: {len(has_warnings)}")
    print(f"❌ Has issues: {len(has_issues)}\n")
    
    if has_issues:
        print(f"❌ SYMBOLS WITH ISSUES ({len(has_issues)}):")
        print(f"{'='*80}")
        for r in has_issues:
            print(f"\n📊 {r['symbol']}_{r['horizon']}d:")
            for issue in r['issues']:
                print(f"   ❌ {issue}")
            if r.get('warnings'):
                for warning in r['warnings']:
                    print(f"   ⚠️ {warning}")
    
    if has_warnings:
        print(f"\n⚠️ SYMBOLS WITH WARNINGS ({len(has_warnings)}):")
        print(f"{'='*80}")
        for r in has_warnings[:10]:  # Show first 10
            print(f"\n📊 {r['symbol']}_{r['horizon']}d:")
            for warning in r['warnings']:
                print(f"   ⚠️ {warning}")
    
    # Detailed statistics
    print(f"\n{'='*80}")
    print("📊 DETAILED STATISTICS")
    print(f"{'='*80}\n")
    
    best_trial_match = sum(1 for r in results if r.get('best_trial_match'))
    best_params_match = sum(1 for r in results if r.get('best_params_match'))
    features_enabled_match = sum(1 for r in results if r.get('features_enabled_match'))
    filter_match = sum(1 for r in results if r.get('filter_match'))
    model_choice_match = sum(1 for r in results if r.get('model_choice_match'))
    
    print(f"✅ Best trial match: {best_trial_match}/{len(results)}")
    print(f"✅ Best params match: {best_params_match}/{len(results)}")
    print(f"✅ Features enabled match: {features_enabled_match}/{len(results)}")
    print(f"✅ Filter match: {filter_match}/{len(results)}")
    print(f"✅ Model choice match: {model_choice_match}/{len(results)}")


if __name__ == '__main__':
    main()

