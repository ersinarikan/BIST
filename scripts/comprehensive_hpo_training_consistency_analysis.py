#!/usr/bin/env python3
"""
Comprehensive analysis of HPO and Training consistency:
1. Check all Python files used in HPO process
2. Analyze filter consistency between HPO and Training
3. Analyze all possible scenarios (filter mismatch, best trial changes, etc.)
4. Determine if previous trainings are invalid
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

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

from scripts.continuous_hpo_training_pipeline import STATE_FILE  # noqa: E402
from scripts.retrain_high_discrepancy_symbols import (  # noqa: E402
    find_study_db
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


def analyze_trial_with_filters(db_file: Path, symbol: str, horizon: int, 
                               trial_number: int) -> Dict:
    """Analyze a specific trial with different filters"""
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        trial = None
        for t in study.trials:
            if t.number == trial_number:
                trial = t
                break
        
        if not trial or trial.state != optuna.trial.TrialState.COMPLETE:
            return {'error': 'Trial not found or incomplete'}
        
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = trial.user_attrs.get('symbol_metrics', {})
        if symbol_key not in symbol_metrics:
            return {'error': 'Symbol metrics not found'}
        
        split_metrics = symbol_metrics[symbol_key].get('split_metrics', [])
        if not split_metrics:
            return {'error': 'Split metrics not found'}
        
        # Analyze with different filters
        results = {
            'trial_number': trial_number,
            'trial_value': trial.value,
            'total_splits': len(split_metrics),
            'filters': {}
        }
        
        # Filter 0: No filter (0/0.0)
        dirhits_0_0 = [s.get('dirhit') for s in split_metrics if s.get('dirhit') is not None]
        results['filters']['0_0'] = {
            'min_mask_count': 0,
            'min_mask_pct': 0.0,
            'included_splits': len(dirhits_0_0),
            'avg_dirhit': sum(dirhits_0_0) / len(dirhits_0_0) if dirhits_0_0 else None
        }
        
        # Filter 1: 10/5.0
        dirhits_10_5 = []
        for s in split_metrics:
            dirhit = s.get('dirhit')
            mask_count = s.get('mask_count', 0)
            mask_pct = s.get('mask_pct', 0.0)
            if dirhit is not None and mask_count >= 10 and mask_pct >= 5.0:
                dirhits_10_5.append(dirhit)
        
        results['filters']['10_5'] = {
            'min_mask_count': 10,
            'min_mask_pct': 5.0,
            'included_splits': len(dirhits_10_5),
            'avg_dirhit': sum(dirhits_10_5) / len(dirhits_10_5) if dirhits_10_5 else None
        }
        
        return results
    except Exception as e:
        return {'error': str(e)}


def find_best_trial_for_each_filter(db_file: Path, symbol: str, horizon: int) -> Dict:
    """Find best trial for each filter scenario"""
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        symbol_key = f"{symbol}_{horizon}d"
        
        # Scenario 1: No filter (0/0.0)
        best_trial_0_0 = None
        best_score_0_0 = float('-inf')
        
        # Scenario 2: 10/5.0 filter
        best_trial_10_5 = None
        best_score_10_5 = float('-inf')
        
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            
            symbol_metrics = trial.user_attrs.get('symbol_metrics', {})
            if symbol_key not in symbol_metrics:
                continue
            
            split_metrics = symbol_metrics[symbol_key].get('split_metrics', [])
            if not split_metrics:
                continue
            
            # Filter 0/0.0: All splits
            dirhits_0_0 = [s.get('dirhit') for s in split_metrics if s.get('dirhit') is not None]
            if dirhits_0_0:
                avg_0_0 = sum(dirhits_0_0) / len(dirhits_0_0)
                if avg_0_0 > best_score_0_0:
                    best_score_0_0 = avg_0_0
                    best_trial_0_0 = trial.number
            
            # Filter 10/5.0: Only splits with sufficient support
            dirhits_10_5 = []
            for s in split_metrics:
                dirhit = s.get('dirhit')
                mask_count = s.get('mask_count', 0)
                mask_pct = s.get('mask_pct', 0.0)
                if dirhit is not None and mask_count >= 10 and mask_pct >= 5.0:
                    dirhits_10_5.append(dirhit)
            
            if dirhits_10_5:
                avg_10_5 = sum(dirhits_10_5) / len(dirhits_10_5)
                if avg_10_5 > best_score_10_5:
                    best_score_10_5 = avg_10_5
                    best_trial_10_5 = trial.number
        
        return {
            'filter_0_0': {
                'best_trial': best_trial_0_0,
                'best_score': best_score_0_0 if best_trial_0_0 else None
            },
            'filter_10_5': {
                'best_trial': best_trial_10_5,
                'best_score': best_score_10_5 if best_trial_10_5 else None
            }
        }
    except Exception as e:
        return {'error': str(e)}


def analyze_symbol(symbol: str, horizon: int, cycle: int) -> Dict:
    """Comprehensive analysis for a symbol"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'cycle': cycle,
        'json_found': False,
        'study_found': False,
        'scenarios': {},
        'issues': [],
        'warnings': []
    }
    
    # Find JSON file
    json_file = find_json_file(symbol, horizon, cycle)
    if not json_file:
        result['issues'].append('JSON file not found')
        return result
    
    result['json_found'] = True
    
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        json_best_trial = json_data.get('best_trial_number')
        json_best_dirhit = json_data.get('best_dirhit')
        json_eval_spec = json_data.get('evaluation_spec', {})
        json_filter_mc = json_eval_spec.get('min_mask_count', 0)
        json_filter_mp = json_eval_spec.get('min_mask_pct', 0.0)
        
        # Find study DB
        db_file = find_study_db(symbol, horizon, cycle)
        if not db_file:
            result['issues'].append('Study DB not found')
            return result
        
        result['study_found'] = True
        
        # Find best trial for each filter
        filter_analysis = find_best_trial_for_each_filter(db_file, symbol, horizon)
        if 'error' in filter_analysis:
            result['issues'].append(f"Filter analysis error: {filter_analysis['error']}")
            return result
        
        # Scenario analysis
        best_trial_0_0 = filter_analysis['filter_0_0']['best_trial']
        best_trial_10_5 = filter_analysis['filter_10_5']['best_trial']
        best_score_0_0 = filter_analysis['filter_0_0']['best_score']
        best_score_10_5 = filter_analysis['filter_10_5']['best_score']
        
        # Scenario 1: HPO ran with 0/0.0, JSON has 0/0.0, best trial matches
        if json_filter_mc == 0 and abs(json_filter_mp - 0.0) < 0.01:
            if json_best_trial == best_trial_0_0:
                result['scenarios']['scenario_1'] = {
                    'description': 'HPO 0/0.0, JSON 0/0.0, best trial matches',
                    'status': 'OK',
                    'hpo_filter': '0/0.0',
                    'json_filter': '0/0.0',
                    'best_trial': json_best_trial,
                    'best_dirhit': json_best_dirhit
                }
            else:
                result['scenarios']['scenario_1'] = {
                    'description': 'HPO 0/0.0, JSON 0/0.0, but best trial mismatch',
                    'status': 'WARNING',
                    'hpo_filter': '0/0.0',
                    'json_filter': '0/0.0',
                    'json_best_trial': json_best_trial,
                    'actual_best_trial_0_0': best_trial_0_0,
                    'best_dirhit': json_best_dirhit
                }
                result['warnings'].append('Best trial mismatch for 0/0.0 filter')
        
        # Scenario 2: HPO ran with 0/0.0, but JSON has 10/5.0 (mismatch)
        elif json_filter_mc == 10 and abs(json_filter_mp - 5.0) < 0.01:
            if best_trial_0_0 == json_best_trial:
                # HPO ran with 0/0.0, found best trial, but JSON says 10/5.0
                result['scenarios']['scenario_2'] = {
                    'description': 'HPO 0/0.0, JSON 10/5.0 (mismatch)',
                    'status': 'ISSUE',
                    'hpo_filter': '0/0.0',
                    'json_filter': '10/5.0',
                    'json_best_trial': json_best_trial,
                    'best_trial_0_0': best_trial_0_0,
                    'best_trial_10_5': best_trial_10_5,
                    'best_score_0_0': best_score_0_0,
                    'best_score_10_5': best_score_10_5
                }
                result['issues'].append('Filter mismatch: HPO used 0/0.0 but JSON has 10/5.0')
                
                # Check if best trial would be different with 10/5.0 filter
                if best_trial_10_5 and best_trial_10_5 != json_best_trial:
                    result['warnings'].append(f'Best trial would be {best_trial_10_5} with 10/5.0 filter (JSON has {json_best_trial})')
            else:
                # JSON was updated with 10/5.0 filter
                if json_best_trial == best_trial_10_5:
                    result['scenarios']['scenario_2b'] = {
                        'description': 'HPO 0/0.0, JSON updated to 10/5.0, best trial matches 10/5.0',
                        'status': 'OK',
                        'hpo_filter': '0/0.0',
                        'json_filter': '10/5.0',
                        'best_trial': json_best_trial,
                        'best_dirhit': json_best_dirhit
                    }
                else:
                    result['scenarios']['scenario_2c'] = {
                        'description': 'HPO 0/0.0, JSON updated to 10/5.0, but best trial still mismatch',
                        'status': 'WARNING',
                        'hpo_filter': '0/0.0',
                        'json_filter': '10/5.0',
                        'json_best_trial': json_best_trial,
                        'actual_best_trial_10_5': best_trial_10_5
                    }
                    result['warnings'].append('Best trial mismatch even after filter update')
        
        # Scenario 3: HPO ran with 10/5.0, JSON has 10/5.0
        # (This would be if HPO was run with 10/5.0 from the start)
        # We can't determine this from JSON alone, but we can check if best_trial_10_5 matches
        
        # Analyze JSON best trial with both filters
        if json_best_trial is not None:
            trial_analysis = analyze_trial_with_filters(db_file, symbol, horizon, json_best_trial)
            if 'error' not in trial_analysis:
                result['json_best_trial_analysis'] = trial_analysis
        
    except Exception as e:
        result['issues'].append(f"Error: {e}")
        import traceback
        result['issues'].append(traceback.format_exc())
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Comprehensive HPO-Training consistency analysis')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to analyze (default: all completed)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to analyze (default: 1)')
    parser.add_argument('--output', type=str,
                       help='Output JSON file for detailed results')
    
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
    
    print("=" * 80)
    print("COMPREHENSIVE HPO-TRAINING CONSISTENCY ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Analyzing {len(completed_symbols)} completed symbols")
    print(f"🔄 Cycle: {current_cycle}")
    print()
    
    # Analyze each symbol
    all_results = []
    for symbol, horizon in sorted(completed_symbols):
        print(f"🔍 Analyzing {symbol}_{horizon}d...")
        result = analyze_symbol(symbol, horizon, current_cycle)
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = len(all_results)
    json_found = sum(1 for r in all_results if r.get('json_found'))
    study_found = sum(1 for r in all_results if r.get('study_found'))
    has_issues = sum(1 for r in all_results if r.get('issues'))
    has_warnings = sum(1 for r in all_results if r.get('warnings'))
    
    print(f"\n📊 Total symbols: {total}")
    print(f"✅ JSON found: {json_found}/{total}")
    print(f"✅ Study found: {study_found}/{total}")
    print(f"❌ Has issues: {has_issues}/{total}")
    print(f"⚠️ Has warnings: {has_warnings}/{total}")
    
    # Scenario breakdown
    scenario_counts = {}
    for r in all_results:
        for scenario_key, scenario_data in r.get('scenarios', {}).items():
            status = scenario_data.get('status', 'UNKNOWN')
            scenario_counts[status] = scenario_counts.get(status, 0) + 1
    
    print("\n📊 Scenario Status:")
    for status, count in sorted(scenario_counts.items()):
        print(f"   {status}: {count}")
    
    # Symbols with issues
    if has_issues:
        print(f"\n❌ SYMBOLS WITH ISSUES ({has_issues}):")
        for r in all_results:
            if r.get('issues'):
                print(f"\n   {r['symbol']}_{r['horizon']}d:")
                for issue in r['issues']:
                    print(f"      ❌ {issue}")
                for warning in r.get('warnings', []):
                    print(f"      ⚠️ {warning}")
    
    # Save detailed results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Detailed results saved to: {args.output}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
