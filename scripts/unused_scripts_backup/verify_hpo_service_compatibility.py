#!/usr/bin/env python3
"""
Verify HPO service compatibility - check state, JSON, and study files are consistent
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

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
from scripts.retrain_high_discrepancy_symbols import find_study_db


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def verify_symbol(symbol: str, horizon: int, cycle: int) -> Dict:
    """Verify a single symbol's compatibility"""
    result = {
        'symbol': symbol,
        'horizon': horizon,
        'key': f"{symbol}_{horizon}d",
        'status': 'unknown',
        'issues': [],
        'warnings': [],
        'checks': {}
    }
    
    # 1. Check state file
    state = load_state()
    tasks = state.get('state', {})
    task = tasks.get(result['key'], {})
    
    if not isinstance(task, dict):
        result['issues'].append("Task not found in state")
        result['status'] = 'error'
        return result
    
    result['checks']['state_exists'] = True
    result['checks']['state_status'] = task.get('status', 'unknown')
    result['checks']['state_cycle'] = task.get('cycle', 0)
    
    # 2. Check JSON file
    best_params_file = task.get('best_params_file')
    if not best_params_file:
        result['issues'].append("best_params_file not set in state")
        result['status'] = 'error'
        return result
    
    json_file = Path(best_params_file)
    if not json_file.exists():
        result['issues'].append(f"JSON file not found: {json_file}")
        result['status'] = 'error'
        return result
    
    result['checks']['json_exists'] = True
    result['checks']['json_file'] = str(json_file)
    
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        # Check JSON structure
        result['checks']['json_has_best_params'] = 'best_params' in json_data
        result['checks']['json_has_best_dirhit'] = 'best_dirhit' in json_data
        result['checks']['json_has_best_trial_number'] = 'best_trial_number' in json_data
        result['checks']['json_has_evaluation_spec'] = 'evaluation_spec' in json_data
        result['checks']['json_has_features_enabled'] = 'features_enabled' in json_data
        
        # Check filter in JSON
        eval_spec = json_data.get('evaluation_spec', {})
        json_filter_mc = eval_spec.get('min_mask_count', 0)
        json_filter_mp = eval_spec.get('min_mask_pct', 0.0)
        result['checks']['json_filter'] = f"{json_filter_mc}/{json_filter_mp}"
        
        # Check best_dirhit
        json_best_dirhit = json_data.get('best_dirhit')
        result['checks']['json_best_dirhit'] = json_best_dirhit
        
        # Check best_trial_number
        json_best_trial = json_data.get('best_trial_number')
        result['checks']['json_best_trial'] = json_best_trial
        
    except Exception as e:
        result['issues'].append(f"Error reading JSON: {e}")
        result['status'] = 'error'
        return result
    
    # 3. Check study database
    db_file = find_study_db(symbol, horizon, cycle)
    if not db_file:
        result['issues'].append("Study database not found")
        result['status'] = 'error'
        return result
    
    result['checks']['study_exists'] = True
    result['checks']['study_file'] = str(db_file)
    
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        # Check if best trial exists
        if study.best_trial is None:
            result['issues'].append("No best trial in study")
            result['status'] = 'error'
            return result
        
        result['checks']['study_has_best_trial'] = True
        result['checks']['study_best_trial'] = study.best_trial.number
        
        # Check if JSON's best_trial_number matches study's best_trial
        if json_best_trial != study.best_trial.number:
            result['warnings'].append(
                f"Best trial mismatch: JSON={json_best_trial}, Study={study.best_trial.number}"
            )
        
        # Check filter consistency
        symbol_key = f"{symbol}_{horizon}d"
        symbol_metrics = study.best_trial.user_attrs.get('symbol_metrics', {})
        symbol_metric = symbol_metrics.get(symbol_key, {}) if isinstance(symbol_metrics, dict) else {}
        split_metrics = symbol_metric.get('split_metrics', []) if isinstance(symbol_metric, dict) else []
        
        if split_metrics:
            first_split = split_metrics[0]
            study_filter_mc = first_split.get('min_mask_count', 0)
            study_filter_mp = first_split.get('min_mask_pct', 0.0)
            result['checks']['study_filter'] = f"{study_filter_mc}/{study_filter_mp}"
            
            if json_filter_mc != study_filter_mc or json_filter_mp != study_filter_mp:
                result['warnings'].append(
                    f"Filter mismatch: JSON={json_filter_mc}/{json_filter_mp}, "
                    f"Study={study_filter_mc}/{study_filter_mp}"
                )
        
        # Check state vs JSON consistency
        state_hpo_dirhit = task.get('hpo_dirhit')
        if state_hpo_dirhit is not None:
            result['checks']['state_hpo_dirhit'] = state_hpo_dirhit
            if abs(state_hpo_dirhit - json_best_dirhit) > 0.01:
                result['warnings'].append(
                    f"HPO DirHit mismatch: State={state_hpo_dirhit:.2f}%, JSON={json_best_dirhit:.2f}%"
                )
        
        state_best_trial = task.get('best_trial_number')
        if state_best_trial is not None:
            result['checks']['state_best_trial'] = state_best_trial
            if state_best_trial != json_best_trial:
                result['warnings'].append(
                    f"Best trial mismatch: State={state_best_trial}, JSON={json_best_trial}"
                )
        
    except Exception as e:
        result['issues'].append(f"Error reading study: {e}")
        result['status'] = 'error'
        return result
    
    # Determine final status
    if result['issues']:
        result['status'] = 'error'
    elif result['warnings']:
        result['status'] = 'warning'
    else:
        result['status'] = 'ok'
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Verify HPO service compatibility')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to verify (default: all completed)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to verify (default: 1)')
    parser.add_argument('--json-output', type=str,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Load state
    state = load_state()
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Get symbols to verify
    if args.symbols:
        symbols_to_verify = [(s, h) for s in args.symbols for h in args.horizons]
    else:
        # Get all completed symbols
        symbols_to_verify = []
        for key, task in tasks.items():
            if not isinstance(task, dict):
                continue
            if task.get('status') == 'completed' and task.get('cycle', 0) == current_cycle:
                parts = key.split('_')
                if len(parts) == 2:
                    symbol = parts[0]
                    try:
                        horizon = int(parts[1].replace('d', ''))
                        if horizon in args.horizons:
                            symbols_to_verify.append((symbol, horizon))
                    except:
                        pass
        
        symbols_to_verify = list(set(symbols_to_verify))
    
    print("=" * 80)
    print("HPO SERVİSİ UYUMLULUK KONTROLÜ")
    print("=" * 80)
    print(f"\n🔄 Cycle: {current_cycle}")
    print(f"📊 Kontrol edilecek sembol sayısı: {len(symbols_to_verify)}")
    
    results = []
    ok_count = 0
    warning_count = 0
    error_count = 0
    
    for symbol, horizon in sorted(symbols_to_verify):
        result = verify_symbol(symbol, horizon, current_cycle)
        results.append(result)
        
        status_icon = {
            'ok': '✅',
            'warning': '⚠️',
            'error': '❌',
            'unknown': '❓'
        }.get(result['status'], '❓')
        
        print(f"\n{status_icon} {result['key']}: {result['status'].upper()}")
        
        if result['issues']:
            for issue in result['issues']:
                print(f"   ❌ {issue}")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"   ⚠️ {warning}")
        
        if result['status'] == 'ok':
            ok_count += 1
        elif result['status'] == 'warning':
            warning_count += 1
        else:
            error_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 ÖZET")
    print(f"{'='*80}")
    print(f"✅ OK: {ok_count}")
    print(f"⚠️  WARNING: {warning_count}")
    print(f"❌ ERROR: {error_count}")
    print(f"📊 Toplam: {len(symbols_to_verify)}")
    
    # Save JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump({
                'cycle': current_cycle,
                'summary': {
                    'ok': ok_count,
                    'warning': warning_count,
                    'error': error_count,
                    'total': len(symbols_to_verify)
                },
                'results': results
            }, f, indent=2)
        print(f"\n✅ Results saved to: {args.json_output}")


if __name__ == '__main__':
    main()

