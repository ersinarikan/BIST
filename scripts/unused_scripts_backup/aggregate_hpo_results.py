#!/usr/bin/env python3
"""
Aggregate HPO results across multiple symbols and horizons.

Reads all /opt/bist-pattern/results/optuna_pilot_h{horizon}_{symbol}_{timestamp}.json files
and creates a consolidated per-horizon aggregation.
"""

import json
from pathlib import Path
from typing import Dict
from collections import defaultdict
import numpy as np


def aggregate_per_horizon(results_dir: str = '/opt/bist-pattern/results') -> Dict:
    """
    Aggregate HPO results per horizon.
    
    Returns:
        {
            '1d': {
                'symbols': ['TUPRS', 'VAKBN', ...],
                'best_params': {...},  # averaged across symbols
                'aggr_dir_hit': 65.8,
                'per_symbol_results': {...}
            },
            '3d': {...},
            ...
        }
    """
    results_dir_path = Path(results_dir)
    per_horizon = defaultdict(list)
    
    # Collect all HPO results
    for json_file in sorted(results_dir_path.glob('optuna_pilot_h*_*.json')):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            horizon = data.get('horizon', 0)
            symbol = data.get('symbols', ['UNKNOWN'])[0]
            best_value = data.get('best_value', 0.0)
            best_params = data.get('best_params', {})
            # ✅ FIX: Also read separate structures if available
            best_params_xgb = data.get('best_params_xgb', {})
            best_params_lgb = data.get('best_params_lgb', {})
            best_params_cat = data.get('best_params_cat', {})
            
            per_horizon[horizon].append({
                'symbol': symbol,
                'dir_hit': best_value,
                'params': best_params,
                'params_xgb': best_params_xgb,  # Separate structures
                'params_lgb': best_params_lgb,
                'params_cat': best_params_cat,
                'source_file': str(json_file)
            })
            
            print(f"📊 Loaded: H={horizon}d, Symbol={symbol}, DirHit={best_value:.2f}%")
            
        except Exception as e:
            print(f"⚠️  Failed to load {json_file}: {e}")
    
    # Aggregate per horizon
    output = {}
    for horizon, trials in per_horizon.items():
        if not trials:
            continue
            
        horizon_key = f'{horizon}d'
        symbols = [t['symbol'] for t in trials]
        dir_hits = [t['dir_hit'] for t in trials]
        
        # Average parameters across symbols
        param_keys = set()
        for t in trials:
            param_keys.update(t['params'].keys())
        
        avg_params = {}
        for key in param_keys:
            values = [t['params'].get(key) for t in trials if key in t['params']]
            if values:
                try:
                    # Try mean for numeric values
                    numeric_vals = [float(v) for v in values]
                    avg_params[key] = float(np.mean(numeric_vals))
                except (ValueError, TypeError):
                    # Keep first value for non-numeric
                    avg_params[key] = values[0]
        
        # ✅ FIX: Also aggregate separate structures if available
        avg_params_xgb = {}
        avg_params_lgb = {}
        avg_params_cat = {}
        
        # Aggregate best_params_xgb
        xgb_keys = set()
        for t in trials:
            if 'params_xgb' in t and t['params_xgb']:
                xgb_keys.update(t['params_xgb'].keys())
        
        for key in xgb_keys:
            values = [t.get('params_xgb', {}).get(key) for t in trials if key in t.get('params_xgb', {})]
            if values:
                try:
                    numeric_vals = [float(v) for v in values]
                    avg_params_xgb[key] = float(np.mean(numeric_vals))
                except (ValueError, TypeError):
                    avg_params_xgb[key] = values[0]
        
        # Aggregate best_params_lgb
        lgb_keys = set()
        for t in trials:
            if 'params_lgb' in t and t['params_lgb']:
                lgb_keys.update(t['params_lgb'].keys())
        
        for key in lgb_keys:
            values = [t.get('params_lgb', {}).get(key) for t in trials if key in t.get('params_lgb', {})]
            if values:
                try:
                    numeric_vals = [float(v) for v in values]
                    avg_params_lgb[key] = float(np.mean(numeric_vals))
                except (ValueError, TypeError):
                    avg_params_lgb[key] = values[0]
        
        # Aggregate best_params_cat
        cat_keys = set()
        for t in trials:
            if 'params_cat' in t and t['params_cat']:
                cat_keys.update(t['params_cat'].keys())
        
        for key in cat_keys:
            values = [t.get('params_cat', {}).get(key) for t in trials if key in t.get('params_cat', {})]
            if values:
                try:
                    numeric_vals = [float(v) for v in values]
                    avg_params_cat[key] = float(np.mean(numeric_vals))
                except (ValueError, TypeError):
                    avg_params_cat[key] = values[0]
        
        # Per-symbol breakdown
        per_symbol_results = {}
        for t in trials:
            per_symbol_results[t['symbol']] = {
                'dir_hit': t['dir_hit'],
                'params': t['params']
            }
        
        # Also aggregate adaptive_k and pattern_weight if present
        adaptive_aggr = None
        pattern_aggr = None
        
        # Check if any trial has adaptive_k or pattern_weight
        has_adaptive = any('adaptive_k' in t['params'] for t in trials)
        has_pattern = any('pattern_weight' in t['params'] for t in trials)
        
        if has_adaptive:
            adaptive_vals = [t['params'].get('adaptive_k') for t in trials if 'adaptive_k' in t['params']]
            adaptive_aggr = float(np.mean(adaptive_vals)) if adaptive_vals else None
        
        if has_pattern:
            pattern_vals = [t['params'].get('pattern_weight') for t in trials if 'pattern_weight' in t['params']]
            pattern_aggr = float(np.mean(pattern_vals)) if pattern_vals else None
        
        output[horizon_key] = {
            'symbols': symbols,
            'best_params': avg_params,
            'best_params_xgb': avg_params_xgb if avg_params_xgb else None,  # ✅ FIX: Preserve separate structures
            'best_params_lgb': avg_params_lgb if avg_params_lgb else None,
            'best_params_cat': avg_params_cat if avg_params_cat else None,
            'aggr_dir_hit': float(np.mean(dir_hits)),
            'min_dir_hit': float(np.min(dir_hits)),
            'max_dir_hit': float(np.max(dir_hits)),
            'per_symbol_results': per_symbol_results,
            'adaptive_k': adaptive_aggr,  # Aggregate adaptive_k across symbols
            'pattern_weight': pattern_aggr  # Aggregate pattern_weight across symbols
        }
        
        print(f"\n✅ Horizon {horizon_key}:")
        print(f"   Symbols: {len(symbols)}")
        print(f"   Avg DirHit: {np.mean(dir_hits):.2f}%")
        print(f"   Range: {np.min(dir_hits):.2f}% - {np.max(dir_hits):.2f}%")
    
    return output


def main():
    print("=" * 80)
    print("🔬 AGGREGATE HPO RESULTS")
    print("=" * 80)
    print()
    
    results = aggregate_per_horizon()
    
    if not results:
        print("⚠️  No HPO results found!")
        return
    
    # Save aggregated results
    config_dir = Path('/opt/bist-pattern/config')
    config_dir.mkdir(exist_ok=True)
    
    output_file = config_dir / 'best_hpo_params.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"✅ Aggregated results saved to: {output_file}")
    print()
    print("📋 Summary:")
    for h, data in results.items():
        print(f"   {h}: {data['aggr_dir_hit']:.2f}% (from {len(data['symbols'])} symbols)")
    
    # Generate shell-exportable environment variables
    print()
    print("🔧 Export commands for start_after_reboot.sh:")
    print("-" * 80)
    
    for h in sorted(results.keys(), key=lambda x: int(x[:-1])):
        data = results[h]
        params = data['best_params']
        horizon_num = h[:-1]  # '1d' -> '1'
        
        print(f"\n# Horizon {h}")
        
        # XGBoost params
        for key, val in sorted(params.items()):
            # Convert to OPTUNA_XGB_* format
            env_key = f"OPTUNA_XGB_{key.upper()}"
            print(f"export {env_key}='{val}'")
        
        # Export adaptive_k and pattern_weight if aggregated
        if data.get('adaptive_k') is not None:
            print(f"export ML_ADAPTIVE_K_{horizon_num}D='{data['adaptive_k']:.4f}'")
        
        if data.get('pattern_weight') is not None:
            print(f"export ML_PATTERN_WEIGHT_SCALE_{horizon_num}D='{data['pattern_weight']:.4f}'")
    
    print()


if __name__ == '__main__':
    main()
