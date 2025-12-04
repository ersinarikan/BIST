#!/usr/bin/env python3
"""
Create HPO JSON files from study databases for symbols that are missing JSON files
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import numpy as np

sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    venv_python = '/opt/bist-pattern/venv/bin/python3'
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        raise

from scripts.continuous_hpo_training_pipeline import STATE_FILE  # noqa: E402
from scripts.retrain_high_discrepancy_symbols import find_study_db  # noqa: E402


def load_state() -> Dict:
    """Load pipeline state"""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def create_json_from_study(db_file: Path, symbol: str, horizon: int, cycle: int, dry_run: bool = False) -> Optional[Path]:
    """Create JSON file from study database"""
    try:
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_file}")
        
        if study.best_trial is None:
            print("  ❌ No best trial found in study")
            return None
        
        best_trial = study.best_trial
        best_params = best_trial.params.copy()
        best_value = float(study.best_value) if study.best_value is not None else 0.0
        
        # Get best dirhit from trial user attrs
        best_dirhit = None
        try:
            _val = best_trial.user_attrs.get('avg_dirhit', None)
            if isinstance(_val, (int, float)) and np.isfinite(_val):
                best_dirhit = float(_val)
        except Exception:
            pass
        
        # Fallback: try symbol-specific avg_dirhit
        if best_dirhit is None:
            try:
                symbol_key = f"{symbol}_{horizon}d"
                symbol_metrics = best_trial.user_attrs.get('symbol_metrics', {})
                if isinstance(symbol_metrics, dict) and symbol_key in symbol_metrics:
                    symbol_metric = symbol_metrics[symbol_key]
                    best_dirhit = symbol_metric.get('avg_dirhit')
                    if best_dirhit is not None:
                        best_dirhit = float(best_dirhit)
            except Exception:
                pass
        
        if best_dirhit is None or not np.isfinite(best_dirhit):
            best_dirhit = best_value  # fallback
        
        # Get feature flags and params
        feature_flags = {}
        feature_params = {}
        hyperparameters = {}
        
        # Feature flags (enable_*)
        for key in best_params.keys():
            if key.startswith('enable_'):
                feature_flags[key] = best_params[key]
        
        # Feature params (ml_adaptive_k_*, ml_pattern_weight_scale_*, external_*, yolo_*)
        feature_params_keys = [
            'external_min_days', 'external_smooth_alpha',
            'yolo_min_conf',
            'ml_loss_mse_weight', 'ml_loss_threshold', 'ml_dir_penalty',
            'n_seeds', 'meta_stacking_alpha',
        ]
        feature_params_keys += [k for k in best_params.keys() if k.startswith('ml_adaptive_k_') or k.startswith('ml_pattern_weight_scale_')]
        feature_params = {k: v for k, v in best_params.items() if k in feature_params_keys}
        
        # Hyperparameters: model parameters (xgb_*, lgb_*, cat_*)
        hyperparameters = {k: v for k, v in best_params.items() if not k.startswith('enable_') and k not in feature_params_keys}
        
        # Get pruned count and avg trial time
        try:
            pruned_count = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
        except Exception:
            pruned_count = 0
        
        try:
            durations = []
            for t in study.trials:
                start = getattr(t, 'datetime_start', None)
                end = getattr(t, 'datetime_complete', None)
                if start is not None and end is not None:
                    durations.append(float((end - start).total_seconds()))
            avg_trial_time = float(np.mean(durations)) if durations else 0.0
        except Exception:
            avg_trial_time = 0.0
        
        # Top-K trials summary (K=3)
        try:
            sorted_trials = sorted(
                [t for t in study.trials if t.value is not None],
                key=lambda tr: float(tr.value) if tr.value is not None else float('-inf'),
                reverse=True
            )
            topk = []
            for t in sorted_trials[:3]:
                topk.append({
                    'number': t.number,
                    'value': float(t.value) if t.value is not None else None,
                    'params': t.params,
                    'attrs': getattr(t, 'user_attrs', {}),
                    'state': str(t.state),
                })
        except Exception:
            topk = []
        
        # Get symbol_metrics from best trial
        symbol_metrics_best = best_trial.user_attrs.get('symbol_metrics', {})
        
        # Check for low_support_warning
        low_support_symbols = []
        if isinstance(symbol_metrics_best, dict):
            for sym_key, sym_metrics in symbol_metrics_best.items():
                if isinstance(sym_metrics, dict) and sym_metrics.get('low_support_warning'):
                    parts = sym_key.rsplit('_', 1)
                    if len(parts) == 2:
                        sym_name = parts[0]
                        try:
                            h = int(parts[1].replace('d', ''))
                            low_support_symbols.append(f"{sym_name}_{h}d")
                        except Exception:
                            pass
        
        # Get evaluation_spec filter values from best trial
        # ✅ CRITICAL: Get filter from symbol-specific metrics, not first symbol
        _min_mc_spec = 0
        _min_mp_spec = 0.0
        try:
            symbol_key = f"{symbol}_{horizon}d"
            if isinstance(symbol_metrics_best, dict) and symbol_key in symbol_metrics_best:
                symbol_metric = symbol_metrics_best[symbol_key]
                split_metrics = symbol_metric.get('split_metrics', [])
                if split_metrics:
                    first_split = split_metrics[0]
                    _min_mc_spec = first_split.get('min_mask_count', 0)
                    _min_mp_spec = first_split.get('min_mask_pct', 0.0)
            elif isinstance(symbol_metrics_best, dict) and len(symbol_metrics_best) > 0:
                # Fallback: use first symbol's metrics
                first_symbol_metrics = next(iter(symbol_metrics_best.values()))
                split_metrics = first_symbol_metrics.get('split_metrics', [])
                if split_metrics:
                    first_split = split_metrics[0]
                    _min_mc_spec = first_split.get('min_mask_count', 0)
                    _min_mp_spec = first_split.get('min_mask_pct', 0.0)
        except Exception as e:
            print(f"  ⚠️ Warning: Could not get filter from study: {e}")
            pass
        
        # Create result JSON
        result = {
            'best_value': float(best_value),
            'best_dirhit': float(best_dirhit),
            'best_params': best_params,
            'best_trial': {
                'number': best_trial.number,
                'value': float(best_trial.value) if best_trial.value is not None else 0.0,
                'state': str(best_trial.state),
            },
            'best_trial_number': int(best_trial.number),
            'best_trial_seed': int(42 + best_trial.number),
            'n_trials': len(study.trials),
            'pruned_count': int(pruned_count),
            'avg_trial_time': float(avg_trial_time),
            'top_k_trials': topk,
            'study_name': study.study_name,
            'symbols': [symbol],
            'horizon': horizon,
            'best_model_choice': best_params.get('model_choice'),
            'feature_flags': feature_flags,
            'feature_params': feature_params,
            'hyperparameters': hyperparameters,
            'features_enabled': {
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
                'ML_USE_ADAPTIVE_LEARNING': '0',  # Always disabled during HPO
                'ENABLE_XGBOOST': '1' if best_params.get('model_choice') in ('xgb', 'all') else '0',
                'ENABLE_LIGHTGBM': '1' if best_params.get('model_choice') in ('lgbm', 'all') else '0',
                'ENABLE_CATBOOST': '1' if best_params.get('model_choice') in ('cat', 'all') else '0',
            },
            'evaluation_spec': {
                'horizon': int(horizon),
                'dirhit_threshold': 0.005,
                'min_mask_count': int(_min_mc_spec),
                'min_mask_pct': float(_min_mp_spec),
                'best_trial_number': int(best_trial.number),
                'best_trial_seed': int(42 + best_trial.number),
            },
            '_created_from_study': True,
            '_created_at': datetime.now().isoformat(),
        }
        
        # Add best_trial_metrics if available
        if isinstance(symbol_metrics_best, dict):
            result['best_trial_metrics'] = symbol_metrics_best
        
        # Add low_support_warnings if any
        if low_support_symbols:
            result['low_support_warnings'] = low_support_symbols
        
        # Create output file path
        output_file = Path(f"/opt/bist-pattern/results/optuna_pilot_features_on_h{horizon}_c{cycle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        if dry_run:
            print(f"  [DRY-RUN] Would create: {output_file.name}")
            print(f"    - Best trial: #{best_trial.number}")
            print(f"    - Best DirHit: {best_dirhit:.2f}%")
            print(f"    - Best value: {best_value:.4f}")
            print(f"    - N trials: {len(study.trials)}")
            return None
        
        # Write JSON file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  ✅ Created: {output_file.name}")
        print(f"    - Best trial: #{best_trial.number}")
        print(f"    - Best DirHit: {best_dirhit:.2f}%")
        print(f"    - Best value: {best_value:.4f}")
        print(f"    - N trials: {len(study.trials)}")
        print(f"    - Filter: min_mask_count={_min_mc_spec}, min_mask_pct={_min_mp_spec}")
        
        return output_file
    
    except Exception as e:
        print(f"  ❌ Error creating JSON: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Create HPO JSON files from study databases')
    parser.add_argument('--symbols', type=str, nargs='+',
                       help='Specific symbols to process (default: missing JSON symbols)')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1],
                       help='Horizons to process (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be created without creating files')
    parser.add_argument('--all-completed', action='store_true',
                       help='Process all completed symbols (not just missing JSON)')
    
    args = parser.parse_args()
    
    # Load state
    state = load_state()
    current_cycle = state.get('cycle', 1)
    tasks = state.get('state', {})
    
    # Find symbols to process
    if args.symbols:
        symbols_to_process = [(s, h) for s in args.symbols for h in args.horizons]
    else:
        # Find completed symbols missing JSON files
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
                # Check if JSON file exists
                results_dir = Path('/opt/bist-pattern/results')
                pattern = f"optuna_pilot_features_on_h{horizon}_c{current_cycle}_*.json"
                json_files = list(results_dir.glob(pattern))
                
                json_found = False
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        if symbol in data.get('symbols', []):
                            json_found = True
                            break
                    except Exception:
                        continue
                
                if not json_found or args.all_completed:
                    completed_symbols.append((symbol, horizon))
        
        symbols_to_process = completed_symbols
    
    print(f"📊 Found {len(symbols_to_process)} symbols to process")
    print(f"🔄 Cycle: {current_cycle}")
    print(f"🔍 Dry-run: {args.dry_run}")
    print()
    
    # Process each symbol
    created = 0
    failed = 0
    
    for symbol, horizon in sorted(symbols_to_process):
        print(f"\n{'='*80}")
        print(f"🔍 {symbol}_{horizon}d")
        print(f"{'='*80}")
        
        # Find study DB
        db_file = find_study_db(symbol, horizon, current_cycle)
        if not db_file:
            print("  ❌ Study DB not found")
            failed += 1
            continue
        
        print(f"  ✅ Study DB: {db_file.name}")
        
        # Create JSON
        json_file = create_json_from_study(db_file, symbol, horizon, current_cycle, args.dry_run)
        if json_file:
            created += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Created: {created}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(symbols_to_process)}")


if __name__ == '__main__':
    main()
