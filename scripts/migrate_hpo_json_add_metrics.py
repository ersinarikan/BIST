#!/usr/bin/env python3
"""
Migration Script: Add best_trial_metrics to existing HPO JSON files

Bu script, mevcut HPO JSON dosyalarına best_trial_metrics ekler.
Best trial'ın parametreleriyle modeli yeniden eğitip metrics hesaplar.

Kullanım:
    python scripts/migrate_hpo_json_add_metrics.py [--dry-run] [--json-file PATH]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

# Set environment
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# ⚡ CRITICAL: Disable prediction logging during migration
os.environ['DISABLE_PREDICTIONS_LOG'] = '1'
os.environ['DISABLE_ML_PREDICTION_DURING_TRAINING'] = '1'

# Ensure DATABASE_URL is set (use PgBouncer port 6432)
# ✅ FIX: Override if it points to wrong port
db_url_env = os.environ.get('DATABASE_URL', '')
if not db_url_env or ':5432/' in db_url_env:
    # Use PgBouncer port 6432
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db'

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from bist_pattern.core.config_manager import ConfigManager
from enhanced_ml_system import EnhancedMLSystem
from scripts.optuna_hpo_with_feature_flags import fetch_prices, generate_walkforward_splits, dirhit, compute_returns


def calculate_best_trial_metrics(
    json_data: Dict[str, Any],
    engine,
    dry_run: bool = False
) -> Optional[Dict[str, Any]]:
    """Calculate best_trial_metrics for a HPO JSON file.
    
    Args:
        json_data: HPO JSON data (loaded from file)
        engine: SQLAlchemy engine for database access
        dry_run: If True, don't actually train models (just validate)
    
    Returns:
        best_trial_metrics dict or None if calculation fails
    """
    try:
        # Extract best trial info
        best_trial = json_data.get('best_trial', {})
        best_trial_number = best_trial.get('number')
        if best_trial_number is None:
            print(f"  ⚠️  No best_trial.number found, skipping")
            return None
        
        best_params = json_data.get('best_params', {})
        if not best_params:
            print(f"  ⚠️  No best_params found, skipping")
            return None
        
        symbols = json_data.get('symbols', [])
        horizon = json_data.get('horizon')
        if not symbols or not horizon:
            print(f"  ⚠️  Missing symbols or horizon, skipping")
            return None
        
        print(f"  📊 Best trial: {best_trial_number}, symbols: {symbols}, horizon: {horizon}d")
        
        # Set best params as environment variables (same as HPO)
        from scripts.train_completed_hpo_with_best_params import set_hpo_params_as_env
        set_hpo_params_as_env(best_params, horizon)
        
        # ✅ CRITICAL: Set feature flags from JSON (if available)
        features_enabled = json_data.get('features_enabled', {})
        if features_enabled:
            # Normalize feature flag keys to uppercase environment variable names
            def _normalize_feature_flag_key(key: str) -> str:
                """Normalize feature flag key to environment variable name."""
                # If already uppercase, return as is
                if key.isupper():
                    return key
                # Map common lowercase patterns to uppercase
                mapping = {
                    'enable_seed_bagging': 'ENABLE_SEED_BAGGING',
                    'enable_directional_loss': 'ML_USE_DIRECTIONAL_LOSS',
                    'enable_adaptive_learning': 'ML_USE_ADAPTIVE_LEARNING',
                    'enable_smart_ensemble': 'ML_USE_SMART_ENSEMBLE',
                    'enable_meta_stacking': 'ML_USE_STACKED_SHORT',
                    'enable_regime_detection': 'ML_USE_REGIME_DETECTION',
                    'enable_yolo_features': 'ENABLE_YOLO_FEATURES',
                    'enable_fingpt_features': 'ENABLE_FINGPT_FEATURES',
                    'enable_external_features': 'ENABLE_EXTERNAL_FEATURES',
                }
                return mapping.get(key.lower(), key.upper())
            
            for key, value in features_enabled.items():
                env_key = _normalize_feature_flag_key(key)
                os.environ[env_key] = str(value)
            print(f"  🔧 Set {len(features_enabled)} feature flags from JSON")
        
        # ✅ CRITICAL: Set feature_params (smart ensemble, etc.) from JSON (if available)
        feature_params = json_data.get('feature_params', {})
        if feature_params:
            # Smart ensemble params
            if 'smart_consensus_weight' in feature_params:
                os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(feature_params['smart_consensus_weight'])
            if 'smart_performance_weight' in feature_params:
                os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(feature_params['smart_performance_weight'])
            if 'smart_sigma' in feature_params:
                os.environ['ML_SMART_SIGMA'] = str(feature_params['smart_sigma'])
            if 'smart_weight_xgb' in feature_params:
                os.environ['ML_SMART_WEIGHT_XGB'] = str(feature_params['smart_weight_xgb'])
            if 'smart_weight_lgbm' in feature_params:
                os.environ['ML_SMART_WEIGHT_LGB'] = str(feature_params['smart_weight_lgbm'])
            if 'smart_weight_cat' in feature_params:
                os.environ['ML_SMART_WEIGHT_CAT'] = str(feature_params['smart_weight_cat'])
            # Other feature params
            if 'ml_loss_mse_weight' in feature_params:
                os.environ['ML_LOSS_MSE_WEIGHT'] = str(feature_params['ml_loss_mse_weight'])
            if 'ml_loss_threshold' in feature_params:
                os.environ['ML_LOSS_THRESHOLD'] = str(feature_params['ml_loss_threshold'])
            if 'ml_dir_penalty' in feature_params:
                os.environ['ML_DIR_PENALTY'] = str(feature_params['ml_dir_penalty'])
            if 'n_seeds' in feature_params:
                os.environ['N_SEEDS'] = str(feature_params['n_seeds'])
            if 'meta_stacking_alpha' in feature_params:
                os.environ['ML_META_STACKING_ALPHA'] = str(feature_params['meta_stacking_alpha'])
            print(f"  🔧 Set feature_params from JSON")
        
        # Set seed (same as HPO: 42 + trial.number)
        seed = 42 + best_trial_number
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ.setdefault('OPTUNA_XGB_RANDOM_STATE', str(seed))
        os.environ.setdefault('OPTUNA_LGB_RANDOM_STATE', str(seed))
        os.environ.setdefault('OPTUNA_CAT_RANDOM_STATE', str(seed))
        
        # Set ML_HORIZONS
        os.environ['ML_HORIZONS'] = str(horizon)
        ConfigManager.clear_cache()
        
        # Create ML system instance
        ml = EnhancedMLSystem()
        ml.base_seeds = [seed]
        ml.prediction_horizons = [horizon]
        
        # Collect metrics for all symbols
        all_symbol_metrics = {}
        
        for symbol in symbols:
            print(f"    🔄 Processing {symbol}...")
            
            # Fetch data (same as HPO: cache bypass)
            df = fetch_prices(engine, symbol, limit=1200)
            if df is None or len(df) < 100:
                print(f"      ⚠️  Insufficient data for {symbol}, skipping")
                continue
            
            # Generate splits (same as HPO)
            total_days = len(df)
            wfv_splits = generate_walkforward_splits(total_days, horizon, n_splits=4)
            if not wfv_splits:
                print(f"      ⚠️  No valid splits for {symbol}, skipping")
                continue
            
            # Collect metrics from all splits (same structure as HPO objective)
            split_metrics_list = []
            split_dirhits = []
            split_nrmses = []
            
            for split_idx, (train_end_idx, test_end_idx) in enumerate(wfv_splits, 1):
                train_df = df.iloc[:train_end_idx].copy()
                test_df = df.iloc[train_end_idx:test_end_idx].copy()
                
                if dry_run:
                    print(f"      [DRY-RUN] Split {split_idx}: train={len(train_df)}, test={len(test_df)}")
                    continue
                
                # Train model
                try:
                    train_result = ml.train_enhanced_models(symbol, train_df)
                    if not train_result:
                        print(f"      ⚠️  Training failed for {symbol} split {split_idx}")
                        continue
                except Exception as e:
                    print(f"      ⚠️  Training error for {symbol} split {split_idx}: {e}")
                    continue
                
                # Get model metrics from trained models
                model_key = f"{symbol}_{horizon}d"
                if model_key not in ml.models:
                    print(f"      ⚠️  Model not found for {symbol} split {split_idx}")
                    continue
                
                horizon_models = ml.models[model_key]
                if not isinstance(horizon_models, dict):
                    print(f"      ⚠️  Invalid model structure for {symbol} split {split_idx}")
                    continue
                
                # Extract metrics from each model
                split_model_metrics = {}
                for model_name in ['xgboost', 'lightgbm', 'catboost']:
                    if model_name not in horizon_models:
                        continue
                    
                    model_info = horizon_models[model_name]
                    if not isinstance(model_info, dict):
                        continue
                    
                    split_model_metrics[model_name] = {
                        'raw_r2': float(model_info.get('raw_r2', 0.0)),
                        'rmse': float(model_info.get('rmse', 0.0)) if not np.isnan(model_info.get('rmse', float('nan'))) else None,
                        'mape': float(model_info.get('mape', 0.0)) if not np.isnan(model_info.get('mape', float('nan'))) else None,
                        'confidence': float(model_info.get('score', 0.5)),
                    }
                
                # Make predictions on test_df and calculate DirHit/nRMSE (same as HPO)
                dirhit_val = None
                rmse_val = None
                mape_val = None
                nrmse_val = None
                mask_count = 0
                mask_pct = 0.0
                valid_count = 0
                
                if len(test_df) > horizon:
                    try:
                        # Calculate true returns
                        y_true = compute_returns(test_df, horizon)
                        
                        # Make predictions for each time step in test_df
                        # ✅ FIX: Initialize preds with same length as test_df (like HPO objective)
                        preds = np.full(len(test_df), np.nan, dtype=float)
                        
                        for t in range(len(test_df) - horizon):
                            try:
                                # Use data up to current time point
                                cur = pd.concat([train_df, test_df.iloc[:t+1]], axis=0).copy()
                                pred = ml.predict_enhanced(symbol, cur)
                                
                                if isinstance(pred, dict):
                                    key = f"{horizon}d"
                                    obj = pred.get(key)
                                    if isinstance(obj, dict):
                                        pred_price = obj.get('ensemble_prediction')
                                        if isinstance(pred_price, (int, float)) and not np.isnan(pred_price):
                                            # Convert price prediction to return (same as HPO)
                                            last_close = float(cur['close'].iloc[-1])
                                            if last_close > 0:
                                                pred_return = float(pred_price) / last_close - 1.0
                                                preds[t] = pred_return
                                    else:
                                        preds[t] = float('nan')
                                else:
                                    preds[t] = float('nan')
                            except Exception:
                                preds[t] = float('nan')
                        
                        # Calculate DirHit, RMSE, MAPE, nRMSE (same as HPO)
                        valid_mask = ~np.isnan(preds) & ~np.isnan(y_true.values)
                        valid_count = int(valid_mask.sum())
                        
                        if valid_count > 0:
                            y_true_valid = y_true.values[valid_mask]
                            preds_valid = preds[valid_mask]
                            
                            # DirHit
                            dirhit_val = dirhit(y_true_valid, preds_valid)
                            
                            # RMSE and MAPE
                            rmse_val = float(np.sqrt(np.mean((y_true_valid - preds_valid) ** 2)))
                            mape_val = float(np.mean(np.abs((y_true_valid - preds_valid) / (y_true_valid + 1e-8))) * 100)
                            
                            # Threshold mask statistics
                            thr = 0.005
                            mask_count = int(((np.abs(y_true_valid) > thr) & (np.abs(preds_valid) > thr)).sum())
                            mask_pct = float((mask_count / valid_count) * 100) if valid_count > 0 else 0.0
                            
                            # nRMSE (normalized by std of y_true_valid)
                            try:
                                std_y = float(np.std(y_true_valid)) if y_true_valid.size > 1 else 0.0
                                if std_y > 0:
                                    nrmse_val = float(rmse_val / std_y)
                                    split_nrmses.append(nrmse_val)
                            except Exception:
                                pass
                            
                            split_dirhits.append(dirhit_val)
                            nrmse_str = f"{nrmse_val:.4f}" if nrmse_val is not None else "N/A"
                            print(f"      Split {split_idx}: DirHit={dirhit_val:.2f}%, RMSE={rmse_val:.6f}, MAPE={mape_val:.2f}%, nRMSE={nrmse_str}")
                    except Exception as e:
                        print(f"      ⚠️  Prediction/evaluation error for {symbol} split {split_idx}: {e}")
                
                # Create split entry (same structure as HPO)
                split_entry = {
                    'split_index': int(split_idx),
                    'train_days': int(len(train_df)),
                    'test_days': int(len(test_df)),
                    'valid_predictions': int(valid_count),
                    'model_metrics': split_model_metrics,
                    'dirhit': float(dirhit_val) if dirhit_val is not None else None,
                    'rmse': float(rmse_val) if rmse_val is not None and not np.isnan(rmse_val) else None,
                    'mape': float(mape_val) if mape_val is not None and not np.isnan(mape_val) else None,
                    'nrmse': float(nrmse_val) if nrmse_val is not None and not np.isnan(nrmse_val) else None,
                    'mask_count': int(mask_count),
                    'mask_pct': float(mask_pct),
                }
                split_metrics_list.append(split_entry)
            
            if not split_metrics_list:
                print(f"      ⚠️  No metrics collected for {symbol}")
                continue
            
            # Calculate average metrics across splits
            avg_model_metrics = {}
            for model_name in ['xgboost', 'lightgbm', 'catboost']:
                model_metrics_list = [
                    split_entry.get('model_metrics', {}).get(model_name)
                    for split_entry in split_metrics_list
                    if model_name in split_entry.get('model_metrics', {})
                ]
                if not model_metrics_list:
                    continue
                
                # Average across splits
                avg_metrics = {
                    'raw_r2': float(np.mean([m['raw_r2'] for m in model_metrics_list])),
                    'rmse': float(np.mean([m['rmse'] for m in model_metrics_list if m['rmse'] is not None])) if any(m['rmse'] is not None for m in model_metrics_list) else None,
                    'mape': float(np.mean([m['mape'] for m in model_metrics_list if m['mape'] is not None])) if any(m['mape'] is not None for m in model_metrics_list) else None,
                    'confidence': float(np.mean([m['confidence'] for m in model_metrics_list])),
                }
                avg_model_metrics[model_name] = avg_metrics
            
            # Calculate average DirHit and nRMSE
            avg_dirhit_value = float(np.mean(split_dirhits)) if split_dirhits else None
            avg_nrmse_value = float(np.mean(split_nrmses)) if split_nrmses else None
            
            if avg_model_metrics:
                all_symbol_metrics[f"{symbol}_{horizon}d"] = {
                    'split_metrics': split_metrics_list,
                    'avg_dirhit': avg_dirhit_value,
                    'avg_nrmse': avg_nrmse_value,
                    'split_count': len(split_metrics_list),
                    'avg_model_metrics': avg_model_metrics,
                }
                dirhit_str = f"{avg_dirhit_value:.2f}%" if avg_dirhit_value is not None else "N/A"
                print(f"      ✅ Collected metrics for {symbol} ({len(split_metrics_list)} splits, avg DirHit={dirhit_str})")
        
        if not all_symbol_metrics:
            print(f"  ⚠️  No metrics collected for any symbol")
            return None
        
        result = {
            'best_trial_metrics': all_symbol_metrics,
        }
        
        print(f"  ✅ Calculated metrics for {len(all_symbol_metrics)} symbol-horizon pairs")
        return result
        
    except Exception as e:
        print(f"  ❌ Error calculating metrics: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def migrate_json_file(json_path: Path, engine, dry_run: bool = False, force: bool = False) -> bool:
    """Migrate a single HPO JSON file.
    
    Returns:
        True if migration successful, False otherwise
    """
    print(f"\n📄 Processing: {json_path.name}")
    
    try:
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check if already migrated
        if (not force) and 'best_trial_metrics' in data and data['best_trial_metrics']:
            print(f"  ✅ Already migrated (best_trial_metrics exists)")
            return True
        
        # Calculate metrics
        metrics_data = calculate_best_trial_metrics(data, engine, dry_run=dry_run)
        
        if not metrics_data:
            print(f"  ⚠️  Failed to calculate metrics")
            return False
        
        if dry_run:
            print(f"  [DRY-RUN] Would add best_trial_metrics to JSON")
            return True
        
        # Merge metrics into JSON data (overwrite if force)
        if force:
            data['best_trial_metrics'] = {}  # clear before overwrite
        data.update(metrics_data)
        
        # Atomic write (same as HPO script)
        tmp_file = str(json_path) + '.tmp'
        try:
            with open(tmp_file, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            
            os.replace(tmp_file, json_path)
            print(f"  ✅ Migration successful")
            return True
            
        except Exception as e:
            # Clean up temp file
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass
            raise e
        
    except Exception as e:
        print(f"  ❌ Migration failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description='Migrate HPO JSON files to add best_trial_metrics')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (don\'t modify files)')
    parser.add_argument('--json-file', type=str, help='Migrate specific JSON file (instead of all)')
    parser.add_argument('--symbols-only', action='store_true', help='Only migrate symbols with DirHit mismatches')
    parser.add_argument('--force', action='store_true', help='Recompute and overwrite existing best_trial_metrics')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔄 HPO JSON Migration: Add best_trial_metrics")
    print("=" * 80)
    if args.dry_run:
        print("⚠️  DRY-RUN MODE: No files will be modified")
    print()
    
    # Define symbols with DirHit mismatches (from user's list)
    # Format: SYMBOL_HORIZONd -> (symbol, horizon)
    mismatch_symbols = {
        'A1CAP': 1, 'A1YEN': 1, 'ACSEL': 1, 'ADEL': 1, 'ADGYO': 1, 'AEFES': 1,
        'AGESA': 1, 'AGHOL': 1, 'AGROT': 1, 'AGYO': 1, 'AHGAZ': 1, 'AHSGY': 1,
        'AKBNK': 1, 'AKCNS': 1, 'AKENR': 1, 'AKFGY': 1, 'AKFIS': 1, 'AKFYE': 1,
        'AKGRT': 1, 'AKMGY': 1, 'AKSA': 1, 'AKSUE': 1, 'AKYHO': 1, 'ALARK': 1,
        'ALBRK': 1, 'ALCAR': 1, 'ALFAS': 1, 'ALGYO': 1, 'ALKLC': 1, 'ALTNY': 1,
        'ALVES': 1, 'ANELE': 1, 'ANGEN': 1, 'ANHYT': 1, 'ANSGR': 1, 'ARASE': 1,
        'ARCLK': 1, 'ARDYZ': 1, 'ARENA': 1, 'ARMGD': 1, 'ARSAN': 1, 'ARTMS': 1,
        'ARZUM': 1, 'ASELS': 1, 'ASGYO': 1, 'ASTOR': 1, 'ASUZU': 1, 'ATAGY': 1,
        'ATAKP': 1, 'ATATP': 1, 'ATEKS': 1, 'ATLAS': 1, 'ATSYH': 1, 'AVGYO': 1,
        'AVHOL': 1, 'AVOD': 1, 'AVPGY': 1, 'AYCES': 1, 'AYDEM': 1, 'AYEN': 1,
        'AYES': 1, 'AZTEK': 1, 'BAGFS': 1, 'BAHKM': 1, 'BAKAB': 1, 'BALAT': 1,
        'BALSU': 1, 'BANVT': 1,
    }
    
    # Find JSON files
    results_dir = Path('/opt/bist-pattern/results')
    if args.json_file:
        json_files = [Path(args.json_file)]
    else:
        json_files = sorted(results_dir.glob('optuna_pilot_features_on_h*.json'))
    
    if not json_files:
        print("⚠️  No HPO JSON files found!")
        return 1
    
    # Filter by symbols if --symbols-only is specified
    # Otherwise, process ALL JSON files that need migration (best_trial_metrics missing)
    if args.symbols_only:
        filtered_files = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                symbols = data.get('symbols', [])
                horizon = data.get('horizon')
                if symbols and horizon:
                    symbol = symbols[0]  # Each JSON contains one symbol
                    if symbol in mismatch_symbols and mismatch_symbols[symbol] == horizon:
                        filtered_files.append(json_file)
            except Exception:
                pass
        json_files = filtered_files
        print(f"🔍 Filtered to {len(json_files)} JSON file(s) with DirHit mismatches")
    elif not args.force:
        # Filter out already migrated files (best_trial_metrics exists)
        # This ensures we process ALL old JSONs that need migration
        unmigrated_files = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                # Skip if already migrated (best_trial_metrics exists and is not empty)
                if 'best_trial_metrics' in data and data.get('best_trial_metrics'):
                    continue
                unmigrated_files.append(json_file)
            except Exception:
                # If we can't read it, include it (will be handled later)
                unmigrated_files.append(json_file)
        json_files = unmigrated_files
        print(f"📋 Found {len(json_files)} unmigrated JSON file(s) (will process all)")
    
    if not json_files:
        print("⚠️  No matching JSON files found!")
        return 1
    
    print()
    
    # Create database engine (use PgBouncer port 6432)
    db_url = os.getenv('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db').strip()
    # ✅ FIX: Use NullPool for migration (single process, no connection pooling needed)
    engine = create_engine(db_url, poolclass=NullPool, pool_pre_ping=True, connect_args={"application_name": "hpo-migration"})
    
    # Migrate each file
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for json_file in json_files:
        if not json_file.exists():
            print(f"\n⚠️  File not found: {json_file}")
            skip_count += 1
            continue
        
        try:
            # Check if already migrated (skip unless forcing)
            with open(json_file, 'r') as f:
                data = json.load(f)
            if (not args.force) and 'best_trial_metrics' in data and data['best_trial_metrics']:
                print(f"\n⏭️  Skipping {json_file.name} (already migrated)")
                skip_count += 1
                continue
        except Exception:
            pass
        
        success = migrate_json_file(json_file, engine, dry_run=args.dry_run, force=args.force)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print()
    print("=" * 80)
    print("📊 Migration Summary")
    print("=" * 80)
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {fail_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"  📄 Total: {len(json_files)}")
    print()
    
    if args.dry_run:
        print("⚠️  DRY-RUN: No files were modified")
    else:
        print("✅ Migration completed!")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

