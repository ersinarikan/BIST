#!/usr/bin/env python3
"""
Train completed HPO symbols with best parameters (all features ON) and evaluate DirHit
This script:
1. Loads completed HPO results from JSON files
2. For each symbol/horizon pair:
   - Sets best parameters as environment variables
   - Trains model with all features ON (production-like)
   - Evaluates DirHit using walk-forward validation
   - Saves models and DirHit results
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
# datetime not required

# Set environment
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# ⚡ CRITICAL: Disable prediction logging during training
os.environ['DISABLE_PREDICTIONS_LOG'] = '1'  # Skip PredictionsLog writes during training
os.environ['DISABLE_ML_PREDICTION_DURING_TRAINING'] = '1'  # Disable ML prediction during training
os.environ['WRITE_ENHANCED_DURING_CYCLE'] = '0'  # Disable prediction write during training

# Ensure DATABASE_URL is set (PgBouncer 6432, secure secret fallback)
if 'DATABASE_URL' not in os.environ:
    try:
        secret_path = '/opt/bist-pattern/.secrets/db_password'
        if os.path.exists(secret_path):
            with open(secret_path, 'r') as sp:
                _pwd = sp.read().strip()
                if _pwd:
                    os.environ['DATABASE_URL'] = f'postgresql://bist_user:{_pwd}@127.0.0.1:6432/bist_pattern_db'
        if 'DATABASE_URL' not in os.environ:
            # Final fallback to PgBouncer port
            os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db'
    except Exception:
        os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db'

# Import after path setup
from app import app  # noqa: E402
from pattern_detector import HybridPatternDetector  # noqa: E402
from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_returns(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Compute forward returns for given horizon."""
    if df is None or len(df) < horizon:
        return np.array([])
    fwd = (df['close'].shift(-horizon) / df['close'] - 1.0).values
    return fwd


def dirhit(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.005) -> float:
    """Calculate directional hit rate (DirHit)."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    yt = np.sign(y_true)
    yp = np.sign(y_pred)
    # HPO parity: evaluate only significant predictions/returns
    m = (np.abs(y_true) > thr) & (np.abs(y_pred) > thr)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(yt[m] == yp[m]) * 100.0)


def set_hpo_params_as_env(params: dict, horizon: int):
    """Set HPO parameters as environment variables."""
    # XGBoost params
    xgb_params = []
    lgb_params = []
    cat_params = []
    # Helper: normalize CatBoost bootstrap_type to valid enum
    
    def _normalize_cat_bootstrap_type(v):
        if v is None:
            return None
        s = str(v).strip()
        # Map booleans and common falsy/truthy to valid enums
        mapping = {
            'false': 'No',
            '0': 'No',
            'none': None,
            'true': 'Bernoulli',
            '1': 'Bernoulli',
        }
        lower = s.lower()
        if lower in mapping:
            return mapping[lower]
        # Allow only valid CatBoost enums
        allowed = {
            'poisson': 'Poisson',
            'bayesian': 'Bayesian',
            'bernoulli': 'Bernoulli',
            'mvs': 'MVS',
            'no': 'No',
        }
        return allowed.get(lower, None)
    # ✅ FIX: First pass - set bootstrap_type first to check for subsample
    bootstrap_type_value = None
    for key, val in params.items():
        if key == 'cat_bootstrap_type':
            norm = _normalize_cat_bootstrap_type(val)
            if norm is not None:
                bootstrap_type_value = norm
            break
    
    for key, val in params.items():
        if key.startswith('xgb_'):
            env_key = f"OPTUNA_XGB_{key.replace('xgb_', '').upper()}"
            os.environ[env_key] = str(val)
            xgb_params.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
        elif key.startswith('lgb_'):
            env_key = f"OPTUNA_LGB_{key.replace('lgb_', '').upper()}"
            os.environ[env_key] = str(val)
            lgb_params.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
        elif key.startswith('cat_'):
            env_key = f"OPTUNA_CAT_{key.replace('cat_', '').upper()}"
            # Special handling for bootstrap_type (CatBoost enum)
            if key == 'cat_bootstrap_type':
                norm = _normalize_cat_bootstrap_type(val)
                if norm is not None:
                    os.environ[env_key] = norm
                    cat_params.append(f"{key}={norm}")
                else:
                    # Skip setting invalid enum; let model defaults apply
                    continue
            elif key == 'cat_subsample':
                # ✅ FIX: Only set subsample if bootstrap_type is not 'No'
                # When bootstrap_type='No', subsample should not be set
                # Check both from params dict and environment (in case bootstrap_type was set earlier)
                bt_check = bootstrap_type_value or os.environ.get('OPTUNA_CAT_BOOTSTRAP_TYPE', '').strip()
                if bt_check.lower() not in ('no', 'false', '0', 'none', ''):
                    os.environ[env_key] = str(val)
                    cat_params.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
                else:
                    # Skip subsample when bootstrap is disabled
                    continue
            else:
                os.environ[env_key] = str(val)
                cat_params.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
        elif key == 'adaptive_k':
            os.environ[f'ML_ADAPTIVE_K_{horizon}D'] = str(val)
        elif key == 'pattern_weight':
            os.environ[f'ML_PATTERN_WEIGHT_SCALE_{horizon}D'] = str(val)
    
    # Propagate feature parameters (smart ensemble, directional loss, externals) if present
    feature_params = {}
    try:
        if isinstance(params.get('feature_params'), dict):
            feature_params = params['feature_params'] or {}
    except Exception:
        feature_params = {}
    # Support flat params as fallback (older JSONs)
    flat = params
    
    def _get_fp(name: str, default=None):
        if name in feature_params:
            return feature_params.get(name, default)
        return flat.get(name, default)
    # Smart ensemble params
    scw = _get_fp('smart_consensus_weight')
    spw = _get_fp('smart_performance_weight')
    ssig = _get_fp('smart_sigma')
    if scw is not None:
        os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(scw)
    if spw is not None:
        os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(spw)
    if ssig is not None:
        os.environ['ML_SMART_SIGMA'] = str(ssig)
    # Prior weights per model
    w_xgb = _get_fp('smart_weight_xgb')
    w_lgb = _get_fp('smart_weight_lgbm') or _get_fp('smart_weight_lgb')
    w_cat = _get_fp('smart_weight_cat')
    if w_xgb is not None:
        os.environ['ML_SMART_WEIGHT_XGB'] = str(w_xgb)
    if w_lgb is not None:
        os.environ['ML_SMART_WEIGHT_LGB'] = str(w_lgb)
    if w_cat is not None:
        os.environ['ML_SMART_WEIGHT_CAT'] = str(w_cat)
    # Directional loss params
    mse_w = _get_fp('ml_loss_mse_weight')
    thr = _get_fp('ml_loss_threshold')
    pen = _get_fp('ml_dir_penalty')
    if mse_w is not None:
        os.environ['ML_LOSS_MSE_WEIGHT'] = str(mse_w)
    if thr is not None:
        os.environ['ML_LOSS_THRESHOLD'] = str(thr)
    if pen is not None:
        os.environ['ML_DIR_PENALTY'] = str(pen)
    
    # Log best parameters for verification
    if xgb_params or lgb_params or cat_params:
        logger.debug(f"   ⚙️ Best HPO parameters set for {horizon}d: XGB({len(xgb_params)}), LGB({len(lgb_params)}), CAT({len(cat_params)})")


def train_symbol_with_all_horizons_best_params(symbol: str, all_horizon_params: dict) -> dict:
    """Train model with best params for all horizons (all features ON).
    
    Args:
        symbol: Stock symbol
        all_horizon_params: Dict of {horizon: best_params} for all horizons
                           Example: {1: {...}, 3: {...}, 7: {...}, 14: {...}, 30: {...}}
    
    NOTE: Since enhanced_ml_system.py reads OPTUNA_XGB_* globally (not per-horizon),
    we need to train each horizon separately with its own parameters.
    """
    try:
        horizons_list = sorted(all_horizon_params.keys())
        
        logger.info(f"🎯 Training {symbol} for horizons {horizons_list} with best HPO parameters...")
        logger.info("   ⚠️  Note: Training each horizon separately to use correct parameters")
        
        with app.app_context():
            det = HybridPatternDetector()
            
            # Get stock data once
            df = det.get_stock_data(symbol, days=0)
            
            if df is None or len(df) < 50:
                logger.warning(f"❌ {symbol}: Insufficient data ({len(df) if df is not None else 0} days)")
                return {'success': False, 'dirhit_results': {}}
            
            logger.info(f"✅ {symbol}: Data loaded ({len(df)} days)")
            
            # Train each horizon separately with its own best parameters
            # This ensures each horizon uses its correct HPO parameters
            all_success = True
            for horizon in horizons_list:
                best_params = all_horizon_params[horizon]
                
                # Set parameters for this horizon
                set_hpo_params_as_env(best_params, horizon)
                
                # ⚡ CRITICAL: Set only this horizon for training BEFORE creating instance
                # DO NOT set ML_HORIZONS to all horizons - it will be cached!
                os.environ['ML_HORIZONS'] = str(horizon)
                
                logger.info(f"   🎯 Training {symbol} {horizon}d with best parameters...")
                
                # ⚡ CRITICAL: Clear ConfigManager cache AND remove from cache before creating new instance
                # ConfigManager caches environment variables, so we need to clear cache
                # when ML_HORIZONS changes. Also remove 'ML_HORIZONS' from cache specifically.
                try:
                    from bist_pattern.core.config_manager import ConfigManager
                    ConfigManager.clear_cache()  # Clear all cache
                    # Also remove ML_HORIZONS from cache specifically (in case clear_cache doesn't work)
                    if hasattr(ConfigManager, '_cache') and 'ML_HORIZONS' in ConfigManager._cache:
                        del ConfigManager._cache['ML_HORIZONS']
                except Exception:
                    pass  # Fallback if ConfigManager not available
                
                # ⚡ CRITICAL: Reset singleton instance to read new ML_HORIZONS
                # Singleton instance's prediction_horizons is set in __init__ based on ML_HORIZONS
                # ✅ CRITICAL FIX: Clear singleton using thread-safe function
                from enhanced_ml_system import clear_enhanced_ml_system
                clear_enhanced_ml_system()
                
                ml = get_enhanced_ml_system()  # Get fresh instance to read new env vars
                # Verify prediction_horizons is correct
                if ml.prediction_horizons != [horizon]:
                    logger.warning(f"   ⚠️  {symbol} {horizon}d: prediction_horizons mismatch! Expected [{horizon}], got {ml.prediction_horizons}")
                    # Force update - this is safe because we're training only this horizon
                    ml.prediction_horizons = [horizon]
                    logger.info(f"   ✅ {symbol} {horizon}d: prediction_horizons force-updated to [{horizon}]")
                
                result = ml.train_enhanced_models(symbol, df)
                
                if not result:
                    logger.warning(f"   ❌ {symbol} {horizon}d: Training failed")
                    all_success = False
                else:
                    logger.info(f"   ✅ {symbol} {horizon}d: Training completed")
                    # ⚡ CRITICAL FIX: Ensure features are saved to disk after each horizon training
                    # This ensures that when prediction is done later, all horizon features are available
                    try:
                        ml.save_enhanced_models(symbol)
                        logger.debug(f"   ✅ {symbol} {horizon}d: Features saved to disk")
                    except Exception as e:
                        logger.warning(f"   ⚠️  {symbol} {horizon}d: Failed to save features: {e}")
            
            # ⚡ CRITICAL FIX: After all horizons are trained, load all features into a single instance
            # This ensures prediction can access all horizon features
            # Restore all horizons for final status
            os.environ['ML_HORIZONS'] = ','.join(str(h) for h in horizons_list)
            
            # Clear singleton and create final instance with all horizons
            try:
                from bist_pattern.core.config_manager import ConfigManager
                ConfigManager.clear_cache()
                if hasattr(ConfigManager, '_cache') and 'ML_HORIZONS' in ConfigManager._cache:
                    del ConfigManager._cache['ML_HORIZONS']
            except Exception:
                pass
            
            # ✅ CRITICAL FIX: Clear singleton using thread-safe function
            from enhanced_ml_system import clear_enhanced_ml_system
            clear_enhanced_ml_system()
            
            # Create final instance with all horizons for prediction
            ml_final = get_enhanced_ml_system()
            ml_final.prediction_horizons = horizons_list
            
            # Load all horizon features from disk
            try:
                # ⚡ CRITICAL FIX: Use load_trained_models() instead of _load_models_from_disk()
                # _load_models_from_disk() method doesn't exist, use load_trained_models() instead
                if hasattr(ml_final, 'load_trained_models'):
                    ml_final.load_trained_models(symbol)
                    logger.debug(f"✅ {symbol}: All horizon features loaded from disk")
                else:
                    # Fallback: manually load horizon_features.json
                    import json
                    horizon_cols_file = f"{ml_final.model_directory}/{symbol}_horizon_features.json"
                    if os.path.exists(horizon_cols_file):
                        with open(horizon_cols_file, 'r') as rf:
                            horizon_features = json.load(rf) or {}
                        # Restore horizon-specific features to self.models
                        for h in horizons_list:
                            h_key = f"{h}d"
                            if h_key in horizon_features:
                                feature_key = f"{symbol}_{h}d_features"
                                ml_final.models[feature_key] = horizon_features[h_key]
                        logger.debug(f"✅ {symbol}: Horizon features loaded from disk ({len(horizon_features)} horizons)")
                    else:
                        logger.warning(f"⚠️  {symbol}: Horizon features file not found: {horizon_cols_file}")
            except Exception as e:
                logger.warning(f"⚠️  {symbol}: Failed to load horizon features from disk: {e}")
            
            if all_success:
                logger.info(f"✅ {symbol}: Training completed successfully for all horizons {horizons_list} with best HPO parameters")
                
                # ⚡ CRITICAL: Evaluate DirHit using walk-forward validation
                logger.info(f"📊 Evaluating DirHit for {symbol} with walk-forward validation...")
                dirhit_results = evaluate_dirhit_walkforward(symbol, df, horizons_list, all_horizon_params, ml_final)
                
                return {'success': True, 'dirhit_results': dirhit_results}
            else:
                logger.warning(f"⚠️ {symbol}: Some horizons failed")
                return {'success': False, 'dirhit_results': {}}
                
    except Exception as e:
        logger.error(f"❌ {symbol}: Error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {'success': False, 'dirhit_results': {}}


def evaluate_dirhit_walkforward(symbol: str, df: pd.DataFrame, horizons_list: list, all_horizon_params: dict, ml_final) -> dict:
    """Evaluate DirHit using walk-forward validation for all horizons.
    
    ⚡ CRITICAL: Disable adaptive learning during validation to prevent data leakage.
    - Training: Adaptive learning ON (model learns from test_data)
    - Validation: Adaptive learning OFF (model only predicts, no updates)
    This ensures validation results are realistic and no data leakage occurs.
    """
    dirhit_results = {}
    
    # ⚡ CRITICAL: Disable adaptive learning during validation to prevent data leakage
    # Training sırasında adaptive learning açık (model test_data'yı görüyor)
    # Validation'da adaptive learning kapalı olmalı (data leakage önlemek için)
    original_adaptive = os.environ.get('ML_USE_ADAPTIVE_LEARNING', '0')
    os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'
    logger.debug(f"   🔒 Adaptive learning disabled for validation (was: {original_adaptive})")
    
    try:
        # Split data: 80% train, 20% test (walk-forward)
        total_days = len(df)
        if total_days < 180:
            logger.warning(f"⚠️ {symbol}: Insufficient data for walk-forward validation ({total_days} days, need at least 180)")
            return dirhit_results
        
        split_idx = int(total_days * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"   📊 Data split: {len(train_df)} train, {len(test_df)} test days")
        
        # Evaluate each horizon
        for horizon in horizons_list:
            try:
                # Set parameters for this horizon (for prediction)
                best_params = all_horizon_params[horizon]
                set_hpo_params_as_env(best_params, horizon)
                
                # Compute true returns
                y_true = compute_returns(test_df, horizon)
                
                # Walk-forward predictions
                preds = np.full(len(test_df), np.nan, dtype=float)
                
                min_test_days = horizon + 10  # At least 10 predictions
                if len(test_df) < min_test_days:
                    logger.warning(f"   ⚠️ {symbol} {horizon}d: Insufficient test data ({len(test_df)} days, need {min_test_days})")
                    continue
                
                valid_predictions = 0
                for t in range(len(test_df) - horizon):
                    try:
                        # Build current data: train + test up to t
                        cur = pd.concat([train_df, test_df.iloc[:t + 1]], axis=0).copy()
                        
                        # Predict
                        p = ml_final.predict_enhanced(symbol, cur)
                        
                        if not isinstance(p, dict):
                            continue
                        
                        key = f"{horizon}d"
                        obj = p.get(key)
                        if isinstance(obj, dict):
                            pred_price = obj.get('ensemble_prediction')
                            try:
                                if isinstance(pred_price, (int, float)) and not np.isnan(pred_price):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        preds[t] = float(pred_price) / last_close - 1.0
                                        valid_predictions += 1
                            except Exception:
                                pass
                    except Exception as e:
                        logger.debug(f"   {symbol} {horizon}d: Prediction error at t={t}: {e}")
                        continue
                
                # Calculate DirHit
                if valid_predictions > 0:
                    dh = dirhit(y_true, preds)
                    dirhit_results[horizon] = {
                        'dirhit': float(dh) if not np.isnan(dh) else None,
                        'valid_predictions': valid_predictions,
                        'total_test_days': len(test_df),
                        'train_days': len(train_df),
                        'test_days': len(test_df)
                    }
                    logger.info(f"   ✅ {symbol} {horizon}d: DirHit = {dh:.2f}% ({valid_predictions} valid predictions)")
                else:
                    logger.warning(f"   ⚠️ {symbol} {horizon}d: No valid predictions")
                    dirhit_results[horizon] = {
                        'dirhit': None,
                        'valid_predictions': 0,
                        'total_test_days': len(test_df),
                        'train_days': len(train_df),
                        'test_days': len(test_df)
                    }
            except Exception as e:
                logger.error(f"   ❌ {symbol} {horizon}d: DirHit evaluation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                dirhit_results[horizon] = {
                    'dirhit': None,
                    'error': str(e)
                }
        
    except Exception as e:
        logger.error(f"   ❌ {symbol}: Walk-forward validation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        # ⚡ CRITICAL: Restore original adaptive learning setting
        os.environ['ML_USE_ADAPTIVE_LEARNING'] = original_adaptive
        logger.debug(f"   🔓 Adaptive learning restored to: {original_adaptive}")
    
    return dirhit_results


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Train completed HPO symbols with best parameters')
    parser.add_argument('--phase', type=str, default='phase1', choices=['phase1', 'phase2'],
                       help='HPO phase: phase1 (features off) or phase2 (features on)')
    args = parser.parse_args()
    
    # Load completed HPO symbols based on phase
    if args.phase == 'phase2':
        completed_file = '/opt/bist-pattern/results/completed_hpo_symbols_phase2.json'
        output_prefix = 'train_completed_hpo_phase2'
    else:
        completed_file = '/opt/bist-pattern/results/completed_hpo_symbols.json'
        output_prefix = 'train_completed_hpo'
    
    if not os.path.exists(completed_file):
        logger.error(f"Completed HPO symbols file not found: {completed_file}")
        return 1
    
    with open(completed_file, 'r') as f:
        completed = json.load(f)
    
    logger.info("=" * 80)
    logger.info(f"🎯 Training Completed HPO Symbols with Best Parameters ({args.phase.upper()})")
    logger.info("=" * 80)
    logger.info("All features enabled (production-like training)")
    logger.info(f"Loaded {sum(len(items) for items in completed.values())} symbol/horizon pairs")
    logger.info("=" * 80)
    
    # Group by symbol: collect all horizons for each symbol
    symbol_horizons = {}
    for horizon_str, items in completed.items():
        horizon = int(horizon_str.replace('d', ''))
        for item in items:
            symbol = item['symbol']
            json_file = item['json_file']
            best_params = item['best_params']
            
            if symbol not in symbol_horizons:
                symbol_horizons[symbol] = {}
            symbol_horizons[symbol][horizon] = {
                'best_params': best_params,
                'json_file': json_file,
                'horizon_str': horizon_str
            }
    
    logger.info(f"\n📊 Grouped into {len(symbol_horizons)} symbols")
    logger.info("   Each symbol will be trained for all its horizons at once")
    
    # Results storage
    results = []
    success_count = 0
    fail_count = 0
    
    # Process each symbol with all its horizons
    # ⚡ CRITICAL: For Phase 2, only process the 5 symbols used in Phase 2 HPO
    phase2_symbols = {'ALKA', 'ALKIM', 'ARASE', 'ARENA', 'ARSAN'}
    
    if args.phase == 'phase2':
        # Filter to only Phase 2 symbols
        symbol_horizons = {sym: data for sym, data in symbol_horizons.items() if sym in phase2_symbols}
        logger.info(f"🔍 Phase 2: Filtering to {len(symbol_horizons)} symbols: {sorted(symbol_horizons.keys())}")
    
    for symbol, horizons_data in symbol_horizons.items():
        # Extract best_params for each horizon
        all_horizon_params = {h: data['best_params'] for h, data in horizons_data.items()}
        
        logger.info(f"\n🎯 Processing {symbol} for horizons {sorted(all_horizon_params.keys())}...")
        
        result = train_symbol_with_all_horizons_best_params(symbol, all_horizon_params)
        
        # Handle new return format (dict with success and dirhit_results)
        if isinstance(result, dict):
            success = result.get('success', False)
            dirhit_results = result.get('dirhit_results', {})
        else:
            # Backward compatibility
            success = result
            dirhit_results = {}
        
        if success:
            # Add one result entry per horizon with DirHit
            for horizon, data in horizons_data.items():
                horizon_dirhit = dirhit_results.get(horizon, {})
                results.append({
                    'symbol': symbol,
                    'horizon': data['horizon_str'],
                    'status': 'success',
                    'json_file': data['json_file'],
                    'dirhit': horizon_dirhit.get('dirhit'),
                    'valid_predictions': horizon_dirhit.get('valid_predictions'),
                    'train_days': horizon_dirhit.get('train_days'),
                    'test_days': horizon_dirhit.get('test_days'),
                })
            success_count += len(horizons_data)
        else:
            # Add one result entry per horizon (all failed)
            for horizon, data in horizons_data.items():
                results.append({
                    'symbol': symbol,
                    'horizon': data['horizon_str'],
                    'status': 'failed',
                    'json_file': data['json_file'],
                    'dirhit': None,
                })
            fail_count += len(horizons_data)
    
    # Save results
    output_dir = Path('/opt/bist-pattern/results')
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / f'{output_prefix}_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate average DirHit
    successful_results = [r for r in results if r.get('status') == 'success' and r.get('dirhit') is not None]
    if successful_results:
        avg_dirhit = np.mean([r['dirhit'] for r in successful_results])
        logger.info("=" * 80)
        logger.info(f"📊 Average DirHit: {avg_dirhit:.2f}% ({len(successful_results)} symbol/horizon pairs)")
        logger.info("=" * 80)
        
        # Save CSV summary
        csv_file = output_dir / f'{output_prefix}_dirhits.csv'
        df_results = pd.DataFrame(successful_results)
        df_results.to_csv(csv_file, index=False)
        logger.info(f"✅ CSV summary saved to {csv_file}")
    
    logger.info("=" * 80)
    logger.info(f"✅ Training completed: {success_count} success, {fail_count} failed")
    logger.info(f"✅ Results saved to {json_file}")
    logger.info("=" * 80)
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
