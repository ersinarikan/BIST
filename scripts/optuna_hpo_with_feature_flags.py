#!/usr/bin/env python3
"""
Optuna HPO with Feature Flag Optimization

Bu script, hem hyperparameter'ları hem de feature flag'leri birlikte optimize eder.
Optuna'nın suggest_categorical() fonksiyonu ile feature flag'leri de optimize edebiliriz.

Örnek kullanım:
    python scripts/optuna_hpo_with_feature_flags.py --symbols ASELS --horizon 7 --trials 200
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import cast, Optional

import numpy as np
import optuna
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from pathlib import Path

sys.path.insert(0, '/opt/bist-pattern')
from bist_pattern.core.config_manager import ConfigManager  # noqa: E402
from enhanced_ml_system import EnhancedMLSystem  # noqa: E402

# Check model availability
try:
    import xgboost  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost  # noqa: F401
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='ASELS', help='Comma-separated symbols')
    ap.add_argument('--horizon', type=int, default=7, help='Horizon in days')
    ap.add_argument('--trials', type=int, default=200, help='Number of Optuna trials')
    ap.add_argument('--timeout', type=int, default=57600, help='Timeout seconds')  # 16h
    return ap.parse_args()


# (Removed Redis-based DB semaphore; PgBouncer provides pooling)


def fetch_prices(engine, symbol: str, limit: int = 1200) -> pd.DataFrame | None:
    """Fetch stock prices - ⚡ CRITICAL: Ensure we get enough data for HPO (at least 200 days).
    
    Cache is bypassed for HPO to ensure fresh data from DB.
    """
    symbol = symbol.strip().upper()
    if symbol.startswith('\ufeff'):
        symbol = symbol[1:]
    
    # ⚡ CRITICAL FIX: Skip cache for HPO to ensure fresh data from DB
    # Cache might be stale or incomplete (e.g., A1YEN has 130 days in cache but 139 in DB)
    # For HPO, we need reliable, complete data, so fetch directly from DB
    # Cache is bypassed to avoid stale or incomplete data

    q = text(
        """
        SELECT p.date, p.open_price, p.high_price, p.low_price, p.close_price, p.volume
        FROM stock_prices p
        JOIN stocks s ON s.id = p.stock_id
        WHERE s.symbol = :sym
        ORDER BY p.date DESC
        LIMIT :lim
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"sym": symbol, "lim": limit}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            'open': float(r[1]),
            'high': float(r[2]),
            'low': float(r[3]),
            'close': float(r[4]),
            'volume': float(r[5])
        }
        for r in rows
    ], index=[r[0] for r in rows])
    # ✅ Ensure datetime index (prevents macro merge fallback and TZ mismatches)
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception:
        try:
            df.index = pd.to_datetime(df.index, errors='coerce').tz_localize(None)
        except Exception:
            pass
    df = df.sort_index()
    
    # ✅ HİBRİT YAKLAŞIM: Duplicate date kontrolü (aynı tarihli kayıtlar varsa temizle)
    if df.index.duplicated().any():
        duplicate_count = df.index.duplicated().sum()
        print(f"[hpo] {symbol}: {duplicate_count} duplicate date found, dropping (keep='last')", file=sys.stderr, flush=True)
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
    
    if limit > 0 and len(df) > limit:
        df = df.iloc[-limit:].copy()
    return cast(pd.DataFrame, df)


def compute_returns(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Compute forward returns for given horizon."""
    return df['close'].shift(-horizon) / df['close'] - 1.0


def dirhit(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.005) -> float:
    """Compute directional hit rate."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    yt = np.sign(y_true)
    yp = np.sign(y_pred)
    m = (np.abs(y_true) > thr) & (np.abs(y_pred) > thr)
    if m.sum() == 0:
        return 0.0
    return float(np.mean(yt[m] == yp[m]) * 100.0)


def calculate_dynamic_split(total_days: int, horizon: int) -> int:
    """Calculate dynamic split index based on total days.
    
    Uses the same logic as the existing split mechanism:
    - 240+ days: last 120 days test, rest train
    - 180-239 days: 2/3 train, 1/3 test
    - <180 days: 2/3 train, 1/3 test
    
    Args:
        total_days: Total number of days in dataset
        horizon: Prediction horizon in days
        
    Returns:
        split_idx: Index where test set starts
    """
    min_test_days = horizon + 10
    
    if total_days >= 240:
        split_idx = total_days - 120
    elif total_days >= 180:
        split_idx = int(total_days * 2 / 3)
    else:
        split_idx = max(1, int(total_days * 2 / 3))
    
    # Ensure test set is large enough
    if total_days - split_idx < min_test_days:
        split_idx = total_days - min_test_days
        if split_idx < 1:
            return -1  # Invalid split
    
    return split_idx


def generate_walkforward_splits(total_days: int, horizon: int, n_splits: int = 4) -> list[tuple[int, int]]:
    """Generate multiple splits for walk-forward validation.
    
    Creates multiple train/test splits using the dynamic split mechanism.
    Each split uses an expanding window approach where:
    - Split 1: Uses initial split_idx as train end
    - Split 2: Expands train by ~30 days, shifts test window
    - Split 3: Expands train further, shifts test window
    - etc.
    
    Args:
        total_days: Total number of days in dataset
        horizon: Prediction horizon in days
        n_splits: Number of splits to generate (default: 4)
        
    Returns:
        List of (train_end_idx, test_end_idx) tuples
    """
    min_test_days = horizon + 10
    splits = []
    
    # Calculate initial split using dynamic mechanism
    initial_split = calculate_dynamic_split(total_days, horizon)
    if initial_split < 1:
        return splits  # Invalid, return empty
    
    # Calculate split window size (how much to expand train for each split)
    # Use approximately 30 days per split, but ensure we have enough test data
    available_test_days = total_days - initial_split
    if available_test_days < min_test_days * n_splits:
        # Not enough data for multiple splits, use single split
        if available_test_days >= min_test_days:
            splits.append((initial_split, total_days))
        return splits
    
    # Calculate step size for expanding train window
    # We want to create n_splits, so divide available test days
    step_size = max(30, (total_days - initial_split) // (n_splits + 1))
    
    # Generate splits
    for i in range(n_splits):
        train_end = initial_split + (i * step_size)
        test_end = min(train_end + step_size, total_days)
        
        # Ensure we have enough test days
        test_days = test_end - train_end
        if test_days < min_test_days:
            continue
        
        # Ensure train_end doesn't exceed total_days
        if train_end >= total_days:
            break
        
        splits.append((train_end, test_end))
    
    return splits


def objective(trial: optuna.Trial, symbols, horizon: int, engine, db_url: str, study=None, max_trials: Optional[int] = None) -> float:
    """Optuna objective function - Feature flags + Hyperparameters birlikte optimize edilir."""
    # ✅ FIX: Check trial limit at the start of each trial to prevent exceeding n_trials
    # This provides a second layer of protection against race conditions in parallel execution
    # Count all trials except the current one (which is just starting)
    if study is not None and max_trials is not None:
        # Count all trials except the current one
        other_trials = [t for t in study.trials if t.number != trial.number]
        if len(other_trials) >= max_trials:
            # Skip this trial if we've already reached the limit
            raise optuna.TrialPruned(f"Trial limit reached ({len(other_trials)}/{max_trials})")
    
    ConfigManager.clear_cache()
    
    # ⚡ NEW: Feature flag'leri optimize et (12 feature - test script ile aynı)
    feature_flags = {
        'enable_external_features': trial.suggest_categorical('enable_external_features', [True, False]),
        'enable_fingpt_features': trial.suggest_categorical('enable_fingpt_features', [True, False]),
        'enable_yolo_features': trial.suggest_categorical('enable_yolo_features', [True, False]),
        'enable_directional_loss': trial.suggest_categorical('enable_directional_loss', [True, False]),
        'enable_seed_bagging': trial.suggest_categorical('enable_seed_bagging', [True, False]),
        'enable_talib_patterns': trial.suggest_categorical('enable_talib_patterns', [True, False]),
        'enable_smart_ensemble': trial.suggest_categorical('enable_smart_ensemble', [True, False]),
        'enable_stacked_short': trial.suggest_categorical('enable_stacked_short', [True, False]),
        'enable_meta_stacking': trial.suggest_categorical('enable_meta_stacking', [True, False]),
        'enable_regime_detection': trial.suggest_categorical('enable_regime_detection', [True, False]),
        'enable_fingpt': trial.suggest_categorical('enable_fingpt', [True, False]),
        # ML_USE_ADAPTIVE_LEARNING: HPO'da her zaman kapalı (data leakage önleme)
    }

    # ⚡ NEW: Algorithm choice (controls which base learners are active)
    # ✅ CRITICAL FIX: Only suggest available models
    # This prevents Optuna from suggesting unavailable models (e.g., XGBoost if not installed)
    available_choices = []
    if XGBOOST_AVAILABLE:
        available_choices.append('xgb')
    if LIGHTGBM_AVAILABLE:
        available_choices.append('lgbm')
    if CATBOOST_AVAILABLE:
        available_choices.append('cat')
    if len(available_choices) > 1:  # Only add 'all' if multiple models are available
        available_choices.append('all')
    
    # ✅ DEBUG: Allow forcing a specific model via environment (e.g., FORCE_MODEL_CHOICE=xgb)
    # Useful for investigating model-specific issues without changing code logic
    force_choice = os.getenv('FORCE_MODEL_CHOICE')
    if force_choice:
        force_choice = force_choice.strip().lower()
        if force_choice in available_choices:
            available_choices = [force_choice]
    
    # Fallback: if no models available, use lgbm as default (should not happen)
    if not available_choices:
        available_choices = ['lgbm']
    
    model_choice = trial.suggest_categorical('model_choice', available_choices)
    enable_xgb = (model_choice in ('xgb', 'all')) and XGBOOST_AVAILABLE
    enable_lgb = (model_choice in ('lgbm', 'all')) and LIGHTGBM_AVAILABLE
    enable_cat = (model_choice in ('cat', 'all')) and CATBOOST_AVAILABLE
    
    # ⚡ NEW: Feature iç parametrelerini optimize et
    feature_params = {}
    
    # Directional Loss parametreleri (sadece enable_directional_loss=True ise optimize et)
    if feature_flags['enable_directional_loss']:
        feature_params['ml_loss_mse_weight'] = trial.suggest_float('ml_loss_mse_weight', 0.2, 0.8)  # MSE vs Directional balance
        feature_params['ml_loss_threshold'] = trial.suggest_float('ml_loss_threshold', 0.001, 0.01, log=True)  # Flat threshold
        feature_params['ml_dir_penalty'] = trial.suggest_float('ml_dir_penalty', 1.0, 5.0)  # Wrong direction penalty
    
    # Seed Bagging parametreleri (sadece enable_seed_bagging=True ise optimize et)
    if feature_flags['enable_seed_bagging']:
        feature_params['n_seeds'] = trial.suggest_int('n_seeds', 2, 5)  # Number of seeds
    
    # Meta Stacking parametreleri (sadece enable_meta_stacking=True ise optimize et)
    if feature_flags['enable_meta_stacking']:
        feature_params['meta_stacking_alpha'] = trial.suggest_float('meta_stacking_alpha', 0.01, 10.0, log=True)  # Ridge alpha
    
    # Adaptive Learning parametreleri (her zaman optimize et, çünkü Phase 2 skip ediliyor ama Phase 1'de kullanılıyor)
    horizon_key = f'{horizon}d'
    feature_params[f'ml_adaptive_k_{horizon_key}'] = trial.suggest_float(f'ml_adaptive_k_{horizon_key}', 1.0, 3.0)  # Adaptive K multiplier
    feature_params[f'ml_pattern_weight_scale_{horizon_key}'] = trial.suggest_float(f'ml_pattern_weight_scale_{horizon_key}', 0.5, 2.0)  # Pattern weight scale
    
    # TALIB (short horizons enable) - optional flag
    feature_flags['enable_talib_patterns_short'] = trial.suggest_categorical('enable_talib_patterns_short', [True, False])
    
    # External features internal params
    feature_params['external_min_days'] = trial.suggest_int('external_min_days', 20, 60)
    feature_params['external_smooth_alpha'] = trial.suggest_float('external_smooth_alpha', 0.05, 0.5)
    
    # Regime detection internal params (only if enabled)
    if feature_flags['enable_regime_detection']:
        feature_params['regime_scale_low'] = trial.suggest_float('regime_scale_low', 0.80, 0.98)
        feature_params['regime_scale_high'] = trial.suggest_float('regime_scale_high', 1.02, 1.20)
    
    # YOLO parametreleri (sadece enable_yolo_features=True ise optimize et)
    if feature_flags['enable_yolo_features']:
        feature_params['yolo_min_conf'] = trial.suggest_float('yolo_min_conf', 0.3, 0.8)  # YOLO confidence threshold
    
    # Smart Ensemble parametreleri (sadece enable_smart_ensemble=True ise optimize et)
    if feature_flags['enable_smart_ensemble']:
        feature_params['smart_consensus_weight'] = trial.suggest_float('smart_consensus_weight', 0.3, 0.8)  # Consensus vs Performance balance
        feature_params['smart_performance_weight'] = trial.suggest_float('smart_performance_weight', 0.2, 0.7)  # Performance weight (complement of consensus)
        feature_params['smart_sigma'] = trial.suggest_float('smart_sigma', 0.001, 0.02, log=True)  # Sensitivity threshold
        # ✅ NEW: Fixed per-model prior weights (applied in smart ensemble)
        feature_params['smart_weight_xgb'] = trial.suggest_float('smart_weight_xgb', 0.5, 1.5)
        feature_params['smart_weight_lgbm'] = trial.suggest_float('smart_weight_lgbm', 0.5, 1.5)
        feature_params['smart_weight_cat'] = trial.suggest_float('smart_weight_cat', 0.5, 1.5)
    
    # FinGPT parametreleri (sadece enable_fingpt_features=True ise optimize et)
    if feature_flags['enable_fingpt_features']:
        feature_params['fingpt_confidence_threshold'] = trial.suggest_float('fingpt_confidence_threshold', 0.2, 0.5)  # Sentiment confidence threshold
    
    # Feature flag'leri environment variable'lara set et (12 feature)
    os.environ['ENABLE_EXTERNAL_FEATURES'] = '1' if feature_flags['enable_external_features'] else '0'
    os.environ['ENABLE_FINGPT_FEATURES'] = '1' if feature_flags['enable_fingpt_features'] else '0'
    os.environ['ENABLE_YOLO_FEATURES'] = '1' if feature_flags['enable_yolo_features'] else '0'
    os.environ['ML_USE_DIRECTIONAL_LOSS'] = '1' if feature_flags['enable_directional_loss'] else '0'
    os.environ['ENABLE_SEED_BAGGING'] = '1' if feature_flags['enable_seed_bagging'] else '0'
    os.environ['ENABLE_TALIB_PATTERNS'] = '1' if feature_flags['enable_talib_patterns'] else '0'
    os.environ['ML_USE_SMART_ENSEMBLE'] = '1' if feature_flags['enable_smart_ensemble'] else '0'
    os.environ['ML_USE_STACKED_SHORT'] = '1' if feature_flags['enable_stacked_short'] else '0'
    os.environ['ENABLE_META_STACKING'] = '1' if feature_flags['enable_meta_stacking'] else '0'
    os.environ['ML_USE_REGIME_DETECTION'] = '1' if feature_flags['enable_regime_detection'] else '0'
    os.environ['ENABLE_FINGPT'] = '1' if feature_flags['enable_fingpt'] else '0'
    # TALIB short horizons
    os.environ['ENABLE_TALIB_PATTERNS_SHORT'] = '1' if feature_flags.get('enable_talib_patterns_short') else '0'

    # Algorithm enable flags
    os.environ['ENABLE_XGBOOST'] = '1' if enable_xgb else '0'
    os.environ['ENABLE_LIGHTGBM'] = '1' if enable_lgb else '0'
    os.environ['ENABLE_CATBOOST'] = '1' if enable_cat else '0'
    
    # Feature iç parametrelerini environment variable'lara set et
    if feature_flags['enable_directional_loss']:
        os.environ['ML_LOSS_MSE_WEIGHT'] = str(feature_params['ml_loss_mse_weight'])
        os.environ['ML_LOSS_THRESHOLD'] = str(feature_params['ml_loss_threshold'])
        os.environ['ML_DIR_PENALTY'] = str(feature_params['ml_dir_penalty'])
    
    if feature_flags['enable_seed_bagging']:
        os.environ['N_SEEDS'] = str(feature_params['n_seeds'])
    else:
        os.environ.pop('N_SEEDS', None)
    
    if feature_flags['enable_meta_stacking']:
        os.environ['ML_META_STACKING_ALPHA'] = str(feature_params['meta_stacking_alpha'])
    
    # Adaptive Learning parametreleri (her zaman set et)
    os.environ[f'ML_ADAPTIVE_K_{horizon_key.upper()}'] = str(feature_params[f'ml_adaptive_k_{horizon_key}'])
    os.environ[f'ML_PATTERN_WEIGHT_SCALE_{horizon_key.upper()}'] = str(feature_params[f'ml_pattern_weight_scale_{horizon_key}'])
    
    if feature_flags['enable_yolo_features']:
        os.environ['YOLO_MIN_CONF'] = str(feature_params['yolo_min_conf'])
    
    if feature_flags['enable_smart_ensemble']:
        os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(feature_params['smart_consensus_weight'])
        os.environ['ML_SMART_PERFORMANCE_WEIGHT'] = str(feature_params['smart_performance_weight'])
        os.environ['ML_SMART_SIGMA'] = str(feature_params['smart_sigma'])
        # ✅ NEW: Export per-model prior weights
        os.environ['ML_SMART_WEIGHT_XGB'] = str(feature_params['smart_weight_xgb'])
        os.environ['ML_SMART_WEIGHT_LGB'] = str(feature_params['smart_weight_lgbm'])
        os.environ['ML_SMART_WEIGHT_CAT'] = str(feature_params['smart_weight_cat'])
    
    if feature_flags['enable_fingpt_features']:
        os.environ['FINGPT_CONFIDENCE_THRESHOLD'] = str(feature_params['fingpt_confidence_threshold'])
    
    # External features params
    os.environ['EXTERNAL_MIN_DAYS'] = str(feature_params['external_min_days'])
    os.environ['EXTERNAL_SMOOTH_ALPHA'] = str(feature_params['external_smooth_alpha'])
    
    # Regime params
    if feature_flags['enable_regime_detection']:
        os.environ['REGIME_SCALE_LOW'] = str(feature_params['regime_scale_low'])
        os.environ['REGIME_SCALE_HIGH'] = str(feature_params['regime_scale_high'])
    
    # ⚡ CRITICAL: Disable adaptive learning during HPO (data leakage önleme)
    os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'
    os.environ['ML_SKIP_ADAPTIVE_PHASE2'] = '1'
    
    # XGBoost hyperparameters (only if enabled)
    if enable_xgb:
        if horizon in (1, 3):
            params_xgb = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 150, 600),
                'max_depth': trial.suggest_int('xgb_max_depth', 2, 6),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.5, 0.9),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 0.9),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-6, 1.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-3, 50.0, log=True),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 5, 15),
                'gamma': trial.suggest_float('xgb_gamma', 1e-4, 1.0, log=True),
                # Advanced
                'grow_policy': trial.suggest_categorical('xgb_grow_policy', ['depthwise', 'lossguide']),
                # GPU guard: only allow gpu_hist if explicitly enabled
                'tree_method': trial.suggest_categorical(
                    'xgb_tree_method',
                    (['hist', 'gpu_hist'] if str(os.getenv('XGB_ENABLE_GPU', '0')).lower() in ('1', 'true', 'yes', 'on') else ['hist'])
                ),
                'max_bin': trial.suggest_int('xgb_max_bin', 128, 512),
            }
        else:
            params_xgb = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 150, 900),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.25, log=True),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-6, 5.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-4, 20.0, log=True),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 3, 12),
                'gamma': trial.suggest_float('xgb_gamma', 1e-5, 1.0, log=True),
                # Advanced
                'grow_policy': trial.suggest_categorical('xgb_grow_policy', ['depthwise', 'lossguide']),
                # GPU guard: only allow gpu_hist if explicitly enabled
                'tree_method': trial.suggest_categorical(
                    'xgb_tree_method',
                    (['hist', 'gpu_hist'] if str(os.getenv('XGB_ENABLE_GPU', '0')).lower() in ('1', 'true', 'yes', 'on') else ['hist'])
                ),
                'max_bin': trial.suggest_int('xgb_max_bin', 128, 512),
            }
        for key, val in params_xgb.items():
            env_key = key.replace('xgb_', '').upper()
            os.environ[f'OPTUNA_XGB_{env_key}'] = str(val)

    # LightGBM hyperparameters (only if enabled)
    if enable_lgb:
        params_lgb = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 120, 600),
            'max_depth': trial.suggest_int('lgb_max_depth', 2, 8),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 7, 63),
            'subsample': trial.suggest_float('lgb_subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-6, 3.0, log=True),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-4, 6.0, log=True),
            # Advanced
            'min_data_in_leaf': trial.suggest_int('lgb_min_data_in_leaf', 10, 100),
            'feature_fraction_bynode': trial.suggest_float('lgb_feature_fraction_bynode', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('lgb_bagging_freq', 0, 10),
            'min_gain_to_split': trial.suggest_float('lgb_min_gain_to_split', 0.0, 1.0),
        }
        for key, val in params_lgb.items():
            env_key = key.replace('lgb_', '').upper()
            os.environ[f'OPTUNA_LGB_{env_key}'] = str(val)

    # CatBoost hyperparameters (only if enabled)
    if enable_cat:
        params_cat = {
            'iterations': trial.suggest_int('cat_iterations', 120, 600),
            'depth': trial.suggest_int('cat_depth', 2, 8),
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1.0, 10.0),
            'subsample': trial.suggest_float('cat_subsample', 0.5, 0.95),
            'rsm': trial.suggest_float('cat_rsm', 0.5, 0.95),
            # Advanced
            'border_count': trial.suggest_int('cat_border_count', 16, 255),
            'random_strength': trial.suggest_float('cat_random_strength', 0.1, 5.0, log=True),
            'leaf_estimation_iterations': trial.suggest_int('cat_leaf_estimation_iterations', 1, 20),
            'bootstrap_type': trial.suggest_categorical('cat_bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS', 'No']),
        }
        key_map = {
            'iterations': 'OPTUNA_CAT_ITERATIONS',
            'depth': 'OPTUNA_CAT_DEPTH',
            'learning_rate': 'OPTUNA_CAT_LEARNING_RATE',
            'l2_leaf_reg': 'OPTUNA_CAT_L2_LEAF_REG',
            'subsample': 'OPTUNA_CAT_SUBSAMPLE',
            'rsm': 'OPTUNA_CAT_RSM',
            'border_count': 'OPTUNA_CAT_BORDER_COUNT',
            'random_strength': 'OPTUNA_CAT_RANDOM_STRENGTH',
            'leaf_estimation_iterations': 'OPTUNA_CAT_LEAF_ESTIMATION_ITERATIONS',
            'bootstrap_type': 'OPTUNA_CAT_BOOTSTRAP_TYPE',
        }
        for k, env_k in key_map.items():
            os.environ[env_k] = str(params_cat[k])
    
    # Ensure DATABASE_URL is set
    os.environ['DATABASE_URL'] = db_url
    
    # ⚡ CRITICAL FIX: Set ML_HORIZONS BEFORE creating EnhancedMLSystem instance
    # This ensures self.prediction_horizons is set correctly
    os.environ['ML_HORIZONS'] = str(horizon)
    
    ConfigManager.clear_cache()
    
    # Create ML system instance (now with correct ML_HORIZONS)
    ml = EnhancedMLSystem()
    
    # Set seed for reproducibility
    ml.base_seeds = [42 + trial.number]
    
    dirhits = []
    nrmses = []
    print(f"[hpo] Trial {trial.number}: Processing {len(symbols)} symbols: {symbols}", file=sys.stderr, flush=True)
    
    for sym in symbols:
        print(f"[hpo] Trial {trial.number}: Fetching prices for {sym}...", file=sys.stderr, flush=True)
        df = fetch_prices(engine, sym)
        if df is None:
            print(f"[hpo] Trial {trial.number}: {sym} - df is None, skipping", file=sys.stderr, flush=True)
            continue
        # ⚡ FIX: Minimum data requirement - all horizons require 100 days
        # ✅ USER REQUEST: All horizons require minimum 100 days (not horizon * 15)
        # This allows more symbols to be processed, especially for longer horizons
        min_required_days = 100  # All horizons require minimum 100 days
        if len(df) < min_required_days:
            print(f"[hpo] Trial {trial.number}: {sym} - len(df)={len(df)} < {min_required_days} (min required for {horizon}d), skipping", file=sys.stderr, flush=True)
            continue
        print(f"[hpo] Trial {trial.number}: {sym} - df shape: {df.shape}, period: {df.index.min()} to {df.index.max()}", file=sys.stderr, flush=True)
        
        # ⚡ NEW: Generate multiple splits for walk-forward validation
        total_days = len(df)
        wfv_splits = generate_walkforward_splits(total_days, horizon, n_splits=4)
        
        if not wfv_splits:
            # Fallback to single split if multiple splits not possible
            split_idx = calculate_dynamic_split(total_days, horizon)
            if split_idx < 1:
                print(f"[hpo] Trial {trial.number}: {sym} - Invalid split, skipping", file=sys.stderr, flush=True)
                continue
            wfv_splits = [(split_idx, total_days)]
        
        print(f"[hpo] Trial {trial.number}: {sym} - Generated {len(wfv_splits)} walk-forward splits", file=sys.stderr, flush=True)
        
        # Evaluate on all splits and average DirHit
        split_dirhits = []
        split_nrmses_local: list[float] = []
        for split_idx, (train_end_idx, test_end_idx) in enumerate(wfv_splits, 1):
            train_df = df.iloc[:train_end_idx].copy()
            test_df = df.iloc[train_end_idx:test_end_idx].copy()
            
            print(f"[hpo] Trial {trial.number}: {sym} - Split {split_idx}/{len(wfv_splits)}: train={len(train_df)} days, test={len(test_df)} days", file=sys.stderr, flush=True)
            
            # Train model
            try:
                result = ml.train_enhanced_models(sym, train_df)
                if not result:
                    print(f"[hpo] Training failed for {sym} {horizon}d split {split_idx}", file=sys.stderr, flush=True)
                    continue
                
                # ⚡ DEBUG: Check if models are in memory after training
                model_key = f"{sym}_{horizon}d"
                models_exist = model_key in ml.models
                if split_idx == 1:  # Log only for first split to avoid spam
                    print(f"[hpo] {sym} {horizon}d: Training completed, model in memory: {models_exist}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[hpo] Training error for {sym} {horizon}d split {split_idx}: {e}", file=sys.stderr, flush=True)
                import traceback
                if split_idx == 1:  # Log traceback only for first split
                    print(f"[hpo] Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                continue
            
            # Walk-forward prediction
            y_true = compute_returns(test_df, horizon)
            preds = np.full(len(test_df), np.nan, dtype=float)
            
            # 🔍 DEBUG: Log data source and statistics (only for first split)
            if split_idx == 1:
                print(f"[hpo-debug] {sym} {horizon}d: DATA SOURCE = DB (cache bypassed)", file=sys.stderr, flush=True)
                print(f"[hpo-debug] {sym} {horizon}d: Total data: {total_days} days", file=sys.stderr, flush=True)
                print(f"[hpo-debug] {sym} {horizon}d: Seed = {42 + trial.number}", file=sys.stderr, flush=True)
                # 🔍 DEBUG: Log feature flags
                feature_flags_str = ", ".join([f"{k}={v}" for k, v in feature_flags.items()])
                print(f"[hpo-debug] {sym} {horizon}d: Feature flags: {feature_flags_str}", file=sys.stderr, flush=True)
            
            print(f"[hpo-debug] {sym} {horizon}d Split {split_idx}: Train period: {train_df.index.min()} to {train_df.index.max()}", file=sys.stderr, flush=True)
            print(f"[hpo-debug] {sym} {horizon}d Split {split_idx}: Test period: {test_df.index.min()} to {test_df.index.max()}", file=sys.stderr, flush=True)
            
            valid_predictions = 0
            prediction_details = []  # Store first 5 and last 5 predictions for analysis (only first split)
            
            for t in range(len(test_df) - horizon):
                try:
                    cur = pd.concat([train_df, test_df.iloc[: t + 1]], axis=0).copy()
                    p = ml.predict_enhanced(sym, cur)
                    
                    if not isinstance(p, dict):
                        if t == 0 and split_idx == 1:  # Log only first failure of first split
                            print(f"[hpo-debug] {sym} {horizon}d t={t}: predict_enhanced returned {type(p).__name__}, expected dict", file=sys.stderr, flush=True)
                        continue
                    
                    key = f"{horizon}d"
                    obj = p.get(key)
                    if not isinstance(obj, dict):
                        if t == 0 and split_idx == 1:  # Log only first failure of first split
                            print(f"[hpo-debug] {sym} {horizon}d t={t}: prediction dict missing key '{key}' or value is not dict. Available keys: {list(p.keys())}", file=sys.stderr, flush=True)
                        continue
                    
                    pred_price = obj.get('ensemble_prediction')
                    if isinstance(pred_price, (int, float)) and not np.isnan(pred_price):
                        last_close = float(cur['close'].iloc[-1])
                        if last_close > 0:
                            pred_return = float(pred_price) / last_close - 1.0
                            preds[t] = pred_return
                            valid_predictions += 1
                            
                            # Store prediction details for first 5 and last 5 (only first split)
                            if split_idx == 1 and (t < 5 or t >= len(test_df) - horizon - 5):
                                true_return = y_true.iloc[t] if not np.isnan(y_true.iloc[t]) else np.nan
                                pred_details = {
                                    't': t,
                                    'date': test_df.index[t] if t < len(test_df) else None,
                                    'last_close': last_close,
                                    'pred_price': pred_price,
                                    'pred_return': pred_return,
                                    'true_return': true_return,
                                    'direction_match': np.sign(pred_return) == np.sign(true_return) if not np.isnan(true_return) else None
                                }
                                prediction_details.append(pred_details)
                    elif t == 0 and split_idx == 1:  # Log only first failure of first split
                        print(f"[hpo-debug] {sym} {horizon}d t={t}: ensemble_prediction invalid: {pred_price} (type: {type(pred_price).__name__})", file=sys.stderr, flush=True)
                except Exception as e:
                    if t == 0 and split_idx == 1:  # Log only first exception of first split
                        print(f"[hpo-debug] {sym} {horizon}d t={t}: Prediction exception: {e}", file=sys.stderr, flush=True)
                    continue
            
            # 🔍 DEBUG: Log prediction statistics (only for first split)
            if split_idx == 1:
                print(f"[hpo-debug] {sym} {horizon}d: Valid predictions: {valid_predictions}/{len(test_df) - horizon}", file=sys.stderr, flush=True)
                if prediction_details:
                    print(f"[hpo-debug] {sym} {horizon}d: Sample predictions (first/last 5):", file=sys.stderr, flush=True)
                    for pred_detail in prediction_details[:5]:
                        print(
                            f"[hpo-debug]   t={pred_detail['t']}, date={pred_detail['date']}, "
                            f"pred_return={pred_detail['pred_return']:.4f}, "
                            f"true_return={pred_detail['true_return']:.4f}, "
                            f"match={pred_detail['direction_match']}",
                            file=sys.stderr, flush=True
                        )
            
            # Calculate DirHit with detailed logging
            valid_mask = ~np.isnan(preds) & ~np.isnan(y_true.values)
            valid_count = valid_mask.sum()
            
            # 🔍 DEBUG: Calculate additional metrics
            if valid_count > 0:
                # DirHit
                dirhit_val = dirhit(y_true.values[valid_mask], preds[valid_mask])
                
                # RMSE and MAPE
                y_true_valid = y_true.values[valid_mask]
                preds_valid = preds[valid_mask]
                rmse = np.sqrt(np.mean((y_true_valid - preds_valid) ** 2))
                mape = np.mean(np.abs((y_true_valid - preds_valid) / (y_true_valid + 1e-8))) * 100
                
                # Threshold mask statistics
                thr = 0.005
                mask_count = ((np.abs(y_true_valid) > thr) & (np.abs(preds_valid) > thr)).sum()
                mask_pct = (mask_count / valid_count) * 100 if valid_count > 0 else 0
                
                # Direction statistics
                direction_matches = (np.sign(y_true_valid) == np.sign(preds_valid)).sum()
                direction_pct = (direction_matches / valid_count) * 100 if valid_count > 0 else 0
                
                # 🔍 DEBUG: Log detailed metrics (only for first split)
                if split_idx == 1:
                    print(f"[hpo-debug] {sym} {horizon}d: METRICS:", file=sys.stderr, flush=True)
                    print(f"[hpo-debug]   Valid predictions: {valid_count}/{len(preds)}", file=sys.stderr, flush=True)
                    print(f"[hpo-debug]   DirHit (threshold={thr}): {dirhit_val:.2f}% (mask_count={mask_count}, mask_pct={mask_pct:.1f}%)", file=sys.stderr, flush=True)
                    print(f"[hpo-debug]   Direction match (all): {direction_pct:.2f}% ({direction_matches}/{valid_count})", file=sys.stderr, flush=True)
                    print(f"[hpo-debug]   RMSE: {rmse:.6f}", file=sys.stderr, flush=True)
                    print(f"[hpo-debug]   MAPE: {mape:.2f}%", file=sys.stderr, flush=True)
                
                # nRMSE (normalized by std of y_true_valid)
                try:
                    std_y = float(np.std(y_true_valid)) if y_true_valid.size > 1 else 0.0
                    if std_y > 0:
                        nrmse_val = float(rmse / std_y)
                        split_nrmses_local.append(nrmse_val)
                except Exception:
                    pass
                split_dirhits.append(dirhit_val)
                print(
                    f"[hpo] {sym} {horizon}d Split {split_idx}: DirHit={dirhit_val:.2f}% "
                    f"(valid={valid_count}/{len(preds)}, mask={mask_count}, RMSE={rmse:.6f}, MAPE={mape:.2f}%)",
                    file=sys.stderr, flush=True
                )
            else:
                preds_valid = np.sum(~np.isnan(preds))
                y_true_valid = np.sum(~np.isnan(y_true.values))
                print(
                    f"[hpo] {sym} {horizon}d Split {split_idx}: No valid predictions! "
                    f"(preds: {preds_valid}/{len(preds)} valid, "
                    f"y_true: {y_true_valid}/{len(y_true)} valid)",
                    file=sys.stderr, flush=True
                )
        
        # Average DirHit across all splits
        if split_dirhits:
            avg_dirhit = float(np.mean(split_dirhits))
            print(
                f"[hpo] {sym} {horizon}d: Average DirHit across {len(split_dirhits)} splits: {avg_dirhit:.2f}% "
                f"(splits: {split_dirhits})",
                file=sys.stderr, flush=True
            )
            dirhits.append(avg_dirhit)
        else:
            print(f"[hpo] {sym} {horizon}d: No valid DirHit from any split", file=sys.stderr, flush=True)
        # Compute per-symbol nRMSE as the average across split nRMSE values
        if split_nrmses_local:
            try:
                nrmses.append(float(np.mean(split_nrmses_local)))
            except Exception:
                pass
    
    if not dirhits:
        print(f"[hpo] Trial {trial.number}: No DirHit values calculated for any symbol! Returning 0.0", file=sys.stderr, flush=True)
        # ⚡ MEMORY LEAK FIX: Clean up models before returning
        ml.models.clear()
        ml.feature_columns = None
        # ✅ CPU PERFORMANCE: gc.collect() sadece gerekli durumlarda
        import gc
        if trial.number % 5 == 0:
            gc.collect()
        return 0.0
    
    avg_dirhit = float(np.mean(dirhits))
    avg_nrmse = float(np.mean(nrmses)) if nrmses else float('inf')
    k = 6.0 if horizon in (1, 3, 7) else 4.0
    score = float(0.7 * avg_dirhit - k * (avg_nrmse if np.isfinite(avg_nrmse) else 3.0))
    print(f"[hpo] Trial {trial.number}: Average DirHit={avg_dirhit:.2f}% (from {len(dirhits)} symbols), nRMSE={avg_nrmse:.3f}, score={score:.2f}", file=sys.stderr, flush=True)
    try:
        trial.set_user_attr('avg_dirhit', avg_dirhit)
        trial.set_user_attr('avg_nrmse', avg_nrmse)
        trial.set_user_attr('model_choice', model_choice)
    except Exception:
        pass
    
    # ⚡ MEMORY LEAK FIX: Clean up models after each trial to prevent memory leak
    # Models accumulate in ml.models dictionary during training, causing RAM usage to grow over time
    # Clear models and feature columns, then force garbage collection
    ml.models.clear()
    ml.feature_columns = None
    # Clear any cached feature data
    if hasattr(ml, 'horizon_features'):
        ml.horizon_features.clear()
    if hasattr(ml, '_feature_cache'):
        ml._feature_cache.clear()
    
    # ✅ CPU PERFORMANCE OPTIMIZATION: Garbage collection'ı optimize et
    # Her trial sonrası gc.collect() yerine her 5 trial'da bir çalıştır
    # Bu CPU kullanımını artırır, memory riski minimal (185GB available)
    import gc
    if trial.number % 5 == 0 or trial.number == 0:
        # Her 5 trial'da bir veya ilk trial'da garbage collection yap
        gc.collect()
    # Her trial'da sadece clear() yap, gc.collect()'i azalt
    
    return score


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    horizon = int(args.horizon)
    n_trials = int(args.trials)
    timeout = int(args.timeout)
    
    print("=" * 100)
    print("🔬 OPTUNA HPO WITH FEATURE FLAG + FEATURE PARAMETER OPTIMIZATION")
    print("=" * 100)
    print(f"Symbols: {symbols}")
    print(f"Horizon: {horizon}d")
    print(f"Trials (requested): {n_trials}")
    print(f"Timeout (requested): {timeout}s ({timeout/3600:.1f}h)")
    print("=" * 100)
    print()
    print("⚡ NEW: TÜM Feature flags + TÜM Feature iç parametreleri + Hyperparameters birlikte optimize ediliyor!")
    print("   - Feature flags: açık/kapalı (11 adet)")
    print("     * ENABLE_EXTERNAL_FEATURES, ENABLE_FINGPT_FEATURES, ENABLE_YOLO_FEATURES")
    print("     * ML_USE_DIRECTIONAL_LOSS, ENABLE_SEED_BAGGING, ENABLE_TALIB_PATTERNS")
    print("     * ML_USE_SMART_ENSEMBLE, ML_USE_STACKED_SHORT, ENABLE_META_STACKING")
    print("     * ML_USE_REGIME_DETECTION, ENABLE_FINGPT")
    print("   - Feature iç parametreleri (10-12 adet, feature açık/kapalı durumuna göre):")
    print("     * Directional Loss (3): ml_loss_mse_weight, ml_loss_threshold, ml_dir_penalty")
    print("     * Seed Bagging (1): n_seeds")
    print("     * Meta Stacking (1): meta_stacking_alpha")
    print("     * Adaptive Learning (2): ml_adaptive_k_{horizon}d, ml_pattern_weight_scale_{horizon}d")
    print("     * YOLO (1): yolo_min_conf")
    print("     * Smart Ensemble (3): smart_consensus_weight, smart_performance_weight, smart_sigma")
    print("     * FinGPT (1): fingpt_confidence_threshold")
    print("   - Hyperparameters: model parametreleri (~15-20 adet: XGBoost, LightGBM, CatBoost)")
    print("   - Not: ML_USE_ADAPTIVE_LEARNING her zaman kapalı (HPO'da data leakage önleme)")
    print("   - Toplam optimize edilen parametre: ~36-43 adet (11 flag + 10-12 feature param + 15-20 hyperparam)")
    print()
    
    # Use DATABASE_URL from environment (PgBouncer-friendly); fallback to direct Postgres
    db_url = os.getenv('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:6432/bist_pattern_db').strip()
    os.environ['DATABASE_URL'] = db_url
    os.environ['ML_HORIZONS'] = str(horizon)
    
    engine = create_engine(
        db_url,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={"application_name": "bist-hpo"},
    )

    # ✅ Dynamic trial/timeout budgeting based on data length and horizon
    def _estimate_days(sym: str) -> int:
        try:
            df_est = fetch_prices(engine, sym, limit=1200)
            return int(len(df_est)) if df_est is not None else 0
        except Exception:
            return 0
    total_days_list = [_estimate_days(s) for s in symbols]
    min_days = min(total_days_list) if total_days_list else 0
    
    # ✅ UNIFORM 1500 TRIALS FOR ALL HORIZONS (USER REQUEST)
    # All horizons get same quality treatment
    target_trials = 1500
    target_timeout = 72 * 3600  # 72h for all
    
    # Data quality gating: If not enough data, reduce to 1000 trials
    # ✅ USER UPDATE: 300 → 1000 (düşük veride de optimizasyon gerekli)
    if min_days < 240:
        dyn_trials = 1000
        dyn_timeout = 24 * 3600  # 24h for lower quality
    else:
        dyn_trials = target_trials
        dyn_timeout = target_timeout
    
    # Apply conservative min(requested, dynamic)
    n_trials = min(n_trials, dyn_trials)
    timeout = min(timeout, dyn_timeout)
    print(f"Trials (effective): {n_trials}  [min_days={min_days}]")
    print(f"Timeout (effective): {timeout}s ({timeout/3600:.1f}h)")
    
    # Create study
    # ✅ CRITICAL FIX: Add cycle number to study name to allow new HPO in each cycle
    # Each cycle should have its own study file to enable incremental learning with new data
    # Previous cycles' studies are preserved for analysis
    symbol_str = '_'.join(symbols) if len(symbols) <= 2 else f"{symbols[0]}_and_{len(symbols)-1}more"
    # Get cycle number from environment (set by pipeline) or default to 1
    cycle_num = int(os.getenv('HPO_CYCLE', '1'))
    study_name = f"hpo_with_features_{symbol_str}_h{horizon}_c{cycle_num}"
    study_path = f"sqlite:////opt/bist-pattern/hpo_studies/{study_name}.db"
    os.makedirs('/opt/bist-pattern/hpo_studies', exist_ok=True)
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=study_path,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
    )
    
    # ✅ FIX: Enable WAL mode for better concurrent read/write performance
    # WAL allows multiple readers and one writer simultaneously, reducing lock contention
    # This is critical when 96 HPO processes are writing to SQLite files concurrently
    try:
        import sqlite3
        db_path = study_path.replace('sqlite:///', '')
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')  # Faster than FULL, still safe with WAL
            conn.close()
            print(f"✅ Enabled WAL mode for {study_name}", flush=True)
    except Exception as e:
        print(f"⚠️ Could not enable WAL mode: {e}", flush=True)
    
    print(f"📊 Study: {study_name}")
    print(f"📁 Storage: {study_path}")
    print()
    
    # ✅ Warm-start: enqueue best params from recent JSON results (top-K = 3)
    try:
        # Skip warm-start if a specific model is forced (to avoid fixed param mismatch)
        if os.getenv('FORCE_MODEL_CHOICE'):
            print("⏭️ Skipping warm-start enqueue (FORCE_MODEL_CHOICE is set)", flush=True)
        else:
        import glob
        prev_jsons = sorted(
            glob.glob(f"/opt/bist-pattern/results/optuna_pilot_features_on_h{horizon}_*.json"),
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )
        enqueued = 0
        for jf in prev_jsons:
            if enqueued >= 3:
                break
                with open(jf, 'r') as rf:
                    data_prev = json.load(rf)
                prev_syms = data_prev.get('symbols', [])
                if not prev_syms:
                    continue
                # Enqueue if at least one symbol overlaps
                if any(s in prev_syms for s in symbols):
                    bp = data_prev.get('best_params')
                    if isinstance(bp, dict) and bp:
                        study.enqueue_trial(bp)
                        enqueued += 1
                        print(f"↪️ Warm-start: enqueued params from {os.path.basename(jf)}", flush=True)
    except Exception:
        pass
    
    # ✅ FIX: Check existing trials before optimizing to prevent exceeding n_trials limit
    # When multiple processes write to the same study, each process may try to add n_trials,
    # causing total trials to exceed the limit (e.g., 1505/1500)
    existing_trials = len(study.trials)
    remaining_trials = max(0, n_trials - existing_trials)
    
    if remaining_trials <= 0:
        print(f"✅ Study already has {existing_trials} trials (target: {n_trials}), skipping optimization", flush=True)
    else:
        print(f"📊 Study has {existing_trials} existing trials, will add {remaining_trials} more (target: {n_trials})", flush=True)
    # Optimize
        # ✅ FIX: Pass study and n_trials to objective for per-trial limit checking
    study.optimize(
            lambda trial: objective(trial, symbols, horizon, engine, db_url, study=study, max_trials=n_trials),
            n_trials=remaining_trials,  # Only add remaining trials
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Save results to JSON (compatible with continuous_hpo_training_pipeline.py)
    best_value = float(study.best_value) if study.best_value is not None else 0.0
    best_params = study.best_params
    best_trial = study.best_trial
    
    # Extract feature flags, feature parameters, and hyperparameters
    feature_flags = {k: v for k, v in best_params.items() if k.startswith('enable_')}
    # Feature parameters: ml_loss_*, n_seeds, meta_stacking_alpha, ml_adaptive_k_*, ml_pattern_weight_scale_*, 
    # yolo_min_conf, smart_consensus_weight, smart_performance_weight, smart_sigma, smart_weight_*, fingpt_confidence_threshold
    feature_params_keys = [
        'ml_loss_mse_weight', 'ml_loss_threshold', 'ml_dir_penalty', 'n_seeds',
        'meta_stacking_alpha', 'yolo_min_conf', 'smart_consensus_weight',
        'smart_performance_weight', 'smart_sigma', 'fingpt_confidence_threshold',
        # ✅ NEW: per-model prior weights for smart ensemble
        'smart_weight_xgb', 'smart_weight_lgbm', 'smart_weight_cat',
        'external_min_days', 'external_smooth_alpha',
        'regime_scale_low', 'regime_scale_high',
    ]
    feature_params_keys += [k for k in best_params.keys() if k.startswith('ml_adaptive_k_') or k.startswith('ml_pattern_weight_scale_')]
    feature_params = {k: v for k, v in best_params.items() if k in feature_params_keys}
    # Hyperparameters: model parameters (xgb_*, lgb_*, cat_*)
    hyperparameters = {k: v for k, v in best_params.items() if not k.startswith('enable_') and k not in feature_params_keys}
    
    # Save JSON file (same format as optuna_hpo_pilot_features_on.py)
    # ✅ Add cycle number to JSON filename for better traceability
    cycle_num = int(os.getenv('HPO_CYCLE', '1'))
    output_file = f"/opt/bist-pattern/results/optuna_pilot_features_on_h{horizon}_c{cycle_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Observability: pruned count, avg trial time
    try:
        from optuna.trial import TrialState
        pruned_count = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
    except Exception:
        pruned_count = 0
    try:
        durations: list[float] = []
        for t in study.trials:
            start = getattr(t, 'datetime_start', None)
            end = getattr(t, 'datetime_complete', None)
            if start is not None and end is not None:
                durations.append(float((end - start).total_seconds()))
        avg_trial_time = float(np.mean(durations)) if durations else 0.0
    except Exception:
        avg_trial_time = 0.0
    
    # Best dirhit from trial user attrs if available (objective returns combined score)
    best_dirhit = None
    try:
        _val = best_trial.user_attrs.get('avg_dirhit', None)
        if isinstance(_val, (int, float)) and np.isfinite(_val):
            best_dirhit = float(_val)
        else:
            best_dirhit = None
    except Exception:
        best_dirhit = None
    if best_dirhit is None or not np.isfinite(best_dirhit):
        best_dirhit = best_value  # fallback
    
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
    
    result = {
        'best_value': float(best_value),  # Combined score
        'best_dirhit': float(best_dirhit),  # DirHit percentage (from user_attrs or fallback)
        'best_params': best_params,
        'best_trial': {
            'number': best_trial.number,
            'value': float(best_trial.value) if best_trial.value is not None else 0.0,
            'state': str(best_trial.state),
        },
        'n_trials': len(study.trials),
        'pruned_count': int(pruned_count),
        'avg_trial_time': float(avg_trial_time),
        'top_k_trials': topk,
        'study_name': study_name,
        'symbols': symbols,
        'horizon': horizon,
        'best_model_choice': best_params.get('model_choice'),
        'feature_flags': feature_flags,  # NEW: Feature flags that were optimized
        'feature_params': feature_params,  # NEW: Feature internal parameters
        'hyperparameters': hyperparameters,  # NEW: Separated hyperparameters
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
            'ML_USE_ADAPTIVE_LEARNING': '0',  # Always disabled during HPO (data leakage önleme)
            # Algorithm enables per best model_choice
            'ENABLE_XGBOOST': '1' if best_params.get('model_choice') in ('xgb', 'all') else '0',
            'ENABLE_LIGHTGBM': '1' if best_params.get('model_choice') in ('lgbm', 'all') else '0',
            'ENABLE_CATBOOST': '1' if best_params.get('model_choice') in ('cat', 'all') else '0',
        }
    }
    
    # ✅ CRITICAL FIX: Add exception handling for JSON file creation
    # If JSON file creation fails, try to recover by reading from study and retrying
    json_saved = False
    try:
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
        print(f"✅ Results saved to: {output_file}", flush=True)
        json_saved = True
    except Exception as json_err:
        print(f"❌ ERROR: Failed to save JSON file {output_file}: {json_err}", file=sys.stderr, flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        
        # ✅ CRITICAL FIX: Try to recover by reading best params from study and recreating JSON
        # This ensures we don't lose 1500+ trials of work
        print("🔄 Attempting to recover JSON file from study...", flush=True)
        try:
            # Recreate result dict (already has all data from study)
            result_recovered = result.copy()
            
            # Try alternative filename (add _recovered suffix)
            output_file_recovered = output_file.replace('.json', '_recovered.json')
            with open(output_file_recovered, 'w') as f:
                json.dump(result_recovered, f, indent=2)
            print(f"✅ Recovered JSON file saved to: {output_file_recovered}", flush=True)
            output_file = output_file_recovered
            json_saved = True
        except Exception as recover_err:
            print(f"❌ ERROR: Failed to recover JSON file: {recover_err}", file=sys.stderr, flush=True)
            print(f"⚠️ WARNING: JSON file not saved, but study file exists at {study_path}", file=sys.stderr, flush=True)
            print("⚠️ WARNING: Best params can be recovered from study file later", file=sys.stderr, flush=True)
            output_file = None  # Mark as failed
            json_saved = False
    
    # Ensure file permissions (only if file was created successfully)
    if output_file and json_saved:
    try:
        from bist_pattern.utils.file_utils import ensure_file_permissions
        ensure_file_permissions(Path(output_file))
    except ImportError:
        pass
        except Exception as perm_err:
            print(f"⚠️ WARNING: Failed to set file permissions for {output_file}: {perm_err}", file=sys.stderr, flush=True)
    
    # Print results
    print()
    print("=" * 100)
    print("📊 BEST TRIAL RESULTS")
    print("=" * 100)
    best_dirhit = study.best_value if study.best_value is not None else 0.0
    print(f"Best DirHit: {best_dirhit:.2f}%")
    print()
    print("Best Feature Flags:")
    for key, value in feature_flags.items():
        print(f"  {key}: {value}")
    print()
    if feature_params:
        print("Best Feature Parameters:")
        for key, value in feature_params.items():
            print(f"  {key}: {value}")
        print()
    print("Best Hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    print()
    if output_file and json_saved:
    print(f"✅ Results saved to: {output_file}")
    else:
        print(f"⚠️ WARNING: Results NOT saved to JSON file (study file exists at {study_path})")
        print("⚠️ WARNING: Best params can be recovered from study file later")
    print("=" * 100)
    
    # ✅ CRITICAL FIX: Exit with error if JSON file was not saved
    # This ensures run_hpo() can detect the failure and handle it appropriately
    if not json_saved:
        print("❌ FATAL: JSON file not saved after recovery attempts", file=sys.stderr, flush=True)
        sys.exit(1)  # Exit with error code so run_hpo() can detect failure


if __name__ == '__main__':
    main()
