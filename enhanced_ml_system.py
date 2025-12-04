"""
Enhanced ML System
XGBoost, LightGBM, CatBoost ile gelişmiş tahmin modelleri

CHANGELOG 2025-10-20:
- Added Directional Loss (hybrid MSE + directional accuracy)
- Increased prediction caps 5x
- Objective: Improve direction accuracy from 19% to 50%+
"""

import numpy as np
import pandas as pd
import os
import math
import logging
import threading
from typing import Optional, Dict
from bist_pattern.core.config_manager import ConfigManager
from bist_pattern.utils.error_handler import ErrorHandler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  # noqa: F401 (reserved for future direction classifier)
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIRECTIONAL LOSS FUNCTIONS (2025-10-20)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def directional_loss_xgboost(y_pred, dtrain):
    """
    Custom XGBoost objective: Hybrid MSE + Directional Loss
    
    Loss = alpha * MSE + (1-alpha) * DirectionalLoss
    
    DirectionalLoss penalizes wrong direction heavily:
    - Correct direction: small penalty (based on magnitude error)
    - Wrong direction: large penalty (proportional to |error|)
    """
    y_true = dtrain.get_label()
    
    # Hyperparameter: weight between MSE and directional
    # FIXED (2025-10-20): Increased from 0.3 to 0.5 for better balance
    # ✅ FIX: Use ConfigManager for consistent config access
    alpha = float(ConfigManager.get('ML_LOSS_MSE_WEIGHT', '0.5'))  # 50% MSE, 50% directional
    threshold = float(ConfigManager.get('ML_LOSS_THRESHOLD', '0.005'))  # 0.5% threshold
    # FIXED (2025-10-20): Reduced from 10.0 to 2.0 to prevent over-penalization
    dir_penalty = float(ConfigManager.get('ML_DIR_PENALTY', '2.0'))  # Penalty multiplier for wrong direction
    
    # MSE component
    error = y_pred - y_true
    mse_grad = error
    mse_hess = np.ones_like(error)
    
    # Directional component
    # Direction: sign with threshold for "flat"
    y_true_dir = np.where(np.abs(y_true) < threshold, 0, np.sign(y_true))
    y_pred_dir = np.where(np.abs(y_pred) < threshold, 0, np.sign(y_pred))
    
    # Direction match
    dir_match = (y_true_dir == y_pred_dir).astype(float)
    
    # Directional loss gradient:
    # - If directions match: small gradient (like MSE)
    # - If directions mismatch: moderate gradient (2x penalty, not 10x!)
    dir_grad = np.where(
        dir_match,
        error,  # Correct direction: normal MSE gradient
        error * dir_penalty  # Wrong direction: amplified gradient (2x)
    )
    
    # Hessian (second derivative) - keep it positive for convexity
    dir_hess = np.where(
        dir_match,
        np.ones_like(error),
        np.ones_like(error) * dir_penalty
    )
    
    # Combine
    grad = alpha * mse_grad + (1 - alpha) * dir_grad
    hess = alpha * mse_hess + (1 - alpha) * dir_hess
    
    return grad, hess


def directional_metric_xgboost(y_pred, dtrain):
    """
    Custom XGBoost eval metric: Direction Accuracy
    """
    y_true = dtrain.get_label()
    # ✅ FIX: Use ConfigManager for consistent config access
    threshold = float(ConfigManager.get('ML_LOSS_THRESHOLD', '0.005'))
    
    y_true_dir = np.where(np.abs(y_true) < threshold, 0, np.sign(y_true))
    y_pred_dir = np.where(np.abs(y_pred) < threshold, 0, np.sign(y_pred))
    
    accuracy = np.mean(y_true_dir == y_pred_dir)
    
    return 'dir_acc', accuracy


# ⚡ PURGED TIME-SERIES SPLIT - Data Leakage Prevention
class PurgedTimeSeriesSplit:
    """
    Time-series cross-validation with purging and embargo.
    
    Purging: Remove samples from training set that are too close to test set
    Embargo: Add gap between train and test to prevent lookahead bias
    
    Based on "Advances in Financial Machine Learning" by Marcos López de Prado
    """
    
    def __init__(self, n_splits=3, purge_gap=5, embargo_td=2):
        """
        Args:
            n_splits: Number of splits
            purge_gap: Number of samples to purge before test (removes overlap)
            embargo_td: Number of samples to embargo after train (future data gap)
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_td = embargo_td
    
    def split(self, X, y=None, groups=None):
        """Generate purged train/test indices."""
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Test set
            test_start = (i + 1) * fold_size
            test_end = test_start + fold_size
            test_indices = indices[test_start:test_end]
            
            # Train set (before test, with purging)
            train_end = test_start - self.purge_gap  # Purge gap
            if train_end <= 0:
                continue
            
            train_indices = indices[:train_end]
            
            # Apply embargo (remove recent samples that overlap with test timing)
            if self.embargo_td > 0 and len(train_indices) > self.embargo_td:
                train_indices = train_indices[:-self.embargo_td]
            
            if len(train_indices) > 10 and len(test_indices) > 3:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# Enhanced ML Models
try:
    import xgboost as xgb  # type: ignore[import]
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb  # type: ignore[import]
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore[assignment]

try:
    import catboost as cb  # type: ignore[import]
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None  # type: ignore[assignment]

# Base ML system
try:
    from ml_prediction_system import MLPredictionSystem  # type: ignore[import]
    BASE_ML_AVAILABLE = True
except ImportError:
    BASE_ML_AVAILABLE = False
    MLPredictionSystem = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ATOMIC FILE OPERATIONS HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _atomic_write_json(file_path: str, data: any, indent: int = 2) -> None:
    """
    Atomically write JSON file (temp file + rename to prevent corrupt files).
    
    Args:
        file_path: Target file path
        data: Data to write (will be serialized as JSON, can be dict, list, etc.)
        indent: JSON indentation (default: 2)
    """
    tmp_path = file_path + '.tmp'
    try:
        # Write to temp file
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=indent)
            f.flush()
            try:
                os.fsync(f.fileno())  # Force write to disk
            except Exception as e:
                logger.debug(f"fsync failed (may not be available on all systems): {e}")
        
        # Atomic rename (this is atomic on POSIX systems)
        os.replace(tmp_path, file_path)
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as cleanup_e:
            logger.debug(f"Failed to remove temp file {tmp_path} during error cleanup: {cleanup_e}")
        raise e


def _atomic_write_pickle(file_path: str, data: any) -> None:
    """
    Atomically write pickle file using joblib (temp file + rename to prevent corrupt files).
    
    Args:
        file_path: Target file path
        data: Data to write (will be serialized using joblib)
    """
    tmp_path = file_path + '.tmp'
    try:
        # Write to temp file
        joblib.dump(data, tmp_path)
        # Force write to disk (joblib doesn't expose file descriptor, so we can't fsync)
        # But rename is still atomic
        
        # Atomic rename (this is atomic on POSIX systems)
        os.replace(tmp_path, file_path)
    except Exception as e:
        # Clean up temp file if it exists
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as cleanup_e:
            logger.debug(f"Failed to remove temp file {tmp_path} during error cleanup: {cleanup_e}")
        raise e


def _atomic_read_modify_write_json(file_path: str, modify_func, default_data: Optional[dict] = None) -> None:
    """
    Atomically read-modify-write JSON file with file locking to prevent race conditions.
    
    Uses a separate lock file to ensure the lock persists across the atomic rename operation.
    This prevents race conditions where the lock would be on the old inode after os.replace().
    
    Args:
        file_path: Target file path
        modify_func: Function that takes existing data dict and returns modified data dict
        default_data: Default data if file doesn't exist (default: {})
    """
    import fcntl
    if default_data is None:
        default_data = {}
    
    # Use a separate lock file that persists across atomic rename
    # This prevents the race condition where the lock is on the old inode after os.replace()
    lock_file_path = file_path + '.lock'
    lock_fd = None
    try:
        # Open or create the lock file and acquire exclusive lock
        lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_RDWR, 0o644)
        
        # Acquire exclusive lock - will be held until explicitly released
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except Exception as e:
            logger.debug(f"Lock acquisition failed (best effort): {e}")
        
        # Read existing data from the actual data file (with lock held)
        existing_data = default_data.copy()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load existing data from {file_path}, using default: {e}")
        
        # Modify data (lock still held)
        modified_data = modify_func(existing_data)
        
        # Atomic write (lock still held)
        tmp_path = file_path + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                json.dump(modified_data, f, indent=2)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception as e:
                    logger.debug(f"fsync failed: {e}")
            
            # Atomic rename (lock still held on separate lock file, preventing concurrent writes)
            # The lock file is separate, so it's not affected by the rename operation
            os.replace(tmp_path, file_path)
        finally:
            # Clean up temp file if it still exists
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.debug(f"Failed to remove temp file {tmp_path}: {e}")
    finally:
        # Release lock and close file descriptor
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except Exception as e:
                logger.debug(f"Failed to unlock: {e}")
            try:
                os.close(lock_fd)
            except Exception as e:
                logger.debug(f"Failed to close lock file descriptor: {e}")


class EnhancedMLSystem:
    """Gelişmiş ML tahmin sistemi"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        # Reference ensemble weights/metrics injected during evaluation (e.g., to mirror HPO behavior)
        self.reference_historical_r2: Dict[str, Dict[str, float]] = {}
        self.use_reference_historical_r2 = False
        # ⚡ NEW: Meta-learners storage (Ridge for stacking)
        self.meta_learners = {}  # {symbol_horizon: Ridge model}
        # ✅ FIX: Store model order for meta-stacking (ensures prediction uses same order as training)
        self.meta_model_orders = {}  # {symbol_horizon: ['xgboost', 'lightgbm', 'catboost']}
        # ✅ FIX: Use ConfigManager for consistent config access
        self.model_directory = ConfigManager.get('ML_MODEL_PATH', "enhanced_ml_models")
        self.prediction_horizons = [1, 3, 7, 14, 30]  # 1D, 3D, 7D, 14D, 30D
        # Optional ENV override: ML_HORIZONS="14,30" or "1,3,7,14,30"
        try:
            # ✅ FIX: Read ML_HORIZONS directly from os.environ (not ConfigManager) to avoid parsing issues
            # ⚡ CRITICAL: ConfigManager.get() parses the value (e.g., '14' -> 14 int), but we need string to split
            # We use os.getenv() directly to get the raw string value
            _h_env = os.getenv('ML_HORIZONS', '').strip()
            if _h_env:
                # Convert to string if ConfigManager parsed it (fallback)
                if not isinstance(_h_env, str):
                    _h_env = str(_h_env)
                parsed = []
                for tok in _h_env.split(','):
                    tok = tok.strip().lower().replace('d', '')
                    if tok:
                        parsed.append(int(tok))
                if parsed:
                    self.prediction_horizons = parsed
                    logger.info(f"✅ ML_HORIZONS override: {self.prediction_horizons}")
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_horizons', level='debug')
            pass
        self.feature_columns = {}  # Initialize feature columns dict for adaptive learning
        # Optional feature flags
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.enable_talib_patterns = str(ConfigManager.get('ENABLE_TALIB_PATTERNS', 'True')).lower() in ('1', 'true', 'yes')
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_talib', level='debug')
            self.enable_talib_patterns = True
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.enable_external_features = str(ConfigManager.get('ENABLE_EXTERNAL_FEATURES', 'True')).lower() in ('1', 'true', 'yes')
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_external', level='debug')
            self.enable_external_features = True
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.enable_fingpt_features = str(ConfigManager.get('ENABLE_FINGPT_FEATURES', 'True')).lower() in ('1', 'true', 'yes')
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_fingpt', level='debug')
            self.enable_fingpt_features = True
        try:
            # ⚡ NEW: Meta-stacking with Ridge learner OOF (default: enabled for short horizons only!)
            # ✅ FIX: Meta-stacking should only be used for short horizons (1d, 3d, 7d) - not long horizons
            # Short horizons have more noise, meta-stacking helps. Long horizons (14d, 30d) are already smooth.
            # ML_USE_STACKED_SHORT=1 means meta-stacking ONLY for short horizons (1d, 3d, 7d)
            # ENABLE_META_STACKING=1 means meta-stacking for ALL horizons (if ML_USE_STACKED_SHORT not set)
            enable_meta_general = str(ConfigManager.get('ENABLE_META_STACKING', 'False')).lower() in ('1', 'true', 'yes')
            use_stacked_short = str(ConfigManager.get('ML_USE_STACKED_SHORT', '1')).lower() in ('1', 'true', 'yes')
            # If ML_USE_STACKED_SHORT=1, enforce short-only mode (even if ENABLE_META_STACKING=1)
            if use_stacked_short:
                # Short-only mode: meta-stacking only for short horizons (1d, 3d, 7d)
                self.enable_meta_stacking = True  # Enabled, but will be filtered by horizon
                self.use_meta_stacking_short_only = True
            elif enable_meta_general:
                # General mode: meta-stacking for all horizons (backward compatibility)
                self.enable_meta_stacking = True
                self.use_meta_stacking_short_only = False
            else:
                # Disabled
                self.enable_meta_stacking = False
                self.use_meta_stacking_short_only = False
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_meta_stacking', level='debug')
            self.enable_meta_stacking = False
            self.use_meta_stacking_short_only = False
        try:
            # ⚡ NEW: Seed bagging (multiple random seeds for variance reduction)
            # ✅ FIX: Use ConfigManager for consistent config access
            self.enable_seed_bagging = str(ConfigManager.get('ENABLE_SEED_BAGGING', 'True')).lower() in ('1', 'true', 'yes')
            self.n_seeds = int(ConfigManager.get('N_SEEDS', '3'))  # 3 seeds by default
            self.base_seeds = [42, 123, 456, 789, 999][:self.n_seeds]  # Use first N seeds
            logger.info(f"📋 Seed Bagging (__init__): enable={self.enable_seed_bagging}, n_seeds={self.n_seeds}, base_seeds={self.base_seeds}")
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_seed_bagging', level='debug')
            self.enable_seed_bagging = True  # Enable by default!
            self.n_seeds = 3
            self.base_seeds = [42, 123, 456]
            logger.warning(f"⚠️ Seed bagging init failed, using defaults: enable={self.enable_seed_bagging}, n_seeds={self.n_seeds}")
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.enable_yolo_features = str(ConfigManager.get('ENABLE_YOLO_FEATURES', 'True')).lower() in ('1', 'true', 'yes')
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_yolo', level='debug')
            self.enable_yolo_features = True
        # Prediction-time feature guard (schema drift tolerance)
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.enable_pred_feature_guard = str(ConfigManager.get('ENABLE_PREDICTION_FEATURE_GUARD', 'False')).lower() in ('1', 'true', 'yes')
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_pred_guard', level='debug')
            self.enable_pred_feature_guard = False
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            _guard_pref = ConfigManager.get('GUARD_ALLOWED_PREFIXES', 'fingpt_,yolo_,usdtry,cds,rate') or 'fingpt_,yolo_,usdtry,cds,rate'
            self.guard_allowed_prefixes = [p.strip() for p in _guard_pref.split(',') if p.strip()]
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_guard_prefixes', level='debug')
            self.guard_allowed_prefixes = ['fingpt_', 'yolo_', 'usdtry', 'cds', 'rate']
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.guard_max_missing_ratio = float(ConfigManager.get('GUARD_MAX_MISSING_RATIO', '0.1'))
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_guard_missing', level='debug')
            self.guard_max_missing_ratio = 0.1
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.guard_penalty_max = float(ConfigManager.get('GUARD_PENALTY_MAX', '0.2'))
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_guard_penalty', level='debug')
            self.guard_penalty_max = 0.2
        # External features directory (backfilled files live here)
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.external_feature_dir = ConfigManager.get('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_external_dir', level='debug')
            self.external_feature_dir = '/opt/bist-pattern/logs/feature_backfill'
        # Training parallelism and early stopping configuration (ENV-driven)
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.train_threads = int(ConfigManager.get('ML_TRAIN_THREADS', '2'))
            if self.train_threads <= 0:
                # Fallback to CPU count if invalid
                cpu_count = os.cpu_count() or 2
                self.train_threads = max(1, int(cpu_count))
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_train_threads', level='debug')
            cpu_count = os.cpu_count() or 2
            self.train_threads = max(1, int(cpu_count))
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.early_stop_rounds = int(ConfigManager.get('ML_EARLY_STOP_ROUNDS', '50'))
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_early_stop', level='debug')
            self.early_stop_rounds = 50
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.early_stop_min_val = int(ConfigManager.get('ML_EARLY_STOP_MIN_VAL', '10'))
        except Exception as e:
            ErrorHandler.handle(e, 'enhanced_ml_init_early_stop_min', level='debug')
            self.early_stop_min_val = 10
        # Optional stop-sentinel file path (for graceful halt)
        # ✅ FIX: Use ConfigManager for consistent config access
        self.stop_file_path = ConfigManager.get('TRAIN_STOP_FILE', '/opt/bist-pattern/.cache/STOP_TRAIN')
        
        # Model klasörünü oluştur
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Base ML system
        if BASE_ML_AVAILABLE:
            self.base_ml = MLPredictionSystem()  # type: ignore[misc]
        
        # CatBoost çalışma dizini (write permission hatalarını önlemek için)
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.catboost_train_dir = ConfigManager.get('CATBOOST_TRAIN_DIR', '/opt/bist-pattern/.cache/catboost')
            os.makedirs(self.catboost_train_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create catboost_train_dir from config, using /tmp: {e}")
            # Son çare: tmp dizini
            self.catboost_train_dir = '/tmp/catboost_info'
            try:
                os.makedirs(self.catboost_train_dir, exist_ok=True)
            except Exception as e2:
                logger.error(f"Failed to create /tmp/catboost_info: {e2}")

        # Model kayıt dizini (yazılabilir)
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.model_directory = ConfigManager.get('ML_MODEL_PATH', '/opt/bist-pattern/.cache/enhanced_ml_models')
            os.makedirs(self.model_directory, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create model_directory from config, using fallback: {e}")
            try:
                self.model_directory = 'enhanced_ml_models'
                os.makedirs(self.model_directory, exist_ok=True)
            except Exception as e2:
                logger.error(f"Failed to create fallback model_directory: {e2}")

        logger.info("🧠 Enhanced ML System başlatıldı")
        logger.info(f"📊 XGBoost: {XGBOOST_AVAILABLE}")
        logger.info(f"📊 LightGBM: {LIGHTGBM_AVAILABLE}")
        logger.info(f"📊 CatBoost: {CATBOOST_AVAILABLE}")
    
    @staticmethod
    def _smape(y_true, y_pred, eps: float = 1e-8) -> float:
        try:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            denom = np.abs(y_true_arr) + np.abs(y_pred_arr) + eps
            return float(np.mean(2.0 * np.abs(y_pred_arr - y_true_arr) / denom))
        except Exception as e:
            logger.debug(f"SMAPE calculation failed: {e}")
            return float('nan')
    
    @staticmethod
    def _calculate_score(dirhit: float, nrmse: float, horizon: int) -> float:
        """
        Calculate composite score (same as HPO):
        Score = 0.7 × DirHit - k × nRMSE
        
        Args:
            dirhit: Directional hit rate (percentage, 0-100)
            nrmse: Normalized RMSE
            horizon: Prediction horizon in days
            
        Returns:
            Composite score (float)
        """
        try:
            k = 6.0 if horizon in (1, 3, 7) else 4.0
            score = 0.7 * dirhit - k * (nrmse if np.isfinite(nrmse) else 3.0)
            return float(score)
        except Exception as e:
            logger.debug(f"Score calculation failed: {e}")
            return float('nan')
    
    @staticmethod
    def _r2_to_confidence(r2_val):
        """
        Convert R² score to confidence [0-1]
        R² can be negative (model worse than mean baseline)
        
        CRITICAL FIX: R² directly as confidence was wrong!
        - R² < 0 → poor model, but still give some confidence (0.3-0.4)
        - R² = 0 → baseline (0.45)
        - R² = 0.5 → moderate (0.7)
        - R² = 0.8 → good (0.85)
        - R² = 1.0 → excellent (0.95, never 1.0 for humility)
        
        Formula: conf = 0.3 + 0.65 / (1 + exp(-5*r2))
        """
        try:
            # Sigmoid transformation shifted and scaled
            exponent = -5.0 * float(r2_val)
            confidence = 0.3 + (0.65 / (1.0 + np.exp(exponent)))
            # Clamp to [0.2, 0.95] for safety
            return float(np.clip(confidence, 0.2, 0.95))
        except Exception as e:
            logger.debug(f"R2 to confidence conversion failed: {e}")
            return 0.5  # Fallback

    def create_advanced_features(self, data, symbol: str | None = None, horizon: int | None = None):
        """Gelişmiş feature engineering
        
        Args:
            data: Price data DataFrame
            symbol: Stock symbol (optional)
            horizon: Prediction horizon in days (optional, for horizon-aware features)
        """
        try:
            if BASE_ML_AVAILABLE:
                # Base features from original system
                df = self.base_ml.create_technical_features(data)
            else:
                df = data.copy()
                # Basic fallback features
                if 'Close' in df.columns:
                    df = df.rename(columns={
                        'Open': 'open', 'High': 'high', 
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    })
            
            # Advanced technical indicators
            self._add_advanced_indicators(df)
            
            # Market microstructure features
            self._add_microstructure_features(df)
            
            # Volatility features
            self._add_volatility_features(df)
            
            # Cyclical features
            self._add_cyclical_features(df)
            
            # Statistical features
            self._add_statistical_features(df)
            
            # ⚡ NEW: Liquidity/Volume tier features
            self._add_liquidity_features(df)
            
            # ⚡ NEW: Macro economic features (USDTRY, CDS, TCMB Rate)
            # ✅ FIX: Use ConfigManager for consistent config access
            if symbol and str(ConfigManager.get('ML_ENABLE_INTERNAL_MACRO', '1')).lower() in ('1', 'true', 'yes'):
                self._add_macro_features(df)
            
            # ⚡ TA-Lib candlestick patterns: DISABLE for short horizons (1d, 3d, 7d) - they're noisy!
            # Only enable for longer horizons (14d, 30d) where patterns are more meaningful
            # Note: Patterns are now added per-horizon in train_enhanced_models() for better control
            # This is a fallback for when create_advanced_features is called without horizon context
            if self.enable_talib_patterns and horizon is None:
                # If horizon unknown, respect env variable (default: enable if ENABLE_TALIB_PATTERNS_SHORT is set)
                # ✅ FIX: Use ConfigManager for consistent config access
                talib_short_ok = str(ConfigManager.get('ENABLE_TALIB_PATTERNS_SHORT', '0')).lower() in ('1', 'true', 'yes')
                if talib_short_ok or True:  # Default: enable if horizon unknown (backward compatibility)
                    self._add_candlestick_features(df)
            
            # NEW: Event-anchored pattern features (H&S measured move, breakout, recency)
            try:
                from bist_pattern.features.pattern_features import build_pattern_features
                df = build_pattern_features(df)
                logger.info("✅ Pattern features merged (hs_signal, hs_conf, hs_neckline, hs_mm, breakout)")
            except Exception as _pf_err:
                logger.warning(f"Pattern features unavailable: {_pf_err}")
            
            # Optional: Merge external backfilled features (FinGPT / YOLO)
            if symbol and self.enable_external_features:
                self._merge_external_features(symbol, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Advanced feature engineering hatası: {e}")
            return data

    def _add_candlestick_features(self, df):
        """Lightweight TA-Lib candlestick features: last-3 day bull/bear counts and today's signal."""
        try:
            # Ensure target columns exist even if TA-Lib unavailable
            def _ensure_pat_cols(_df: pd.DataFrame) -> None:
                cols = [
                    'pat_bull3', 'pat_bear3', 'pat_net3', 'pat_today',
                    'pat_bull5', 'pat_bear5', 'pat_net5',
                    'pat_bull10', 'pat_bear10', 'pat_net10',
                ]
                for c in cols:
                    if c not in _df.columns:
                        _df[c] = 0.0
            try:
                import talib  # type: ignore
            except Exception as e:
                logger.debug(f"TA-Lib import failed, skipping candlestick features: {e}")
                _ensure_pat_cols(df)
                return
            if not all(c in df.columns for c in ('open', 'high', 'low', 'close')):
                _ensure_pat_cols(df)
                return
            op = df['open'].astype(float)
            hi = df['high'].astype(float)
            lo = df['low'].astype(float)
            cl = df['close'].astype(float)

            # Subset of robust patterns; TA-Lib returns 100/-100/0 values
            pat_series = []
            try:
                pat_series.append(talib.CDLENGULFING(op, hi, lo, cl))
            except Exception as e:
                logger.debug(f"TA-Lib CDLENGULFING failed: {e}")
            try:
                pat_series.append(talib.CDLHAMMER(op, hi, lo, cl))
            except Exception as e:
                logger.debug(f"TA-Lib CDLHAMMER failed: {e}")
            try:
                pat_series.append(talib.CDLSHOOTINGSTAR(op, hi, lo, cl))
            except Exception as e:
                logger.debug(f"TA-Lib CDLSHOOTINGSTAR failed: {e}")
            try:
                pat_series.append(talib.CDLHARAMI(op, hi, lo, cl))
            except Exception as e:
                logger.debug(f"TA-Lib CDLHARAMI failed: {e}")
            try:
                pat_series.append(talib.CDLDOJI(op, hi, lo, cl))
            except Exception as e:
                logger.debug(f"TA-Lib CDLDOJI failed: {e}")

            if not pat_series:
                _ensure_pat_cols(df)
                return

            pats = None
            try:
                import pandas as _pd  # local alias
                pats = sum(pat_series)
                # Ensure pandas Series
                if not hasattr(pats, 'tail'):
                    pats = _pd.Series(pats, index=df.index)
            except Exception as e:
                logger.debug(f"Pattern series creation failed: {e}")
                return

            # ⚡ Enhanced: Pattern strength and confidence features (gürültüyü önlemek için)
            # Pattern strength: abs(value) / 100.0 (0.0-1.0 range)
            # Multiple pattern confirmation: birden fazla pattern aynı yönde ise confidence artar

            # Leakage-safe rolling features per row (uses only up-to-date info)
            try:
                # Binary flags
                bull_flag = (pats > 0).astype(float)
                bear_flag = (pats < 0).astype(float)
                
                # Pattern strength (pattern'in gücü, 0.0-1.0)
                pat_strength = np.clip(np.abs(pats) / 100.0, 0.0, 1.0).astype(float)
                
                # Basic counts (existing) - 3-day window
                df['pat_bull3'] = bull_flag.rolling(3, min_periods=1).sum().astype(float)
                df['pat_bear3'] = bear_flag.rolling(3, min_periods=1).sum().astype(float)
                df['pat_net3'] = (df['pat_bull3'] - df['pat_bear3']).astype(float)
                df['pat_today'] = np.clip(pats / 100.0, -1.0, 1.0).astype(float)
                
                # Extended windows (5-day and 10-day) - MUST be calculated BEFORE confidence features
                df['pat_bull5'] = bull_flag.rolling(5, min_periods=1).sum().astype(float)
                df['pat_bear5'] = bear_flag.rolling(5, min_periods=1).sum().astype(float)
                df['pat_net5'] = (df['pat_bull5'] - df['pat_bear5']).astype(float)
                df['pat_bull10'] = bull_flag.rolling(10, min_periods=1).sum().astype(float)
                df['pat_bear10'] = bear_flag.rolling(10, min_periods=1).sum().astype(float)
                df['pat_net10'] = (df['pat_bull10'] - df['pat_bear10']).astype(float)
                
                # ⚡ NEW: Pattern strength features (pattern'in gücü)
                df['pat_strength3'] = pat_strength.rolling(3, min_periods=1).mean().astype(float)  # Average strength
                df['pat_strength5'] = pat_strength.rolling(5, min_periods=1).mean().astype(float)
                df['pat_strength10'] = pat_strength.rolling(10, min_periods=1).mean().astype(float)
                df['pat_strength_today'] = pat_strength.astype(float)
                
                # ⚡ NEW: Multiple pattern confirmation (birden fazla pattern aynı yönde ise)
                # Confidence = base (0.5) + multiple_pattern_bonus (0.0-0.3) + strength_bonus (0.0-0.2)
                # NOTE: pat_bull5/bear5 and pat_bull10/bear10 MUST be calculated BEFORE this
                df['pat_conf3'] = np.clip(
                    0.5 +  # Base confidence
                    np.minimum(df['pat_bull3'].fillna(0) + df['pat_bear3'].fillna(0), 3.0) / 10.0 +  # Multiple pattern bonus (0.0-0.3)
                    df['pat_strength3'].fillna(0) * 0.2,  # Strength bonus (0.0-0.2)
                    0.0, 1.0
                ).astype(float)
                
                df['pat_conf5'] = np.clip(
                    0.5 +
                    np.minimum(df['pat_bull5'].fillna(0) + df['pat_bear5'].fillna(0), 5.0) / 15.0 +
                    df['pat_strength5'].fillna(0) * 0.2,
                    0.0, 1.0
                ).astype(float)
                
                df['pat_conf10'] = np.clip(
                    0.5 +
                    np.minimum(df['pat_bull10'].fillna(0) + df['pat_bear10'].fillna(0), 10.0) / 30.0 +
                    df['pat_strength10'].fillna(0) * 0.2,
                    0.0, 1.0
                ).astype(float)
                
                # ⚡ NEW: YOLO confirmation boost (eğer YOLO feature'ları varsa)
                # YOLO ile uyumlu pattern'ler daha güvenilir (1.2x-1.5x boost)
                # Note: YOLO features are merged in _merge_external_features() before this function is called
                if 'yolo_density' in df.columns and 'yolo_bull' in df.columns and 'yolo_bear' in df.columns:
                    try:
                        # YOLO bull/bear flags
                        yolo_bull = (df['yolo_bull'].fillna(0) > 0.5).astype(float)
                        yolo_bear = (df['yolo_bear'].fillna(0) > 0.5).astype(float)
                        
                        # TA-Lib pattern direction ile YOLO uyumu kontrol et
                        # Bullish pattern + YOLO bullish = boost
                        # Bearish pattern + YOLO bearish = boost
                        pat_bullish = (bull_flag > 0).astype(float)
                        pat_bearish = (bear_flag > 0).astype(float)
                        
                        # YOLO confirmation multiplier (rolling window: 1.0 = no boost, 1.3 = 30% boost)
                        # Rolling mean to match the confidence window
                        yolo_confirm_mult_3 = 1.0 + (
                            ((pat_bullish.rolling(3, min_periods=1).sum() > 0) & 
                              (yolo_bull.rolling(3, min_periods=1).sum() > 0)).astype(float) * 0.3 +  # Bullish match
                            ((pat_bearish.rolling(3, min_periods=1).sum() > 0) & 
                              (yolo_bear.rolling(3, min_periods=1).sum() > 0)).astype(float) * 0.3    # Bearish match
                        )
                        
                        yolo_confirm_mult_5 = 1.0 + (
                            ((pat_bullish.rolling(5, min_periods=1).sum() > 0) & 
                              (yolo_bull.rolling(5, min_periods=1).sum() > 0)).astype(float) * 0.3 +
                            ((pat_bearish.rolling(5, min_periods=1).sum() > 0) & 
                              (yolo_bear.rolling(5, min_periods=1).sum() > 0)).astype(float) * 0.3
                        )
                        
                        yolo_confirm_mult_10 = 1.0 + (
                            ((pat_bullish.rolling(10, min_periods=1).sum() > 0) & 
                              (yolo_bull.rolling(10, min_periods=1).sum() > 0)).astype(float) * 0.3 +
                            ((pat_bearish.rolling(10, min_periods=1).sum() > 0) & 
                              (yolo_bear.rolling(10, min_periods=1).sum() > 0)).astype(float) * 0.3
                        )
                        
                        # Apply YOLO confirmation to confidence features
                        df['pat_conf3'] = np.clip(df['pat_conf3'] * yolo_confirm_mult_3, 0.0, 1.0).astype(float)
                        df['pat_conf5'] = np.clip(df['pat_conf5'] * yolo_confirm_mult_5, 0.0, 1.0).astype(float)
                        df['pat_conf10'] = np.clip(df['pat_conf10'] * yolo_confirm_mult_10, 0.0, 1.0).astype(float)
                    except Exception as _yolo_e:
                        logger.debug(f"YOLO confirmation boost failed: {_yolo_e}")
            except Exception as e:
                logger.debug(f"Candlestick features failed, skipping: {e}")
                # If any issue occurs, skip silently to avoid breaking pipeline
                _ensure_pat_cols(df)
        except Exception as e:
            logger.debug(f"TA-Lib candlestick features skipped: {e}")

    def _merge_external_features(self, symbol: str, df: pd.DataFrame) -> None:
        """Merge offline backfilled features (FinGPT sentiment and YOLO pattern density) if available.

        Expected files (CSV) under EXTERNAL_FEATURE_DIR:
          - fingpt/{SYMBOL}.csv with columns like: date, sentiment_score (or score/sentiment), news_count
          - yolo/{SYMBOL}.csv   with columns like: date, yolo_density (or density), yolo_bull, yolo_bear, yolo_score
        """
        try:
            if df is None or len(df) == 0:
                return
            dates = pd.to_datetime(df.index).normalize()

            def _load_csv_safe(path: str) -> pd.DataFrame | None:
                try:
                    if not os.path.exists(path):
                        return None
                    tmp = pd.read_csv(path)
                    # Flexible date parsing
                    if 'date' in tmp.columns:
                        tmp['date'] = pd.to_datetime(tmp['date']).dt.normalize()
                        tmp = tmp.set_index('date').sort_index()
                    elif 'timestamp' in tmp.columns:
                        tmp['timestamp'] = pd.to_datetime(tmp['timestamp']).dt.normalize()
                        tmp = tmp.set_index('timestamp').sort_index()
                    else:
                        # try to parse index if unnamed
                        tmp.index = pd.to_datetime(tmp.index).normalize()
                    return tmp
                except Exception as _e:
                    logger.debug(f"External feature load failed: {path} → {_e}")
                    return None

            # Read external feature knobs
            try:
                min_days_required = int(ConfigManager.get('EXTERNAL_MIN_DAYS', '0') or '0')
            except Exception as e:
                logger.debug(f"Failed to get EXTERNAL_MIN_DAYS, using 0: {e}")
                min_days_required = 0
            try:
                smooth_alpha = float(ConfigManager.get('EXTERNAL_SMOOTH_ALPHA', '0') or '0')
            except Exception as e:
                logger.debug(f"Failed to get EXTERNAL_SMOOTH_ALPHA, using 0.0: {e}")
                smooth_alpha = 0.0
            use_smoothing = smooth_alpha > 0.0 and smooth_alpha < 1.0
            
            # FinGPT features
            if self.enable_fingpt_features:
                f_csv = os.path.join(self.external_feature_dir, 'fingpt', f'{symbol}.csv')
                fdf = _load_csv_safe(f_csv)
                if fdf is not None and len(fdf) > 0:
                    if min_days_required > 0 and len(fdf) < min_days_required:
                        # Not enough days → zero out FinGPT features
                        df['fingpt_sent'] = 0.0
                        df['fingpt_news'] = 0.0
                    else:
                        # pick score column
                        score_cols = [c for c in (
                            'sentiment_score', 'score', 'avg_score', 'sentiment', 'sentiment_avg', 'polarity'
                        ) if c in fdf.columns]
                        count_cols = [c for c in ('news_count', 'count', 'n') if c in fdf.columns]
                        # Align to DF dates
                        # ⚡ FIX: Use method='ffill' to prevent lookahead bias (same as macro features)
                        fdf = fdf.reindex(dates, method='ffill')
                        if score_cols:
                            try:
                                df['fingpt_sent'] = pd.to_numeric(fdf[score_cols[0]], errors='coerce').fillna(0.0).astype(float)
                                if use_smoothing:
                                    df['fingpt_sent'] = df['fingpt_sent'].ewm(alpha=smooth_alpha, adjust=False).mean()
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent fail
                                logger.debug(f"Failed to merge FinGPT sentiment for {symbol}: {e}")
                                df['fingpt_sent'] = 0.0
                        else:
                            df['fingpt_sent'] = 0.0
                        if count_cols:
                            try:
                                df['fingpt_news'] = pd.to_numeric(fdf[count_cols[0]], errors='coerce').fillna(0.0).astype(float)
                                if use_smoothing:
                                    df['fingpt_news'] = df['fingpt_news'].ewm(alpha=smooth_alpha, adjust=False).mean()
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent fail
                                logger.debug(f"Failed to merge FinGPT news count for {symbol}: {e}")
                                df['fingpt_news'] = 0.0
                        else:
                            df['fingpt_news'] = 0.0

            # YOLO features
            if self.enable_yolo_features:
                y_csv = os.path.join(self.external_feature_dir, 'yolo', f'{symbol}.csv')
                ydf = _load_csv_safe(y_csv)
                if ydf is not None and len(ydf) > 0:
                    if min_days_required > 0 and len(ydf) < min_days_required:
                        df['yolo_density'] = 0.0
                        df['yolo_bull'] = 0.0
                        df['yolo_bear'] = 0.0
                        df['yolo_score'] = 0.0
                    else:
                        dens_cols = [c for c in ('yolo_density', 'density', 'det_density') if c in ydf.columns]
                        bull_cols = [c for c in ('yolo_bull', 'bull', 'bull_count') if c in ydf.columns]
                        bear_cols = [c for c in ('yolo_bear', 'bear', 'bear_count') if c in ydf.columns]
                        score_cols = [c for c in ('yolo_score', 'score', 'align') if c in ydf.columns]
                        # ⚡ FIX: Use method='ffill' to prevent lookahead bias (same as macro features)
                        ydf = ydf.reindex(dates, method='ffill')
                        if dens_cols:
                            try:
                                df['yolo_density'] = pd.to_numeric(ydf[dens_cols[0]], errors='coerce').fillna(0.0).astype(float)
                                if use_smoothing:
                                    df['yolo_density'] = df['yolo_density'].ewm(alpha=smooth_alpha, adjust=False).mean()
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent fail
                                logger.debug(f"Failed to merge YOLO density for {symbol}: {e}")
                                df['yolo_density'] = 0.0
                        else:
                            df['yolo_density'] = 0.0
                        if bull_cols:
                            try:
                                df['yolo_bull'] = pd.to_numeric(ydf[bull_cols[0]], errors='coerce').fillna(0.0).astype(float)
                                if use_smoothing:
                                    df['yolo_bull'] = df['yolo_bull'].ewm(alpha=smooth_alpha, adjust=False).mean()
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent fail
                                logger.debug(f"Failed to merge YOLO bull for {symbol}: {e}")
                                df['yolo_bull'] = 0.0
                        else:
                            df['yolo_bull'] = 0.0
                        if bear_cols:
                            try:
                                df['yolo_bear'] = pd.to_numeric(ydf[bear_cols[0]], errors='coerce').fillna(0.0).astype(float)
                                if use_smoothing:
                                    df['yolo_bear'] = df['yolo_bear'].ewm(alpha=smooth_alpha, adjust=False).mean()
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent fail
                                logger.debug(f"Failed to merge YOLO bear for {symbol}: {e}")
                                df['yolo_bear'] = 0.0
                        else:
                            df['yolo_bear'] = 0.0
                        if score_cols:
                            try:
                                df['yolo_score'] = pd.to_numeric(ydf[score_cols[0]], errors='coerce').fillna(0.0).astype(float)
                                if use_smoothing:
                                    df['yolo_score'] = df['yolo_score'].ewm(alpha=smooth_alpha, adjust=False).mean()
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent fail
                                logger.debug(f"Failed to merge YOLO score for {symbol}: {e}")
                                df['yolo_score'] = 0.0
                        else:
                            df['yolo_score'] = 0.0
        except Exception as e:
            logger.debug(f"External feature merge skipped: {e}")
    
    def _add_advanced_indicators(self, df):
        """Gelişmiş teknik indikatörler"""
        try:
            # ATR (Average True Range)
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift())
            df['low_close'] = np.abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            for period in [14, 21]:
                df[f'atr_{period}'] = df['true_range'].rolling(period).mean()
            
            # Commodity Channel Index (CCI)
            for period in [14, 20]:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                mean_typical = typical_price.rolling(period).mean()
                mean_deviation = typical_price.rolling(period).apply(
                    lambda x: np.mean(np.abs(x - x.mean()))
                )
                df[f'cci_{period}'] = (typical_price - mean_typical) / (0.015 * mean_deviation + 1e-10)
                df[f'cci_{period}'] = np.clip(df[f'cci_{period}'], -200.0, 200.0)  # Prevent overflow (CCI typically -100 to +100)
            
            # Money Flow Index (MFI)
            if 'volume' in df.columns:
                for period in [14, 21]:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    money_flow = typical_price * df['volume']
                    
                    positive_flow = money_flow.where(df['close'] > df['close'].shift(), 0)
                    negative_flow = money_flow.where(df['close'] < df['close'].shift(), 0)
                    
                    positive_sum = positive_flow.rolling(period).sum()
                    negative_sum = negative_flow.rolling(period).sum()
                    
                    money_ratio = positive_sum / (negative_sum + 1e-10)
                    money_ratio = np.clip(money_ratio, 0.0, 1000.0)  # Prevent overflow
                    df[f'mfi_{period}'] = 100 - (100 / (1 + money_ratio))
                    df[f'mfi_{period}'] = np.clip(df[f'mfi_{period}'], 0.0, 100.0)  # MFI should be 0-100
            
            # Parabolic SAR (proper calculation)
            try:
                # Simple SAR approximation (trend following)
                high = df['high']
                low = df['low']
                close = df['close']
                
                # Start with first close
                sar = close.copy()
                trend = 1  # 1 = uptrend, -1 = downtrend
                af = 0.02  # acceleration factor
                ep = high.iloc[0]  # extreme point
                
                for i in range(1, len(df)):
                    # Update SAR based on trend
                    if trend == 1:  # Uptrend
                        sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                        if low.iloc[i] < sar.iloc[i]:
                            trend = -1
                            sar.iloc[i] = ep
                            ep = low.iloc[i]
                            af = 0.02
                        elif high.iloc[i] > ep:
                            ep = high.iloc[i]
                            af = min(af + 0.02, 0.2)
                    else:  # Downtrend
                        sar.iloc[i] = sar.iloc[i-1] - af * (sar.iloc[i-1] - ep)
                        if high.iloc[i] > sar.iloc[i]:
                            trend = 1
                            sar.iloc[i] = ep
                            ep = high.iloc[i]
                            af = 0.02
                        elif low.iloc[i] < ep:
                            ep = low.iloc[i]
                            af = min(af + 0.02, 0.2)
                
                df['sar'] = sar
            except Exception as e:
                logger.debug(f"SAR calculation failed, using EMA fallback: {e}")
                # Fallback: simple EMA if SAR fails
                df['sar'] = df['close'].ewm(span=20).mean()
            
            # Awesome Oscillator
            sma_5 = ((df['high'] + df['low']) / 2).rolling(5).mean()
            sma_34 = ((df['high'] + df['low']) / 2).rolling(34).mean()
            df['awesome_oscillator'] = sma_5 - sma_34
            
        except Exception as e:
            logger.error(f"Advanced indicators hatası: {e}")
    
    def _add_microstructure_features(self, df):
        """Market microstructure features"""
        try:
            # OHLC ratios
            # ✅ FIX: Use epsilon instead of replace(0, np.nan) to prevent INF from very small denominators
            # ✅ FIX: Aggressive clipping to prevent overflow (>1e10 values)
            high_low_diff = (df['high'] - df['low'])
            df['body_ratio'] = np.abs(df['close'] - df['open']) / (high_low_diff + 1e-10)
            df['body_ratio'] = np.clip(df['body_ratio'], 0.0, 100.0)  # Prevent overflow
            
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 1e-10)
            df['shadow_ratio'] = np.clip(df['shadow_ratio'], -100.0, 100.0)  # Prevent overflow
            
            # Gap analysis
            df['gap'] = df['open'] - df['close'].shift()
            close_shift = df['close'].shift()
            df['gap_ratio'] = df['gap'] / (close_shift.abs() + 1e-10)
            df['gap_ratio'] = np.clip(df['gap_ratio'], -100.0, 100.0)  # Prevent overflow
            
            # Intraday returns
            df['intraday_return'] = (df['close'] - df['open']) / (df['open'].abs() + 1e-10)
            df['intraday_return'] = np.clip(df['intraday_return'], -10.0, 10.0)  # Prevent overflow (returns should be reasonable)
            
            df['overnight_return'] = (df['open'] - close_shift) / (close_shift.abs() + 1e-10)
            df['overnight_return'] = np.clip(df['overnight_return'], -10.0, 10.0)  # Prevent overflow (returns should be reasonable)
            
            # Volume-price trend
            if 'volume' in df.columns:
                close_change = df['close'] - close_shift
                df['vpt'] = df['volume'] * (close_change / (close_shift.abs() + 1e-10))
                df['vpt_sma'] = df['vpt'].rolling(10).mean()
            
            # ⚡ NEW: Bollinger Bands (3 features)
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_upper'] = sma_20 + (2 * std_20)
            df['bb_lower'] = sma_20 - (2 * std_20)
            # ✅ FIX: Use epsilon instead of replace(0, np.nan) to prevent INF
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma_20.abs() + 1e-10)
            
            # ⚡ NEW: EMA (2 features)
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # ⚡ NEW: Stochastic Oscillator (2 features)
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # ⚡ NEW: ROC - Rate of Change (1 feature)
            df['roc'] = df['close'].pct_change(12) * 100  # 12-period ROC
            
            # ⚡ NEW: Williams %R (1 feature)
            df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
            
            # ⚡ NEW: TRIX (1 feature)
            ema1 = df['close'].ewm(span=15).mean()
            ema2 = ema1.ewm(span=15).mean()
            ema3 = ema2.ewm(span=15).mean()
            df['trix'] = ema3.pct_change() * 100
            
        except Exception as e:
            logger.error(f"Microstructure features hatası: {e}")
    
    def _add_volatility_features(self, df):
        """Volatility features"""
        try:
            # Different volatility measures
            for window in [5, 10, 20, 30]:
                returns = df['close'].pct_change()
                df[f'volatility_{window}'] = returns.rolling(window).std()
                df[f'volatility_rank_{window}'] = df[f'volatility_{window}'].rolling(window*2).rank(pct=True)
            
            # GARCH-like features
            returns = df['close'].pct_change()
            df['return_squared'] = returns ** 2
            df['volatility_garch'] = df['return_squared'].ewm(alpha=0.1).mean()
            
            # Volatility regime
            vol_20 = df['close'].pct_change().rolling(20).std()
            vol_ma = vol_20.rolling(60).mean()
            # ✅ FIX: Use epsilon instead of replace(0, np.nan) to prevent INF from very small denominators
            df['vol_regime'] = ((vol_20 / (vol_ma + 1e-10)) - 1).ffill().fillna(0)
            
            # ⚡ NEW: ADX (Average Directional Index) - Trend Strength
            try:
                high = df['high']
                low = df['low']
                close = df['close']
                
                # True Range
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                
                # Directional Movement
                up_move = high - high.shift()
                down_move = low.shift() - low
                
                # Convert to Series with proper index
                plus_dm = pd.Series(np.where((up_move > 0) & (up_move > down_move), up_move, 0), index=df.index)
                minus_dm = pd.Series(np.where((down_move > 0) & (down_move > up_move), down_move, 0), index=df.index)
                
                # Directional Indicators
                plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-10)
                minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-10)
                
                # ADX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                df['adx'] = dx.rolling(14).mean()
                df['adx_trending'] = (df['adx'] > 25).astype(int)  # 1 if trending, 0 if ranging
                
            except Exception as e:
                logger.debug(f"ADX calculation error: {e}")
                df['adx'] = 0
                df['adx_trending'] = 0
            
            # ⚡ NEW: Realized Volatility (Annualized)
            try:
                returns = df['close'].pct_change()
                # Realized vol over different windows
                df['realized_vol_5d'] = returns.rolling(5).std() * np.sqrt(252)
                df['realized_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
                df['realized_vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
                
                # Volatility regime based on quantiles
                vol_5d = df['realized_vol_5d']
                df['vol_regime_high'] = (vol_5d > vol_5d.quantile(0.75)).astype(int)
                df['vol_regime_low'] = (vol_5d < vol_5d.quantile(0.25)).astype(int)
                
            except Exception as e:
                logger.debug(f"Realized vol calculation error: {e}")
            
        except Exception as e:
            logger.error(f"Volatility features hatası: {e}")
    
    def _add_cyclical_features(self, df):
        """Cyclical time features"""
        try:
            # Assuming df.index is datetime
            dates = pd.to_datetime(df.index)
            
            # Day of week effects
            df['day_of_week'] = dates.dayofweek
            df['is_monday'] = (dates.dayofweek == 0).astype(int)
            df['is_friday'] = (dates.dayofweek == 4).astype(int)
            
            # Month effects
            df['month'] = dates.month
            df['quarter'] = dates.quarter
            df['is_month_end'] = dates.is_month_end.astype(int)
            df['is_quarter_end'] = dates.is_quarter_end.astype(int)
            
            # Cyclical encoding
            df['day_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
            
            # TA-Lib Hilbert Transform (dominant cycle) features
            try:
                import talib  # type: ignore
                if 'close' in df.columns:
                    cl = df['close'].astype(float).values
                    # Dominant Cycle Period & Phase
                    df['ht_dcperiod'] = pd.Series(talib.HT_DCPERIOD(cl), index=df.index)
                    df['ht_dcphase'] = pd.Series(talib.HT_DCPHASE(cl), index=df.index)
                    # Trend Mode (1=trend, 0=cycle)
                    df['ht_trendmode'] = pd.Series(talib.HT_TRENDMODE(cl), index=df.index)
                    # Phasor components
                    inphase, quadrature = talib.HT_PHASOR(cl)
                    df['ht_inphase'] = pd.Series(inphase, index=df.index)
                    df['ht_quadrature'] = pd.Series(quadrature, index=df.index)
                    # Sine/LeadSine
                    sine, leadsine = talib.HT_SINE(cl)
                    df['ht_sine'] = pd.Series(sine, index=df.index)
                    df['ht_leadsine'] = pd.Series(leadsine, index=df.index)
            except Exception as _e:
                logger.debug(f"Hilbert cycle features skipped: {_e}")
            
        except Exception as e:
            logger.error(f"Cyclical features hatası: {e}")
    
    def _add_statistical_features(self, df):
        """Statistical features"""
        try:
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'skewness_{window}'] = df['close'].rolling(window).skew()
                df[f'kurtosis_{window}'] = df['close'].rolling(window).kurt()
                
                # Percentile features
                df[f'percentile_25_{window}'] = df['close'].rolling(window).quantile(0.25)
                df[f'percentile_75_{window}'] = df['close'].rolling(window).quantile(0.75)
                
                # Z-score
                mean = df['close'].rolling(window).mean()
                std = df['close'].rolling(window).std()
                # ✅ FIX: Use epsilon instead of replace(0, np.nan) to prevent INF from very small denominators
                df[f'zscore_{window}'] = (df['close'] - mean) / (std + 1e-10)
            
            # Entropy-like measures
            for window in [10, 20]:
                returns = df['close'].pct_change()
                abs_returns = np.abs(returns)
                df[f'entropy_{window}'] = abs_returns.rolling(window).apply(
                    lambda x: -np.sum(x * np.log(x + 1e-10)) if len(x) > 0 else 0
                )
            
        except Exception as e:
            logger.error(f"Statistical features hatası: {e}")
    
    def _add_liquidity_features(self, df):
        """Liquidity and volume tier features"""
        try:
            volume = df['volume']
            close = df['close']
            
            # Volume statistics
            vol_mean = volume.mean()
            vol_std = volume.std()
            
            # Relative volume (vs rolling average)
            for window in [5, 20, 60]:
                vol_ma = volume.rolling(window).mean()
                df[f'relative_volume_{window}'] = volume / (vol_ma + 1e-10)
            
            # Volume tier classification (based on percentiles)
            vol_q25 = volume.quantile(0.25)
            vol_q75 = volume.quantile(0.75)
            
            # Tier features (one-hot style)
            df['volume_tier_high'] = (volume > vol_q75).astype(int)  # Top 25%
            df['volume_tier_low'] = (volume < vol_q25).astype(int)   # Bottom 25%
            df['volume_tier_mid'] = ((volume >= vol_q25) & (volume <= vol_q75)).astype(int)
            
            # Continuous volume score (normalized)
            # ✅ FIX: Use epsilon instead of conditional NaN to prevent INF from very small denominators
            df['volume_zscore'] = (volume - vol_mean) / (vol_std + 1e-10)
            
            # Volume regime (high activity vs low activity)
            vol_rank = volume.rolling(60).rank(pct=True)
            df['volume_regime'] = vol_rank  # 0-1, higher = more active lately
            
            # Dollar volume (proxy for liquidity)
            df['dollar_volume'] = volume * close
            dollar_vol_ma = df['dollar_volume'].rolling(20).mean()
            df['relative_dollar_volume'] = df['dollar_volume'] / (dollar_vol_ma + 1e-10)
            
            # Volume-price relationship
            # High volume + up move = strong momentum
            # High volume + down move = strong sell-off
            returns = close.pct_change()
            df['volume_price_corr_5'] = volume.rolling(5).corr(returns)
            df['volume_price_corr_20'] = volume.rolling(20).corr(returns)
            
            logger.debug("Liquidity/volume features added")
            
        except Exception as e:
            logger.error(f"Liquidity features hatası: {e}")
    
    def _add_macro_features(self, df):
        """Macro economic features from VT (USDTRY, CDS, TCMB Rate)"""
        try:
            logger.debug("_add_macro_features() called")
            
            # Get macro data from VT (SQL query)
            query = """
                SELECT date, usdtry_close, turkey_cds, tcmb_policy_rate
                FROM macro_indicators
                WHERE date >= :start_date AND date <= :end_date
                ORDER BY date
            """
            
            # Date range from df
            # If df.index is datetime, use it; otherwise use 'date' column if available
            try:
                # Try to get date from index (if it's DatetimeIndex)
                if isinstance(df.index, pd.DatetimeIndex):
                    start_date = df.index.min().date()
                    end_date = df.index.max().date()
                elif hasattr(df.index.min(), 'date'):
                    # Index has date attribute (Timestamp)
                    start_date = df.index.min().date()
                    end_date = df.index.max().date()
                elif 'date' in df.columns:
                    # Use 'date' column
                    start_date = pd.to_datetime(df['date']).min().date()
                    end_date = pd.to_datetime(df['date']).max().date()
                else:
                    # Fallback: assume df.index is integer; use last 2 years as a safe range
                    logger.warning("No date column/index found for macro features; using last 2 years")
                    end_ts = pd.Timestamp.now()
                    start_ts = end_ts - pd.Timedelta(days=730)  # type: ignore[operator]
                    end_date = end_ts.date()
                    start_date = start_ts.date()
            except (AttributeError, TypeError) as e:
                # ✅ FIX: Handle case where index is integer or other non-date type
                logger.warning(f"Could not extract date from index/column: {e}. Using last 2 years as fallback")
                end_ts = pd.Timestamp.now()
                start_ts = end_ts - pd.Timedelta(days=730)  # type: ignore[operator]
                end_date = end_ts.date()
                start_date = start_ts.date()
            params = {'start_date': start_date, 'end_date': end_date}
            # Always use direct SQLAlchemy engine to avoid Flask app context dependency
            logger.info("Macro loader: using direct SQLAlchemy engine (no Flask context)")
            # ✅ FIX: Use ConfigManager for consistent config access
            db_url = ConfigManager.get('DATABASE_URL')
            if not db_url:
                # Robust fallback: read secret and construct DSN
                try:
                    secret_path = '/opt/bist-pattern/.secrets/db_password'
                    if os.path.exists(secret_path):
                        with open(secret_path, 'r') as sp:
                            _pwd = sp.read().strip()
                            if _pwd:
                                db_url = f"postgresql://bist_user:{_pwd}@127.0.0.1:6432/bist_pattern_db"
                                logger.warning("DATABASE_URL not set; using fallback DSN from secret file")
                                # Also export for downstream steps in this process
                                os.environ['DATABASE_URL'] = db_url
                except Exception as _e:
                    raise RuntimeError(f"DATABASE_URL not set and secret fallback failed: {_e}")
            if not db_url:
                raise RuntimeError("DATABASE_URL not set (macro fallback)")
            from sqlalchemy import create_engine, text as sqla_text
            from sqlalchemy.pool import NullPool
            engine = create_engine(
                db_url,
                pool_pre_ping=True,
                poolclass=NullPool,
                connect_args={"connect_timeout": 5, "application_name": "bist-hpo"},
            )
            try:
                with engine.connect() as conn:
                    rows = conn.execute(sqla_text(query), params).fetchall()
            finally:
                # ✅ CRITICAL FIX: Dispose engine in finally block to ensure cleanup
                try:
                    engine.dispose()
                except Exception as e:
                    logger.debug(f"Failed to dispose macro engine: {e}")
            macro_data = pd.DataFrame(rows, columns=['date', 'usdtry', 'cds', 'rate'])
            
            if len(macro_data) > 0:
                # Convert date to datetime for merge
                macro_data['date'] = pd.to_datetime(macro_data['date'])
                macro_data = macro_data.set_index('date')
                
                # ⚡ CRITICAL FIX: Timezone normalization to prevent join failures
                # Remove timezone info from both indices to ensure alignment
                df_index_normalized = pd.to_datetime(df.index).tz_localize(None)
                macro_index_normalized = pd.to_datetime(macro_data.index).tz_localize(None)
                
                # Temporarily set normalized index for merge
                df_original_index = df.index.copy()
                df.index = df_index_normalized
                macro_data.index = macro_index_normalized
                
                # ⚡ FIX: Merge macro data (reindex to match df dates)
                # Don't use join - use reindex for alignment
                macro_data = macro_data.reindex(df.index, method='ffill')
                
                # Restore original index after merge
                df.index = df_original_index
                
                # Add columns directly (in-place!) + Convert to float64!
                # ⚡ FIX: Only ffill (no bfill - prevents lookahead!)
                df['usdtry'] = pd.to_numeric(macro_data['usdtry'], errors='coerce').ffill().fillna(0).astype('float64')
                df['cds'] = pd.to_numeric(macro_data['cds'], errors='coerce').ffill().fillna(0).astype('float64')
                df['rate'] = pd.to_numeric(macro_data['rate'], errors='coerce').ffill().fillna(0).astype('float64')
                logger.info("✅ Macro base features added: usdtry, cds, rate (dtype=float64)")
                
                # Create derivative features
                df['usdtry_change_1d'] = df['usdtry'].pct_change().fillna(0)
                df['usdtry_change_5d'] = df['usdtry'].pct_change(5).fillna(0)
                df['usdtry_change_20d'] = df['usdtry'].pct_change(20).fillna(0)
                df['cds_change_5d'] = df['cds'].pct_change(5).fillna(0)
                df['rate_change_20d'] = df['rate'].pct_change(20).fillna(0)
                logger.info("✅ Macro derivative features added (5 changes)")
                
                logger.info(f"✅ Macro features complete: {len(macro_data)} days merged, 8 features")
            else:
                logger.warning("No macro data found in VT")
                
        except Exception as e:
            import traceback
            logger.error(f"Macro features error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback: zero features
            for col in ['usdtry', 'usdtry_change_1d', 'usdtry_change_5d', 'usdtry_change_20d',
                       'cds', 'cds_change_5d', 'rate', 'rate_change_20d']:
                if col not in df.columns:
                    df[col] = 0.0
    
    def _clean_data(self, df):
        """Veri temizleme - INF, NaN ve aşırı değerleri temizle"""
        try:
            logger.info(f"🧹 Veri temizleme başlatılıyor - Shape: {df.shape}")
            
            # INF değerleri temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Numeric sütunları al
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # CRITICAL FIX: Softened outlier removal (was too aggressive)
            # Previous: 3 sigma + 1-99 percentile → Market shocks were removed!
            # New: 5 sigma + 0.5-99.5 percentile → Keep real market events
            for col in numeric_columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue  # Ana price sütunlarını dokunma
                
                # Z-score ile outlier tespiti (5 sigma - yumuşatıldı)
                if df[col].std() > 0:  # Std > 0 kontrolü
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df.loc[z_scores > 5, col] = np.nan  # 3 → 5 sigma
                
                # Çok büyük değerleri sınırla (yumuşatıldı)
                percentile_high = df[col].quantile(0.995)  # 0.99 → 0.995
                percentile_low = df[col].quantile(0.005)   # 0.01 → 0.005
                
                if not np.isnan(percentile_high) and not np.isnan(percentile_low):
                    df[col] = df[col].clip(lower=percentile_low, upper=percentile_high)
            
            # NaN değerleri forward fill ile doldur
            df = df.ffill()
            
            # Hala NaN varsa 0 ile doldur
            df = df.fillna(0)
            
            # Final check - hala INF var mı?
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                logger.warning(f"⚠️ {inf_count} INF değer hala mevcut, 0 ile değiştiriliyor")
                df = df.replace([np.inf, -np.inf], 0)
            
            logger.info(f"✅ Veri temizleme tamamlandı - Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Veri temizleme hatası: {e}")
            return df
    
    def _should_halt(self) -> bool:
        """Eğitim sırasında dıştan durdurma talebi var mı kontrol et"""
        try:
            return bool(self.stop_file_path and os.path.exists(self.stop_file_path))
        except Exception as e:
            logger.debug(f"_should_halt check failed: {e}")
            return False

    def train_enhanced_models(self, symbol, data):
        """Gelişmiş modelleri eğit - Adaptive Learning destekli"""
        try:
            logger.info("="*80)
            logger.info(f"🧠 TRAIN_ENHANCED_MODELS: {symbol}")
            logger.info("="*80)
            logger.info(f"📊 Input data shape: {data.shape}")
            logger.info(f"📊 Data period: {data.index.min()} - {data.index.max()}")
            
            # ⚡ DIAGNOSTIC: Log all feature flags and their impact
            critical_vars = [
                'ML_USE_ADAPTIVE_LEARNING', 'ML_SKIP_ADAPTIVE_PHASE2',
                'ENABLE_SEED_BAGGING', 'N_SEEDS',
                'ENABLE_TALIB_PATTERNS', 'ML_USE_SMART_ENSEMBLE',
                'ML_USE_STACKED_SHORT', 'ENABLE_META_STACKING',
                'ML_USE_REGIME_DETECTION', 'ML_USE_DIRECTIONAL_LOSS',
                'ENABLE_EXTERNAL_FEATURES', 'ENABLE_FINGPT_FEATURES', 'ENABLE_YOLO_FEATURES',
                'ENABLE_FINGPT'
            ]
            logger.info("📋 Feature Configuration (in train_enhanced_models):")
            feature_status = {}
            for var in critical_vars:
                val = ConfigManager.get(var, 'NOT SET')
                is_enabled = str(val).lower() in ('1', 'true', 'yes')
                feature_status[var] = is_enabled
                status_icon = "✅" if is_enabled else "❌"
                logger.info(f"   {status_icon} {var} = {val}")
            
            # ⚡ DIAGNOSTIC: Log feature summary
            enabled_features = [f for f, enabled in feature_status.items() if enabled]
            disabled_features = [f for f, enabled in feature_status.items() if not enabled]
            logger.info(f"📊 Feature Summary: {len(enabled_features)} enabled, {len(disabled_features)} disabled")
            if enabled_features:
                logger.info(f"   Enabled: {', '.join(enabled_features)}")
            if disabled_features:
                logger.info(f"   Disabled: {', '.join(disabled_features)}")
            
            # Data validation
            if data is None or len(data) == 0:
                logger.error(f"{symbol} için veri bulunamadı")
                return False
            
            # ⚡ ADAPTIVE LEARNING: Train/Test Split (HPO ile aynı mantık)
            # ✅ FIX: Use ConfigManager for consistent config access
            use_adaptive = str(ConfigManager.get('ML_USE_ADAPTIVE_LEARNING', '0')).lower() in ('1', 'true', 'yes')
            # ✅ CRITICAL FIX: Evaluation mode - skip Phase 2 when training with pre-split train_df
            # This ensures evaluation uses same amount of data as HPO (no internal split)
            skip_phase2 = str(ConfigManager.get('ML_SKIP_ADAPTIVE_PHASE2', '0')).lower() in ('1', 'true', 'yes')
            
            logger.info("📋 Adaptive Learning Settings:")
            logger.info(f"   use_adaptive = {use_adaptive}")
            logger.info(f"   skip_phase2 = {skip_phase2}")
            
            if use_adaptive and not skip_phase2:
                # ✅ FIX: HPO ile aynı mantık - 300 gün çok yüksek, 180 gün yeterli
                total_days = len(data)
                logger.info(f"📊 Adaptive Learning ENABLED: Calculating split for {total_days} days...")
                
                # HPO'daki gibi: daha esnek split mantığı
                if total_days >= 240:
                    # 240+ gün: last 120 days validation, rest train
                    test_days = 120
                    train_days = total_days - test_days
                    logger.info("   Split rule: >=240 days → last 120 test, rest train")
                elif total_days >= 180:
                    # 180-239 gün: 2/3 train, 1/3 validation
                    train_days = int(total_days * 2 / 3)
                    test_days = total_days - train_days
                    logger.info("   Split rule: 180-239 days → 2/3 train, 1/3 test")
                else:
                    # 3-179 gün: 2/3 train, 1/3 validation (minimum 1 validation)
                    train_days = max(1, int(total_days * 2 / 3))
                    test_days = total_days - train_days
                    logger.info("   Split rule: <180 days → 2/3 train, 1/3 test")
                
                # Minimum test kontrolü: en az 3 gün olmalı (HPO ile uyumlu)
                if test_days < 3:
                    logger.warning(f"⚠️ Adaptive learning disabled: Insufficient data for split ({total_days} days, test={test_days} < 3)")
                    data_for_training = data
                    train_days = len(data)
                    test_days = 0
                else:
                    # Split data
                    train_data = data.iloc[:train_days].copy()
                    test_data = data.iloc[train_days:].copy()
                    
                    logger.info(f"🎯 ADAPTIVE LEARNING SPLIT: {total_days} gün → Train: {train_days} ({train_days/total_days*100:.1f}%), Test: {test_days} ({test_days/total_days*100:.1f}%)")
                    logger.info(f"   Train period: {train_data.index.min().date()} - {train_data.index.max().date()}")
                    logger.info(f"   Test period: {test_data.index.min().date()} - {test_data.index.max().date()}")
                    
                    # First train on train_data
                    logger.info(f"🔄 Phase 1: Initial training on {train_days} days")
                    data_for_training = train_data
            else:
                # ✅ CRITICAL FIX: Evaluation mode or adaptive OFF - use all data for Phase 1, skip Phase 2
                # This ensures evaluation uses same amount of data as HPO (no internal split)
                if skip_phase2:
                    logger.info(f"🎯 EVALUATION MODE: Using all {len(data)} days for Phase 1 (skip Phase 2 to match HPO data usage)")
                else:
                    logger.info(f"🎯 Adaptive Learning DISABLED: Using all {len(data)} days for training")
                data_for_training = data
                train_days = len(data)
                test_days = 0
            
            # Feature engineering (use data_for_training for initial model)
            df_features = self.create_advanced_features(data_for_training, symbol=symbol)
            
            # ⚡ CRITICAL FIX: Don't drop all NaN rows!
            # Rolling features (ADX, realized_vol_60d) have NaN at start
            # dropna() would remove 60-100 days unnecessarily!
            # Strategy: Forward fill + mean fill (not 0!)
            # Target NaN will be removed later (per horizon)
            
            # Forward fill + mean fill for features (better than fillna(0))
            for col in df_features.columns:
                if df_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Step 1: Forward fill (use previous values)
                    # ✅ FIX: Use ffill() instead of fillna(method='ffill') (deprecated in pandas 2.0+)
                    df_features[col] = df_features[col].ffill()
                    
                    # Step 2: For remaining NaN (at start), use column mean
                    # This is better than 0 (e.g., realized_vol_60d mean ~0.025, not 0!)
                    if df_features[col].isna().any():
                        col_mean = df_features[col].mean()
                        if pd.notna(col_mean):  # If mean is valid
                            df_features[col] = df_features[col].fillna(col_mean)
                        else:
                            # Fallback: If all NaN, use 0
                            df_features[col] = df_features[col].fillna(0)
            
            # Clean infinite and large values (delegated)
            try:
                from bist_pattern.features.cleaning import clean_dataframe as _clean_df
                df_features = _clean_df(df_features)
            except Exception as e:
                logger.debug(f"External clean_dataframe failed, using internal: {e}")
                df_features = self._clean_data(df_features)
            
            try:
                # Read minimum data days from environment; default 50 for HPO compatibility
                # ✅ FIX: Use ConfigManager for consistent config access
                min_days = int(ConfigManager.get('ML_MIN_DATA_DAYS', ConfigManager.get('ML_MIN_DAYS', '50')))
            except Exception as e:
                logger.debug(f"Failed to get ML_MIN_DATA_DAYS, using 50: {e}")
                min_days = 50
            if len(df_features) < min_days:
                logger.warning(f"{symbol} için yeterli veri yok (Enhanced ML için {min_days}+ gün gerekli, sadece {len(df_features)} mevcut)")
                return False
            
            # Feature selection (base set)
            feature_cols = [col for col in df_features.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']
                          and df_features[col].dtype in ['float64', 'float32', 'int64', 'int32']  # ⚡ FIX: Include int32!
                          and not df_features[col].isnull().all()]
            
            logger.info(f"📊 {len(feature_cols)} feature kullanılacak (base set)")
            
            # ⚡ NEW: Calculate market regime for adaptive model parameters
            # Regime score: 0 = low volatility/trending, 1 = high volatility/ranging
            try:
                vol_20 = df_features['close'].pct_change().rolling(20).std()
                vol_60 = df_features['close'].pct_change().rolling(60).std()
                v20_last = vol_20.iloc[-1] if len(vol_20) > 0 and not pd.isna(vol_20.iloc[-1]) else 0.5
                v60_last = vol_60.iloc[-1] if len(vol_60) > 0 and not pd.isna(vol_60.iloc[-1]) else 1.0
                regime_score = float(v20_last / v60_last) if v60_last > 1e-10 else 0.5
                regime_score = max(0.0, min(1.0, regime_score))  # Clamp to [0, 1]
                logger.info(f"📊 Market regime score: {regime_score:.3f} (0=calm, 1=volatile)")
            except Exception as e:
                logger.debug(f"Regime score calculation failed, using 0.5: {e}")
                regime_score = 0.5  # Neutral fallback
            
            results = {}
            
            # Her tahmin ufku için model eğit
            for horizon in self.prediction_horizons:
                # ✅ Algorithm enable/disable (env-driven; defaults enabled)
                # Note: XGBoost availability is checked via XGBOOST_AVAILABLE flag
                try:
                    enable_lgb = str(ConfigManager.get('ENABLE_LIGHTGBM', '1')).lower() in ('1', 'true', 'yes', 'on')
                except Exception as e:
                    logger.debug(f"Failed to get ENABLE_LIGHTGBM, defaulting to True: {e}")
                    enable_lgb = True
                try:
                    enable_cat = str(ConfigManager.get('ENABLE_CATBOOST', '1')).lower() in ('1', 'true', 'yes', 'on')
                except Exception as e:
                    logger.debug(f"Failed to get ENABLE_CATBOOST, defaulting to True: {e}")
                    enable_cat = True
                # Graceful stop kontrolü
                if self._should_halt():
                    logger.warning("⛔ Stop sentinel tespit edildi, eğitim durduruluyor")
                    # Kısmi sonuçları kaydetmeden çık
                    return False
                logger.info(f"📈 {symbol} - {horizon} gün tahmini için model eğitimi")
                
                # ⚡ TA-Lib patterns: Add per-horizon (ALL horizons enabled)
                # Candlestick patterns are daily-based but their effects can extend to longer horizons:
                # - Short-term (1d, 3d, 7d): Pattern itself is meaningful
                # - Long-term (14d, 30d): Pattern's trend impact is meaningful, especially for low-volume stocks
                # Patterns can indicate trend formation that lasts weeks, especially in low-liquidity stocks
                if self.enable_talib_patterns:
                    # Add TA-Lib candlestick patterns if not already present
                    if 'pat_bull3' not in df_features.columns:
                        self._add_candlestick_features(df_features)
                        logger.debug(f"✅ TA-Lib patterns added for {horizon}d horizon")
                
                # Target variable: forward return (percentage)
                target = f'target_ret_{horizon}d'
                df_features[target] = (
                    df_features['close'].shift(-horizon) / df_features['close'] - 1.0
                )
                # Target audit: compare with pct_change variant and log summary stats
                try:
                    _chk = df_features['close'].pct_change(periods=horizon).shift(-horizon)
                    _diff = (df_features[target] - _chk).abs()
                    _n = int(_diff.dropna().shape[0])
                    if _n > 0:
                        _mdiff = float(_diff.dropna().mean())
                        _p95 = float(np.nanpercentile(np.abs(df_features[target].values), 95))
                        logger.info(
                            f"🎯 Target audit {symbol} {horizon}d: samples={_n}, mean_abs_diff={_mdiff:.6f}, |ret|_p95={_p95:.3f}"
                        )
                except Exception as e:
                    logger.debug(f"Target audit failed: {e}")
                
                # Remove last horizon rows FIRST (before feature selection)
                df_model = df_features[:-horizon].copy()

                # Compute empirical cap from training target distribution (e.g., P95/97 of |return|)
                try:
                    # ✅ FIX: Use ConfigManager for consistent config access
                    cap_perc = float(ConfigManager.get('TRAIN_CAP_PERCENTILE', '95'))
                except Exception as e:
                    logger.debug(f"Failed to get TRAIN_CAP_PERCENTILE, using 95.0: {e}")
                    cap_perc = 95.0
                try:
                    cap_val = float(np.nanpercentile(np.abs(df_model[target].values), cap_perc))
                    # Store in memory for manifest and later inference preference
                    self.models[f"{symbol}_{horizon}d_cap"] = float(max(0.0, min(0.90, cap_val)))
                    logger.info(f"📏 Empirical cap {symbol} {horizon}d: P{cap_perc:.0f}(|ret|)={cap_val:.3f}")
                except Exception as e:
                    logger.debug(f"Failed to calculate cap_val: {e}")
                
                # ⚡ FEATURE REDUCTION: Reduce from 107 → ~50-60 features
                # Strategy: Remove low-variance and highly-correlated features
                # This improves sample/feature ratio and reduces overfitting
                
                # Step 1: Variance-based filtering (remove features with very low variance)
                try:
                    from bist_pattern.features.selection import variance_and_correlation_filter
                    reduced_features = variance_and_correlation_filter(
                        pd.DataFrame(df_model[feature_cols]),
                        feature_cols,
                        var_thr=0.01,
                        corr_thr=0.90,
                    )
                    logger.debug(f"Variance+Correlation filter: {len(feature_cols)} → {len(reduced_features)} features")
                    # ✅ FIX: Validate reduced_features is not empty
                    if not reduced_features:
                        logger.warning("Variance+Correlation filter returned empty list! Using all features")
                        reduced_features = feature_cols
                except Exception as e:
                    logger.warning(f"Feature selection (var+corr) failed: {e}")
                    reduced_features = feature_cols
                
                # Step 3: FEATURE IMPORTANCE SELECTION
                # Train a quick temporary model to get feature importance
                # Then select top N features based on importance
                # ✅ FIX: Use ConfigManager for consistent config access
                max_features = int(ConfigManager.get('ML_MAX_FEATURES', '32'))  # tightened for short horizons
                
                if len(reduced_features) > max_features:
                    try:
                        logger.info("🎯 Training temporary model(s) for feature importance selection...")
                        
                        # Prepare data with reduced features
                        X_temp = df_model[reduced_features].values
                        y_temp = df_model[target].values
                        
                        # Stability-based selection across seeds (fast, shallow models)
                        # ✅ FIX: Use ConfigManager for consistent config access
                        use_stability = str(ConfigManager.get('ML_USE_FEATURE_STABILITY', '1')).lower() in ('1', 'true', 'yes')
                        max_k = max_features
                        min_share = float(ConfigManager.get('ML_FEATURE_STABILITY_MIN_SHARE', '0.55'))
                        # ⚡ CRITICAL FIX: Feature selection için seed'leri base_seeds'den al
                        # HPO'da: base_seeds = [trial.number] (örneğin [42]) → best trial'ın seed'i kullanılır
                        # Test'te: base_seeds = [best_trial_number] (örneğin [42]) → best trial'ın seed'i kullanılır
                        # Bu sayede feature selection HPO'daki BEST TRIAL ile aynı seed'i kullanır!
                        # "En iyi 32" = ML_MAX_FEATURES=32 (feature sayısı, seed değil)
                        base_seeds_attr = getattr(self, 'base_seeds', [42])
                        if not base_seeds_attr:
                            base_seeds_attr = [42]
                        # ⚡ CRITICAL: Feature selection için seed sayısını base_seeds uzunluğundan al
                        # HPO'da: 1 seed (best_trial_number) → 1 feature selection model → aynı 32 feature seçilir
                        # Test'te (seed bagging kapalı): 1 seed (best_trial_number) → 1 feature selection model → aynı 32 feature seçilir
                        # Test'te (seed bagging açık): 3 seed (best_trial_number, ...) → 3 feature selection model (stability için)
                        num_seeds = len(base_seeds_attr)
                        # ✅ FIX: Eğer stability kapalıysa veya tek seed varsa, tek seed kullan (best seed)
                        if not use_stability or num_seeds == 1:
                            num_seeds = 1
                            seed_list = [base_seeds_attr[0]]  # Best seed kullanılır
                            logger.info(f"🎯 Feature selection: Using BEST SEED={seed_list[0]} (from base_seeds={base_seeds_attr}) → Selecting top {max_k} features")
                        else:
                            # Stability açık ve birden fazla seed varsa, hepsini kullan
                            seed_list = list(base_seeds_attr)[:num_seeds]
                            logger.info(f"🎯 Feature selection: Using {num_seeds} seeds for stability (from base_seeds={base_seeds_attr}) → Selecting top {max_k} features")
                        
                        importance_matrix = []  # list of np.arrays aligned to reduced_features
                        for seed in seed_list:
                            try:
                                model_seed = xgb.XGBRegressor(  # type: ignore[union-attr]
                                    n_estimators=50,
                                    max_depth=3,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    n_jobs=self.train_threads,
                                    random_state=int(seed),
                                    eval_metric='rmse'
                                )
                                model_seed.fit(X_temp, y_temp)
                                importance_matrix.append(np.asarray(model_seed.feature_importances_, dtype=float))
                            except Exception as seed_err:
                                # ✅ FIX: Log exception instead of silent skip
                                logger.debug(f"Feature importance model (seed={seed}) failed: {seed_err}")
                                continue
                        
                        if not importance_matrix:
                            # Fallback: single quick model
                            # ✅ FIX: Use first seed from base_seeds instead of hardcoded 42
                            fallback_seed = self.base_seeds[0] if self.base_seeds else 42
                            try:
                                temp_model = xgb.XGBRegressor(  # type: ignore[union-attr]
                                    n_estimators=50,
                                    max_depth=3,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    n_jobs=self.train_threads,
                                    random_state=int(fallback_seed),
                                    eval_metric='rmse'
                                )
                                temp_model.fit(X_temp, y_temp)
                                importance_matrix = [np.asarray(temp_model.feature_importances_, dtype=float)]
                            except Exception as fallback_err:
                                # ✅ FIX: Log fallback failure and raise exception
                                logger.error(f"Fallback feature importance model also failed: {fallback_err}")
                                raise  # Re-raise to be caught by outer try-except
                        
                        imp_arr = np.vstack(importance_matrix)
                        avg_imp = np.nanmean(imp_arr, axis=0)
                        
                        # Rank-based stability: fraction of seeds where feature is in top K
                        if use_stability and imp_arr.shape[0] > 1:
                            top_k_flags = []
                            for i in range(imp_arr.shape[0]):
                                ranks = np.argsort(-imp_arr[i])  # descending
                                top_idx = set(ranks[:max_k].tolist())
                                flags = np.array([1 if j in top_idx else 0 for j in range(len(reduced_features))], dtype=float)
                                top_k_flags.append(flags)
                            top_k_flags = np.vstack(top_k_flags)
                            share_in_topk = np.nanmean(top_k_flags, axis=0)
                        else:
                            share_in_topk = np.ones_like(avg_imp)
                        
                        # Combine: first take stable features (share>=min_share), sort by avg importance
                        tuples = list(zip(reduced_features, avg_imp.tolist(), share_in_topk.tolist()))
                        stable = [(f, a, s) for (f, a, s) in tuples if s >= min_share]
                        stable.sort(key=lambda x: x[1], reverse=True)
                        selected = stable[:max_k]
                        
                        if len(selected) < max_k:
                            # Fill remaining by avg importance regardless of share
                            remaining = [(f, a, s) for (f, a, s) in tuples if (f not in {x[0] for x in selected})]
                            remaining.sort(key=lambda x: x[1], reverse=True)
                            selected.extend(remaining[: (max_k - len(selected))])
                            
                            # ✅ FIX: Final validation - ensure we have enough features
                            if len(selected) < max_k:
                                logger.warning(f"Only {len(selected)}/{max_k} features selected. Using all available from reduced_features.")
                                # Use all reduced_features if still not enough
                                if len(reduced_features) > len(selected):
                                    missing = [f for f in reduced_features if f not in {x[0] for x in selected}]
                                    selected.extend([(f, 0.0, 0.0) for f in missing[:max_k - len(selected)]])
                                # If still not enough, this will be caught by empty check later
                        
                        horizon_feature_cols = [f for (f, _, __) in selected]
                        
                        # ✅ FIX: Validate selected features are not empty
                        if not horizon_feature_cols:
                            logger.error(f"Feature selection returned empty list! Using fallback: first {max_features} features")
                            horizon_feature_cols = reduced_features[:max_features] if len(reduced_features) >= max_features else reduced_features
                            if not horizon_feature_cols:
                                logger.error(f"Fallback also empty! Using first {max_features} from all features")
                                horizon_feature_cols = feature_cols[:max_features] if len(feature_cols) >= max_features else feature_cols
                        
                        # ✅ FIX: Final validation - ensure we have at least 1 feature
                        if not horizon_feature_cols:
                            logger.error("CRITICAL: No features selected! Cannot proceed with training.")
                            return False
                        
                        logger.info(
                            f"✅ Feature selection: {len(reduced_features)} → {len(horizon_feature_cols)} (stability={use_stability}, seeds={len(importance_matrix)}, min_share={min_share})"
                        )
                        logger.debug(f"Top 5 stable features: {[f for f, _, __ in selected[:5]]}")
                        
                    except Exception as e:
                        logger.warning(f"Feature importance selection failed: {e}, using simple truncation")
                        horizon_feature_cols = reduced_features[:max_features] if len(reduced_features) >= max_features else reduced_features
                        # ✅ FIX: Validate fallback is not empty
                        if not horizon_feature_cols:
                            logger.error(f"Fallback truncation also empty! Using first {max_features} from all features")
                            horizon_feature_cols = feature_cols[:max_features] if len(feature_cols) >= max_features else feature_cols
                            if not horizon_feature_cols:
                                logger.error("CRITICAL: No features available! Cannot proceed with training.")
                                return False
                else:
                    horizon_feature_cols = reduced_features
                    # ✅ FIX: Validate reduced_features is not empty
                    if not horizon_feature_cols:
                        logger.error(f"reduced_features is empty! Using first {max_features} from all features")
                        horizon_feature_cols = feature_cols[:max_features] if len(feature_cols) >= max_features else feature_cols
                        if not horizon_feature_cols:
                            logger.error("CRITICAL: No features available! Cannot proceed with training.")
                            return False
                
                logger.info(f"🎯 {horizon}d: Using {len(horizon_feature_cols)} features (reduced from {len(feature_cols)}) - PER-SYMBOL OPTIMIZED")
                
                # ✅ FIX: Final validation before using horizon_feature_cols
                if not horizon_feature_cols:
                    logger.error(f"CRITICAL: horizon_feature_cols is empty for {symbol} {horizon}d! Cannot proceed.")
                    return False
                
                # ✅ FIX: Check if all selected features exist in df_model
                missing_cols = [c for c in horizon_feature_cols if c not in df_model.columns]
                if missing_cols:
                    logger.error(f"Missing features in df_model for {symbol} {horizon}d: {missing_cols}")
                    # Remove missing features
                    horizon_feature_cols = [c for c in horizon_feature_cols if c in df_model.columns]
                    if not horizon_feature_cols:
                        logger.error("All selected features missing in df_model! Cannot proceed.")
                        return False
                    logger.warning(f"Using {len(horizon_feature_cols)} features after removing {len(missing_cols)} missing ones")
                
                # ⚡ ADAPTIVE LEARNING: Save feature columns for Phase 2
                if not hasattr(self, 'feature_columns'):
                    self.feature_columns = {}
                self.feature_columns[f'{horizon}d'] = horizon_feature_cols
                
                # Sample weights (event-focused): prioritize recent/confirmed H&S events
                try:
                    w: np.ndarray = np.ones(len(df_model), dtype=float)
                    # H&S weighting
                    if 'hs_conf' in df_model.columns:
                        _s = pd.to_numeric(df_model['hs_conf'], errors='coerce')
                        hs_conf_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        hs_conf_arr = np.nan_to_num(hs_conf_arr, nan=0.0)
                        hs_conf_arr = np.clip(hs_conf_arr, 0.0, 1.0)
                        w = w * (1.0 + hs_conf_arr)
                    if 'hs_breakout' in df_model.columns:
                        _s = pd.to_numeric(df_model['hs_breakout'], errors='coerce')
                        hs_breakout: np.ndarray = np.asarray(_s.values, dtype=float)
                        hs_breakout = np.nan_to_num(hs_breakout, nan=0.0)
                        w = w * (1.0 + 0.75 * hs_breakout)
                    if 'hs_event_window_14' in df_model.columns:
                        _s = pd.to_numeric(df_model['hs_event_window_14'], errors='coerce')
                        hs_ev14: np.ndarray = np.asarray(_s.values, dtype=float)
                        hs_ev14 = np.nan_to_num(hs_ev14, nan=0.0)
                        w = w * (1.0 + 0.35 * hs_ev14)
                    # DTB weighting
                    if 'dtb_conf' in df_model.columns:
                        _s = pd.to_numeric(df_model['dtb_conf'], errors='coerce')
                        dtb_conf: np.ndarray = np.asarray(_s.values, dtype=float)
                        dtb_conf = np.nan_to_num(dtb_conf, nan=0.0)
                        dtb_conf = np.clip(dtb_conf, 0.0, 1.0)
                        w = w * (1.0 + dtb_conf)
                    if 'dtb_breakout' in df_model.columns:
                        _s = pd.to_numeric(df_model['dtb_breakout'], errors='coerce')
                        dtb_breakout: np.ndarray = np.asarray(_s.values, dtype=float)
                        dtb_breakout = np.nan_to_num(dtb_breakout, nan=0.0)
                        w = w * (1.0 + 0.5 * dtb_breakout)
                    if 'dtb_event_window_14' in df_model.columns:
                        _s = pd.to_numeric(df_model['dtb_event_window_14'], errors='coerce')
                        dtb_ev14: np.ndarray = np.asarray(_s.values, dtype=float)
                        dtb_ev14 = np.nan_to_num(dtb_ev14, nan=0.0)
                        w = w * (1.0 + 0.25 * dtb_ev14)
                    # Triangle weighting
                    if 'tri_conf' in df_model.columns:
                        _s = pd.to_numeric(df_model['tri_conf'], errors='coerce')
                        tri_conf_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        tri_conf_arr = np.nan_to_num(tri_conf_arr, nan=0.0)
                        tri_conf_arr = np.clip(tri_conf_arr, 0.0, 1.0)
                        w = w * (1.0 + tri_conf_arr)
                    if 'tri_breakout' in df_model.columns:
                        _s = pd.to_numeric(df_model['tri_breakout'], errors='coerce')
                        tri_breakout_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        tri_breakout_arr = np.nan_to_num(tri_breakout_arr, nan=0.0)
                        w = w * (1.0 + 0.5 * tri_breakout_arr)
                    if 'tri_event_window_14' in df_model.columns:
                        _s = pd.to_numeric(df_model['tri_event_window_14'], errors='coerce')
                        tri_ev14_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        tri_ev14_arr = np.nan_to_num(tri_ev14_arr, nan=0.0)
                        w = w * (1.0 + 0.25 * tri_ev14_arr)

                    # Flag weighting
                    if 'flag_conf' in df_model.columns:
                        _s = pd.to_numeric(df_model['flag_conf'], errors='coerce')
                        flag_conf_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        flag_conf_arr = np.nan_to_num(flag_conf_arr, nan=0.0)
                        flag_conf_arr = np.clip(flag_conf_arr, 0.0, 1.0)
                        w = w * (1.0 + flag_conf_arr)
                    if 'flag_breakout' in df_model.columns:
                        _s = pd.to_numeric(df_model['flag_breakout'], errors='coerce')
                        flag_breakout_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        flag_breakout_arr = np.nan_to_num(flag_breakout_arr, nan=0.0)
                        w = w * (1.0 + 0.5 * flag_breakout_arr)
                    if 'flag_event_window_14' in df_model.columns:
                        _s = pd.to_numeric(df_model['flag_event_window_14'], errors='coerce')
                        flag_ev14_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        flag_ev14_arr = np.nan_to_num(flag_ev14_arr, nan=0.0)
                        w = w * (1.0 + 0.25 * flag_ev14_arr)

                    # Wedge weighting
                    if 'wedge_conf' in df_model.columns:
                        _s = pd.to_numeric(df_model['wedge_conf'], errors='coerce')
                        wedge_conf_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        wedge_conf_arr = np.nan_to_num(wedge_conf_arr, nan=0.0)
                        wedge_conf_arr = np.clip(wedge_conf_arr, 0.0, 1.0)
                        w = w * (1.0 + wedge_conf_arr)
                    if 'wedge_breakout' in df_model.columns:
                        _s = pd.to_numeric(df_model['wedge_breakout'], errors='coerce')
                        wedge_breakout_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        wedge_breakout_arr = np.nan_to_num(wedge_breakout_arr, nan=0.0)
                        w = w * (1.0 + 0.5 * wedge_breakout_arr)
                    if 'wedge_event_window_14' in df_model.columns:
                        _s = pd.to_numeric(df_model['wedge_event_window_14'], errors='coerce')
                        wedge_ev14_arr: np.ndarray = np.asarray(_s.values, dtype=float)
                        wedge_ev14_arr = np.nan_to_num(wedge_ev14_arr, nan=0.0)
                        w = w * (1.0 + 0.25 * wedge_ev14_arr)

                    # Aggregated pattern features weighting
                    if 'pattern_breakout_any' in df_model.columns:
                        _s = pd.to_numeric(df_model['pattern_breakout_any'], errors='coerce')
                        patt_brk_any: np.ndarray = np.asarray(_s.values, dtype=float)
                        patt_brk_any = np.nan_to_num(patt_brk_any, nan=0.0)
                        w = w * (1.0 + 0.25 * patt_brk_any)
                    if 'pattern_event_window_14' in df_model.columns:
                        _s = pd.to_numeric(df_model['pattern_event_window_14'], errors='coerce')
                        patt_ev14: np.ndarray = np.asarray(_s.values, dtype=float)
                        patt_ev14 = np.nan_to_num(patt_ev14, nan=0.0)
                        w = w * (1.0 + 0.25 * patt_ev14)
                    if 'pattern_dir_consensus' in df_model.columns:
                        _s = pd.to_numeric(df_model['pattern_dir_consensus'], errors='coerce').abs()
                        patt_dirc: np.ndarray = np.asarray(_s.values, dtype=float)
                        patt_dirc = np.nan_to_num(patt_dirc, nan=0.0)
                        # Modest emphasis on strong directional consensus
                        w = w * (1.0 + 0.20 * patt_dirc)

                    # Short-horizon multipliers: emphasize recent events and net candlestick bias
                    try:
                        if horizon in (1, 3):
                            # Boost event windows (7-day variants if present)
                            if 'pattern_event_window_7' in df_model.columns:
                                _s = pd.to_numeric(df_model['pattern_event_window_7'], errors='coerce')
                                patt_ev7 = np.nan_to_num(np.asarray(_s.values, dtype=float), nan=0.0)
                                w = w * (1.0 + 0.20 * patt_ev7)
                            # Boost any breakout flags slightly more
                            w = w * (1.10)
                            # Net pattern candlestick bias if available (pat_net5/10)
                            for _col in ('pat_net5', 'pat_net10'):
                                if _col in df_model.columns:
                                    _s = pd.to_numeric(df_model[_col], errors='coerce').abs()
                                    _arr = np.asarray(_s.values, dtype=float)
                                    _arr = np.nan_to_num(_arr, nan=0.0)
                                    w = w * (1.0 + 0.10 * np.clip(_arr, 0.0, 1.0))
                        elif horizon == 7:
                            # Moderate boost for week horizon
                            if 'pattern_event_window_7' in df_model.columns:
                                _s = pd.to_numeric(df_model['pattern_event_window_7'], errors='coerce')
                                patt_ev7: np.ndarray = np.asarray(_s.values, dtype=float)
                                patt_ev7 = np.nan_to_num(patt_ev7, nan=0.0)
                                w = w * (1.0 + 0.10 * patt_ev7)
                            w = w * (1.05)
                    except Exception as e:
                        logger.debug(f"Pattern weight boost failed: {e}")

                    # Optional: global pattern weight scale per horizon (env override)
                    try:
                        # ✅ FIX: Use ConfigManager for consistent config access
                        scale_map = {
                            1: float(ConfigManager.get('ML_PATTERN_WEIGHT_SCALE_1D', '1.0')),
                            3: float(ConfigManager.get('ML_PATTERN_WEIGHT_SCALE_3D', '1.0')),
                            7: float(ConfigManager.get('ML_PATTERN_WEIGHT_SCALE_7D', '1.0')),
                            14: float(ConfigManager.get('ML_PATTERN_WEIGHT_SCALE_14D', '1.0')),
                            30: float(ConfigManager.get('ML_PATTERN_WEIGHT_SCALE_30D', '1.0')),
                        }
                        w = w * float(scale_map.get(horizon, 1.0))
                    except Exception as e:
                        logger.debug(f"Pattern weight scale failed: {e}")

                    # Clip weights to avoid instability
                    w = np.asarray(np.clip(w, 0.5, 5.0), dtype=float)
                except Exception as e:
                    logger.debug(f"Weight calculation failed, using uniform weights: {e}")
                    w = np.ones(len(df_model), dtype=float)
                
                # ✅ FIX: Final validation before creating X
                try:
                    X = df_model[horizon_feature_cols].values
                    if X.shape[1] == 0:
                        logger.error(f"CRITICAL: X has 0 features for {symbol} {horizon}d! Cannot proceed.")
                        return False
                    if X.shape[0] == 0:
                        logger.error(f"CRITICAL: X has 0 samples for {symbol} {horizon}d! Cannot proceed.")
                        return False
                except (KeyError, IndexError) as e:
                    logger.error(f"CRITICAL: Failed to create X matrix for {symbol} {horizon}d: {e}")
                    logger.error(f"  horizon_feature_cols ({len(horizon_feature_cols)}): {horizon_feature_cols[:10]}...")
                    logger.error(f"  df_model columns ({len(df_model.columns)}): {list(df_model.columns)[:10]}...")
                    return False
                
                y = df_model[target].values
                
                # Time series split
                # ⚡ USE PURGED CV: Prevents data leakage with purging + embargo
                # ⚡ HORIZON-AWARE: scale purge/embargo with prediction horizon to avoid label overlap leakage
                purge_gap = max(int(horizon), 5)
                embargo_td = max(int(horizon // 2), 1)
                tscv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=purge_gap, embargo_td=embargo_td)
                logger.info(f"✅ Using Purged Time-Series CV (n_splits=3, purge={purge_gap}, embargo={embargo_td}) - HORIZON-AWARE")
                
                # Train models
                horizon_models = {}
                
                # 1. XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        # ⚡ IMPROVED: Regime-based and horizon-based adaptive parameters for ALL horizons
                        # High volatility → more regularization, less depth
                        # Longer horizon → more conservative, slower learning
                        
                        # ⚡ ANTI-OVERFITTING: Conservative hyperparameters
                        # Previous: n_est=700, max_d=10 → massive overfitting
                        # New: Lower complexity, higher regularization
                        
                        if horizon == 1:
                            # 1d: Aggressive parameters to overcome underfitting
                            n_est = int(600 * (1.0 - 0.05 * regime_score))  # tighten: −25%
                            max_d = int(8 * (1.0 - 0.1 * regime_score))   # 8-7 (deeper)
                            lr = 0.15  # Faster learning (was 0.08)
                            reg_a = 0.0  # No L1
                            reg_l = 0.3  # slightly higher L2
                            logger.info(f"🎯 1d AGGRESSIVE FIX: n_est={n_est}, max_depth={max_d}, lr={lr:.3f}, reg_alpha={reg_a:.2f}, reg_lambda={reg_l:.1f}")
                        
                        elif horizon == 3:
                            # 3d: Aggressive parameters to overcome underfitting
                            n_est = int(560 * (1.0 - 0.05 * regime_score))  # tighten: −25%
                            max_d = int(8 * (1.0 - 0.1 * regime_score))   # 8-7 (deeper)
                            lr = 0.14  # Faster learning (was 0.07)
                            reg_a = 0.0  # No L1
                            reg_l = 0.3
                            logger.info(f"🎯 3d AGGRESSIVE FIX: n_est={n_est}, max_depth={max_d}, lr={lr:.3f}, reg_alpha={reg_a:.2f}, reg_lambda={reg_l:.1f}")
                        
                        elif horizon == 7:
                            # 7d: Aggressive parameters to overcome underfitting
                            n_est = int(525 * (1.0 - 0.05 * regime_score))  # tighten: −25%
                            max_d = int(7 * (1.0 - 0.1 * regime_score))    # 7-6 (deeper)
                            lr = 0.13  # Faster learning (was 0.06)
                            reg_a = 0.0  # No L1
                            reg_l = 0.35
                            logger.info(f"🎯 7d AGGRESSIVE FIX: n_est={n_est}, max_depth={max_d}, lr={lr:.3f}, reg_alpha={reg_a:.2f}, reg_lambda={reg_l:.1f}")
                        
                        elif horizon == 14:
                            # 14d: Aggressive parameters to overcome underfitting
                            n_est = int(650 * (1.0 - 0.05 * regime_score))  # 650-617 (3.6x more!)
                            max_d = int(7 * (1.0 - 0.1 * regime_score))    # 7-6 (deeper)
                            lr = 0.12  # Faster learning (was 0.05)
                            reg_a = 0.0  # No L1
                            reg_l = 0.2  # Minimal L2 (was 1.0)
                            logger.info(f"🎯 14d AGGRESSIVE FIX: n_est={n_est}, max_depth={max_d}, lr={lr:.3f}, reg_alpha={reg_a:.2f}, reg_lambda={reg_l:.1f}")
                        
                        elif horizon == 30:
                            # 30d: Aggressive parameters to overcome underfitting
                            n_est = int(600 * (1.0 - 0.05 * regime_score))  # 600-570 (4x more!)
                            max_d = int(6 * (1.0 - 0.15 * regime_score))   # 6-5 (deeper)
                            lr = 0.11  # Faster learning (was 0.05)
                            reg_a = 0.0  # No L1
                            reg_l = 0.25  # Minimal L2 (was 1.2)
                            logger.info(f"🎯 30d AGGRESSIVE FIX: n_est={n_est}, max_depth={max_d}, lr={lr:.3f}, reg_alpha={reg_a:.1f}, reg_lambda={reg_l:.1f}")
                        
                        else:
                            # Unknown horizon: safe defaults
                            n_est = 150
                            max_d = 4
                            lr = 0.04
                            reg_a = 0.7
                            reg_l = 2.5
                            logger.warning(f"⚠️ Unknown horizon {horizon}d, using safe default XGBoost parameters")
                        
                        # Env overrides for HPO (Optuna) or manual tuning
                        try:
                            # ✅ FIX: Use ConfigManager for consistent config access
                            _env_n_est = ConfigManager.get('OPTUNA_XGB_N_ESTIMATORS')
                            if _env_n_est not in (None, ''):
                                n_est = int(float(_env_n_est))
                                logger.info(f"⚙️ {symbol} {horizon}d: Using HPO param OPTUNA_XGB_N_ESTIMATORS={n_est}")
                            _env_md = ConfigManager.get('OPTUNA_XGB_MAX_DEPTH')
                            if _env_md not in (None, ''):
                                max_d = int(float(_env_md))
                                logger.info(f"⚙️ {symbol} {horizon}d: Using HPO param OPTUNA_XGB_MAX_DEPTH={max_d}")
                            _env_lr = ConfigManager.get('OPTUNA_XGB_LEARNING_RATE')
                            if _env_lr not in (None, ''):
                                lr = float(_env_lr)
                                logger.info(f"⚙️ {symbol} {horizon}d: Using HPO param OPTUNA_XGB_LEARNING_RATE={lr}")
                            _env_sub = ConfigManager.get('OPTUNA_XGB_SUBSAMPLE')
                            if _env_sub not in (None, ''):
                                sub_override = float(_env_sub)
                            else:
                                sub_override = None
                            _env_col = ConfigManager.get('OPTUNA_XGB_COLSAMPLE_BYTREE')
                            if _env_col not in (None, ''):
                                col_override = float(_env_col)
                            else:
                                col_override = None
                            _env_ra = ConfigManager.get('OPTUNA_XGB_REG_ALPHA')
                            if _env_ra not in (None, ''):
                                reg_a = float(_env_ra)
                            _env_rl = ConfigManager.get('OPTUNA_XGB_REG_LAMBDA')
                            if _env_rl not in (None, ''):
                                reg_l = float(_env_rl)
                            _env_mcw = ConfigManager.get('OPTUNA_XGB_MIN_CHILD_WEIGHT')
                            if _env_mcw not in (None, ''):
                                mcw_override = int(float(_env_mcw))
                            else:
                                mcw_override = None
                            _env_gamma = ConfigManager.get('OPTUNA_XGB_GAMMA')
                            if _env_gamma not in (None, ''):
                                gamma_override = float(_env_gamma)
                            else:
                                gamma_override = None
                        except Exception as e:
                            logger.debug(f"XGBoost param override parsing failed: {e}")
                            sub_override = None
                            col_override = None
                            mcw_override = None
                            gamma_override = None
                        
                        # DIRECTIONAL LOSS FIX (2025-10-20): Custom objective for direction accuracy
                        # Previous: Pure MSE → conservative predictions (46% <0.5%, only 13.5% real <0.5%)
                        # New: Hybrid MSE + Directional Loss → penalize wrong direction heavily
                        # FIXED: Now using native XGBoost DMatrix API for custom objective support
                        # ✅ FIX: Use ConfigManager for consistent config access
                        use_directional = bool(int(ConfigManager.get('ML_USE_DIRECTIONAL_LOSS', '1')))  # Enabled by default!
                        
                        # XGBoost parameters dict
                        # Note: For native API, use num_boost_round instead of n_estimators
                        # For sklearn API, n_estimators is added separately
                        xgb_params = {
                            'max_depth': max_d,
                            'learning_rate': lr,
                            'subsample': sub_override if sub_override is not None else 0.8,
                            'colsample_bytree': col_override if col_override is not None else 0.8,
                            'nthread': self.train_threads,
                            'n_jobs': self.train_threads,
                            'min_child_weight': mcw_override if mcw_override is not None else 2,
                            'gamma': gamma_override if gamma_override is not None else 0.0,
                            'reg_alpha': reg_a,
                            'reg_lambda': reg_l,
                            'verbosity': 0
                        }
                        # Advanced XGBoost params from HPO (optional)
                        try:
                            _tm = ConfigManager.get('OPTUNA_XGB_TREE_METHOD')
                            if _tm not in (None, ''):
                                xgb_params['tree_method'] = str(_tm)
                        except Exception as e:
                            logger.debug(f"Failed to get OPTUNA_XGB_TREE_METHOD: {e}")
                        try:
                            _gp = ConfigManager.get('OPTUNA_XGB_GROW_POLICY')
                            if _gp not in (None, ''):
                                xgb_params['grow_policy'] = str(_gp)
                        except Exception as e:
                            logger.debug(f"Failed to get OPTUNA_XGB_GROW_POLICY: {e}")
                        try:
                            _mb = ConfigManager.get('OPTUNA_XGB_MAX_BIN')
                            if _mb not in (None, ''):
                                xgb_params['max_bin'] = int(float(_mb))
                        except Exception as e:
                            logger.debug(f"Failed to get OPTUNA_XGB_MAX_BIN: {e}")
                        
                        # Short-horizon deadband: zero out tiny moves for training (reduces noise)
                        # Supports fixed thresholds via env and optional adaptive (std/ATR) overrides
                        try:
                            # ✅ FIX: Use ConfigManager for consistent config access
                            deadband_map = {
                                1: float(ConfigManager.get('ML_DEADBAND_1D', '0.006')),
                                3: float(ConfigManager.get('ML_DEADBAND_3D', '0.006')),
                                7: float(ConfigManager.get('ML_DEADBAND_7D', '0.005')),
                                14: float(ConfigManager.get('ML_DEADBAND_14D', '0.000')),
                                30: float(ConfigManager.get('ML_DEADBAND_30D', '0.000')),
                            }
                        except Exception as e:
                            ErrorHandler.handle(e, 'enhanced_ml_init_deadband', level='debug')
                            deadband_map = {1: 0.006, 3: 0.006, 7: 0.005, 14: 0.0, 30: 0.0}
                        deadband_thr = float(deadband_map.get(horizon, 0.0))

                        # Adaptive override (std/ATR) if configured
                        try:
                            # ✅ FIX: Use ConfigManager for consistent config access
                            adapt_mode = ConfigManager.get('ML_ADAPTIVE_DEADBAND_MODE', '').lower().strip()
                            k_env = None
                            if horizon == 1:
                                k_env = ConfigManager.get('ML_ADAPTIVE_K_1D')
                            elif horizon == 3:
                                k_env = ConfigManager.get('ML_ADAPTIVE_K_3D')
                            elif horizon == 7:
                                k_env = ConfigManager.get('ML_ADAPTIVE_K_7D')
                            elif horizon == 14:
                                k_env = ConfigManager.get('ML_ADAPTIVE_K_14D')
                            elif horizon == 30:
                                k_env = ConfigManager.get('ML_ADAPTIVE_K_30D')
                            if adapt_mode in ('std', 'atr') and k_env not in (None, ''):
                                k_val = float(k_env)
                                # std-based scale of horizon returns; ATR not wired here, fallback to std
                                scale = float(np.nanstd(y))
                                if scale > 0.0 and k_val > 0.0:
                                    deadband_thr = float(k_val * scale)
                        except Exception as e:
                            logger.debug(f"Adaptive deadband calculation failed: {e}")

                        try:
                            y_db_full = np.asarray(y, dtype=float).copy()
                            if deadband_thr > 0.0 and horizon in (1, 3, 7, 14, 30):
                                mask_db = np.abs(y_db_full) < deadband_thr
                                y_db_full[mask_db] = 0.0
                            else:
                                y_db_full = y_db_full
                        except Exception as e:
                            logger.debug(f"Deadband calculation failed, using original y: {e}")
                            y_db_full = y
                        
                        # Cross-validation (on returns) + OOF collection for meta-learner
                        xgb_scores = []
                        xgb_oof_preds = np.full(len(X), np.nan)  # OOF predictions storage
                        
                        # ✅ CRITICAL FIX: Check xgboost availability before use
                        if not XGBOOST_AVAILABLE or xgb is None:
                            logger.error("XGBoost not available but xgboost model requested! Skipping...")
                            continue
                        
                        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                            try:
                                X_train, X_val = X[train_idx], X[val_idx]
                                y_train, y_val = y[train_idx], y[val_idx]
                                # Deadbanded training labels
                                y_train_db = y_db_full[train_idx] if isinstance(y_db_full, np.ndarray) else y_train
                                
                                if use_directional:
                                    # ⚡ NATIVE XGBOOST API with custom objective
                                    dtrain = xgb.DMatrix(X_train, label=y_train_db, weight=w[train_idx])  # type: ignore[attr-defined]
                                    dval = xgb.DMatrix(X_val, label=y_val)  # type: ignore[attr-defined]
                                    
                                    # Train with custom objective and metric
                                    bst = xgb.train(  # type: ignore[attr-defined]
                                        xgb_params,
                                        dtrain,
                                        num_boost_round=n_est,
                                        obj=directional_loss_xgboost,
                                        custom_metric=directional_metric_xgboost,
                                        evals=[(dval, 'eval')],
                                        early_stopping_rounds=self.early_stop_rounds,
                                        verbose_eval=True
                                    )
                                    
                                    # Predict
                                    pred = bst.predict(dval)
                                    
                                    # Store model for later (will be converted to sklearn-compatible format)
                                    xgb_model_fold = bst
                                else:
                                    # ⚡ SKLEARN API (fallback - standard MSE)
                                    # ✅ FIX: Use first seed from base_seeds instead of hardcoded 42
                                    fallback_seed = self.base_seeds[0] if self.base_seeds else 42
                                    xgb_params_clean = {k: v for k, v in xgb_params.items() if k not in ['seed']}
                                    xgb_model_fold = xgb.XGBRegressor(n_estimators=n_est, objective='reg:squarederror', random_state=int(fallback_seed), **xgb_params_clean)  # type: ignore[union-attr]
                                    try:
                                        xgb_model_fold.fit(
                                            X_train, y_train_db,
                                            sample_weight=w[train_idx],
                                    eval_set=[(X_val, y_val)],
                                            early_stopping_rounds=self.early_stop_rounds,
                                    verbose=False
                                )
                                    except Exception as e:
                                        logger.debug(f"XGBoost fit with sample_weight failed, retrying without: {e}")
                                        xgb_model_fold.fit(
                                            X_train, y_train_db,
                                            eval_set=[(X_val, y_val)],
                                            early_stopping_rounds=self.early_stop_rounds,
                                            verbose=False
                                        )
                                    pred = xgb_model_fold.predict(X_val)
                                    try:
                                        best_iter = getattr(xgb_model_fold, 'best_iteration', None)
                                        if best_iter is not None:
                                            logger.info(f"XGB sklearn fold {fold}: best_iteration={best_iter}")
                                    except Exception as e:
                                        logger.debug(f"Failed to get best_iteration: {e}")
                                
                                # Optional: horizon-wise cap using percentile of training labels (avoid leakage)
                                try:
                                    cap_map = {
                                        # ✅ FIX: Use ConfigManager for consistent config access
                                        1: ConfigManager.get('ML_CAP_PCTL_1D', ''),
                                        3: ConfigManager.get('ML_CAP_PCTL_3D', ''),
                                        7: ConfigManager.get('ML_CAP_PCTL_7D', ''),
                                        14: ConfigManager.get('ML_CAP_PCTL_14D', ''),
                                        30: ConfigManager.get('ML_CAP_PCTL_30D', ''),
                                    }
                                    _cap_p = cap_map.get(horizon, '')
                                    if _cap_p not in (None, ''):
                                        p = float(_cap_p)
                                        # Use absolute distribution from training fold labels (non-deadbanded)
                                        cap_abs = float(np.percentile(np.abs(y_train), p)) if len(y_train) > 0 else float('nan')
                                        if cap_abs == cap_abs and cap_abs > 0.0:  # not NaN
                                            pred = np.asarray(np.clip(pred, -cap_abs, cap_abs), dtype=float)
                                except Exception as e:
                                    logger.debug(f"Prediction cap calculation failed: {e}")

                                xgb_oof_preds[val_idx] = pred  # ⚡ Save OOF predictions!
                                score = r2_score(y_val, pred)
                                xgb_scores.append(score)
                                
                                # Calculate direction hit for this fold with horizon-aware threshold
                                try:
                                    # ✅ FIX: Use ConfigManager for consistent config access
                                    base_threshold_eval = float(ConfigManager.get('ML_LOSS_THRESHOLD', '0.005'))
                                except Exception as e:
                                    logger.debug(f"Failed to get ML_LOSS_THRESHOLD, using 0.005: {e}")
                                    base_threshold_eval = 0.005
                                # Directional evaluation threshold: independent override via env, else fallback
                                thr_eval = base_threshold_eval
                                try:
                                    eval_map = {
                                        # ✅ FIX: Use ConfigManager for consistent config access
                                        1: ConfigManager.get('ML_DIR_EVAL_THRESH_1D', ''),
                                        3: ConfigManager.get('ML_DIR_EVAL_THRESH_3D', ''),
                                        7: ConfigManager.get('ML_DIR_EVAL_THRESH_7D', ''),
                                        14: ConfigManager.get('ML_DIR_EVAL_THRESH_14D', ''),
                                        30: ConfigManager.get('ML_DIR_EVAL_THRESH_30D', ''),
                                    }
                                    _v = eval_map.get(horizon, '')
                                    if _v not in (None, ''):
                                        thr_eval = float(_v)
                                    else:
                                        # Fallback: use larger of base threshold and training deadband for short horizons
                                        if horizon in (1, 3, 7):
                                            thr_eval = max(base_threshold_eval, float(deadband_thr))
                                except Exception as e:
                                    logger.debug(f"Dir eval threshold calculation failed: {e}")
                                # Original (flat-to-zero) method
                                y_val_dir = np.where(np.abs(y_val) < thr_eval, 0, np.sign(y_val))
                                pred_dir = np.where(np.abs(pred) < thr_eval, 0, np.sign(pred))
                                dir_hit_fold = np.mean(y_val_dir == pred_dir) * 100
                                # HPO-style masked method (evaluate only significant true AND predicted moves)
                                try:
                                    m_mask = (np.abs(y_val) > thr_eval) & (np.abs(pred) > thr_eval)
                                    if np.any(m_mask):
                                        dir_hit_fold_masked = float(
                                            np.mean(np.sign(y_val[m_mask]) == np.sign(pred[m_mask])) * 100.0
                                        )
                                    else:
                                        dir_hit_fold_masked = float('nan')
                                except Exception as e:
                                    logger.debug(f"Dir hit masked calculation failed: {e}")
                                    dir_hit_fold_masked = float('nan')
                                logger.info(
                                    f"XGBoost fold {fold}: R²={score:.3f}, DirHit(all)={dir_hit_fold:.1f}%"
                                    + (f", DirHit(mask)={dir_hit_fold_masked:.1f}%" if dir_hit_fold_masked == dir_hit_fold_masked else "")
                                )
                            except Exception as e:
                                logger.error(f"XGBoost fold {fold} error: {e}")
                                import traceback
                                logger.error(traceback.format_exc())
                                # Don't raise - continue with other folds
                        
                        # Compute OOF metrics (more realistic than train-tail)
                        try:
                            _mask = ~np.isnan(xgb_oof_preds)
                            if np.any(_mask):
                                xgb_rmse_oof = float(np.sqrt(mean_squared_error(y[_mask], xgb_oof_preds[_mask])))
                                xgb_mape_oof = float(self._smape(y[_mask], xgb_oof_preds[_mask]))
                                # Additional: normalized RMSE and direction hit-rate (with eval threshold)
                                try:
                                    y_mask = np.asarray(y[_mask], dtype=float)
                                    pred_mask = np.asarray(xgb_oof_preds[_mask], dtype=float)
                                    std_y = float(np.std(y_mask)) if y_mask.size > 1 else float('nan')
                                    nrmse = float(xgb_rmse_oof / std_y) if (not np.isnan(std_y) and std_y > 0) else float('inf')
                                    # Compute evaluation threshold for this horizon
                                    try:
                                        # ✅ FIX: Use ConfigManager for consistent config access
                                        base_threshold_eval = float(ConfigManager.get('ML_LOSS_THRESHOLD', '0.005'))
                                    except Exception as e:
                                        logger.debug(f"Failed to get ML_LOSS_THRESHOLD, using 0.005: {e}")
                                        base_threshold_eval = 0.005
                                    thr_eval = base_threshold_eval
                                    try:
                                        eval_map = {
                                            # ✅ FIX: Use ConfigManager for consistent config access
                                            1: ConfigManager.get('ML_DIR_EVAL_THRESH_1D', ''),
                                            3: ConfigManager.get('ML_DIR_EVAL_THRESH_3D', ''),
                                            7: ConfigManager.get('ML_DIR_EVAL_THRESH_7D', ''),
                                            14: ConfigManager.get('ML_DIR_EVAL_THRESH_14D', ''),
                                            30: ConfigManager.get('ML_DIR_EVAL_THRESH_30D', ''),
                                        }
                                        _v = eval_map.get(horizon, '')
                                        if _v not in (None, ''):
                                            thr_eval = float(_v)
                                        else:
                                            if horizon in (1, 3, 7):
                                                thr_eval = max(base_threshold_eval, float(deadband_thr))
                                    except Exception as e:
                                        logger.debug(f"Dir eval threshold override failed: {e}")
                                    # Original (flat-to-zero) dirhit (fraction 0-1)
                                    y_dir_oof = np.where(np.abs(y_mask) < thr_eval, 0, np.sign(y_mask))
                                    pred_dir_oof = np.where(np.abs(pred_mask) < thr_eval, 0, np.sign(pred_mask))
                                    dir_hit = float(np.mean(y_dir_oof == pred_dir_oof))
                                    # HPO-style masked dirhit (fraction 0-1)
                                    try:
                                        m_mask = (np.abs(y_mask) > thr_eval) & (np.abs(pred_mask) > thr_eval)
                                        if np.any(m_mask):
                                            dir_hit_masked = float(
                                                np.mean(np.sign(y_mask[m_mask]) == np.sign(pred_mask[m_mask]))
                                            )
                                        else:
                                            dir_hit_masked = float('nan')
                                    except Exception as e:
                                        logger.debug(f"OOF dir hit masked calculation failed: {e}")
                                        dir_hit_masked = float('nan')
                                    
                                    # ⚡ DIAGNOSTIC: Log feature impact summary
                                    logger.info(
                                        f"📊 XGBoost OOF Performance ({symbol} {horizon}d): "
                                        f"RMSE={xgb_rmse_oof:.4f}, MAPE={xgb_mape_oof:.4f}, "
                                        f"DirHit={dir_hit*100:.2f}% (mask={dir_hit_masked*100:.2f}%), "
                                        f"n_samples={_mask.sum()}, "
                                        f"features=[DirectionalLoss={self.use_directional_loss}, "
                                        f"SeedBagging={self.enable_seed_bagging}, "
                                        f"Talib={self.enable_talib_patterns}, "
                                        f"SmartEnsemble={self.use_smart_ensemble}, "
                                        f"MetaStacking={self.enable_meta_stacking}, "
                                        f"RegimeDetection={self.use_regime_detection}]"
                                    )
                                except Exception as e:
                                    logger.debug(f"NRMSE/dir hit calculation failed: {e}")
                                    nrmse = float('nan')
                                    dir_hit = float('nan')
                                    dir_hit_masked = float('nan')
                            else:
                                xgb_rmse_oof = float('nan')
                                xgb_mape_oof = float('nan')
                                nrmse = float('nan')
                                dir_hit = float('nan')
                                dir_hit_masked = float('nan')
                        except Exception as e:
                            logger.debug(f"XGBoost OOF metrics calculation failed: {e}")
                            xgb_rmse_oof = float('nan')
                            xgb_mape_oof = float('nan')
                            nrmse = float('nan')
                            dir_hit = float('nan')
                            dir_hit_masked = float('nan')

                        # ⚡ SEED BAGGING: Train with multiple seeds for variance reduction
                        # UPDATED: Now supports directional loss with native XGBoost API
                        xgb_models = []
                        
                        # ✅ CRITICAL FIX: Check xgboost availability before seed bagging
                        if not XGBOOST_AVAILABLE or xgb is None:
                            logger.error("XGBoost not available but seed bagging requested! Skipping...")
                        else:
                            for seed in self.base_seeds:
                                try:
                                    if use_directional:
                                        # Native XGBoost with custom objective
                                        # Train final model on deadbanded labels for short horizons
                                        # Use last 10% as validation for early stopping
                                        val_size = max(1, int(len(X) * 0.1))
                                        X_train_full = X[:-val_size]
                                        y_train_full = (y_db_full[:-val_size] if isinstance(y_db_full, np.ndarray) else y[:-val_size])
                                        w_train_full = w[:-val_size]
                                        X_val_full = X[-val_size:]
                                        y_val_full = (y_db_full[-val_size:] if isinstance(y_db_full, np.ndarray) else y[-val_size:])
                                        
                                        dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, weight=w_train_full)  # type: ignore[attr-defined]
                                        dval_full = xgb.DMatrix(X_val_full, label=y_val_full)  # type: ignore[attr-defined]
                                        xgb_params_seed = xgb_params.copy()
                                        xgb_params_seed['seed'] = seed
                                        
                                        seed_model = xgb.train(  # type: ignore[attr-defined]
                                            xgb_params_seed,
                                            dtrain_full,
                                            num_boost_round=n_est,
                                            obj=directional_loss_xgboost,
                                            custom_metric=directional_metric_xgboost,
                                            evals=[(dval_full, 'eval')],
                                            early_stopping_rounds=self.early_stop_rounds,
                                            verbose_eval=False
                                        )
                                    else:
                                        # Sklearn API (fallback)
                                        # Use last 10% as validation for early stopping
                                        val_size = max(1, int(len(X) * 0.1))
                                        X_train_full = X[:-val_size]
                                        y_train_full = (y_db_full[:-val_size] if isinstance(y_db_full, np.ndarray) else y[:-val_size])
                                        w_train_full = w[:-val_size]
                                        X_val_full = X[-val_size:]
                                        y_val_full = (y_db_full[-val_size:] if isinstance(y_db_full, np.ndarray) else y[-val_size:])
                                        
                                        xgb_params_seed = {k: v for k, v in xgb_params.items() if k not in ['seed', 'random_state']}
                                        seed_model = xgb.XGBRegressor(  # type: ignore[union-attr]
                                            n_estimators=n_est,
                                            objective='reg:squarederror',
                                            random_state=seed,
                                            **xgb_params_seed
                                        )
                                        # Train final model on deadbanded labels for short horizons with sample weights and early stopping
                                        seed_model.fit(
                                            X_train_full, y_train_full,
                                            sample_weight=w_train_full,
                                            eval_set=[(X_val_full, y_val_full)],
                                            early_stopping_rounds=self.early_stop_rounds,
                                            verbose=False
                                        )
                                    
                                    xgb_models.append(seed_model)
                                except Exception as e:
                                    logger.error(f"XGBoost seed {seed} error: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            
                        # Use first model as primary (for feature importance, etc)
                        # Note: Native XGBoost models (Booster) don't have sklearn interface
                        xgb_model = xgb_models[0] if xgb_models else None
                        
                        # CRITICAL FIX: R² to confidence conversion
                        raw_r2 = np.mean(xgb_scores)
                        confidence = self._r2_to_confidence(raw_r2)
                        
                        horizon_models['xgboost'] = {
                            'model': xgb_model,      # Primary model
                            'models': xgb_models,     # All seed models (for ensemble prediction)
                            'score': confidence,      # Confidence [0-1]
                            'raw_r2': float(raw_r2),  # Keep raw R² for debugging
                            'rmse': xgb_rmse_oof,
                            'mape': xgb_mape_oof,
                        }
                        
                        # Feature importance
                        # Note: Native XGBoost Booster has different API for feature importance
                        # ✅ FIX: Use horizon_feature_cols (selected features) instead of feature_cols (all features)
                        try:
                            if xgb_model is not None:
                                # Check if native Booster or sklearn wrapper
                                if hasattr(xgb_model, 'get_booster'):
                                    # Sklearn wrapper: use feature_importances_
                                    # ✅ FIX: Use horizon_feature_cols (32 features) not feature_cols (107 features)
                                    self.feature_importance[f"{symbol}_{horizon}d_xgb"] = dict(
                                        zip(horizon_feature_cols, xgb_model.feature_importances_)
                                    )
                                elif hasattr(xgb_model, 'get_score'):
                                    # Native XGBoost Booster: use get_score()
                                    importance_dict = xgb_model.get_score(importance_type='weight')
                                    # Map feature names (f0, f1, ...) to actual names
                                    # ✅ FIX: Use horizon_feature_cols (32 features) not feature_cols (107 features)
                                    self.feature_importance[f"{symbol}_{horizon}d_xgb"] = {
                                        horizon_feature_cols[int(k[1:])]: v for k, v in importance_dict.items() if k.startswith('f')
                                    }
                        except Exception as e:
                            logger.warning(f"Could not extract feature importance: {e}")
                        
                        _hit_pct = (dir_hit * 100.0) if not np.isnan(dir_hit) else float('nan')
                        logger.info(f"XGBoost {horizon}D - R²: {raw_r2:.3f}, MAPE: {xgb_mape_oof:.2f}% nRMSE:{nrmse:.2f} Hit:{_hit_pct:.1f}% → Confidence: {confidence:.3f}")
                        # Store concise metrics for pilot summaries
                        try:
                            if 'metrics' not in results:
                                results['metrics'] = {}
                            results['metrics'][f"{horizon}d"] = {
                                'r2': float(raw_r2),
                                'mape': float(xgb_mape_oof) if not np.isnan(xgb_mape_oof) else None,
                                'nrmse': float(nrmse) if not np.isnan(nrmse) else None,
                                'dir_hit_pct': float(_hit_pct) if not np.isnan(_hit_pct) else None,
                                # HPO-style masked dirhit (percentage)
                                'dir_hit_pct_masked': float(dir_hit_masked * 100.0) if (dir_hit_masked == dir_hit_masked) else None,
                            }
                        except Exception as e:
                            logger.debug(f"XGBoost results dict creation failed: {e}")

                        # ⚡ REMOVED: Stacked approach (direction classifier + isotonic calibration)
                        # REASON: Kullanılmıyor (sadece metrics için), HPO optimize ettiği şey değil
                        # HPO zaten base model parametrelerini optimize ediyor, bu yeterli
                        # Isotonic calibration adaptive learning ile çakışıyor (negatif spiral riski)
                        
                    except Exception as e:
                        logger.error(f"XGBoost eğitim hatası: {e}")
                
                # 2. LightGBM
                if LIGHTGBM_AVAILABLE and enable_lgb:
                    try:
                        # ⚡ ANTI-OVERFITTING: Conservative hyperparameters (matched with XGBoost)
                        if horizon == 1:
                            # 1d: Conservative
                            n_est_lgb = int(200 * (1.0 - 0.15 * regime_score))  # 200-170
                            max_d_lgb = int(5 * (1.0 - 0.2 * regime_score))     # 5-4
                            lr_lgb = 0.05
                            reg_a_lgb = 0.5 * (1.0 + 0.5 * regime_score)   # 0.5-0.75
                            reg_l_lgb = 2.0 * (1.0 + 0.5 * regime_score)   # 2.0-3.0
                            num_leaves_lgb = int(31 * (1.0 - 0.2 * regime_score))  # 31-25
                            logger.info(f"🎯 1d LightGBM anti-overfit: n_est={n_est_lgb}, max_depth={max_d_lgb}, num_leaves={num_leaves_lgb}")
                        
                        elif horizon == 3:
                            # 3d: Balanced but conservative
                            n_est_lgb = int(180 * (1.0 - 0.15 * regime_score))  # 180-153
                            max_d_lgb = int(5 * (1.0 - 0.2 * regime_score))     # 5-4
                            lr_lgb = 0.05
                            reg_a_lgb = 0.6 * (1.0 + 0.5 * regime_score)   # 0.6-0.9
                            reg_l_lgb = 2.2 * (1.0 + 0.5 * regime_score)   # 2.2-3.3
                            num_leaves_lgb = int(31 * (1.0 - 0.2 * regime_score))  # 31-25
                            logger.info(f"🎯 3d LightGBM anti-overfit: n_est={n_est_lgb}, max_depth={max_d_lgb}, num_leaves={num_leaves_lgb}")
                        
                        elif horizon == 7:
                            # 7d: More conservative
                            n_est_lgb = int(160 * (1.0 - 0.15 * regime_score))  # 160-136
                            max_d_lgb = int(4 * (1.0 - 0.25 * regime_score))    # 4-3
                            lr_lgb = 0.04
                            reg_a_lgb = 0.7 * (1.0 + 0.6 * regime_score)   # 0.7-1.1
                            reg_l_lgb = 2.5 * (1.0 + 0.6 * regime_score)   # 2.5-4.0
                            num_leaves_lgb = int(15 * (1.0 - 0.2 * regime_score))  # 15-12
                            logger.info(f"🎯 7d LightGBM anti-overfit: n_est={n_est_lgb}, max_depth={max_d_lgb}, num_leaves={num_leaves_lgb}")
                        
                        elif horizon == 14:
                            # 14d: Very conservative
                            n_est_lgb = int(140 * (1.0 - 0.15 * regime_score))  # 140-119
                            max_d_lgb = int(4 * (1.0 - 0.25 * regime_score))    # 4-3
                            lr_lgb = 0.04
                            reg_a_lgb = 0.8 * (1.0 + 0.7 * regime_score)   # 0.8-1.4
                            reg_l_lgb = 2.8 * (1.0 + 0.7 * regime_score)   # 2.8-4.8
                            num_leaves_lgb = int(15 * (1.0 - 0.2 * regime_score))  # 15-12
                            logger.info(f"🎯 14d LightGBM anti-overfit: n_est={n_est_lgb}, max_depth={max_d_lgb}, num_leaves={num_leaves_lgb}")
                        
                        elif horizon == 30:
                            # 30d: Ultra conservative
                            n_est_lgb = int(120 * (1.0 - 0.15 * regime_score))  # 120-102
                            max_d_lgb = int(3 * (1.0 - 0.33 * regime_score))    # 3-2
                            lr_lgb = 0.03
                            reg_a_lgb = 1.0 * (1.0 + regime_score)         # 1.0-2.0
                            reg_l_lgb = 3.0 * (1.0 + regime_score)         # 3.0-6.0
                            num_leaves_lgb = int(7 * (1.0 - 0.3 * regime_score))   # 7-5
                            logger.info(f"🎯 30d LightGBM anti-overfit: n_est={n_est_lgb}, max_depth={max_d_lgb}, num_leaves={num_leaves_lgb}")
                        
                        else:
                            # Unknown horizon: safe defaults
                            n_est_lgb = 150
                            max_d_lgb = 4
                            lr_lgb = 0.04
                            reg_a_lgb = 0.7
                            reg_l_lgb = 2.5
                            num_leaves_lgb = 15
                            logger.warning(f"⚠️ Unknown horizon {horizon}d, using safe default LightGBM parameters")
                        
                        # HPO overrides (Optuna) - per horizon
                        try:
                            # ✅ FIX: Use ConfigManager for consistent config access
                            _env_n_est_lgb = ConfigManager.get('OPTUNA_LGB_N_ESTIMATORS')
                            if _env_n_est_lgb not in (None, ''):
                                n_est_lgb = int(float(_env_n_est_lgb))
                            _env_md_lgb = ConfigManager.get('OPTUNA_LGB_MAX_DEPTH')
                            if _env_md_lgb not in (None, ''):
                                max_d_lgb = int(float(_env_md_lgb))
                            _env_lr_lgb = ConfigManager.get('OPTUNA_LGB_LEARNING_RATE')
                            if _env_lr_lgb not in (None, ''):
                                lr_lgb = float(_env_lr_lgb)
                            _env_nl_lgb = ConfigManager.get('OPTUNA_LGB_NUM_LEAVES')
                            if _env_nl_lgb not in (None, ''):
                                num_leaves_lgb = int(float(_env_nl_lgb))
                            _env_sub_lgb = ConfigManager.get('OPTUNA_LGB_SUBSAMPLE')
                            sub_override_lgb = float(_env_sub_lgb) if _env_sub_lgb not in (None, '') else None
                            _env_col_lgb = ConfigManager.get('OPTUNA_LGB_COLSAMPLE_BYTREE')
                            col_override_lgb = float(_env_col_lgb) if _env_col_lgb not in (None, '') else None
                            _env_ra_lgb = ConfigManager.get('OPTUNA_LGB_REG_ALPHA')
                            if _env_ra_lgb not in (None, ''):
                                reg_a_lgb = float(_env_ra_lgb)
                            _env_rl_lgb = ConfigManager.get('OPTUNA_LGB_REG_LAMBDA')
                            if _env_rl_lgb not in (None, ''):
                                reg_l_lgb = float(_env_rl_lgb)
                        except Exception as e:
                            logger.debug(f"LightGBM param override parsing failed: {e}")
                            sub_override_lgb = None
                            col_override_lgb = None
                        
                        # ✨ IMPROVED: Optimized hyperparameters (matched with XGBoost quality)
                        # ✅ FIX: Use first seed from base_seeds instead of hardcoded 42
                        fallback_seed = self.base_seeds[0] if self.base_seeds else 42
                        lgb_model = lgb.LGBMRegressor(  # type: ignore[union-attr]
                            n_estimators=n_est_lgb,     # Adaptive: 210-500
                            max_depth=max_d_lgb,        # Adaptive: 4-8
                            learning_rate=lr_lgb,       # Adaptive: 0.03-0.05
                            num_leaves=num_leaves_lgb,  # Adaptive: 22-63
                            # More conservative leaves to reduce overfitting on noise
                            min_data_in_leaf=max(25, int(num_leaves_lgb * 0.8)),
                            num_threads=self.train_threads,  # ENV: Parallelism
                            subsample=sub_override_lgb if sub_override_lgb is not None else 0.7,  # Row sampling
                            colsample_bytree=col_override_lgb if col_override_lgb is not None else 0.7,  # Feature sampling
                            reg_alpha=reg_a_lgb,        # Adaptive L1: 0.1-0.4
                            reg_lambda=reg_l_lgb,       # Adaptive L2: 1.0-3.0
                            random_state=int(fallback_seed),
                            n_jobs=self.train_threads,
                            verbose=-1
                        )
                        # Advanced LightGBM params from HPO (optional)
                        try:
                            _minleaf = ConfigManager.get('OPTUNA_LGB_MIN_DATA_IN_LEAF')
                            if _minleaf not in (None, ''):
                                lgb_model.set_params(min_data_in_leaf=int(float(_minleaf)))
                        except Exception as e:
                            logger.debug(f"Failed to set OPTUNA_LGB_MIN_DATA_IN_LEAF: {e}")
                        try:
                            _ffbn = ConfigManager.get('OPTUNA_LGB_FEATURE_FRACTION_BY_NODE')
                            if _ffbn not in (None, ''):
                                lgb_model.set_params(feature_fraction_bynode=float(_ffbn))
                        except Exception as e:
                            logger.debug(f"Failed to set OPTUNA_LGB_FEATURE_FRACTION_BY_NODE: {e}")
                        try:
                            _bf = ConfigManager.get('OPTUNA_LGB_BAGGING_FREQ')
                            if _bf not in (None, ''):
                                lgb_model.set_params(bagging_freq=int(float(_bf)))
                        except Exception as e:
                            logger.debug(f"Failed to set OPTUNA_LGB_BAGGING_FREQ: {e}")
                        try:
                            _mgs = ConfigManager.get('OPTUNA_LGB_MIN_GAIN_TO_SPLIT')
                            if _mgs not in (None, ''):
                                lgb_model.set_params(min_split_gain=float(_mgs))
                        except Exception as e:
                            logger.debug(f"Failed to set OPTUNA_LGB_MIN_GAIN_TO_SPLIT: {e}")
                        
                        # Cross-validation (on returns) + OOF
                        lgb_scores = []
                        lgb_oof_preds = np.full(len(X), np.nan)
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            # Ensure numpy arrays for LightGBM sklearn wrapper type hints
                            import numpy as _np
                            y_train = _np.asarray(y_train).astype(float)
                            y_val = _np.asarray(y_val).astype(float)
                            X_train = _np.asarray(X_train)
                            X_val = _np.asarray(X_val)

                            if len(val_idx) >= self.early_stop_min_val:
                                # LightGBM sklearn wrapper: use callbacks for early stopping
                                lgb_model.fit(  # type: ignore[arg-type]
                                    X_train, y_train,
                                    eval_set=[(X_val, y_val)],
                                    eval_metric='rmse',
                                    callbacks=[lgb.early_stopping(self.early_stop_rounds, verbose=False)],  # type: ignore[union-attr]
                                )
                            else:
                                lgb_model.fit(X_train, y_train)  # type: ignore[arg-type]

                            pred = lgb_model.predict(X_val)
                            lgb_oof_preds[val_idx] = pred  # Save OOF
                            score = r2_score(y_val, pred)
                            lgb_scores.append(score)
                        
                        # Compute OOF metrics
                        try:
                            _mask = ~np.isnan(lgb_oof_preds)
                            if np.any(_mask):
                                lgb_rmse_oof = float(np.sqrt(mean_squared_error(y[_mask], lgb_oof_preds[_mask])))
                                lgb_mape_oof = float(self._smape(y[_mask], lgb_oof_preds[_mask]))
                            else:
                                lgb_rmse_oof = float('nan')
                                lgb_mape_oof = float('nan')
                        except Exception as e:
                            logger.debug(f"LightGBM OOF metrics calculation failed: {e}")
                            lgb_rmse_oof = float('nan')
                            lgb_mape_oof = float('nan')

                        # ⚡ SEED BAGGING: Train with multiple seeds
                        lgb_models = []
                        for seed in self.base_seeds:
                            try:
                                seed_model = lgb.LGBMRegressor(  # type: ignore[union-attr]
                                    n_estimators=n_est_lgb,
                                    max_depth=max_d_lgb,
                                    learning_rate=lr_lgb,
                                    num_leaves=num_leaves_lgb,
                                    min_data_in_leaf=max(25, int(num_leaves_lgb * 0.8)),
                                    num_threads=self.train_threads,
                                    subsample=sub_override_lgb if sub_override_lgb is not None else 0.8,
                                    colsample_bytree=col_override_lgb if col_override_lgb is not None else 0.8,
                                    reg_alpha=reg_a_lgb,
                                    reg_lambda=reg_l_lgb,
                                    random_state=seed,
                                    n_jobs=self.train_threads,
                                    verbose=-1
                                )
                                seed_model.fit(np.asarray(X), np.asarray(y))  # type: ignore[arg-type]
                                lgb_models.append(seed_model)
                            except Exception as e:
                                logger.error(f"LightGBM seed {seed} error: {e}")
                            
                        lgb_model = lgb_models[0] if lgb_models else lgb_model
                        
                        # CRITICAL FIX: R² to confidence conversion
                        raw_r2 = np.mean(lgb_scores)
                        confidence = self._r2_to_confidence(raw_r2)
                        
                        horizon_models['lightgbm'] = {
                            'model': lgb_model,
                            'models': lgb_models,
                            'score': confidence,
                            'raw_r2': float(raw_r2),
                            'rmse': lgb_rmse_oof,
                            'mape': lgb_mape_oof,
                        }
                        
                        # Feature importance
                        # ✅ FIX: Use horizon_feature_cols (selected features) instead of feature_cols (all features)
                        self.feature_importance[f"{symbol}_{horizon}d_lgb"] = dict(
                            zip(horizon_feature_cols, lgb_model.feature_importances_)
                        )
                        
                        logger.info(f"LightGBM {horizon}D - R²: {raw_r2:.3f}, MAPE: {lgb_mape_oof:.2f}% → Confidence: {confidence:.3f}")
                        
                    except Exception as e:
                        logger.error(f"LightGBM eğitim hatası: {e}")
                
                # 3. CatBoost
                if CATBOOST_AVAILABLE and enable_cat:
                    try:
                        # ⚡ ANTI-OVERFITTING: Conservative hyperparameters (matched with XGBoost)
                        if horizon == 1:
                            # 1d: Conservative
                            n_iter_cat = int(200 * (1.0 - 0.15 * regime_score))  # 200-170
                            depth_cat = int(5 * (1.0 - 0.2 * regime_score))      # 5-4
                            lr_cat = 0.05
                            l2_cat = 3.0 * (1.0 + 0.5 * regime_score)  # 3.0-4.5
                            logger.info(f"🎯 1d CatBoost anti-overfit: iterations={n_iter_cat}, depth={depth_cat}, l2_leaf_reg={l2_cat:.1f}")
                        
                        elif horizon == 3:
                            # 3d: Balanced but conservative
                            n_iter_cat = int(180 * (1.0 - 0.15 * regime_score))  # 180-153
                            depth_cat = int(5 * (1.0 - 0.2 * regime_score))      # 5-4
                            lr_cat = 0.05
                            l2_cat = 3.5 * (1.0 + 0.5 * regime_score)  # 3.5-5.25
                            logger.info(f"🎯 3d CatBoost anti-overfit: iterations={n_iter_cat}, depth={depth_cat}, l2_leaf_reg={l2_cat:.1f}")
                        
                        elif horizon == 7:
                            # 7d: More conservative
                            n_iter_cat = int(160 * (1.0 - 0.15 * regime_score))  # 160-136
                            depth_cat = int(4 * (1.0 - 0.25 * regime_score))     # 4-3
                            lr_cat = 0.04
                            l2_cat = 4.0 * (1.0 + 0.6 * regime_score)  # 4.0-6.4
                            logger.info(f"🎯 7d CatBoost anti-overfit: iterations={n_iter_cat}, depth={depth_cat}, l2_leaf_reg={l2_cat:.1f}")
                        
                        elif horizon == 14:
                            # 14d: Very conservative
                            n_iter_cat = int(140 * (1.0 - 0.15 * regime_score))  # 140-119
                            depth_cat = int(4 * (1.0 - 0.25 * regime_score))     # 4-3
                            lr_cat = 0.04
                            l2_cat = 4.5 * (1.0 + 0.7 * regime_score)  # 4.5-7.7
                            logger.info(f"🎯 14d CatBoost anti-overfit: iterations={n_iter_cat}, depth={depth_cat}, l2_leaf_reg={l2_cat:.1f}")
                        
                        elif horizon == 30:
                            # 30d: Ultra conservative
                            n_iter_cat = int(120 * (1.0 - 0.15 * regime_score))  # 120-102
                            depth_cat = int(3 * (1.0 - 0.33 * regime_score))     # 3-2
                            lr_cat = 0.03
                            l2_cat = 5.0 * (1.0 + regime_score)        # 5.0-10.0
                            logger.info(f"🎯 30d CatBoost anti-overfit: iterations={n_iter_cat}, depth={depth_cat}, l2_leaf_reg={l2_cat:.1f}")
                        
                        else:
                            # Unknown horizon: safe defaults
                            n_iter_cat = 150
                            depth_cat = 4
                            lr_cat = 0.04
                            l2_cat = 4.0
                            logger.warning(f"⚠️ Unknown horizon {horizon}d, using safe default CatBoost parameters")
                        
                        # HPO overrides (Optuna) - per horizon
                        try:
                            # ✅ FIX: Use ConfigManager for consistent config access
                            _env_iter_cat = ConfigManager.get('OPTUNA_CAT_ITERATIONS')
                            if _env_iter_cat not in (None, ''):
                                n_iter_cat = int(float(_env_iter_cat))
                            _env_depth_cat = ConfigManager.get('OPTUNA_CAT_DEPTH')
                            if _env_depth_cat not in (None, ''):
                                depth_cat = int(float(_env_depth_cat))
                            _env_lr_cat = ConfigManager.get('OPTUNA_CAT_LEARNING_RATE')
                            if _env_lr_cat not in (None, ''):
                                lr_cat = float(_env_lr_cat)
                            _env_l2_cat = ConfigManager.get('OPTUNA_CAT_L2_LEAF_REG')
                            if _env_l2_cat not in (None, ''):
                                l2_cat = float(_env_l2_cat)
                            _env_sub_cat = ConfigManager.get('OPTUNA_CAT_SUBSAMPLE')
                            sub_override_cat = float(_env_sub_cat) if _env_sub_cat not in (None, '') else None
                            _env_rsm_cat = ConfigManager.get('OPTUNA_CAT_RSM')
                            rsm_override_cat = float(_env_rsm_cat) if _env_rsm_cat not in (None, '') else None
                        except Exception as e:
                            logger.debug(f"CatBoost param override parsing failed: {e}")
                            sub_override_cat = None
                            rsm_override_cat = None
                        
                        # ✨ IMPROVED: Optimized hyperparameters (matched with XGBoost quality)
                        # ✅ FIX: Use first seed from base_seeds instead of hardcoded 42
                        fallback_seed_cat = self.base_seeds[0] if self.base_seeds else 42
                        # Determine bootstrap_type first to respect CatBoost constraints (e.g., Bayesian forbids subsample)
                        bootstrap_type_norm = None
                        try:
                            _bt_env = ConfigManager.get('OPTUNA_CAT_BOOTSTRAP_TYPE')
                            if _bt_env not in (None, ''):
                                _raw_bt = str(_bt_env).strip()
                                _bt_map = {
                                    'false': 'No',
                                    '0': 'No',
                                    'none': None,
                                    'true': 'Bernoulli',
                                    '1': 'Bernoulli',
                                    'poisson': 'Poisson',
                                    'bayesian': 'Bayesian',
                                    'bernoulli': 'Bernoulli',
                                    'mvs': 'MVS',
                                    'no': 'No',
                                }
                                bootstrap_type_norm = _bt_map.get(_raw_bt.lower(), None)
                        except Exception as e:
                            logger.debug(f"CatBoost bootstrap_type parsing failed: {e}")
                            bootstrap_type_norm = None
                        
                        cat_init_params = {
                            'iterations': n_iter_cat,
                            'depth': depth_cat,
                            'learning_rate': lr_cat,
                            'l2_leaf_reg': l2_cat,
                            'border_count': 128,
                            'thread_count': self.train_threads,
                            'rsm': rsm_override_cat if rsm_override_cat is not None else 0.8,
                            'random_seed': int(fallback_seed_cat),
                            'allow_writing_files': False,
                            'train_dir': self.catboost_train_dir,
                            'logging_level': 'Silent',
                            'od_type': 'Iter',
                            'od_wait': self.early_stop_rounds,
                        }
                        # Only include subsample when bootstrap_type is not Bayesian (unsupported) and not 'No' (bootstrap disabled)
                        # ✅ FIX: CatBoost error: "you shoudn't provide bootstrap options if bootstrap is disabled"
                        # When bootstrap_type='No', subsample should not be set
                        if bootstrap_type_norm not in ('Bayesian', 'No', None):
                            cat_init_params['subsample'] = sub_override_cat if sub_override_cat is not None else 0.8
                        # Include bootstrap_type if provided
                        if bootstrap_type_norm:
                            cat_init_params['bootstrap_type'] = bootstrap_type_norm
                        
                        cat_model = cb.CatBoostRegressor(**cat_init_params)  # type: ignore[union-attr]
                        # Advanced CatBoost params from HPO (optional)
                        try:
                            _bc = ConfigManager.get('OPTUNA_CAT_BORDER_COUNT')
                            if _bc not in (None, ''):
                                cat_model.set_params(border_count=int(float(_bc)))
                        except Exception as e:
                            logger.debug(f"Failed to set OPTUNA_CAT_BORDER_COUNT: {e}")
                        try:
                            _rs = ConfigManager.get('OPTUNA_CAT_RANDOM_STRENGTH')
                            if _rs not in (None, ''):
                                cat_model.set_params(random_strength=float(_rs))
                        except Exception as e:
                            logger.debug(f"Failed to set OPTUNA_CAT_RANDOM_STRENGTH: {e}")
                        try:
                            _lei = ConfigManager.get('OPTUNA_CAT_LEAF_ESTIMATION_ITERATIONS')
                            if _lei not in (None, ''):
                                cat_model.set_params(leaf_estimation_iterations=int(float(_lei)))
                        except Exception as e:
                            logger.debug(f"Failed to set OPTUNA_CAT_LEAF_ESTIMATION_ITERATIONS: {e}")
                        try:
                            _bt = ConfigManager.get('OPTUNA_CAT_BOOTSTRAP_TYPE')
                            if _bt not in (None, ''):
                                # Normalize to valid CatBoost enums
                                raw = str(_bt).strip()
                                mapping = {
                                    'false': 'No',
                                    '0': 'No',
                                    'none': None,
                                    'true': 'Bernoulli',
                                    '1': 'Bernoulli',
                                    'poisson': 'Poisson',
                                    'bayesian': 'Bayesian',
                                    'bernoulli': 'Bernoulli',
                                    'mvs': 'MVS',
                                    'no': 'No',
                                }
                                norm = mapping.get(raw.lower(), None)
                                if norm:
                                    # ✅ FIX: If bootstrap_type is 'No' or 'Bayesian', ensure subsample is not set
                                    # CatBoost error: "you shoudn't provide bootstrap options if bootstrap is disabled"
                                    if norm in ('No', 'Bayesian'):
                                        # Remove subsample if it exists (model might have been created with subsample)
                                        try:
                                            # Check if subsample is in model params
                                            model_params = cat_model.get_params()
                                            if 'subsample' in model_params and model_params['subsample'] is not None:
                                                cat_model.set_params(subsample=None)
                                        except Exception as e:
                                            logger.debug(f"Failed to clear subsample param: {e}")
                                    cat_model.set_params(bootstrap_type=norm)
                                elif norm is None and raw.lower() == 'none':
                                    # Explicitly skip when None-like provided
                                    pass
                                else:
                                    try:
                                        logger.warning(f"Invalid OPTUNA_CAT_BOOTSTRAP_TYPE='{raw}', skipping (allowed: Poisson,Bayesian,Bernoulli,MVS,No)")
                                    except Exception as e:
                                        logger.debug(f"Failed to log bootstrap_type warning: {e}")
                        except Exception as e:
                            logger.debug(f"CatBoost bootstrap_type setup failed: {e}")
                        
                        # Cross-validation (on returns) + OOF
                        cat_scores = []
                        cat_oof_preds = np.full(len(X), np.nan)
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]

                            if len(val_idx) >= self.early_stop_min_val:
                                cat_model.fit(
                                    X_train, y_train,
                                    eval_set=(X_val, y_val),
                                    use_best_model=True
                                )  # logging_level in model params
                            else:
                                cat_model.fit(X_train, y_train)  # logging_level in model params

                            pred = cat_model.predict(X_val)
                            cat_oof_preds[val_idx] = pred  # Save OOF
                            score = r2_score(y_val, pred)
                            cat_scores.append(score)
                        
                        # ⚡ SEED BAGGING: Train with multiple seeds
                        cat_models = []
                        for seed in self.base_seeds:
                            try:
                                seed_params = {
                                    'iterations': n_iter_cat,
                                    'depth': depth_cat,
                                    'learning_rate': lr_cat,
                                    'l2_leaf_reg': l2_cat,
                                    'border_count': 128,
                                    'thread_count': self.train_threads,
                                    'rsm': rsm_override_cat if rsm_override_cat is not None else 0.8,
                                    'random_seed': seed,
                                    'allow_writing_files': False,
                                    'train_dir': self.catboost_train_dir,
                                    'logging_level': 'Silent',
                                    'od_type': 'Iter',
                                    'od_wait': self.early_stop_rounds,
                                }
                                # ✅ FIX: Only include subsample when bootstrap_type is not Bayesian (unsupported) and not 'No' (bootstrap disabled)
                                # When bootstrap_type='No', subsample should not be set
                                if bootstrap_type_norm not in ('Bayesian', 'No', None):
                                    seed_params['subsample'] = sub_override_cat if sub_override_cat is not None else 0.8
                                if bootstrap_type_norm:
                                    seed_params['bootstrap_type'] = bootstrap_type_norm
                                
                                seed_model = cb.CatBoostRegressor(**seed_params)  # type: ignore[union-attr]
                                seed_model.fit(X, y)
                                cat_models.append(seed_model)
                            except Exception as e:
                                logger.error(f"CatBoost seed {seed} error: {e}")
                            
                        cat_model = cat_models[0] if cat_models else cat_model
                        
                        # CRITICAL FIX: R² to confidence conversion
                        raw_r2 = np.mean(cat_scores)
                        confidence = self._r2_to_confidence(raw_r2)
                        
                        # Compute OOF metrics
                        try:
                            _mask = ~np.isnan(cat_oof_preds)
                            if np.any(_mask):
                                cat_rmse_oof = float(np.sqrt(mean_squared_error(y[_mask], cat_oof_preds[_mask])))
                                cat_mape_oof = float(self._smape(y[_mask], cat_oof_preds[_mask]))
                            else:
                                cat_rmse_oof = float('nan')
                                cat_mape_oof = float('nan')
                        except Exception as e:
                            logger.debug(f"CatBoost OOF metrics calculation failed: {e}")
                            cat_rmse_oof = float('nan')
                            cat_mape_oof = float('nan')

                        horizon_models['catboost'] = {
                            'model': cat_model,
                            'models': cat_models,
                            'score': confidence,
                            'raw_r2': float(raw_r2),
                            'rmse': cat_rmse_oof,
                            'mape': cat_mape_oof,
                        }
                        
                        # Feature importance
                        # ✅ FIX: Use horizon_feature_cols (selected features) instead of feature_cols (all features)
                        self.feature_importance[f"{symbol}_{horizon}d_cat"] = dict(
                            zip(horizon_feature_cols, cat_model.feature_importances_)
                        )
                        
                        logger.info(f"CatBoost {horizon}D - R²: {raw_r2:.3f}, MAPE: {cat_mape_oof:.2f}% → Confidence: {confidence:.3f}")
                        
                    except Exception as e:
                        logger.error(f"CatBoost eğitim hatası: {e}")
                
                # ⚡ META-LEARNER: Train Ridge on OOF predictions
                # ✅ FIX: Meta-stacking should only be used for short horizons (1d, 3d, 7d)
                # Short horizons have more noise, meta-stacking helps. Long horizons (14d, 30d) are already smooth.
                use_meta_for_horizon = self.enable_meta_stacking
                if hasattr(self, 'use_meta_stacking_short_only') and self.use_meta_stacking_short_only:
                    # Short-only mode: only enable for short horizons
                    use_meta_for_horizon = horizon in (1, 3, 7)
                
                if use_meta_for_horizon and len(horizon_models) >= 2:
                    try:
                        # Collect OOF predictions from all models (check if exist)
                        oof_list = []
                        if 'xgboost' in horizon_models:
                            try:
                                oof_list.append(xgb_oof_preds)  # type: ignore[name-defined]
                            except NameError:
                                pass
                        if 'lightgbm' in horizon_models:
                            try:
                                oof_list.append(lgb_oof_preds)  # type: ignore[name-defined]
                            except NameError:
                                pass
                        if 'catboost' in horizon_models:
                            try:
                                oof_list.append(cat_oof_preds)  # type: ignore[name-defined]
                            except NameError:
                                pass
                        
                        if len(oof_list) >= 2:
                            # ✅ FIX: Record which models were used for meta-learner training
                            # This ensures prediction uses the same model order
                            model_order = []  # Track model order: ['xgboost', 'lightgbm', 'catboost']
                            if 'xgboost' in horizon_models:
                                try:
                                    _ = xgb_oof_preds  # type: ignore[name-defined]
                                    model_order.append('xgboost')
                                except NameError:
                                    pass
                            if 'lightgbm' in horizon_models:
                                try:
                                    _ = lgb_oof_preds  # type: ignore[name-defined]
                                    model_order.append('lightgbm')
                                except NameError:
                                    pass
                            if 'catboost' in horizon_models:
                                try:
                                    _ = cat_oof_preds  # type: ignore[name-defined]
                                    model_order.append('catboost')
                                except NameError:
                                    pass
                            
                            # Stack OOF predictions as features
                            meta_X = np.column_stack(oof_list)  # Shape: (n_samples, n_models)
                            # Filter rows where all models have valid OOF preds
                            try:
                                _valid_mask = ~np.any(np.isnan(meta_X), axis=1)
                            except Exception as e:
                                logger.debug(f"Meta X validation mask creation failed: {e}")
                                _valid_mask = np.ones(meta_X.shape[0], dtype=bool)
                            meta_X = meta_X[_valid_mask]
                            meta_y = y[_valid_mask]  # True targets are returns
                            
                            # ⚡ FIX: Scale features for Ridge (linear model needs scaling!)
                            meta_scaler = StandardScaler()
                            meta_X_scaled = meta_scaler.fit_transform(meta_X)
                            
                            # Train Ridge meta-learner
                            from sklearn.linear_model import Ridge
                            # r2_score and mean_squared_error already imported at top of file
                            
                            # ⚡ DIAGNOSTIC: Analyze OOF predictions quality before meta-learning
                            oof_correlations = {}
                            oof_r2_scores = {}
                            oof_rmse_scores = {}
                            
                            for i, oof_pred in enumerate(oof_list):
                                model_name = ['XGBoost', 'LightGBM', 'CatBoost'][i] if i < 3 else f'Model_{i}'
                                try:
                                    # Correlation with target
                                    corr = float(np.corrcoef(oof_pred[_valid_mask], meta_y)[0, 1]) if len(meta_y) > 1 else 0.0
                                    oof_correlations[model_name] = corr
                                    
                                    # R² score
                                    r2 = float(r2_score(meta_y, oof_pred[_valid_mask])) if len(meta_y) > 1 else 0.0
                                    oof_r2_scores[model_name] = r2
                                    
                                    # RMSE
                                    rmse = float(np.sqrt(mean_squared_error(meta_y, oof_pred[_valid_mask]))) if len(meta_y) > 1 else float('inf')
                                    oof_rmse_scores[model_name] = rmse
                                except Exception as e:
                                    logger.debug(f"OOF diagnostic failed for {model_name}: {e}")
                            
                            # Cross-validation for optimal alpha (or use HPO-provided alpha)
                            from sklearn.model_selection import cross_val_score
                            
                            # ⚡ NEW: Check if HPO provided alpha via environment variable
                            hpo_alpha = ConfigManager.get('ML_META_STACKING_ALPHA')
                            if hpo_alpha is not None and hpo_alpha != '':
                                try:
                                    best_alpha = float(hpo_alpha)
                                    best_cv_score = None  # Not calculated when using HPO alpha
                                    logger.debug(f"Using HPO-provided meta-stacking alpha: {best_alpha:.2f}")
                                except (ValueError, TypeError):
                                    best_alpha = 1.0
                                    best_cv_score = float('-inf')
                            else:
                                # Use CV to find best alpha (original behavior)
                                alpha_candidates = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
                                best_alpha = 1.0
                                best_cv_score = float('-inf')
                            
                            if best_cv_score is not None and len(meta_X_scaled) >= 20:  # Only do CV if not using HPO alpha
                                try:
                                    cv_scores = {}
                                    for alpha in alpha_candidates:
                                        temp_model = Ridge(alpha=alpha)
                                        scores = cross_val_score(temp_model, meta_X_scaled, meta_y, cv=min(5, len(meta_X_scaled) // 4), scoring='r2')
                                        cv_scores[alpha] = float(np.mean(scores))
                                        if cv_scores[alpha] > best_cv_score:
                                            best_cv_score = cv_scores[alpha]
                                            best_alpha = alpha
                                except Exception as e:
                                    logger.debug(f"Alpha CV failed: {e}, using default alpha=1.0")
                            
                            meta_model = Ridge(alpha=best_alpha)
                            if len(meta_X_scaled) >= self.early_stop_min_val:
                                meta_model.fit(meta_X_scaled, meta_y)
                                
                                # ⚡ DIAGNOSTIC: Evaluate meta-learner performance
                                meta_train_pred = meta_model.predict(meta_X_scaled)
                                meta_train_r2 = float(r2_score(meta_y, meta_train_pred)) if len(meta_y) > 1 else 0.0
                                meta_train_rmse = float(np.sqrt(mean_squared_error(meta_y, meta_train_pred))) if len(meta_y) > 1 else float('inf')
                                
                                # Base ensemble (weighted average) performance for comparison
                                base_ensemble_pred = np.mean(meta_X, axis=1)  # Simple average
                                base_ensemble_r2 = float(r2_score(meta_y, base_ensemble_pred)) if len(meta_y) > 1 else 0.0
                                base_ensemble_rmse = float(np.sqrt(mean_squared_error(meta_y, base_ensemble_pred))) if len(meta_y) > 1 else float('inf')
                                
                                # OOF predictions correlation matrix
                                oof_corr_matrix = np.corrcoef(meta_X.T) if meta_X.shape[1] > 1 else np.array([[1.0]])
                                
                                # ⚡ FIX: Handle None values in format strings
                                cv_score_str = f"CV R²={best_cv_score:.3f}" if best_cv_score is not None else "CV R²=N/A (HPO alpha)"
                                logger.info(
                                    f"✅ Meta-learner trained for {symbol} {horizon}d: "
                                    f"alpha={best_alpha:.2f} ({cv_score_str}), "
                                    f"train R²={meta_train_r2:.3f}, train RMSE={meta_train_rmse:.4f}, "
                                    f"base ensemble R²={base_ensemble_r2:.3f}, base ensemble RMSE={base_ensemble_rmse:.4f}"
                                )
                                # ⚡ FIX: Handle empty dicts in format strings
                                oof_corr_str = ', '.join([f'{k}={v:.3f}' for k, v in oof_correlations.items()]) if oof_correlations else 'N/A'
                                oof_r2_str = ', '.join([f'{k}={v:.3f}' for k, v in oof_r2_scores.items()]) if oof_r2_scores else 'N/A'
                                oof_rmse_str = ', '.join([f'{k}={v:.4f}' for k, v in oof_rmse_scores.items()]) if oof_rmse_scores else 'N/A'
                                logger.info(
                                    f"   OOF correlations: {oof_corr_str}, "
                                    f"OOF R²: {oof_r2_str}, "
                                    f"OOF RMSE: {oof_rmse_str}"
                                )
                                logger.info(
                                    f"   OOF correlation matrix: {oof_corr_matrix.tolist()}, "
                                    f"meta-learner improvement: R²={meta_train_r2 - base_ensemble_r2:+.3f}, "
                                    f"RMSE={base_ensemble_rmse - meta_train_rmse:+.4f}"
                                )
                            else:
                                raise RuntimeError("Insufficient OOF rows for meta-learner")
                            
                            # Store meta-learner + scaler
                            meta_key = f"{symbol}_{horizon}d_meta"
                            self.meta_learners[meta_key] = meta_model
                            # Store scaler with special key
                            scaler_key = f"{symbol}_{horizon}d_meta_scaler"
                            self.scalers[scaler_key] = meta_scaler
                            # ✅ FIX: Store model order to ensure prediction uses same order
                            model_order_key = f"{symbol}_{horizon}d_meta_model_order"
                            if not hasattr(self, 'meta_model_orders'):
                                self.meta_model_orders = {}
                            self.meta_model_orders[model_order_key] = model_order
                            
                            logger.info(f"✅ Meta-learner saved for {symbol} {horizon}d (alpha={best_alpha:.2f}, models={model_order})")
                    except Exception as e:
                        logger.error(f"Meta-learner training error: {e}")
                
                # Store models and results
                self.models[f"{symbol}_{horizon}d"] = horizon_models
                results[f"{horizon}d"] = horizon_models
                
                # ⚡ CRITICAL: Save horizon-specific feature columns for prediction!
                # Each horizon may use different features after reduction
                self.models[f"{symbol}_{horizon}d_features"] = horizon_feature_cols
                
                # Note: self.feature_columns is now a dict (horizon -> features) for adaptive learning
                # No longer overwriting with feature_cols (backward compatibility removed)
            
            # Save models
            self.save_enhanced_models(symbol)
            
            # Store performance
            self.model_performance[symbol] = results
            
            # Auto-backtest: varsayılan AÇIK (ölçüm için)
            # ✅ FIX: Use ConfigManager for consistent config access
            backtest_enabled = str(ConfigManager.get('ENABLE_AUTO_BACKTEST', 'True')).lower() in ('1', 'true', 'yes', 'on')
            if backtest_enabled:
                try:
                    # ⚡ CRITICAL: Verify horizon_features are in memory before backtest
                    # If not, backtest will fail with "Horizon-specific features not found"
                    horizon_features_ready = all(
                        f"{symbol}_{h}d_features" in self.models 
                        for h in self.prediction_horizons
                    )
                    
                    if not horizon_features_ready:
                        logger.warning(f"⚠️ {symbol}: Horizon features not in memory, SKIPPING backtest")
                        logger.warning(f"Available keys: {[k for k in self.models.keys() if 'features' in k]}")
                        results['backtest'] = {
                            'status': 'skipped',
                            'reason': 'Horizon-specific features not found in memory'
                        }
                    else:
                        logger.debug(f"✅ {symbol}: All horizon features ready for backtest")
                        
                        from bist_pattern.ml.ml_backtester import get_ml_backtester
                        from bist_pattern.ml.backtest_service import run_backtest
                        backtester = get_ml_backtester()
                        backtest_results = run_backtest(
                            backtester,
                            symbol,
                            self,
                            data,
                            [f"{h}d" for h in self.prediction_horizons],
                        )
                    
                        # Store backtest results
                        # ✅ FIX: Properly handle backtest_results and prevent KeyError
                        if isinstance(backtest_results, dict) and backtest_results.get('status') == 'success':
                            overall = backtest_results.get('overall', {})
                            results['backtest'] = {
                                'sharpe_ratio': overall.get('avg_sharpe_ratio', 0.0),
                                'mape': overall.get('avg_mape', 0.0),
                                'hit_rate': overall.get('avg_hit_rate', 0.0),
                                'quality': overall.get('quality', 'UNKNOWN')
                            }
                            
                            # Log successful backtest
                            backtest_data = results.get('backtest', {})
                            logger.info(
                                f"📊 Backtest {symbol}: Sharpe={backtest_data.get('sharpe_ratio', 0):.2f}, "
                                f"Hit Rate={backtest_data.get('hit_rate', 0):.1%}, "
                                f"Quality={backtest_data.get('quality', 'UNKNOWN')}"
                            )
                            
                            # Warn if poor performance
                            # ✅ FIX: Use ConfigManager for consistent config access
                            min_sharpe = float(ConfigManager.get('BACKTEST_MIN_SHARPE', '0.3'))
                            sharpe_ratio = backtest_data.get('sharpe_ratio', 0)
                            if sharpe_ratio < min_sharpe:
                                logger.warning(
                                    f"⚠️ {symbol} model has low Sharpe ratio: "
                                    f"{sharpe_ratio:.2f} < {min_sharpe}"
                                )
                        else:
                            # Backtest failed or returned invalid result
                            results['backtest'] = {
                                'status': 'failed',
                                'reason': 'Invalid backtest result or status != success',
                                'sharpe_ratio': 0.0,
                                'mape': 0.0,
                                'hit_rate': 0.0,
                                'quality': 'UNKNOWN'
                            }
                            logger.debug(f"⚠️ {symbol}: Backtest failed or invalid result")
                    
                except Exception as e:
                    logger.warning(f"Backtest error for {symbol}: {e}")
            
            # ⚡ ADAPTIVE LEARNING Phase 2: GERÇEK incremental learning
            # Model 382 gün ile eğitildi, şimdi 164 gün test data ile öğrenecek
            if use_adaptive and test_days > 0:
                logger.info(f"🔄 Phase 2: GERÇEK Adaptive Learning - {test_days} test günü ile incremental öğrenme")
                logger.info(f"   📊 Phase 2 data: train_data: {train_days} days, test_data: {test_days} days")
                logger.info(f"   📅 Test period: {test_data.index.min().date()} - {test_data.index.max().date()}")
                logger.info("   Model gerçekleşmeleri görecek ve kendini düzeltecek")
                
                try:
                    # Feature engineering on test data
                    df_test_features = self.create_advanced_features(test_data, symbol=symbol)
                    
                    # Clean test data
                    for col in df_test_features.columns:
                        if df_test_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            df_test_features[col] = df_test_features[col].ffill()
                            if df_test_features[col].isna().any():
                                col_mean = df_test_features[col].mean()
                                if pd.notna(col_mean):
                                    df_test_features[col] = df_test_features[col].fillna(col_mean)
                                else:
                                    df_test_features[col] = df_test_features[col].fillna(0)
                    
                    try:
                        from bist_pattern.features.cleaning import clean_dataframe as _clean_df
                        df_test_features = _clean_df(df_test_features)
                    except Exception as e:
                        logger.debug(f"External clean_dataframe failed for test, using internal: {e}")
                        df_test_features = self._clean_data(df_test_features)
                    
                    # Incremental learning for each horizon
                    adapted_count = 0
                    # Note: xgb is already imported at module level (line 154)
                    
                    for h in self.prediction_horizons:
                        try:
                            # Create target for test data
                            # ✅ FIX: Use same target calculation as training (consistent!)
                            # Training: close.shift(-horizon) / close - 1.0
                            # Adaptive: (close.shift(-h) / close - 1.0) for consistency
                            target_col = f'target_{h}d'
                            if 'close' in df_test_features.columns:
                                # ✅ FIX: Use same formula as training (shift before division, not pct_change)
                                df_test_features[target_col] = (
                                    df_test_features['close'].shift(-h) / df_test_features['close'] - 1.0
                                )
                            else:
                                logger.warning(f"   ⚠️ {h}d: 'close' column missing, skipping")
                                continue
                            
                            # Drop NaN targets
                            df_test_clean = df_test_features.dropna(subset=[target_col]).copy()
                            if len(df_test_clean) < 10:
                                logger.warning(f"   ⚠️ {h}d: Insufficient test data ({len(df_test_clean)} samples)")
                                continue
                            
                            # Get feature columns from Phase 1
                            if not hasattr(self, 'feature_columns') or f'{h}d' not in self.feature_columns:
                                logger.warning(f"   ⚠️ {h}d: Feature columns not saved in Phase 1, skipping")
                                continue
                            
                            feature_cols = self.feature_columns[f'{h}d']
                            
                            # Ensure all features exist
                            missing_cols = [c for c in feature_cols if c not in df_test_clean.columns]
                            if missing_cols:
                                logger.warning(f"   ⚠️ {h}d: Missing {len(missing_cols)} features, skipping")
                                continue
                            
                            X_test = df_test_clean[feature_cols].values
                            y_test = df_test_clean[target_col].values
                            
                            # Get Phase 1 model
                            # Models are stored as: self.models[f"{symbol}_{h}d"] = {'xgboost': model, ...}
                            horizon_key = f"{symbol}_{h}d"
                            if horizon_key not in self.models:
                                logger.warning(f"   ⚠️ {h}d: Horizon models not found, skipping")
                                continue
                            
                            horizon_models = self.models[horizon_key]
                            if not isinstance(horizon_models, dict) or 'xgboost' not in horizon_models:
                                logger.warning(f"   ⚠️ {h}d: XGBoost model not found in horizon models, skipping")
                                continue
                            
                            # XGBoost model is stored as {'model': XGBRegressor, 'score': ..., ...}
                            xgb_entry = horizon_models['xgboost']
                            if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                phase1_model = xgb_entry['model']
                            else:
                                # Direct model (old format)
                                phase1_model = xgb_entry
                            
                            # ✅ KULLANICI MANTIĞI: Önce tahmin yap, sonra gerçek değerlerle karşılaştır
                            # 1. Model test_data üzerinde tahmin yap (hiç görmediği veriler)
                            logger.info(f"   🔮 {h}d Phase 2 Step 1: Model test_data üzerinde tahmin yapıyor (hiç görmediği veriler)")
                            try:
                                # ✅ FIX: Check if model is sklearn wrapper or native XGBoost
                                # Sklearn wrapper accepts numpy array, native XGBoost needs DMatrix
                                if hasattr(phase1_model, 'predict') and hasattr(phase1_model, 'get_booster'):
                                    # Sklearn wrapper - can use numpy array directly
                                    y_pred_before = phase1_model.predict(X_test)
                                else:
                                    # Native XGBoost or other - convert to DMatrix
                                    try:
                                        import xgboost as _xgb_local  # avoid shadowing module-level 'xgb'
                                        dtest = _xgb_local.DMatrix(X_test, feature_names=feature_cols)
                                        y_pred_before = phase1_model.predict(dtest)
                                    except Exception as e:
                                        logger.debug(f"DMatrix prediction failed, using numpy array fallback: {e}")
                                        # Fallback: try numpy array anyway
                                        y_pred_before = phase1_model.predict(X_test)
                                
                                # 2. Gerçek değerlerle karşılaştır ve hataları hesapla
                                logger.info(f"   📊 {h}d Phase 2 Step 2: Gerçek değerlerle karşılaştırılıyor")
                                errors = y_pred_before - y_test
                                mse_before = np.mean(errors ** 2)
                                mae_before = np.mean(np.abs(errors))
                                
                                # Direction accuracy (kullanıcının istediği: hangi tahminler doğru/yanlış)
                                thr = 0.005
                                y_true_dir = np.where(np.abs(y_test) < thr, 0, np.sign(y_test))
                                y_pred_dir = np.where(np.abs(y_pred_before) < thr, 0, np.sign(y_pred_before))
                                dir_match = (y_true_dir == y_pred_dir)
                                dir_accuracy_before = np.mean(dir_match) * 100
                                
                                logger.info(
                                    f"   📈 {h}d Phase 2 ÖNCE (tahmin vs gerçek): "
                                    f"MSE={mse_before:.6f}, MAE={mae_before:.6f}, "
                                    f"DirAccuracy={dir_accuracy_before:.2f}% "
                                    f"({dir_match.sum()}/{len(dir_match)} doğru yön)"
                                )
                                
                                # 3. Model'e gerçek değerleri göster ve "disiplin et"
                                logger.info(
                                    f"   🎓 {h}d Phase 2 Step 3: Model'e gerçek değerler gösteriliyor, "
                                    f"hangi tahminlerin doğru/yanlış olduğu öğretiliyor"
                                )
                            except Exception as e:
                                logger.warning(f"   ⚠️ {h}d: Prediction before adaptive learning failed: {e}")
                                y_pred_before = None
                            
                            # XGBoost incremental learning
                            try:
                                # Convert to DMatrix
                                dtrain_test = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
                                
                                # Get booster from Phase 1 model
                                if hasattr(phase1_model, 'get_booster'):
                                    # Sklearn wrapper: obtain underlying Booster
                                    booster = phase1_model.get_booster()
                                    params = phase1_model.get_params()
                                    n_rounds = max(10, int(params.get('n_estimators', 100) * 0.1))
                                    xgb_params = {
                                        'max_depth': params.get('max_depth', 6),
                                        'learning_rate': params.get('learning_rate', 0.1) * 0.5,
                                        'subsample': params.get('subsample', 0.8),
                                        'colsample_bytree': params.get('colsample_bytree', 0.8),
                                        'objective': 'reg:squarederror',
                                        'verbosity': 0
                                    }
                                    adapted_booster = xgb.train(
                                        xgb_params,
                                        dtrain_test,
                                        num_boost_round=n_rounds,
                                        xgb_model=booster,
                                        verbose_eval=False
                                    )
                                    # Update wrapper's booster
                                    try:
                                        phase1_model._Booster = adapted_booster  # type: ignore[attr-defined]
                                    except Exception as e:
                                        # ✅ FIX: Log exception instead of silent pass
                                        logger.warning(f"Failed to update model booster for {symbol} {h}d adaptive learning: {e}")
                                        # Don't raise - continue with training
                                    if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                        xgb_entry['model'] = phase1_model
                                        horizon_models['xgboost'] = xgb_entry
                                    else:
                                        horizon_models['xgboost'] = phase1_model
                                    self.models[horizon_key] = horizon_models
                                    adapted_count += 1
                                    
                                    # ✅ KULLANICI MANTIĞI: Sonra tekrar tahmin yap, iyileşmeyi göster
                                    if y_pred_before is not None:
                                        try:
                                            y_pred_after = phase1_model.predict(X_test)
                                            errors_after = y_pred_after - y_test
                                            mse_after = np.mean(errors_after ** 2)
                                            mae_after = np.mean(np.abs(errors_after))
                                            
                                            y_pred_dir_after = np.where(np.abs(y_pred_after) < thr, 0, np.sign(y_pred_after))
                                            dir_match_after = (y_true_dir == y_pred_dir_after)
                                            dir_accuracy_after = np.mean(dir_match_after) * 100
                                            
                                            logger.info(
                                                f"   📈 {h}d Phase 2 SONRA (tahmin vs gerçek): "
                                                f"MSE={mse_after:.6f}, MAE={mae_after:.6f}, "
                                                f"DirAccuracy={dir_accuracy_after:.2f}% "
                                                f"({dir_match_after.sum()}/{len(dir_match_after)} doğru yön)"
                                            )
                                            logger.info(
                                                f"   🎯 {h}d Phase 2 İYİLEŞME: "
                                                f"MSE: {mse_before:.6f} → {mse_after:.6f} "
                                                f"({((mse_before-mse_after)/mse_before*100):.1f}%), "
                                                f"DirAccuracy: {dir_accuracy_before:.2f}% → {dir_accuracy_after:.2f}% "
                                                f"({dir_accuracy_after-dir_accuracy_before:+.2f}%)"
                                            )
                                        except Exception as e:
                                            logger.debug(f"Failed to log phase2 metrics: {e}")
                                    
                                    logger.info(f"   ✅ {h}d GERÇEK incremental learning (sklearn): +{len(X_test)} test samples, +{n_rounds} rounds")
                                    logger.info(f"   📈 {h}d Phase 2: Model adapted with {len(X_test)} test samples, {n_rounds} additional rounds")
                                elif hasattr(phase1_model, 'save_model') or hasattr(phase1_model, 'get_score'):
                                    # Native Booster path
                                    booster = phase1_model
                                    n_rounds = 50  # conservative default for fine-tuning
                                    xgb_params = {
                                        'objective': 'reg:squarederror',
                                        'verbosity': 0
                                    }
                                    adapted_booster = xgb.train(
                                        xgb_params,
                                        dtrain_test,
                                        num_boost_round=n_rounds,
                                        xgb_model=booster,
                                        verbose_eval=False
                                    )
                                    # Store back the adapted booster
                                    if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                        xgb_entry['model'] = adapted_booster
                                        horizon_models['xgboost'] = xgb_entry
                                    else:
                                        horizon_models['xgboost'] = adapted_booster
                                    self.models[horizon_key] = horizon_models
                                    adapted_count += 1
                                    
                                    # ✅ KULLANICI MANTIĞI: Sonra tekrar tahmin yap, iyileşmeyi göster
                                    if y_pred_before is not None:
                                        try:
                                            # Native booster için predict
                                            dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
                                            y_pred_after = adapted_booster.predict(dtest)
                                            errors_after = y_pred_after - y_test
                                            mse_after = np.mean(errors_after ** 2)
                                            mae_after = np.mean(np.abs(errors_after))
                                            
                                            y_pred_dir_after = np.where(np.abs(y_pred_after) < thr, 0, np.sign(y_pred_after))
                                            dir_match_after = (y_true_dir == y_pred_dir_after)
                                            dir_accuracy_after = np.mean(dir_match_after) * 100
                                            
                                            logger.info(
                                                f"   📈 {h}d Phase 2 SONRA (tahmin vs gerçek): "
                                                f"MSE={mse_after:.6f}, MAE={mae_after:.6f}, "
                                                f"DirAccuracy={dir_accuracy_after:.2f}% "
                                                f"({dir_match_after.sum()}/{len(dir_match_after)} doğru yön)"
                                            )
                                            logger.info(
                                                f"   🎯 {h}d Phase 2 İYİLEŞME: "
                                                f"MSE: {mse_before:.6f} → {mse_after:.6f} "
                                                f"({((mse_before-mse_after)/mse_before*100):.1f}%), "
                                                f"DirAccuracy: {dir_accuracy_before:.2f}% → {dir_accuracy_after:.2f}% "
                                                f"({dir_accuracy_after-dir_accuracy_before:+.2f}%)"
                                            )
                                        except Exception as e:
                                            logger.debug(f"Failed to log phase2 metrics: {e}")
                                    
                                    logger.info(f"   ✅ {h}d GERÇEK incremental learning (booster): +{len(X_test)} test samples, +{n_rounds} rounds")
                                    logger.info(f"   📈 {h}d Phase 2: Model adapted with {len(X_test)} test samples, {n_rounds} additional rounds")
                                else:
                                    logger.warning(f"   ⚠️ {h}d: Unknown XGBoost model type for incremental learning, skipping")
                                    
                            except Exception as e:
                                logger.warning(f"   ⚠️ {h}d incremental learning error: {e}")
                                import traceback
                                logger.debug(traceback.format_exc())
                            
                        except Exception as e:
                            logger.warning(f"   ⚠️ {h}d adaptive learning error: {e}")
                    
                    if adapted_count > 0:
                        logger.info(f"✅ Phase 1.5 (Test ile Disiplin) tamamlandı: {adapted_count}/{len(self.prediction_horizons)} model güncellendi")
                        
                        # ✅ PHASE 1.6: Tekrar 300 günlük veri ile eğitim (güncellenmiş model ile)
                        # Böylece 300 günlük öğrenmeler korunuyor, test verisiyle düzelterek eğittiğimiz model ezilmiyor
                        logger.info(f"🔄 Phase 1.6: Tekrar {train_days} günlük veri ile eğitim (güncellenmiş model ile)")
                        logger.info("   📊 Bu adım, test verisiyle ince ayar yaptığımız modelin 300 günlük öğrenmelerini koruyor")
                        
                        try:
                            # Feature engineering on train_data (300 gün)
                            df_train_features_phase16 = self.create_advanced_features(train_data, symbol=symbol)
                            
                            # Clean train data
                            for col in df_train_features_phase16.columns:
                                if df_train_features_phase16[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                    df_train_features_phase16[col] = df_train_features_phase16[col].ffill()
                                    if df_train_features_phase16[col].isna().any():
                                        col_mean = df_train_features_phase16[col].mean()
                                        if pd.notna(col_mean):
                                            df_train_features_phase16[col] = df_train_features_phase16[col].fillna(col_mean)
                                        else:
                                            df_train_features_phase16[col] = df_train_features_phase16[col].fillna(0)
                            
                            try:
                                from bist_pattern.features.cleaning import clean_dataframe as _clean_df
                                df_train_features_phase16 = _clean_df(df_train_features_phase16)
                            except Exception as e:
                                logger.debug(f"External clean_dataframe failed for phase16, using internal: {e}")
                                df_train_features_phase16 = self._clean_data(df_train_features_phase16)
                            
                            # Incremental learning on train_data for each horizon
                            phase16_adapted_count = 0
                            for h in self.prediction_horizons:
                                try:
                                    target_col = f'target_{h}d'
                                    if 'close' not in df_train_features_phase16.columns:
                                        continue
                                    
                                    df_train_features_phase16[target_col] = (
                                        df_train_features_phase16['close'].shift(-h) / df_train_features_phase16['close'] - 1.0
                                    )
                                    
                                    df_train_clean_phase16 = df_train_features_phase16.dropna(subset=[target_col]).copy()
                                    if len(df_train_clean_phase16) < 10:
                                        continue
                                    
                                    if f'{h}d' not in self.feature_columns:
                                        continue
                                    
                                    feature_cols = self.feature_columns[f'{h}d']
                                    missing_cols = [c for c in feature_cols if c not in df_train_clean_phase16.columns]
                                    if missing_cols:
                                        continue
                                    
                                    X_train_phase16 = df_train_clean_phase16[feature_cols].values
                                    y_train_phase16 = df_train_clean_phase16[target_col].values
                                    
                                    horizon_key = f"{symbol}_{h}d"
                                    if horizon_key not in self.models:
                                        continue
                                    
                                    horizon_models = self.models[horizon_key]
                                    if not isinstance(horizon_models, dict) or 'xgboost' not in horizon_models:
                                        continue
                                    
                                    xgb_entry = horizon_models['xgboost']
                                    if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                        phase15_model = xgb_entry['model']
                                    else:
                                        phase15_model = xgb_entry
                                    
                                    # Incremental learning on train_data
                                    try:
                                        dtrain_phase16 = xgb.DMatrix(X_train_phase16, label=y_train_phase16, feature_names=feature_cols)
                                        
                                        if hasattr(phase15_model, 'get_booster'):
                                            booster = phase15_model.get_booster()
                                            params = phase15_model.get_params()
                                            n_rounds = max(10, int(params.get('n_estimators', 100) * 0.1))
                                            xgb_params = {
                                                'max_depth': params.get('max_depth', 6),
                                                'learning_rate': params.get('learning_rate', 0.1) * 0.5,
                                                'subsample': params.get('subsample', 0.8),
                                                'colsample_bytree': params.get('colsample_bytree', 0.8),
                                                'objective': 'reg:squarederror',
                                                'verbosity': 0
                                            }
                                            adapted_booster_phase16 = xgb.train(
                                                xgb_params,
                                                dtrain_phase16,
                                                num_boost_round=n_rounds,
                                                xgb_model=booster,
                                                verbose_eval=False
                                            )
                                            try:
                                                phase15_model._Booster = adapted_booster_phase16  # type: ignore[attr-defined]
                                            except Exception as e:
                                                logger.debug(f"Failed to set phase15_model._Booster: {e}")
                                            if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                                xgb_entry['model'] = phase15_model
                                                horizon_models['xgboost'] = xgb_entry
                                            else:
                                                horizon_models['xgboost'] = phase15_model
                                            self.models[horizon_key] = horizon_models
                                            phase16_adapted_count += 1
                                            logger.info(f"   ✅ {h}d Phase 1.6: Model {train_days} günlük veri ile yeniden eğitildi (+{n_rounds} rounds)")
                                        elif hasattr(phase15_model, 'save_model') or hasattr(phase15_model, 'get_score'):
                                            booster = phase15_model
                                            n_rounds = 50
                                            xgb_params = {
                                                'objective': 'reg:squarederror',
                                                'verbosity': 0
                                            }
                                            adapted_booster_phase16 = xgb.train(
                                                xgb_params,
                                                dtrain_phase16,
                                                num_boost_round=n_rounds,
                                                xgb_model=booster,
                                                verbose_eval=False
                                            )
                                            if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                                xgb_entry['model'] = adapted_booster_phase16
                                                horizon_models['xgboost'] = xgb_entry
                                            else:
                                                horizon_models['xgboost'] = adapted_booster_phase16
                                            self.models[horizon_key] = horizon_models
                                            phase16_adapted_count += 1
                                            logger.info(f"   ✅ {h}d Phase 1.6: Model {train_days} günlük veri ile yeniden eğitildi (+{n_rounds} rounds)")
                                    except Exception as e:
                                        logger.warning(f"   ⚠️ {h}d Phase 1.6 incremental learning error: {e}")
                                except Exception as e:
                                    logger.warning(f"   ⚠️ {h}d Phase 1.6 error: {e}")
                            
                            if phase16_adapted_count > 0:
                                logger.info(f"✅ Phase 1.6 tamamlandı: {phase16_adapted_count}/{len(self.prediction_horizons)} model {train_days} günlük veri ile yeniden eğitildi")
                            else:
                                logger.warning("⚠️ Phase 1.6: Hiçbir model yeniden eğitilmedi")
                        except Exception as e:
                            logger.error(f"❌ Phase 1.6 failed: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        
                        # ✅ PHASE 2: 100 günlük veriyi de eğitim setine ekle (300+100=400 gün) ve 400 gün ile incremental learning yap
                        logger.info(f"🔄 Phase 2: {test_days} günlük veriyi eğitim setine ekle (300+100=400 gün) ve tam eğitim yap")
                        logger.info("   📊 Tüm veri setiyle (400 gün) son ince ayar")
                        
                        try:
                            # Combine train_data + test_data (400 gün)
                            full_data = pd.concat([train_data, test_data], axis=0)
                            
                            # Feature engineering on full data
                            df_full_features = self.create_advanced_features(full_data, symbol=symbol)
                            
                            # Clean full data
                            for col in df_full_features.columns:
                                if df_full_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                                    df_full_features[col] = df_full_features[col].ffill()
                                    if df_full_features[col].isna().any():
                                        col_mean = df_full_features[col].mean()
                                        if pd.notna(col_mean):
                                            df_full_features[col] = df_full_features[col].fillna(col_mean)
                                        else:
                                            df_full_features[col] = df_full_features[col].fillna(0)
                            
                            try:
                                from bist_pattern.features.cleaning import clean_dataframe as _clean_df
                                df_full_features = _clean_df(df_full_features)
                            except Exception as e:
                                logger.debug(f"External clean_dataframe failed for full, using internal: {e}")
                                df_full_features = self._clean_data(df_full_features)
                            
                            # Incremental learning on full data for each horizon
                            phase2_adapted_count = 0
                            for h in self.prediction_horizons:
                                try:
                                    target_col = f'target_{h}d'
                                    if 'close' not in df_full_features.columns:
                                        continue
                                    
                                    df_full_features[target_col] = (
                                        df_full_features['close'].shift(-h) / df_full_features['close'] - 1.0
                                    )
                                    
                                    df_full_clean = df_full_features.dropna(subset=[target_col]).copy()
                                    if len(df_full_clean) < 10:
                                        continue
                                    
                                    if f'{h}d' not in self.feature_columns:
                                        continue
                                    
                                    feature_cols = self.feature_columns[f'{h}d']
                                    missing_cols = [c for c in feature_cols if c not in df_full_clean.columns]
                                    if missing_cols:
                                        continue
                                    
                                    X_full = df_full_clean[feature_cols].values
                                    y_full = df_full_clean[target_col].values
                                    
                                    horizon_key = f"{symbol}_{h}d"
                                    if horizon_key not in self.models:
                                        continue
                                    
                                    horizon_models = self.models[horizon_key]
                                    if not isinstance(horizon_models, dict) or 'xgboost' not in horizon_models:
                                        continue
                                    
                                    xgb_entry = horizon_models['xgboost']
                                    if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                        phase16_model = xgb_entry['model']
                                    else:
                                        phase16_model = xgb_entry
                                    
                                    # Incremental learning on full data
                                    try:
                                        dtrain_full = xgb.DMatrix(X_full, label=y_full, feature_names=feature_cols)
                                        
                                        if hasattr(phase16_model, 'get_booster'):
                                            booster = phase16_model.get_booster()
                                            params = phase16_model.get_params()
                                            n_rounds = max(10, int(params.get('n_estimators', 100) * 0.1))
                                            xgb_params = {
                                                'max_depth': params.get('max_depth', 6),
                                                'learning_rate': params.get('learning_rate', 0.1) * 0.5,
                                                'subsample': params.get('subsample', 0.8),
                                                'colsample_bytree': params.get('colsample_bytree', 0.8),
                                                'objective': 'reg:squarederror',
                                                'verbosity': 0
                                            }
                                            adapted_booster_full = xgb.train(
                                                xgb_params,
                                                dtrain_full,
                                                num_boost_round=n_rounds,
                                                xgb_model=booster,
                                                verbose_eval=False
                                            )
                                            try:
                                                phase16_model._Booster = adapted_booster_full  # type: ignore[attr-defined]
                                            except Exception as e:
                                                logger.debug(f"Failed to set phase16_model._Booster: {e}")
                                            if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                                xgb_entry['model'] = phase16_model
                                                horizon_models['xgboost'] = xgb_entry
                                            else:
                                                horizon_models['xgboost'] = phase16_model
                                            self.models[horizon_key] = horizon_models
                                            phase2_adapted_count += 1
                                            logger.info(f"   ✅ {h}d Phase 2: Model {len(full_data)} günlük veri ile tam eğitim yapıldı (+{n_rounds} rounds)")
                                        elif hasattr(phase16_model, 'save_model') or hasattr(phase16_model, 'get_score'):
                                            booster = phase16_model
                                            n_rounds = 50
                                            xgb_params = {
                                                'objective': 'reg:squarederror',
                                                'verbosity': 0
                                            }
                                            adapted_booster_full = xgb.train(
                                                xgb_params,
                                                dtrain_full,
                                                num_boost_round=n_rounds,
                                                xgb_model=booster,
                                                verbose_eval=False
                                            )
                                            if isinstance(xgb_entry, dict) and 'model' in xgb_entry:
                                                xgb_entry['model'] = adapted_booster_full
                                                horizon_models['xgboost'] = xgb_entry
                                            else:
                                                horizon_models['xgboost'] = adapted_booster_full
                                            self.models[horizon_key] = horizon_models
                                            phase2_adapted_count += 1
                                            logger.info(f"   ✅ {h}d Phase 2: Model {len(full_data)} günlük veri ile tam eğitim yapıldı (+{n_rounds} rounds)")
                                    except Exception as e:
                                        logger.warning(f"   ⚠️ {h}d Phase 2 incremental learning error: {e}")
                                except Exception as e:
                                    logger.warning(f"   ⚠️ {h}d Phase 2 error: {e}")
                            
                            if phase2_adapted_count > 0:
                                logger.info(f"✅ Phase 2 tamamlandı: {phase2_adapted_count}/{len(self.prediction_horizons)} model {len(full_data)} günlük veri ile tam eğitim yapıldı")
                            else:
                                logger.warning("⚠️ Phase 2: Hiçbir model tam eğitim yapılmadı")
                        except Exception as e:
                            logger.error(f"❌ Phase 2 failed: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        
                        # ⚡ CRITICAL: Save Phase 2 models to disk!
                        self.save_enhanced_models(symbol)
                        logger.info(f"💾 {symbol} Phase 2 modeli kaydedildi")
                    else:
                        logger.warning("⚠️ Adaptive learning: Hiçbir model güncellenmedi")
                    
                except Exception as e:
                    logger.error(f"❌ Adaptive learning Phase 2 failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            logger.info(f"✅ {symbol} enhanced model eğitimi tamamlandı")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced model eğitim hatası: {e}")
            return False
    
    def predict_enhanced(self, symbol, current_data, sentiment_score=None):
        """
        Enhanced predictions with optional sentiment adjustment
        
        ✅ AŞAMA 6: Model Prediction Hatası Kontrolü
        - Model yükleme kontrolü
        - Feature engineering kontrolü
        - Feature mismatch kontrolü
        - Prediction hataları detaylı loglanır
        
        Returns:
            dict: Predictions dict veya None (hata durumunda)
        """
        try:
            # ✅ DEBUG: Log prediction call
            logger.debug(f"🔮 PREDICT_ENHANCED: {symbol}, data shape: {current_data.shape if current_data is not None else 'None'}")
            if current_data is not None and len(current_data) > 0:
                logger.debug(f"   Data period: {current_data.index.min()} - {current_data.index.max()}")
            
            # ✅ FIX: Check if models are already in memory first (from training)
            # Only load from disk if models are not in memory
            model_key = f"{symbol}_{self.prediction_horizons[0]}d" if self.prediction_horizons else None
            
            # ⚡ CRITICAL FIX: Check if ANY horizon model exists for this symbol (more flexible)
            # This handles HPO case where only one horizon is trained
            models_in_memory = any(f"{symbol}_{h}d" in self.models for h in self.prediction_horizons)
            
            # ⚡ ADDITIONAL CHECK: Also check if specific horizon models exist (for HPO)
            # HPO trains only one horizon, so check if that specific model exists
            if not models_in_memory and self.prediction_horizons:
                # Check all possible horizon formats
                for h in self.prediction_horizons:
                    if f"{symbol}_{h}d" in self.models:
                        models_in_memory = True
                        break
            
            logger.debug(f"   Models in memory: {models_in_memory}, prediction_horizons: {self.prediction_horizons}, models keys: {list(self.models.keys())[:5]}")
            
            # Auto-load models for this symbol if not already loaded
            if not models_in_memory and (not self.feature_columns or len(self.models) == 0):
                logger.info(f"🔄 {symbol}: Auto-loading trained models...")
                if self.has_trained_models(symbol):
                    loaded = self.load_trained_models(symbol)
                    if not loaded:
                        logger.warning(f"⚠️ {symbol}: Failed to load trained models")
                        return None
                    logger.info(f"✅ {symbol}: Models loaded successfully ({len(self.feature_columns)} features)")
                else:
                    logger.warning(f"⚠️ {symbol}: No trained models found (has_trained_models=False, models_in_memory={models_in_memory}, models_count={len(self.models)})")
                    return None
            
            # Feature engineering
            df_features = self.create_advanced_features(current_data, symbol=symbol)
            
            # ⚡ CONSISTENT WITH TRAINING: Use same NaN handling strategy
            # Forward fill + mean fill (not dropna!)
            for col in df_features.columns:
                if df_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Step 1: Forward fill
                    # ✅ FIX: Use ffill() instead of fillna(method='ffill') (deprecated in pandas 2.0+)
                    df_features[col] = df_features[col].ffill()
                    
                    # Step 2: For remaining NaN, use column mean
                    if df_features[col].isna().any():
                        col_mean = df_features[col].mean()
                        if pd.notna(col_mean):
                            df_features[col] = df_features[col].fillna(col_mean)
                        else:
                            df_features[col] = df_features[col].fillna(0)
            
            # Clean data
            df_features = self._clean_data(df_features)
            
            if len(df_features) == 0:
                return None
            
            # Get latest features
            if not self.feature_columns:
                logger.error("Feature columns not set. Model training required.")
                return None
                
            # Check if all feature columns exist
            # Handle both dict and list formats
            if isinstance(self.feature_columns, dict):
                # Dict format: collect all features from all horizons
                all_feature_cols = set()
                for h_features in self.feature_columns.values():
                    if isinstance(h_features, list):
                        all_feature_cols.update(h_features)
                feature_list = list(all_feature_cols)
            else:
                # List format (old)
                feature_list = self.feature_columns if self.feature_columns else []
            
            # ✅ AŞAMA 5: Feature Mismatch Kontrolü
            missing_cols = [col for col in feature_list if col not in df_features.columns]
            confidence_scale = 1.0
            guard_info = None
            if missing_cols:
                # ✅ FIX: Feature mismatch detaylı log
                logger.warning(f"⚠️ AŞAMA 5: {symbol}: Feature mismatch detected - {len(missing_cols)} missing features")
                logger.debug(f"   Missing features: {missing_cols[:10]}...")  # İlk 10 feature
                logger.debug(f"   Expected features: {len(feature_list)}, Available features: {len(df_features.columns)}")
                
                # Attempt guard if enabled and only allowed prefixes are missing
                if self.enable_pred_feature_guard:
                    allowed_missing = []
                    disallowed_missing = []
                    for col in missing_cols:
                        if any(col.startswith(pref) for pref in self.guard_allowed_prefixes):
                            allowed_missing.append(col)
                        else:
                            disallowed_missing.append(col)
                    
                    # Calculate total feature count (handle both dict and list formats)
                    if isinstance(self.feature_columns, dict):
                        # Dict format: {horizon: [features]}
                        all_features = set()
                        for h_features in self.feature_columns.values():
                            if isinstance(h_features, list):
                                all_features.update(h_features)
                        total_features = len(all_features) if all_features else 1
                    else:
                        # List format (old)
                        total_features = len(self.feature_columns) if self.feature_columns else 1
                    
                    missing_ratio = float(len(missing_cols)) / float(max(1, total_features))
                    if disallowed_missing or missing_ratio > self.guard_max_missing_ratio:
                        logger.error(
                            f"❌ AŞAMA 5: {symbol}: Missing disallowed features or too many missing (ratio={missing_ratio:.3f})"
                        )
                        logger.error(f"   Disallowed missing: {disallowed_missing[:5]}...")
                        logger.error(f"   Allowed missing: {allowed_missing[:5]}...")
                        return None
                    # Create allowed missing columns with neutral fills (0.0)
                    for col in allowed_missing:
                        try:
                            df_features[col] = 0.0
                        except Exception as e:
                            logger.debug(f"Failed to set missing feature {col} to 0.0: {e}")
                    # Confidence penalty proportional to missing ratio (capped)
                    penalty = min(self.guard_penalty_max, missing_ratio * 2.0)
                    confidence_scale = max(0.5, 1.0 - penalty)
                    guard_info = {
                        'missing_total': len(missing_cols),
                        'missing_allowed': len(allowed_missing),
                        'missing_ratio': missing_ratio,
                        'confidence_scale': confidence_scale,
                    }
                    logger.warning(
                        f"Prediction feature guard applied: missing={len(missing_cols)} ratio={missing_ratio:.3f} scale={confidence_scale:.2f}"
                    )
                else:
                    logger.error(f"❌ AŞAMA 5: {symbol}: Missing feature columns: {missing_cols[:10]}...")
                    logger.error("   Feature guard disabled - prediction cannot proceed")
                    return None
            
            predictions = {}
            
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.models:
                    horizon_models = self.models[model_key]
                    
                    # ⚡ CRITICAL: Use horizon-specific features (after reduction)!
                    horizon_feature_key = f"{symbol}_{horizon}d_features"
                    if horizon_feature_key in self.models:
                        horizon_feature_cols = self.models[horizon_feature_key]
                        logger.debug(f"Using {len(horizon_feature_cols)} horizon-specific features for {horizon}d prediction")
                    else:
                        # ⚡ CRITICAL FIX: Try to load horizon features from disk if not in memory
                        # This can happen when training is done per-horizon but prediction needs all horizons
                        horizon_feature_cols = None
                        try:
                            horizon_cols_file = f"{self.model_directory}/{symbol}_horizon_features.json"
                            if os.path.exists(horizon_cols_file):
                                with open(horizon_cols_file, 'r') as rf:
                                    horizon_features = json.load(rf) or {}
                                
                                h_key = f"{horizon}d"
                                if h_key in horizon_features:
                                    horizon_feature_cols = horizon_features[h_key]
                                    # Store in memory for future use
                                    self.models[horizon_feature_key] = horizon_feature_cols
                                    logger.debug(f"✅ {symbol} {horizon}d: Horizon features loaded from disk ({len(horizon_feature_cols)} features)")
                                else:
                                    logger.warning(f"⚠️ {symbol}: Horizon {h_key} not found in horizon_features.json (available: {list(horizon_features.keys())})")
                                    # Fallback: use global features
                                    horizon_feature_cols = feature_list or list(df_features.columns)
                            else:
                                logger.warning(f"⚠️ {symbol}: horizon_features.json not found at {horizon_cols_file}, using global features")
                                # Fallback: use global features
                                horizon_feature_cols = feature_list or list(df_features.columns)
                        except Exception as load_err:
                            # ✅ FIX: Improved error handling with detailed logging
                            logger.error(f"❌ {symbol}: Failed to load horizon features for {horizon}d: {load_err}")
                            # Strict validation (default ON) to prevent 40 vs 107 mismatch
                            strict_hf = str(ConfigManager.get('STRICT_HORIZON_FEATURES', '1')).lower() in ('1', 'true', 'yes', 'on')
                            if strict_hf:
                                # ✅ FIX: Skip this horizon instead of returning None (allow other horizons)
                                logger.warning(f"⚠️ {symbol}: Horizon-specific features not found for {horizon}d (load error: {load_err}) - skipping this horizon")
                                continue  # Skip this horizon, try next one
                            # Fallback to global features (backward compatibility)
                            horizon_feature_cols = feature_list or list(df_features.columns)
                        
                        # ✅ FIX: Ensure horizon_feature_cols is set (fallback to global features)
                        if horizon_feature_cols is None:
                            # IMPORTANT: self.feature_columns may be a dict { '1d': [...], '3d': [...], ... }
                            # Use the union list we already computed as feature_list; never pass a dict as indexer
                            if isinstance(self.feature_columns, dict):
                                horizon_feature_cols = feature_list or list(df_features.columns)
                            else:
                                horizon_feature_cols = list(self.feature_columns or [])
                            logger.warning(f"⚠️ {symbol} {horizon}d: Horizon-specific features not found, using global {len(horizon_feature_cols)} features")

                        # Try to infer expected feature count from an existing model and align
                        try:
                            expected_n = None
                            expected_names = None
                            # Probe candidate underlying models to infer num features
                            for _mn, _mi in horizon_models.items():
                                # Prefer first seed if available
                                if isinstance(_mi, dict) and 'models' in _mi and _mi['models']:
                                    _cands = list(_mi['models']) + [
                                        _mi.get('model')
                                    ]
                                else:
                                    _cands = [_mi.get('model') if isinstance(_mi, dict) else None]
                                for _m in [c for c in _cands if c is not None]:
                                    try:
                                        # XGBoost Booster direct
                                        if hasattr(_m, 'num_features') and callable(getattr(_m, 'num_features')):
                                            expected_n = int(_m.num_features())
                                            # Try names as well if exposed
                                            try:
                                                _names = getattr(_m, 'feature_names', None)
                                                if isinstance(_names, (list, tuple)) and len(_names) == expected_n:
                                                    expected_names = list(_names)
                                            except Exception as e:
                                                logger.debug(f"Failed to get feature names from XGBoost Booster: {e}")
                                            break
                                    except Exception as e:
                                        logger.debug(f"Failed to get num_features from XGBoost Booster: {e}")
                                    try:
                                        # XGBRegressor style
                                        if hasattr(_m, 'get_booster'):
                                            _b = _m.get_booster()
                                            if hasattr(_b, 'num_features') and callable(getattr(_b, 'num_features')):
                                                expected_n = int(_b.num_features())
                                                try:
                                                    _names = getattr(_b, 'feature_names', None)
                                                    if isinstance(_names, (list, tuple)) and len(_names) == expected_n:
                                                        expected_names = list(_names)
                                                except Exception as e:
                                                    logger.debug(f"Failed to get feature names from XGBRegressor: {e}")
                                                break
                                    except Exception as e:
                                        logger.debug(f"Failed to get num_features from XGBRegressor: {e}")
                                    try:
                                        # LightGBM Booster
                                        if hasattr(_m, 'booster_') and hasattr(_m.booster_, 'num_feature'):
                                            expected_n = int(_m.booster_.num_feature())
                                            try:
                                                _names = _m.booster_.feature_name()
                                                if isinstance(_names, (list, tuple)) and len(_names) == expected_n:
                                                    expected_names = list(_names)
                                            except Exception as e:
                                                logger.debug(f"Failed to get feature names from LightGBM: {e}")
                                            break
                                        if hasattr(_m, 'num_feature') and callable(getattr(_m, 'num_feature')):
                                            expected_n = int(_m.num_feature())
                                            break
                                    except Exception as e:
                                        logger.debug(f"Failed to get num_feature from LightGBM: {e}")
                                    try:
                                        # Generic sklearn models
                                        if hasattr(_m, 'n_features_in_'):
                                            expected_n = int(getattr(_m, 'n_features_in_', 0) or 0)
                                            try:
                                                _names = getattr(_m, 'feature_names_in_', None)
                                                if isinstance(_names, (list, tuple, np.ndarray)) and len(_names) == expected_n:
                                                    expected_names = list(_names)
                                            except Exception as e:
                                                logger.debug(f"Failed to get feature_names_in_ from sklearn model: {e}")
                                            if expected_n > 0:
                                                break
                                    except Exception as e:
                                        logger.debug(f"Failed to get n_features_in_ from sklearn model: {e}")
                                if expected_n is not None:
                                    break

                            if expected_n is not None and expected_n > 0:
                                # If model provides names, align by names (safer than slicing)
                                if expected_names:
                                    # Ensure all expected columns exist
                                    for _c in expected_names:
                                        if _c not in df_features.columns:
                                            try:
                                                df_features[_c] = 0.0
                                            except Exception as e:
                                                logger.debug(f"Failed to set missing feature {_c} to 0.0: {e}")
                                    # Keep model's order exactly
                                    horizon_feature_cols = [c for c in expected_names if c in df_features.columns]
                                    logger.warning(
                                        f"Aligned fallback features by model names: {len(horizon_feature_cols)}"
                                    )
                                # Otherwise, truncate to model dimensionality
                                if len(horizon_feature_cols) > expected_n:
                                    horizon_feature_cols = list(horizon_feature_cols)[:expected_n]
                                    logger.warning(
                                        f"Aligned fallback features to model dims: {expected_n}"
                                    )
                        except Exception as e:
                            logger.debug(f"Feature alignment failed: {e}")
                    
                    # Extract features for this horizon
                    # Defensive: ensure horizon_feature_cols is a list-like of column names
                    if isinstance(horizon_feature_cols, dict):
                        # If a dict slipped through (e.g., horizon->features mapping), use the precomputed union
                        horizon_feature_cols = feature_list or list(df_features.columns)
                    elif not isinstance(horizon_feature_cols, (list, tuple, np.ndarray, pd.Index)):
                        # Any other unexpected type → fall back to all columns
                        try:
                            horizon_feature_cols = list(horizon_feature_cols)
                        except Exception as e:
                            logger.debug(f"Failed to convert horizon_feature_cols to list: {e}")
                            horizon_feature_cols = list(df_features.columns)
                    # Filter to existing columns only
                    horizon_feature_cols = [c for c in horizon_feature_cols if c in df_features.columns]
                    
                    # ✅ FIX: Validate horizon_feature_cols is not empty
                    if not horizon_feature_cols:
                        logger.error(f"All horizon features missing in df_features for {symbol} {horizon}d! Cannot predict.")
                        continue  # Skip this horizon
                    
                    # ✅ FIX: Validate latest_features has valid shape
                    try:
                        latest_features = df_features[horizon_feature_cols].iloc[-1:].values
                        if latest_features.shape[1] == 0:
                            logger.error(f"latest_features has 0 features for {symbol} {horizon}d! Cannot predict.")
                            continue  # Skip this horizon
                        if latest_features.shape[0] == 0:
                            logger.error(f"latest_features has 0 samples for {symbol} {horizon}d! Cannot predict.")
                            continue  # Skip this horizon
                    except (KeyError, IndexError) as e:
                        logger.error(f"Failed to extract features for {symbol} {horizon}d: {e}")
                        continue  # Skip this horizon
                    
                    model_predictions = {}
                    
                    # ✅ FIX: Validate horizon_models is not None or empty
                    if not horizon_models or not isinstance(horizon_models, dict):
                        logger.error(f"horizon_models is None or empty for {symbol} {horizon}d! Cannot predict.")
                        continue  # Skip this horizon
                    
                    for model_name, model_info in horizon_models.items():
                        try:
                            # ✅ Algorithm-level prediction gating (env-driven)
                            try:
                                if model_name == 'xgboost':
                                    allow = str(ConfigManager.get('ENABLE_XGBOOST', '1')).lower() in ('1', 'true', 'yes', 'on')
                                    if not allow:
                                        logger.debug("Skipping XGBoost prediction due to ENABLE_XGBOOST=0")
                                        continue
                                elif model_name == 'lightgbm':
                                    allow = str(ConfigManager.get('ENABLE_LIGHTGBM', '1')).lower() in ('1', 'true', 'yes', 'on')
                                    if not allow:
                                        logger.debug("Skipping LightGBM prediction due to ENABLE_LIGHTGBM=0")
                                        continue
                                elif model_name == 'catboost':
                                    allow = str(ConfigManager.get('ENABLE_CATBOOST', '1')).lower() in ('1', 'true', 'yes', 'on')
                                    if not allow:
                                        logger.debug("Skipping CatBoost prediction due to ENABLE_CATBOOST=0")
                                        continue
                            except Exception as e:
                                logger.debug(f"Model enable check failed: {e}")
                            # ✅ FIX: Validate model_info is dict and has 'model' key
                            if not isinstance(model_info, dict):
                                logger.error(f"model_info is not dict for {symbol} {horizon}d {model_name}! Skipping.")
                                continue
                            
                            if 'model' not in model_info:
                                logger.error(f"'model' key missing in model_info for {symbol} {horizon}d {model_name}! Skipping.")
                                continue
                            
                            if model_info['model'] is None:
                                logger.error(f"model is None for {symbol} {horizon}d {model_name}! Skipping.")
                                continue
                            
                            # ✅ FIX: Validate model has predict method
                            model = model_info['model']
                            if not hasattr(model, 'predict'):
                                logger.error(f"model has no 'predict' method for {symbol} {horizon}d {model_name}! Skipping.")
                                continue
                            
                            # ⚡ ENSEMBLE PREDICTION: If multiple seed models, average their predictions
                            if 'models' in model_info and len(model_info['models']) > 1:
                                # Predict with all seed models and average
                                seed_preds = []
                                for seed_model in model_info['models']:
                                    try:
                                        # Check if native XGBoost Booster or sklearn model
                                        if hasattr(seed_model, 'predict') and hasattr(seed_model, 'get_score'):
                                            # Native XGBoost Booster - needs DMatrix
                                            # Ensure feature names align with training (use horizon feature names)
                                            dtest = xgb.DMatrix(latest_features, feature_names=horizon_feature_cols)  # type: ignore[attr-defined]
                                            pred_raw = seed_model.predict(dtest)[0]
                                        else:
                                            # Sklearn-style model
                                            pred_raw = seed_model.predict(latest_features)[0]
                                        
                                        # ⚡ CRITICAL FIX: Validate seed prediction before adding
                                        if math.isnan(pred_raw) or math.isinf(pred_raw):
                                            logger.debug(f"Seed model prediction is {pred_raw} (NaN/Inf) for {symbol} {horizon}d {model_name}, skipping")
                                            continue  # Skip this seed model
                                        
                                        if abs(pred_raw) > 10.0:
                                            logger.warning(f"⚠️ Seed model prediction is {pred_raw:.2e} (extremely large) for {symbol} {horizon}d {model_name}, skipping")
                                            continue  # Skip this seed model
                                        
                                        seed_preds.append(float(pred_raw))
                                    except Exception as e:
                                        logger.debug(f"Seed model prediction failed: {e}")
                                
                                if seed_preds:
                                    # ⚡ CRITICAL FIX: Validate seed predictions before averaging
                                    # Remove any remaining NaN/Inf values (double-check)
                                    valid_seed_preds = [p for p in seed_preds if not (math.isnan(p) or math.isinf(p))]
                                    if not valid_seed_preds:
                                        logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: All seed predictions are NaN/Inf - skipping")
                                        continue  # Skip this model
                                    
                                    # Check for extremely large values
                                    extreme_count = sum(1 for p in valid_seed_preds if abs(p) > 10.0)
                                    if extreme_count > 0:
                                        logger.error(
                                            f"❌ {symbol} {horizon}d {model_name}: {extreme_count}/{len(valid_seed_preds)} "
                                            "seed predictions are extremely large (>10.0) - this indicates a model/feature error"
                                        )
                                        # Remove extreme values from average
                                        valid_seed_preds = [p for p in valid_seed_preds if abs(p) <= 10.0]
                                        if not valid_seed_preds:
                                            logger.error(f"❌ {symbol} {horizon}d {model_name}: All seed predictions are extreme - SKIPPING")
                                            continue  # Skip this model
                                    
                                    pred_ret = float(np.mean(valid_seed_preds))
                                    logger.debug(f"{model_name} ensemble: {len(valid_seed_preds)} models, preds={valid_seed_preds}, mean={pred_ret:.4f}")
                                else:
                                    # Fallback to primary model
                                    # ✅ FIX: Validate primary model exists (already validated above, but double-check)
                                    if model is None:
                                        logger.error(f"Fallback model is None for {symbol} {horizon}d {model_name}! Cannot predict.")
                                        continue  # Skip this model
                                    
                                    try:
                                        if hasattr(model, 'predict') and hasattr(model, 'get_score'):
                                            # Native XGBoost Booster
                                            dtest = xgb.DMatrix(latest_features, feature_names=horizon_feature_cols)  # type: ignore[attr-defined]
                                            pred_ret = float(model.predict(dtest)[0])
                                        else:
                                            # Sklearn-style model
                                            pred_ret = float(model.predict(latest_features)[0])
                                    except Exception as fallback_err:
                                        logger.error(f"Fallback prediction also failed for {symbol} {horizon}d {model_name}: {fallback_err}")
                                        continue  # Skip this model
                            else:
                                # Single model prediction
                                # ✅ FIX: Use already-validated model variable (model is already validated above)
                                try:
                                    if hasattr(model, 'predict') and hasattr(model, 'get_score'):
                                        # Native XGBoost Booster
                                        # ⚡ CRITICAL FIX: Get model's expected feature names and align prediction features
                                        try:
                                            # Get model's expected feature names (from training)
                                            if hasattr(model, 'feature_names'):
                                                expected_feature_names = model.feature_names  # type: ignore[attr-defined]
                                            elif hasattr(model, 'get_booster'):
                                                booster = model.get_booster()  # type: ignore[attr-defined]
                                                if hasattr(booster, 'feature_names'):
                                                    expected_feature_names = booster.feature_names  # type: ignore[attr-defined]
                                                else:
                                                    expected_feature_names = None
                                            else:
                                                expected_feature_names = None
                                            
                                            # If we have expected feature names, align prediction features to match
                                            if expected_feature_names:
                                                # latest_features is a numpy array (from iloc[-1:].values)
                                                # We need to create a DataFrame from df_features to align features properly
                                                # Get the last row of df_features as DataFrame
                                                latest_features_df = df_features[horizon_feature_cols].iloc[-1:]
                                                
                                                # Create aligned feature DataFrame
                                                aligned_features = pd.DataFrame(index=latest_features_df.index)
                                                for feat_name in expected_feature_names:
                                                    if feat_name in latest_features_df.columns:
                                                        aligned_features[feat_name] = latest_features_df[feat_name]
                                                    elif feat_name in df_features.columns:
                                                        aligned_features[feat_name] = df_features[feat_name].iloc[-1]
                                                    else:
                                                        # Missing feature: fill with 0 (or use mean/median if available)
                                                        aligned_features[feat_name] = 0.0
                                                        logger.debug(f"Missing feature {feat_name} for {symbol} {horizon}d {model_name}, filling with 0.0")
                                                
                                                # Ensure exact order matches expected_feature_names
                                                aligned_features = aligned_features[expected_feature_names]
                                                aligned_feature_array = aligned_features.values
                                                aligned_feature_names = list(aligned_features.columns)
                                            else:
                                                # Fallback: use original feature names
                                                aligned_feature_array = latest_features
                                                aligned_feature_names = horizon_feature_cols
                                            
                                            # ⚡ CRITICAL FIX: Validate feature array before creating DMatrix
                                            # Check for NaN, Inf, or extremely large values
                                            # ⚡ ROOT CAUSE ANALYSIS: Instead of skipping, find root cause and fix
                                            if np.any(np.isnan(aligned_feature_array)) or np.any(np.isinf(aligned_feature_array)):
                                                nan_count = np.sum(np.isnan(aligned_feature_array))
                                                inf_count = np.sum(np.isinf(aligned_feature_array))
                                                # ⚡ ROOT CAUSE: NaN/Inf in aligned features - likely from feature engineering
                                                # Find which features have NaN/Inf
                                                nan_features = [
                                                    aligned_feature_names[i] for i in range(len(aligned_feature_names))
                                                    if (np.isnan(aligned_feature_array[0, i]) or np.isinf(aligned_feature_array[0, i]))
                                                ]
                                                logger.error(
                                                    f"❌ {symbol} {horizon}d {model_name}: Feature array contains "
                                                    f"{nan_count} NaN and {inf_count} Inf values - ROOT CAUSE: Feature engineering issue"
                                                )
                                                logger.error(f"   Problematic features: {nan_features[:10]}")  # Log first 10
                                                # ⚡ FIX: Try to fix by forward-filling and mean-filling
                                                fixed_array = aligned_feature_array.copy()
                                                for i, feat_name in enumerate(aligned_feature_names):
                                                    if np.isnan(fixed_array[0, i]) or np.isinf(fixed_array[0, i]):
                                                        # Try to get from df_features
                                                        if feat_name in df_features.columns:
                                                            fixed_value = df_features[feat_name].iloc[-1]
                                                            if not (np.isnan(fixed_value) or np.isinf(fixed_value)):
                                                                fixed_array[0, i] = fixed_value
                                                            else:
                                                                # Use mean or 0
                                                                col_mean = df_features[feat_name].mean()
                                                                fixed_array[0, i] = col_mean if not (np.isnan(col_mean) or np.isinf(col_mean)) else 0.0
                                                        else:
                                                            fixed_array[0, i] = 0.0
                                                aligned_feature_array = fixed_array
                                                logger.warning("   ✅ Attempted fix: Forward-filled and mean-filled NaN/Inf features")
                                                # Re-check after fix
                                                if np.any(np.isnan(aligned_feature_array)) or np.any(np.isinf(aligned_feature_array)):
                                                    logger.error("   ❌ Fix failed: Still have NaN/Inf - SKIPPING")
                                                    continue  # Skip only if fix failed
                                            
                                            # Check for extremely large values (>1e10) that could cause overflow
                                            large_count = np.sum(np.abs(aligned_feature_array) > 1e10)
                                            if large_count > 0:
                                                logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: Feature array contains {large_count} values > 1e10 - clamping to prevent overflow")
                                                aligned_feature_array = np.clip(aligned_feature_array, -1e10, 1e10)
                                            
                                            # Create DMatrix with validated features
                                            dtest = xgb.DMatrix(aligned_feature_array, feature_names=aligned_feature_names)  # type: ignore[attr-defined]
                                            
                                            # ⚡ CRITICAL FIX: Validate prediction result immediately
                                            pred_ret_raw = model.predict(dtest)[0]
                                            if math.isnan(pred_ret_raw) or math.isinf(pred_ret_raw):
                                                # ⚡ ROOT CAUSE: Model prediction is NaN/Inf - likely model overflow or feature issue
                                                logger.error(f"❌ {symbol} {horizon}d {model_name}: Model prediction is {pred_ret_raw} (NaN/Inf) - ROOT CAUSE: Model overflow or feature issue")
                                                logger.error(f"   Feature stats: min={aligned_feature_array.min():.2e}, max={aligned_feature_array.max():.2e}, mean={aligned_feature_array.mean():.2e}")
                                                logger.error(f"   Feature array shape: {aligned_feature_array.shape}, feature names count: {len(aligned_feature_names)}")
                                                continue  # Skip this model - cannot fix NaN/Inf prediction
                                            
                                            # Check for extremely large values (>1.0 for returns is already suspicious)
                                            if abs(pred_ret_raw) > 10.0:
                                                # ⚡ ROOT CAUSE: Extremely large prediction - likely model overflow or feature scaling issue
                                                logger.error(
                                                    f"❌ {symbol} {horizon}d {model_name}: Model prediction is {pred_ret_raw:.2e} "
                                                    "(extremely large) - ROOT CAUSE: Model overflow or feature scaling issue"
                                                )
                                                feat_min = aligned_feature_array.min()
                                                feat_max = aligned_feature_array.max()
                                                feat_mean = aligned_feature_array.mean()
                                                logger.error(
                                                    f"   Feature stats: min={feat_min:.2e}, max={feat_max:.2e}, mean={feat_mean:.2e}"
                                                )
                                                logger.error(
                                                    f"   Feature array shape: {aligned_feature_array.shape}, "
                                                    f"feature names count: {len(aligned_feature_names)}"
                                                )
                                                # ⚡ FIX: Clamp to reasonable range instead of skipping
                                                pred_ret_raw = np.clip(pred_ret_raw, -10.0, 10.0)
                                                logger.warning("   ✅ Attempted fix: Clamped prediction to [-10.0, 10.0]")
                                            
                                            pred_ret = float(pred_ret_raw)
                                        except Exception as align_err:
                                            # Fallback to original approach if alignment fails
                                            logger.warning(f"Feature alignment failed for {symbol} {horizon}d {model_name}: {align_err}, trying original approach")
                                            
                                            # ⚡ CRITICAL FIX: Validate features before fallback prediction
                                            # ⚡ ROOT CAUSE ANALYSIS: Instead of skipping, find root cause and fix
                                            if np.any(np.isnan(latest_features)) or np.any(np.isinf(latest_features)):
                                                nan_count = np.sum(np.isnan(latest_features))
                                                inf_count = np.sum(np.isinf(latest_features))
                                                logger.error(
                                                    f"❌ {symbol} {horizon}d {model_name}: Fallback feature array contains "
                                                    f"{nan_count} NaN and {inf_count} Inf values - ROOT CAUSE: Feature engineering issue"
                                                )
                                                # ⚡ FIX: Try to fix by forward-filling and mean-filling
                                                fixed_features = latest_features.copy()
                                                for i in range(len(fixed_features)):
                                                    if np.isnan(fixed_features[i]) or np.isinf(fixed_features[i]):
                                                        feat_name = horizon_feature_cols[i] if i < len(horizon_feature_cols) else f"feature_{i}"
                                                        if feat_name in df_features.columns:
                                                            fixed_value = df_features[feat_name].iloc[-1]
                                                            if not (np.isnan(fixed_value) or np.isinf(fixed_value)):
                                                                fixed_features[i] = fixed_value
                                                            else:
                                                                col_mean = df_features[feat_name].mean()
                                                                fixed_features[i] = col_mean if not (np.isnan(col_mean) or np.isinf(col_mean)) else 0.0
                                                        else:
                                                            fixed_features[i] = 0.0
                                                latest_features = fixed_features
                                                logger.warning("   ✅ Attempted fix: Forward-filled and mean-filled NaN/Inf features")
                                                # Re-check after fix
                                                if np.any(np.isnan(latest_features)) or np.any(np.isinf(latest_features)):
                                                    logger.error("   ❌ Fix failed: Still have NaN/Inf - SKIPPING")
                                                    continue  # Skip only if fix failed
                                            
                                            # Check for extremely large values
                                            large_count = np.sum(np.abs(latest_features) > 1e10)
                                            if large_count > 0:
                                                logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: Fallback feature array contains {large_count} values > 1e10 - clamping")
                                                latest_features = np.clip(latest_features, -1e10, 1e10)
                                            
                                            dtest = xgb.DMatrix(latest_features, feature_names=horizon_feature_cols)  # type: ignore[attr-defined]
                                            
                                            # ⚡ CRITICAL FIX: Validate fallback prediction result
                                            pred_ret_raw = model.predict(dtest)[0]
                                            if math.isnan(pred_ret_raw) or math.isinf(pred_ret_raw):
                                                # ⚡ ROOT CAUSE: Fallback prediction is NaN/Inf
                                                logger.error(f"❌ {symbol} {horizon}d {model_name}: Fallback prediction is {pred_ret_raw} (NaN/Inf) - ROOT CAUSE: Model overflow or feature issue")
                                                logger.error(f"   Feature stats: min={latest_features.min():.2e}, max={latest_features.max():.2e}, mean={latest_features.mean():.2e}")
                                                continue  # Skip this model - cannot fix NaN/Inf prediction
                                            
                                            if abs(pred_ret_raw) > 10.0:
                                                # ⚡ ROOT CAUSE: Extremely large fallback prediction
                                                logger.error(
                                                    f"❌ {symbol} {horizon}d {model_name}: Fallback prediction is {pred_ret_raw:.2e} "
                                                    "(extremely large) - ROOT CAUSE: Model overflow or feature scaling issue"
                                                )
                                                feat_min = latest_features.min()
                                                feat_max = latest_features.max()
                                                feat_mean = latest_features.mean()
                                                logger.error(f"   Feature stats: min={feat_min:.2e}, max={feat_max:.2e}, mean={feat_mean:.2e}")
                                                # ⚡ FIX: Clamp to reasonable range instead of skipping
                                                pred_ret_raw = np.clip(pred_ret_raw, -10.0, 10.0)
                                                logger.warning("   ✅ Attempted fix: Clamped prediction to [-10.0, 10.0]")
                                            
                                            pred_ret = float(pred_ret_raw)
                                    else:
                                        # Sklearn-style model
                                        # ⚡ CRITICAL FIX: Validate features before prediction
                                        # ⚡ ROOT CAUSE ANALYSIS: Instead of skipping, find root cause and fix
                                        if np.any(np.isnan(latest_features)) or np.any(np.isinf(latest_features)):
                                            nan_count = np.sum(np.isnan(latest_features))
                                            inf_count = np.sum(np.isinf(latest_features))
                                            logger.error(
                                                f"❌ {symbol} {horizon}d {model_name}: Feature array contains "
                                                f"{nan_count} NaN and {inf_count} Inf values - ROOT CAUSE: Feature engineering issue"
                                            )
                                            # ⚡ FIX: Try to fix by forward-filling and mean-filling
                                            fixed_features = latest_features.copy()
                                            for i in range(len(fixed_features)):
                                                if np.isnan(fixed_features[i]) or np.isinf(fixed_features[i]):
                                                    feat_name = horizon_feature_cols[i] if i < len(horizon_feature_cols) else f"feature_{i}"
                                                    if feat_name in df_features.columns:
                                                        fixed_value = df_features[feat_name].iloc[-1]
                                                        if not (np.isnan(fixed_value) or np.isinf(fixed_value)):
                                                            fixed_features[i] = fixed_value
                                                        else:
                                                            col_mean = df_features[feat_name].mean()
                                                            fixed_features[i] = col_mean if not (np.isnan(col_mean) or np.isinf(col_mean)) else 0.0
                                                    else:
                                                        fixed_features[i] = 0.0
                                            latest_features = fixed_features
                                            logger.warning("   ✅ Attempted fix: Forward-filled and mean-filled NaN/Inf features")
                                            # Re-check after fix
                                            if np.any(np.isnan(latest_features)) or np.any(np.isinf(latest_features)):
                                                logger.error("   ❌ Fix failed: Still have NaN/Inf - SKIPPING")
                                                continue  # Skip only if fix failed
                                        
                                        # Check for extremely large values
                                        large_count = np.sum(np.abs(latest_features) > 1e10)
                                        if large_count > 0:
                                            logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: Feature array contains {large_count} values > 1e10 - clamping to prevent overflow")
                                            latest_features = np.clip(latest_features, -1e10, 1e10)
                                        
                                        # Predict and validate result
                                        pred_ret_raw = model.predict(latest_features)[0]
                                        if math.isnan(pred_ret_raw) or math.isinf(pred_ret_raw):
                                            # ⚡ ROOT CAUSE: Sklearn model prediction is NaN/Inf
                                            logger.error(f"❌ {symbol} {horizon}d {model_name}: Model prediction is {pred_ret_raw} (NaN/Inf) - ROOT CAUSE: Model overflow or feature issue")
                                            logger.error(f"   Feature stats: min={latest_features.min():.2e}, max={latest_features.max():.2e}, mean={latest_features.mean():.2e}")
                                            continue  # Skip this model - cannot fix NaN/Inf prediction
                                        
                                        # Check for extremely large values
                                        if abs(pred_ret_raw) > 10.0:
                                            # ⚡ ROOT CAUSE: Extremely large sklearn prediction
                                            logger.error(
                                                f"❌ {symbol} {horizon}d {model_name}: Model prediction is {pred_ret_raw:.2e} "
                                                "(extremely large) - ROOT CAUSE: Model overflow or feature scaling issue"
                                            )
                                            feat_min = latest_features.min()
                                            feat_max = latest_features.max()
                                            feat_mean = latest_features.mean()
                                            logger.error(f"   Feature stats: min={feat_min:.2e}, max={feat_max:.2e}, mean={feat_mean:.2e}")
                                            # ⚡ FIX: Clamp to reasonable range instead of skipping
                                            pred_ret_raw = np.clip(pred_ret_raw, -10.0, 10.0)
                                            logger.warning("   ✅ Attempted fix: Clamped prediction to [-10.0, 10.0]")
                                        
                                        pred_ret = float(pred_ret_raw)
                                except Exception as single_err:
                                    logger.error(f"Single model prediction failed for {symbol} {horizon}d {model_name}: {single_err}")
                                    continue  # Skip this model
                            
                            # ✅ FIX: Validate current_data before accessing
                            try:
                                if current_data is None or len(current_data) == 0:
                                    logger.error(f"current_data is empty for {symbol}! Cannot predict.")
                                    continue  # Skip this model
                                
                                if 'close' not in current_data.columns:
                                    logger.error(f"'close' column missing in current_data for {symbol}! Cannot predict.")
                                    continue  # Skip this model
                                
                                current_px = float(current_data['close'].iloc[-1])
                                if pd.isna(current_px) or current_px <= 0:
                                    logger.error(f"Invalid current price for {symbol}: {current_px}")
                                    continue  # Skip this model
                            except (IndexError, KeyError) as e:
                                logger.error(f"Failed to get current price for {symbol}: {e}")
                                continue  # Skip this model
                            
                            # ⚡ CRITICAL FIX: Validate and clamp pred_ret to reasonable range
                            # pred_ret is a return value (e.g., 0.01 = 1%), should be between -1.0 and +1.0 typically
                            # Very large values (1e20) indicate a calculation error, not HPO best parameters
                            if math.isinf(pred_ret) or math.isnan(pred_ret):
                                logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: pred_ret is inf/nan, skipping")
                                continue  # Skip this model
                            
                            # Clamp pred_ret to reasonable range (-1.0 to +1.0 for returns)
                            # More extreme values are possible but 1e20 is definitely an error
                            if abs(pred_ret) > 1.0:
                                # Clamp to reasonable max/min based on context
                                if pred_ret > 1.0:
                                    logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: pred_ret={pred_ret:.2e} is too large, clamping to 1.0")
                                    pred_ret = 1.0  # Max 100% return
                                else:
                                    logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: pred_ret={pred_ret:.2e} is too small, clamping to -0.5")
                                    pred_ret = -0.5  # Max -50% return (crash scenario)
                            
                            pred = current_px * (1.0 + pred_ret)
                            
                            # ⚡ CRITICAL FIX: Validate prediction value is reasonable
                            # Prediction should be positive and within reasonable bounds
                            if pred <= 0 or math.isinf(pred) or math.isnan(pred):
                                logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: prediction={pred} is invalid, skipping")
                                continue  # Skip this model
                            
                            # Clamp prediction to reasonable range (0.01 * current_px to 100 * current_px)
                            # This prevents extreme values while preserving reasonable predictions
                            if pred > current_px * 100:
                                logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: prediction={pred:.2e} is too large (>{current_px * 100:.2e}), clamping")
                                pred = current_px * 100  # Max 100x price (extreme but possible)
                            elif pred < current_px * 0.01:
                                logger.warning(f"⚠️ {symbol} {horizon}d {model_name}: prediction={pred:.2e} is too small (<{current_px * 0.01:.2e}), clamping")
                                pred = current_px * 0.01  # Min 1% of price (crash scenario)

                            model_predictions[model_name] = {
                                'prediction': float(pred),
                                'pred_ret': float(pred_ret),
                                'confidence': float(model_info['score']),
                                'rmse': float(model_info['rmse']),
                                'mape': float(model_info['mape']),
                                # ✅ FIX: Include raw_r2 for smart ensemble performance weighting
                                'raw_r2': float(model_info.get('raw_r2', 0.0))
                            }
                        except Exception as e:
                            logger.error(f"{model_name} prediction error: {e}")
                    
                    # ✅ FIX: Check if any predictions were successful
                    if not model_predictions:
                        logger.error(f"No successful predictions for {symbol} {horizon}d! All models failed.")
                        continue  # Skip this horizon
                    
                    # Ensemble prediction (weighted by performance OR meta-stacking)
                    if model_predictions:
                        weights = [info['confidence'] for info in model_predictions.values()]
                        predictions_list = [info['prediction'] for info in model_predictions.values()]
                        returns_list = [info.get('pred_ret', 0.0) for info in model_predictions.values()]
                        
                        # ⚡ NEW: Meta-Stacking with Ridge Learner (if enabled)
                        # ✅ FIX: Meta-stacking should only be used for short horizons (1d, 3d, 7d)
                        # Short horizons have more noise, meta-stacking helps. Long horizons (14d, 30d) are already smooth.
                        use_meta_for_horizon = self.enable_meta_stacking
                        if hasattr(self, 'use_meta_stacking_short_only') and self.use_meta_stacking_short_only:
                            # Short-only mode: only enable for short horizons
                            use_meta_for_horizon = horizon in (1, 3, 7)
                        
                        historical_r2_map = None
                        historical_r2_source = 'model'
                        ensemble_weights_map = None
                        
                        if use_meta_for_horizon and len(predictions_list) >= 2:
                            try:
                                # Meta-features: base predictions as features
                                meta_key = f"{symbol}_{horizon}d_meta"
                                scaler_key = f"{symbol}_{horizon}d_meta_scaler"
                                
                                # Check if meta-learner exists (trained during model training)
                                if meta_key in self.meta_learners and scaler_key in self.scalers:
                                    meta_model = self.meta_learners[meta_key]
                                    meta_scaler = self.scalers[scaler_key]
                                    
                                    # ✅ FIX: Get model order from training (ensures same order as training)
                                    model_order_key = f"{symbol}_{horizon}d_meta_model_order"
                                    training_model_order = None
                                    if hasattr(self, 'meta_model_orders') and model_order_key in self.meta_model_orders:
                                        training_model_order = self.meta_model_orders[model_order_key]
                                    
                                    # ⚡ CRITICAL FIX: Reorder predictions to match training order
                                    # Meta-learner was trained with specific model order (e.g., ['xgboost', 'lightgbm'])
                                    # Prediction must use the same order, even if more models are available
                                    expected_n_models = None
                                    if hasattr(meta_scaler, 'n_features_in_'):
                                        expected_n_models = meta_scaler.n_features_in_
                                    elif hasattr(meta_scaler, 'mean_') and meta_scaler.mean_ is not None:
                                        expected_n_models = meta_scaler.mean_.shape[0]
                                    elif hasattr(meta_scaler, 'scale_') and meta_scaler.scale_ is not None:
                                        expected_n_models = meta_scaler.scale_.shape[0]
                                    
                                    # Reorder predictions to match training order
                                    if training_model_order and len(training_model_order) > 0:
                                        # Create ordered lists matching training order
                                        ordered_returns = []
                                        ordered_predictions = []
                                        ordered_weights = []
                                        
                                        # Map model names to their predictions
                                        model_to_pred = {}
                                        model_to_ret = {}
                                        model_to_weight = {}
                                        for model_name, pred_info in model_predictions.items():
                                            model_to_pred[model_name] = pred_info['prediction']
                                            model_to_ret[model_name] = pred_info.get('pred_ret', 0.0)
                                            model_to_weight[model_name] = pred_info['confidence']
                                        
                                        # Build ordered lists based on training order
                                        for model_name in training_model_order:
                                            if model_name in model_to_pred:
                                                ordered_predictions.append(model_to_pred[model_name])
                                                ordered_returns.append(model_to_ret[model_name])
                                                ordered_weights.append(model_to_weight[model_name])
                                        
                                        # Check if we have the expected number of models
                                        if len(ordered_returns) == expected_n_models:
                                            # Use ordered predictions for meta-stacking
                                            meta_X = np.array(ordered_returns, dtype=float).reshape(1, -1)
                                            meta_X_scaled = meta_scaler.transform(meta_X)
                                            ensemble_ret = float(meta_model.predict(meta_X_scaled)[0])
                                            current_px = float(current_data['close'].iloc[-1])
                                            ensemble_pred = float(current_px * (1.0 + ensemble_ret))
                                            avg_confidence = (np.mean(ordered_weights) * 1.1) * confidence_scale
                                            logger.debug(f"Meta-stacking used for {symbol} {horizon}d (ordered: {training_model_order})")
                                        else:
                                            # Model count mismatch: fallback to weighted average
                                            logger.warning(
                                                f"Meta-stacking model count mismatch for {symbol} {horizon}d: "
                                                f"expected {expected_n_models} models ({training_model_order}), "
                                                f"got {len(ordered_returns)} available. Falling back to weighted average."
                                            )
                                            ensemble_pred = np.average(predictions_list, weights=weights) if sum(weights) > 0 else float(np.mean(predictions_list))
                                            avg_confidence = (np.mean(weights) if sum(weights) > 0 else 0.55) * confidence_scale
                                    elif expected_n_models is not None and len(returns_list) != expected_n_models:
                                        # Model count mismatch (no order info): fallback to weighted average
                                        logger.warning(
                                            f"Meta-stacking model count mismatch for {symbol} {horizon}d: "
                                            f"expected {expected_n_models} models, got {len(returns_list)}. "
                                            "Falling back to weighted average."
                                        )
                                        ensemble_pred = np.average(predictions_list, weights=weights) if sum(weights) > 0 else float(np.mean(predictions_list))
                                        avg_confidence = (np.mean(weights) if sum(weights) > 0 else 0.55) * confidence_scale
                                    else:
                                        # Stack predictions as features (returns domain) - original order
                                        meta_X = np.array(returns_list, dtype=float).reshape(1, -1)
                                        
                                        # ⚡ FIX: Scale features before prediction
                                        meta_X_scaled = meta_scaler.transform(meta_X)
                                        
                                        ensemble_ret = float(meta_model.predict(meta_X_scaled)[0])
                                        current_px = float(current_data['close'].iloc[-1])
                                        ensemble_pred = float(current_px * (1.0 + ensemble_ret))
                                        avg_confidence = (np.mean(weights) * 1.1) * confidence_scale  # Meta-stacking bonus + guard penalty
                                        logger.debug(f"Meta-stacking used for {symbol} {horizon}d (scaled)")
                                else:
                                    # Fallback to weighted average if meta-learner not trained yet
                                    ensemble_pred = np.average(predictions_list, weights=weights) if sum(weights) > 0 else float(np.mean(predictions_list))
                                    avg_confidence = (np.mean(weights) if sum(weights) > 0 else 0.55) * confidence_scale
                            except Exception as e:
                                logger.error(f"Meta-stacking error: {e}, falling back to weighted average")
                                ensemble_pred = np.average(predictions_list, weights=weights) if sum(weights) > 0 else float(np.mean(predictions_list))
                                avg_confidence = (np.mean(weights) if sum(weights) > 0 else 0.55) * confidence_scale
                        
                        else:
                            # ⚡ NEW: Smart Ensemble (Hybrid: Consensus + Performance)
                            # ✅ FIX: Use ConfigManager for consistent config access
                            use_smart_ensemble = str(ConfigManager.get('ML_USE_SMART_ENSEMBLE', '1')).lower() in ('1', 'true', 'yes')
                            
                            if use_smart_ensemble and len(predictions_list) >= 2:
                                try:
                                    # Import smart ensemble utility
                                    import sys
                                    sys.path.insert(0, '/opt/bist-pattern/scripts')
                                    from ensemble_utils import smart_ensemble
                                    
                                    # ✅ FIX: Historical R² for performance weighting
                                    # Use raw_r2 from metrics if available, fallback to confidence (converted from R²)
                                    historical_r2 = []
                                    for info in model_predictions.values():
                                        r2_val = info.get('raw_r2')
                                        if r2_val is not None and isinstance(r2_val, (int, float)):
                                            historical_r2.append(float(r2_val))
                                        else:
                                            conf = float(info.get('confidence', 0.5))
                                            approx_r2 = max(-0.5, min(0.8, (conf - 0.3) / 0.65 * 0.8))
                                            historical_r2.append(approx_r2)
                                    historical_r2 = np.array(historical_r2, dtype=float)
                                    
                                    try:
                                        ref_map = getattr(self, 'reference_historical_r2', {})
                                        key = f"{symbol}_{horizon}d"
                                        use_reference = getattr(self, 'use_reference_historical_r2', False)
                                        ref_values = ref_map.get(key) if isinstance(ref_map, dict) else None
                                        if use_reference and isinstance(ref_values, dict) and ref_values:
                                            overridden = []
                                            for idx, model_name in enumerate(model_predictions.keys()):
                                                ref_val = ref_values.get(model_name)
                                                if isinstance(ref_val, (int, float)):
                                                    overridden.append(float(ref_val))
                                                else:
                                                    overridden.append(float(historical_r2[idx]))
                                            historical_r2 = np.array(overridden, dtype=float)
                                            historical_r2_source = 'reference'
                                    except Exception as e:
                                        logger.debug(f"Failed to load reference historical R2, using model: {e}")
                                        historical_r2_source = 'model'
                                    
                                    historical_r2_map = {
                                        model_name: float(historical_r2[idx])
                                        for idx, model_name in enumerate(model_predictions.keys())
                                    }
                                    
                                    # Smart ensemble: consensus + performance (HPO-optimizable)
                                    # ✅ FIX: Use correct parameter names (sigma, consensus_weight, performance_weight)
                                    # ⚡ NEW: Read from environment if HPO provided, else use defaults
                                    consensus_weight = float(ConfigManager.get('ML_SMART_CONSENSUS_WEIGHT', '0.6'))
                                    performance_weight = float(ConfigManager.get('ML_SMART_PERFORMANCE_WEIGHT', '0.4'))
                                    smart_sigma = float(ConfigManager.get('ML_SMART_SIGMA', '0.005'))
                                    
                                    # Optional: fixed prior weights per model (from HPO)
                                    try:
                                        w_xgb = float(ConfigManager.get('ML_SMART_WEIGHT_XGB', '1.0'))
                                    except Exception as e:
                                        logger.debug(f"Failed to get ML_SMART_WEIGHT_XGB, using 1.0: {e}")
                                        w_xgb = 1.0
                                    try:
                                        w_lgb = float(ConfigManager.get('ML_SMART_WEIGHT_LGB', '1.0'))
                                    except Exception as e:
                                        logger.debug(f"Failed to get ML_SMART_WEIGHT_LGB, using 1.0: {e}")
                                        w_lgb = 1.0
                                    try:
                                        w_cat = float(ConfigManager.get('ML_SMART_WEIGHT_CAT', '1.0'))
                                    except Exception as e:
                                        logger.debug(f"Failed to get ML_SMART_WEIGHT_CAT, using 1.0: {e}")
                                        w_cat = 1.0
                                    # Map in the same order as model_predictions
                                    prior_map = {
                                        'xgboost': w_xgb,
                                        'lightgbm': w_lgb,
                                        'catboost': w_cat,
                                    }
                                    prior_weights = np.array([prior_map.get(k, 1.0) for k in model_predictions.keys()], dtype=float)
                                    
                                    ensemble_pred, final_weights = smart_ensemble(
                                        predictions=np.array(predictions_list),
                                        historical_r2=historical_r2,
                                        consensus_weight=consensus_weight,
                                        performance_weight=performance_weight,
                                        sigma=smart_sigma,
                                        prior_weights=prior_weights
                                    )
                                    
                                    # Confidence: weighted average with disagreement penalty
                                    avg_confidence = np.average(weights, weights=final_weights) * confidence_scale
                                    try:
                                        ensemble_weights_map = {
                                            model_name: float(final_weights[idx])
                                            for idx, model_name in enumerate(model_predictions.keys())
                                        }
                                    except Exception as e:
                                        logger.debug(f"Failed to create ensemble_weights_map: {e}")
                                        ensemble_weights_map = None
                                    
                                    # Disagreement penalty (if models diverge)
                                    pred_std = np.std(predictions_list)
                                    pred_mean = np.mean(predictions_list)
                                    disagreement_ratio = pred_std / max(abs(pred_mean), 1e-8)
                                    
                                    if disagreement_ratio > 0.05:
                                        disagreement_penalty = min(0.3, disagreement_ratio * 2)
                                        avg_confidence = max(0.25, avg_confidence * (1 - disagreement_penalty))
                                        logger.debug(f"{symbol} {horizon}d: Smart ensemble, disagreement {disagreement_ratio*100:.1f}%, confidence {avg_confidence:.2f}")
                                    
                                except Exception as e:
                                    logger.warning(f"Smart ensemble failed: {e}, falling back to weighted average")
                                    # Fallback to original weighted average
                                    if sum(weights) > 0:
                                        ensemble_pred = np.average(predictions_list, weights=weights)
                                        avg_confidence = np.mean(weights) * confidence_scale
                                    else:
                                        ensemble_pred = float(np.mean(predictions_list))
                                        avg_confidence = 0.55 * confidence_scale
                            else:
                                # Original: Performance-based weighting + disagreement penalty
                                if sum(weights) > 0:
                                    ensemble_pred = np.average(predictions_list, weights=weights)
                                    avg_confidence = np.mean(weights) * confidence_scale
                                    
                                    # ✨ NEW: Reduce confidence if models disagree significantly
                                    if len(predictions_list) > 1:
                                        pred_std = np.std(predictions_list)
                                        pred_mean = np.mean(predictions_list)
                                        disagreement_ratio = pred_std / max(abs(pred_mean), 1e-8)
                                        
                                        # Penalize confidence if disagreement > 5%
                                        if disagreement_ratio > 0.05:
                                            disagreement_penalty = min(0.3, disagreement_ratio * 2)
                                            avg_confidence = max(0.25, avg_confidence * (1 - disagreement_penalty))
                                            logger.debug(f"{symbol} {horizon}d: Model disagreement {disagreement_ratio*100:.1f}%, confidence adjusted to {avg_confidence:.2f}")
                                else:
                                    ensemble_pred = float(np.mean(predictions_list))
                                    # conservative default confidence
                                    avg_confidence = 0.55 * confidence_scale
                        predictions[f"{horizon}d"] = {
                            'ensemble_prediction': float(ensemble_pred),
                            'confidence': float(avg_confidence),
                            'models': model_predictions,
                            'current_price': float(current_data['close'].iloc[-1]),
                            'model_count': len(model_predictions),
                            'historical_r2_used': historical_r2_map,
                            'historical_r2_source': historical_r2_source,
                            'ensemble_weights': ensemble_weights_map,
                            **({'guard': guard_info} if guard_info else {}),
                        }
            
            # ⚡ NEW: Sentiment-based prediction adjustment (optional)
            if sentiment_score is not None and isinstance(sentiment_score, (int, float)):
                try:
                    sent = float(sentiment_score)
                    # Dynamic adjustment based on sentiment strength (like Basic ML)
                    if sent > 0.7:  # Strong bullish
                        adjustment_factor = 1.10  # +10%
                    elif sent < 0.3:  # Strong bearish
                        adjustment_factor = 0.90  # -10%
                    elif sent > 0.6:  # Moderate bullish
                        adjustment_factor = 1.05  # +5%
                    elif sent < 0.4:  # Moderate bearish
                        adjustment_factor = 0.95  # -5%
                    else:  # Neutral (0.4-0.6)
                        adjustment_factor = 1.0  # No adjustment
                    
                    if adjustment_factor != 1.0:
                        for h_key in predictions:
                            if 'ensemble_prediction' in predictions[h_key]:
                                original = predictions[h_key]['ensemble_prediction']
                                adjusted = original * adjustment_factor
                                predictions[h_key]['ensemble_prediction'] = float(adjusted)
                                predictions[h_key]['sentiment_adjusted'] = True
                                predictions[h_key]['sentiment_score'] = float(sent)
                        logger.debug(f"Sentiment adjustment applied: {sent:.2f} → {adjustment_factor:.2f}x")
                except Exception as e:
                    logger.error(f"Sentiment adjustment error: {e}")
            
            # ⚡ NEW: Regime-based prediction adjustment (volatility-based)
            # ✅ FIX: Use ConfigManager for consistent config access
            use_regime_detection = str(ConfigManager.get('ML_USE_REGIME_DETECTION', '1')).lower() in ('1', 'true', 'yes')
            
            if use_regime_detection:
                try:
                    # Calculate current volatility regime
                    returns = current_data['close'].pct_change().dropna()
                    
                    if len(returns) >= 60:
                        vol_20 = returns.rolling(20).std().iloc[-1]
                        
                        # Historical volatility for percentile calculation
                        vol_history = returns.rolling(20).std().dropna()
                        
                        if len(vol_history) >= 100:
                            # Percentile-based thresholds (33rd and 67th percentiles)
                            vol_p33 = vol_history.quantile(0.33)
                            vol_p67 = vol_history.quantile(0.67)
                            
                            # Determine regime
                            if vol_20 < vol_p33:
                                regime = 'LOW'
                                try:
                                    regime_scale = float(ConfigManager.get('REGIME_SCALE_LOW', '0.85'))
                                except Exception as e:
                                    logger.debug(f"Failed to get REGIME_SCALE_LOW, using 0.85: {e}")
                                    regime_scale = 0.85  # Low volatility → reduce prediction magnitude (more conservative)
                            elif vol_20 > vol_p67:
                                regime = 'HIGH'
                                try:
                                    regime_scale = float(ConfigManager.get('REGIME_SCALE_HIGH', '1.15'))
                                except Exception as e:
                                    logger.debug(f"Failed to get REGIME_SCALE_HIGH, using 1.15: {e}")
                                    regime_scale = 1.15  # High volatility → increase prediction magnitude (more aggressive)
                            else:
                                regime = 'MEDIUM'
                                regime_scale = 1.0  # Normal volatility → no adjustment
                            
                            # Apply regime scaling to all predictions
                            if regime_scale != 1.0:
                                for h_key in predictions:
                                    if 'ensemble_prediction' in predictions[h_key]:
                                        current_price = predictions[h_key]['current_price']
                                        pred_price = predictions[h_key]['ensemble_prediction']
                                        
                                        # Scale the return, not the price
                                        pred_return = (pred_price - current_price) / current_price
                                        scaled_return = pred_return * regime_scale
                                        scaled_price = current_price * (1.0 + scaled_return)
                                        
                                        predictions[h_key]['ensemble_prediction'] = float(scaled_price)
                                        predictions[h_key]['regime'] = regime
                                        predictions[h_key]['regime_scale'] = float(regime_scale)
                                        predictions[h_key]['vol_20d'] = float(vol_20)
                                        predictions[h_key]['vol_p33'] = float(vol_p33)
                                        predictions[h_key]['vol_p67'] = float(vol_p67)
                                
                                logger.debug(f"{symbol}: Regime {regime} (vol={vol_20:.4f}, p33={vol_p33:.4f}, p67={vol_p67:.4f}), scale={regime_scale:.2f}")
                        
                except Exception as e:
                    logger.warning(f"Regime detection error: {e}")
            
            # Empirical horizon caps (clip unrealistic deltas per horizon)
            try:
                # Parse caps from ENV or use aggressive defaults (5x) aligned with directional loss tuning
                # ✅ FIX: Use ConfigManager for consistent config access
                caps_env = ConfigManager.get(
                    'EMPIRICAL_HORIZON_CAPS',
                    '1d:0.30,3d:0.60,7d:0.90,14d:1.50,30d:2.25'
                )
                cap_map = {}
                try:
                    for part in (caps_env or '').split(','):
                        if not part:
                            continue
                        k, v = part.split(':', 1)
                        cap_map[k.strip()] = float(v)
                except Exception as e:
                    logger.debug(f"Failed to parse EMPIRICAL_HORIZON_CAPS, using defaults: {e}")
                    cap_map = {
                        '1d': 0.30,
                        '3d': 0.60,
                        '7d': 0.90,
                        '14d': 1.50,
                        '30d': 2.25,
                    }

                for h_key in list(predictions.keys()):
                    try:
                        current_price = float(predictions[h_key]['current_price'])
                        pred_px = float(predictions[h_key]['ensemble_prediction'])
                        raw_delta = (pred_px - current_price) / max(current_price, 1e-12)
                        # Prefer training-time empirical cap if available for this symbol/horizon
                        try:
                            _h = int(h_key.replace('d', ''))
                            cap_key = f"{symbol}_{_h}d_cap"
                            cap_train = float(self.models.get(cap_key, float('nan')))
                        except Exception as e:
                            logger.debug(f"Failed to get training cap for {cap_key}: {e}")
                            cap_train = float('nan')
                        cap_env = float(cap_map.get(h_key, cap_map.get(str(h_key), 0.30)))
                        cap = float(cap_train) if (cap_train == cap_train) else cap_env  # use train cap if not NaN
                        clipped_delta = float(np.clip(raw_delta, -cap, cap))

                        predictions[h_key]['raw_delta'] = float(raw_delta)
                        predictions[h_key]['cap'] = float(cap)
                        if clipped_delta != raw_delta:
                            predictions[h_key]['cap_applied'] = True
                            predictions[h_key]['delta_clipped'] = float(clipped_delta)
                            predictions[h_key]['ensemble_prediction'] = float(
                                current_price * (1.0 + clipped_delta)
                            )
                            logger.debug(
                                f"{symbol} {h_key}: delta clipped {raw_delta:.3f} → {clipped_delta:.3f} (cap={cap:.2f})"
                            )
                        else:
                            predictions[h_key]['cap_applied'] = False
                    except Exception as _e:
                        logger.debug(f"Empirical cap step error for {symbol} {h_key}: {_e}")
            except Exception as e:
                logger.error(f"Empirical caps error: {e}")

            # ⚡ NEW: Volatility-based calibration (tanh scaling)
            try:
                # Calculate recent volatility
                returns = current_data['close'].pct_change().tail(20)
                vol_20d = float(returns.std()) if len(returns) > 5 else 0.02
                
                # Calibration factor (higher vol → wider predictions)
                for h_key in predictions:
                    pred = predictions[h_key]['ensemble_prediction']
                    current_price = predictions[h_key]['current_price']
                    
                    # Delta in percentage
                    delta_pct = (pred - current_price) / current_price
                    
                    # Tanh calibration (compress extreme predictions)
                    # High vol → less compression
                    # Low vol → more compression
                    scale = 1.0 + vol_20d * 10  # vol 0.02 → scale 1.2, vol 0.05 → scale 1.5
                    calibrated_delta = np.tanh(delta_pct * scale) / scale
                    
                    # Apply calibrated delta
                    calibrated_pred = current_price * (1 + calibrated_delta)
                    predictions[h_key]['ensemble_prediction'] = float(calibrated_pred)
                    predictions[h_key]['calibrated'] = True
                    predictions[h_key]['vol_20d'] = float(vol_20d)
                    
                logger.debug(f"Volatility calibration applied (vol={vol_20d:.4f})")
            except Exception as e:
                logger.error(f"Calibration error: {e}")

            # Horizon-to-horizon smoothing (median of neighbors to reduce single outliers)
            try:
                # ✅ FIX: Use ConfigManager for consistent config access
                enable_smoothing = str(ConfigManager.get('ENABLE_HORIZON_SMOOTHING', '1')).lower() in ('1', 'true', 'yes', 'on')
                if enable_smoothing and predictions:
                    ordered = ['1d', '3d', '7d', '14d', '30d']
                    # Compute deltas after prior steps
                    deltas = {}
                    prices = {}
                    for hk in predictions:
                        cp = float(predictions[hk]['current_price'])
                        px = float(predictions[hk]['ensemble_prediction'])
                        prices[hk] = cp
                        deltas[hk] = (px - cp) / max(cp, 1e-12)

                    def median_of(vals):
                        arr = np.array(vals, dtype=float)
                        return float(np.median(arr)) if arr.size > 0 else 0.0

                    smoothed_any = False
                    for i, hk in enumerate(ordered):
                        if hk not in predictions:
                            continue
                        group = []
                        # neighbors: previous, self, next if exist
                        for j in (i-1, i, i+1):
                            if 0 <= j < len(ordered) and ordered[j] in deltas:
                                group.append(deltas[ordered[j]])
                        smoothed = median_of(group)
                        if len(group) >= 2 and abs(smoothed - deltas.get(hk, 0.0)) > 1e-9:
                            smoothed_any = True
                            predictions[hk]['delta_smoothed'] = float(smoothed)
                            predictions[hk]['ensemble_prediction'] = float(
                                prices[hk] * (1.0 + smoothed)
                            )
                    if smoothed_any:
                        logger.debug(f"{symbol}: horizon smoothing applied")
            except Exception as e:
                logger.error(f"Horizon smoothing error: {e}")
            
            # Horizon consistency test and optional force smoothing
            try:
                ordered = ['1d', '3d', '7d', '14d', '30d']
                inconsistencies = []
                # ✅ FIX: Use ConfigManager for consistent config access
                ratio_max = float(ConfigManager.get('INCONSISTENCY_RATIO_MAX', '2.5'))
                min_check = float(ConfigManager.get('INCONSISTENCY_MIN_ABS', '0.05'))  # 5%
                deltas = {}
                for hk in predictions:
                    cp = float(predictions[hk]['current_price'])
                    px = float(predictions[hk]['ensemble_prediction'])
                    deltas[hk] = (px - cp) / max(cp, 1e-12)

                for i in range(len(ordered) - 1):
                    a, b = ordered[i], ordered[i + 1]
                    if a in deltas and b in deltas:
                        da = abs(deltas[a])
                        db = abs(deltas[b])
                        if db > max(da * ratio_max, min_check) and db > min_check:
                            inconsistencies.append((a, b, da, db))
                if inconsistencies:
                    for hk in predictions:
                        predictions[hk]['inconsistency_flag'] = True
                    logger.debug(f"{symbol}: horizon inconsistency flagged: {inconsistencies}")

                # Optional: force smoothing when inconsistency detected
                # ✅ FIX: Use ConfigManager for consistent config access
                if inconsistencies and str(ConfigManager.get('FORCE_SMOOTH_ON_INCONSISTENCY', '1')).lower() in ('1', 'true', 'yes', 'on'):
                    # Reuse previous median smoothing result (already applied); nothing extra to do
                    # Note: consistency_enforced flag is handled in metrics logging, not in predictions dict
                    pass
            except Exception as e:
                logger.error(f"Horizon consistency check error: {e}")

            # Write lightweight metrics snapshot (symbol-level)
            try:
                import json as _json
                from datetime import datetime as _dt
                # ✅ FIX: Use ConfigManager for consistent config access
                log_dir = ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                os.makedirs(log_dir, exist_ok=True)
                fpath = os.path.join(log_dir, 'metrics_horizon.json')

                entry = {
                    'symbol': symbol,
                    'timestamp': _dt.now().isoformat(),
                    'items': []
                }
                for hk in predictions:
                    if hk in ('consistency_enforced',):
                        continue
                    try:
                        cp = float(predictions[hk]['current_price'])
                        px = float(predictions[hk]['ensemble_prediction'])
                        delta = (px - cp) / max(cp, 1e-12)
                        entry['items'].append({
                            'horizon': hk,
                            'delta': float(delta),
                            'cap_applied': bool(predictions[hk].get('cap_applied', False)),
                            'guard_scale': float(predictions[hk].get('guard', {}).get('confidence_scale', 1.0)) if isinstance(predictions[hk].get('guard'), dict) else 1.0,
                            'inconsistency': bool(predictions[hk].get('inconsistency_flag', False)),
                        })
                    except Exception as e:
                        logger.debug(f"Failed to append prediction entry for {hk}: {e}")
                        continue

                # Append mode with rolling limit
                data_obj = []
                try:
                    if os.path.exists(fpath):
                        with open(fpath, 'r') as rf:
                            data_obj = _json.load(rf) or []
                except Exception as e:
                    logger.debug(f"Failed to load prediction history from {fpath}: {e}")
                    data_obj = []
                data_obj.append(entry)
                # keep last 2000 entries to bound file size
                if len(data_obj) > 2000:
                    data_obj = data_obj[-2000:]
                with open(fpath, 'w') as wf:
                    _json.dump(data_obj, wf)
            except Exception as e:
                logger.debug(f"metrics_horizon write skipped: {e}")

            return predictions
            
        except Exception as e:
            # ✅ FIX: AŞAMA 6 - Model prediction hatası detaylı log
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ AŞAMA 6: {symbol}: Enhanced ML tahmin hatası: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.debug(f"   Traceback:\n{error_trace}")
            return None
    
    def save_enhanced_models(self, symbol):
        """Enhanced modelleri kaydet"""
        try:
            # ✅ CRITICAL FIX: Remove old model files before saving new ones
            # This prevents using old models from previous cycles when model_choice changes
            # Example: Cycle 1: model_choice='all' (3 models), Cycle 2: model_choice='xgb' (1 model)
            # Without cleanup, old lightgbm/catboost models would remain on disk
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                for m in ('xgboost', 'lightgbm', 'catboost'):
                    old_file = f"{self.model_directory}/{model_key}_{m}.pkl"
                    if os.path.exists(old_file):
                        try:
                            os.remove(old_file)
                            logger.debug(f"🗑️ Removed old model: {old_file}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not remove old model {old_file}: {e}")
            
            # Now save new models
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.models:
                    models = self.models[model_key]
                    
                    for model_name, model_info in models.items():
                        filename = f"{self.model_directory}/{model_key}_{model_name}.pkl"
                        # ✅ CRITICAL FIX: Atomic write to prevent corrupt model files
                        _atomic_write_pickle(filename, model_info['model'])
            
            # Save per-horizon metrics (score/rmse/mape/raw_r2) to a JSON in model directory
            try:
                metrics = {}
                for h in self.prediction_horizons:
                    key = f"{symbol}_{h}d"
                    if key in self.models:
                        entry = {}
                        for m, info in (self.models.get(key, {}) or {}).items():
                            try:
                                entry[m] = {
                                    'score': float(info.get('score', 0.0)),
                                    'rmse': float(info.get('rmse', 0.0)),
                                    'mape': float(info.get('mape', 0.0)),
                                    'raw_r2': float(info.get('raw_r2', 0.0)),
                                }
                            except Exception as e:
                                logger.debug(f"Failed to create metrics entry for {m}: {e}")
                                continue
                        metrics[f"{h}d"] = entry
                metrics_file = f"{self.model_directory}/{symbol}_metrics.json"
                # ✅ CRITICAL FIX: Atomic write to prevent corrupt metrics file
                _atomic_write_json(metrics_file, metrics)
            except Exception as e:
                logger.warning(f"Failed to save metrics to {metrics_file}: {e}")

            # Feature importance kaydet
            importance_file = f"{self.model_directory}/{symbol}_feature_importance.pkl"
            symbol_importance = {k: v for k, v in self.feature_importance.items() if k.startswith(symbol)}
            # ✅ CRITICAL FIX: Atomic write to prevent corrupt feature importance file
            _atomic_write_pickle(importance_file, symbol_importance)
            
            # ⚡ META-LEARNERS kaydet
            symbol_meta = {k: v for k, v in self.meta_learners.items() if k.startswith(symbol)}
            if symbol_meta:
                meta_file = f"{self.model_directory}/{symbol}_meta_learners.pkl"
                # ✅ CRITICAL FIX: Atomic write to prevent corrupt meta learners file
                _atomic_write_pickle(meta_file, symbol_meta)
                logger.debug(f"Meta-learners saved: {len(symbol_meta)} models")
            
            # ⚡ META-SCALERS kaydet (Ridge için gerekli!)
            symbol_scalers = {k: v for k, v in self.scalers.items() if k.startswith(symbol) and 'meta_scaler' in k}
            if symbol_scalers:
                scalers_file = f"{self.model_directory}/{symbol}_meta_scalers.pkl"
                # ✅ CRITICAL FIX: Atomic write to prevent corrupt meta scalers file
                _atomic_write_pickle(scalers_file, symbol_scalers)
                logger.debug(f"Meta-scalers saved: {len(symbol_scalers)} scalers")
            
            # ✅ FIX: Save meta model orders (ensures prediction uses same order as training)
            if hasattr(self, 'meta_model_orders'):
                symbol_meta_orders = {k: v for k, v in self.meta_model_orders.items() if k.startswith(symbol)}
                if symbol_meta_orders:
                    orders_file = f"{self.model_directory}/{symbol}_meta_model_orders.pkl"
                    _atomic_write_pickle(orders_file, symbol_meta_orders)
                    logger.debug(f"Meta model orders saved: {len(symbol_meta_orders)} orders")
            
            # Feature columns'ı ayrı JSON olarak kaydet (prediction için gerekli)
            try:
                cols_file = f"{self.model_directory}/{symbol}_feature_columns.json"
                # ✅ CRITICAL FIX: Atomic write to prevent corrupt feature columns file
                _atomic_write_json(cols_file, list(self.feature_columns or []))
            except Exception as e:
                logger.warning(f"Failed to save feature columns to {cols_file}: {e}")
            
            # ⚡ CRITICAL FIX: Save horizon-specific feature columns!
            # Each horizon uses different features after reduction (40 features per horizon)
            # ⚡ CRITICAL: When training per-horizon, we need to APPEND to existing file, not overwrite!
            try:
                horizon_cols_file = f"{self.model_directory}/{symbol}_horizon_features.json"
                
                # ⚡ CRITICAL FIX: Load existing horizon features first (if file exists)
                # This prevents overwriting when training per-horizon
                existing_horizon_features = {}
                if os.path.exists(horizon_cols_file):
                    try:
                        with open(horizon_cols_file, 'r') as rf:
                            existing_horizon_features = json.load(rf) or {}
                        logger.debug(f"Loaded existing horizon features: {list(existing_horizon_features.keys())}")
                    except Exception as load_err:
                        logger.debug(f"Could not load existing horizon features: {load_err}")
                        existing_horizon_features = {}
                
                # ✅ CRITICAL FIX: Atomic read-modify-write with file locking to prevent race conditions
                # This ensures concurrent training processes don't overwrite each other's horizon features
                def merge_horizon_features(existing: dict) -> dict:
                    """Merge current horizon features with existing ones"""
                    merged = existing.copy()
                    for h in self.prediction_horizons:
                        feature_key = f"{symbol}_{h}d_features"
                        if feature_key in self.models:
                            merged[f"{h}d"] = list(self.models[feature_key])
                            logger.debug(f"Adding {h}d horizon features ({len(self.models[feature_key])} features)")
                    return merged
                
                if any(f"{symbol}_{h}d_features" in self.models for h in self.prediction_horizons):
                    _atomic_read_modify_write_json(horizon_cols_file, merge_horizon_features, default_data={})
                    # Read back to get final state for logging
                    try:
                        with open(horizon_cols_file, 'r') as rf:
                            final_horizon_features = json.load(rf) or {}
                        logger.debug(f"Horizon-specific features saved: {len(final_horizon_features)} horizons ({list(final_horizon_features.keys())})")
                    except Exception as e:
                        logger.debug(f"Failed to read back horizon features: {e}")
            except Exception as e:
                logger.warning(f"Failed to save horizon-specific features: {e}")

            # ⚡ NEW: Write per-symbol ModelManifest.json (lightweight)
            try:
                manifest_path = f"{self.model_directory}/{symbol}_manifest.json"
                
                # ✅ CRITICAL FIX: Atomic read-modify-write with file locking to prevent race conditions
                # This ensures concurrent training processes don't overwrite each other's manifest updates
                def merge_manifest(existing: dict) -> dict:
                    """Merge current manifest with existing one"""
                    # Existing structures
                    ex_horizons = set(existing.get('horizons', []))
                    ex_enabled = existing.get('enabled_models', {}) or {}
                    ex_hfeat = existing.get('horizon_features', {}) or {}
                    ex_hcaps = existing.get('horizon_caps', {}) or {}
                    
                    # Compute current horizon entries
                    cur_enabled = {}
                    cur_hfeat = {}
                    cur_hcaps = {}
                    cur_meta_order = {}  # ✅ FIX: Store meta-learner model order in manifest
                    cur_hlist = []
                    for h in self.prediction_horizons:
                        hk = f"{h}d"
                        cur_hlist.append(hk)
                        model_key = f"{symbol}_{h}d"
                        # Enabled models for this horizon (only if trained in this run)
                        if model_key in self.models:
                            cur_enabled[hk] = list(self.models[model_key].keys())
                        # ✅ FIX: Get meta-learner model order if available
                        if hasattr(self, 'meta_model_orders'):
                            meta_order_key = f"{symbol}_{h}d_meta_model_order"
                            if meta_order_key in self.meta_model_orders:
                                cur_meta_order[hk] = self.meta_model_orders[meta_order_key]
                        # Horizon features (if available in memory)
                        cur_hfeat[hk] = list(self.models.get(f"{symbol}_{h}d_features", []))
                        # Horizon empirical cap (if available)
                        cap_val = self.models.get(f"{symbol}_{h}d_cap", np.nan)
                        try:
                            cur_hcaps[hk] = float(cap_val)
                        except Exception as e:
                            logger.debug(f"Failed to get cap value for {hk}: {e}")
                    
                    # Merge
                    new_enabled = dict(ex_enabled)
                    new_enabled.update({k: v for k, v in cur_enabled.items() if v})
                    new_hfeat = dict(ex_hfeat)
                    for k, v in cur_hfeat.items():
                        if v:
                            new_hfeat[k] = v
                    new_hcaps = dict(ex_hcaps)
                    for k, v in cur_hcaps.items():
                        try:
                            if np.isfinite(v):  # type: ignore[attr-defined]
                                new_hcaps[k] = float(v)
                        except Exception as e:
                            logger.debug(f"Failed to check if cap value is finite for {k}: {e}")
                            # If numpy not available in this code path, accept any float-like value
                            try:
                                fv = float(v)
                                new_hcaps[k] = fv
                            except Exception as e2:
                                logger.debug(f"Failed to convert cap value to float: {e2}")
                    
                    # ✅ FIX: Merge meta-learner model orders
                    ex_meta_order = existing.get('meta_model_orders', {}) or {}
                    new_meta_order = dict(ex_meta_order)
                    new_meta_order.update({k: v for k, v in cur_meta_order.items() if v})
                    
                    all_horizons = sorted(set(ex_horizons).union(cur_hlist))
                    
                    return {
                        'symbol': symbol,
                        'trained_at': datetime.now().isoformat(),
                        'horizons': all_horizons,
                        'feature_count': int(len(getattr(self, 'feature_columns', []) or [])),
                        'has_meta_scalers': bool(any(k.startswith(symbol) and 'meta_scaler' in k for k in (self.scalers or {}).keys())),
                        'horizon_features': new_hfeat,
                        'horizon_caps': new_hcaps,
                        'enabled_models': new_enabled,
                        'meta_model_orders': new_meta_order,  # ✅ FIX: Store meta-learner model order in manifest
                    }
                
                # Atomic read-modify-write with file locking
                _atomic_read_modify_write_json(manifest_path, merge_manifest, default_data={})
                # Read back to get final state for logging
                try:
                    with open(manifest_path, 'r') as rf:
                        final_manifest = json.load(rf) or {}
                    all_horizons = final_manifest.get('horizons', [])
                    logger.debug(f"Model manifest merged & written: {manifest_path} (horizons={all_horizons})")
                except Exception as e:
                    logger.debug(f"Failed to read back manifest: {e}")
            except Exception as e:
                logger.warning(f"Failed to write model manifest for {symbol}: {e}")

            # Eğitim meta verisini kaydet (dashboard için)
            try:
                # ✅ FIX: Use ConfigManager for consistent config access
                log_dir = ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                os.makedirs(log_dir, exist_ok=True)
                meta_dir = os.path.join(log_dir, 'model_performance')
                os.makedirs(meta_dir, exist_ok=True)
                meta = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'horizons': self.prediction_horizons,
                    'models': {
                        f"{h}d": {
                            m: {
                                'score': float(info.get('score', 0.0)),
                                'rmse': float(info.get('rmse', 0.0)),
                                'mape': float(info.get('mape', 0.0)),
                            }
                            for m, info in (self.models.get(f"{symbol}_{h}d", {}) or {}).items()
                        }
                        for h in self.prediction_horizons
                    },
                    'feature_count': len(getattr(self, 'feature_columns', [])),
                }
                with open(os.path.join(meta_dir, f"{symbol}.json"), 'w') as wf:
                    json.dump(meta, wf)
            except Exception as e:
                logger.warning(f"Failed to save training metadata for {symbol}: {e}")

            logger.info(f"💾 {symbol} enhanced modelleri kaydedildi")
            
        except Exception as e:
            logger.error(f"Enhanced model kaydetme hatası: {e}")

    def has_trained_models(self, symbol: str) -> bool:
        """
        Diskte bu sembol için en az bir horizon model dosyası var mı?
        
        ✅ AŞAMA 1: Model Dosyası Yok Kontrolü
        - Her horizon için (1d, 3d, 7d, 14d, 30d)
        - Her model tipi için (xgboost, lightgbm, catboost)
        - En az bir model dosyası varsa True döndürür
        
        Returns:
            bool: Model dosyası var mı?
        """
        try:
            found_models = []
            for h in self.prediction_horizons:
                for m in ('xgboost', 'lightgbm', 'catboost'):
                    fpath = f"{self.model_directory}/{symbol}_{h}d_{m}.pkl"
                    if os.path.exists(fpath):
                        found_models.append(f"{h}d_{m}")
                        return True  # En az bir model bulundu
            
            # ✅ FIX: Detaylı log mesajı - hangi model dosyaları eksik
            if not found_models:
                logger.debug(f"⚠️ AŞAMA 1: Enhanced ML model yok: {symbol} (hiç model dosyası bulunamadı)")
                # Hangi dosyaların eksik olduğunu belirt
                missing_files = []
                for h in self.prediction_horizons:
                    for m in ('xgboost', 'lightgbm', 'catboost'):
                        fpath = f"{self.model_directory}/{symbol}_{h}d_{m}.pkl"
                        if not os.path.exists(fpath):
                            missing_files.append(f"{symbol}_{h}d_{m}.pkl")
                if missing_files:
                    logger.debug(f"   Eksik model dosyaları: {missing_files[:5]}...")  # İlk 5 dosya
            
            return False
        except Exception as e:
            logger.error(f"❌ AŞAMA 1: has_trained_models kontrolü hatası {symbol}: {e}")
            return False

    def load_trained_models(self, symbol: str) -> bool:
        """
        Diskten sembol modellerini ve feature kolonlarını yükle (varsa).
        
        ✅ AŞAMA 2: Model Dosyası Yüklenemiyor Kontrolü
        - Her horizon için model dosyaları yüklenir
        - Yükleme hatası olursa loglanır ve continue edilir
        - En az bir model yüklenirse True döndürür
        
        ✅ CRITICAL FIX: Load enabled_models from manifest and set environment variables
        - Only load models that were trained (HPO model_choice)
        - Set ENABLE_XGBOOST, ENABLE_LIGHTGBM, ENABLE_CATBOOST based on manifest
        
        Returns:
            bool: Model yüklendi mi?
        """
        try:
            # ✅ CRITICAL FIX: Load enabled_models from manifest first
            # This ensures we only use models that were trained (HPO model_choice)
            enabled_models_per_horizon = {}
            manifest_path = f"{self.model_directory}/{symbol}_manifest.json"
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as mf:
                        manifest_obj = json.load(mf) or {}
                    enabled_models_per_horizon = manifest_obj.get('enabled_models', {})
                    logger.debug(f"📋 Loaded enabled_models from manifest: {enabled_models_per_horizon}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load manifest for {symbol}: {e}")
            
            loaded_any = False
            load_errors = []
            for h in self.prediction_horizons:
                model_key = f"{symbol}_{h}d"
                horizon_models = {}
                
                # ✅ CRITICAL FIX: Only load models that are in enabled_models (if manifest exists)
                # If manifest doesn't exist, load all available models (backward compatibility)
                models_to_load = enabled_models_per_horizon.get(f"{h}d", ['xgboost', 'lightgbm', 'catboost'])
                
                for m in ('xgboost', 'lightgbm', 'catboost'):
                    # Skip if this model is not in enabled_models (HPO didn't train it)
                    if enabled_models_per_horizon and m not in models_to_load:
                        logger.debug(f"⏭️ Skipping {symbol} {h}d {m}: Not in enabled_models (HPO model_choice)")
                        continue
                    
                    fpath = f"{self.model_directory}/{symbol}_{h}d_{m}.pkl"
                    if os.path.exists(fpath):
                        try:
                            model_obj = joblib.load(fpath)
                            horizon_models[m] = {
                                'model': model_obj,
                                'score': 0.0,
                                'rmse': 0.0,
                                'mape': 0.0,
                            }
                            loaded_any = True
                            logger.debug(f"✅ {symbol} {h}d {m}: Model yüklendi")
                        except Exception as load_err:
                            # ✅ FIX: AŞAMA 2 - Yükleme hatası loglanır
                            error_msg = f"{symbol}_{h}d_{m}.pkl yüklenemedi: {load_err}"
                            load_errors.append(error_msg)
                            logger.warning(f"⚠️ AŞAMA 2: {error_msg}")
                            continue
                if horizon_models:
                    self.models[model_key] = horizon_models
                    logger.debug(f"✅ {symbol} {h}d: {len(horizon_models)} model yüklendi")
            
            # ✅ CRITICAL FIX: Set environment variables based on loaded models
            # This ensures prediction uses only the models that were trained (HPO model_choice)
            if enabled_models_per_horizon:
                # Check all horizons to determine which models are enabled
                all_enabled_models = set()
                for horizon_models in enabled_models_per_horizon.values():
                    all_enabled_models.update(horizon_models)
                
                # Set environment variables
                os.environ['ENABLE_XGBOOST'] = '1' if 'xgboost' in all_enabled_models else '0'
                os.environ['ENABLE_LIGHTGBM'] = '1' if 'lightgbm' in all_enabled_models else '0'
                os.environ['ENABLE_CATBOOST'] = '1' if 'catboost' in all_enabled_models else '0'
                logger.info(f"🔧 {symbol}: Set ENABLE flags from manifest: XGB={os.environ.get('ENABLE_XGBOOST')}, LGB={os.environ.get('ENABLE_LIGHTGBM')}, CAT={os.environ.get('ENABLE_CATBOOST')}")
                # Clear ConfigManager cache to ensure new values are read
                ConfigManager.clear_cache()
            
            # ✅ FIX: Yükleme sonucu loglanır
            if not loaded_any:
                logger.warning(f"⚠️ AŞAMA 2: {symbol}: Failed to load trained models (hiç model yüklenemedi)")
                if load_errors:
                    logger.warning(f"   Yükleme hataları: {load_errors[:3]}...")  # İlk 3 hata
                return False  # Hiç model yüklenemedi

            # Reload metrics if available to restore scores/metrics for weighting
            try:
                metrics_file = f"{self.model_directory}/{symbol}_metrics.json"
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as rf:
                        metrics_obj = json.load(rf) or {}
                    for h in self.prediction_horizons:
                        key = f"{symbol}_{h}d"
                        hkey = f"{h}d"
                        if key in self.models and hkey in metrics_obj:
                            for m, vals in (metrics_obj[hkey] or {}).items():
                                try:
                                    if m in self.models[key]:
                                        self.models[key][m]['score'] = float(vals.get('score', 0.0))
                                        self.models[key][m]['rmse'] = float(vals.get('rmse', 0.0))
                                        self.models[key][m]['mape'] = float(vals.get('mape', 0.0))
                                        if 'raw_r2' in vals:
                                            self.models[key][m]['raw_r2'] = float(vals.get('raw_r2', 0.0))
                                except Exception as e:
                                    logger.debug(f"Failed to load metrics for {key}/{m}: {e}")
                                    continue
            except Exception as e:
                logger.debug(f"Failed to load metrics file: {e}")

            # ✅ AŞAMA 3: Feature Columns Yüklenemiyor Kontrolü
            # Feature columns: JSON → fallback importance → fallback empty
            cols_file = f"{self.model_directory}/{symbol}_feature_columns.json"
            cols = []
            try:
                if os.path.exists(cols_file):
                    with open(cols_file, 'r') as rf:
                        cols = json.load(rf) or []
                    if cols:
                        logger.debug(f"✅ AŞAMA 3: {symbol}: Feature columns yüklendi ({len(cols)} features)")
            except Exception as e:
                logger.warning(f"⚠️ AŞAMA 3: {symbol}: feature_columns.json okunamadı: {e}")
                cols = []
            if not cols:
                try:
                    importance_file = f"{self.model_directory}/{symbol}_feature_importance.pkl"
                    if os.path.exists(importance_file):
                        imp = joblib.load(importance_file) or {}
                        # Union of feature names (order fallback: sorted)
                        keys = set()
                        for _k, _v in (imp.items() if isinstance(imp, dict) else []):
                            try:
                                keys.update(list(_v.keys()))
                            except Exception as e:
                                logger.debug(f"Failed to update keys from feature importance: {e}")
                                continue
                        cols = sorted(keys)
                        if cols:
                            logger.debug(f"✅ AŞAMA 3: {symbol}: Feature columns fallback (importance) yüklendi ({len(cols)} features)")
                except Exception as e:
                    logger.warning(f"⚠️ AŞAMA 3: {symbol}: feature_importance.pkl okunamadı: {e}")
                    cols = []
            
            # ✅ FIX: Feature columns yüklenemedi kontrolü
            if not cols and not self.feature_columns:
                logger.warning(f"⚠️ AŞAMA 3: {symbol}: Feature columns not set. Model training required.")
                logger.warning(f"   Eksik dosyalar: {symbol}_feature_columns.json, {symbol}_feature_importance.pkl")
            # ⚡ FEATURE COLUMNS yükle (dict format for adaptive learning)
            # Load per-horizon feature columns
            if not hasattr(self, 'feature_columns'):
                self.feature_columns = {}
            
            # ✅ AŞAMA 4: Horizon Features Yüklenemiyor Kontrolü
            # ⚡ CRITICAL FIX: Load horizon-specific features from horizon_features.json
            # This is critical for prediction - each horizon uses different features after reduction
            horizon_cols_file = f"{self.model_directory}/{symbol}_horizon_features.json"
            if os.path.exists(horizon_cols_file):
                try:
                    with open(horizon_cols_file, 'r') as rf:
                        horizon_features = json.load(rf) or {}
                    
                    # ⚡ CRITICAL FIX: Restore horizon-specific features to self.models
                    # This is needed for prediction - horizon_feature_cols lookup
                    loaded_horizons = []
                    for h in self.prediction_horizons:
                        h_key = f"{h}d"
                        if h_key in horizon_features:
                            feature_key = f"{symbol}_{h}d_features"
                            self.models[feature_key] = horizon_features[h_key]
                            loaded_horizons.append(h_key)
                            logger.debug(f"✅ AŞAMA 4: {symbol} {h_key}: Horizon features yüklendi ({len(horizon_features[h_key])} features)")
                        else:
                            logger.warning(f"⚠️ AŞAMA 4: {symbol}: Horizon {h_key} not found in horizon_features.json")
                    
                    # Also set feature_columns as dict for backward compatibility
                    self.feature_columns = horizon_features
                    logger.debug(f"✅ AŞAMA 4: {symbol}: Horizon features yüklendi ({len(loaded_horizons)}/{len(self.prediction_horizons)}) horizons)")
                except Exception as e:
                    logger.error(f"❌ AŞAMA 4: {symbol}: horizon_features.json yüklenemedi: {e}")
            else:
                logger.warning(f"⚠️ AŞAMA 4: {symbol}: horizon_features.json not found at {horizon_cols_file}")
            
            # Fallback: use cols (union of all features) if per-horizon not available
            if not self.feature_columns and cols:
                for h in self.prediction_horizons:
                    self.feature_columns[f'{h}d'] = cols
            
            # ⚡ Load per-horizon empirical caps from manifest (if available)
            try:
                manifest_path = f"{self.model_directory}/{symbol}_manifest.json"
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as mf:
                        manifest_obj = json.load(mf) or {}
                    hcaps = manifest_obj.get('horizon_caps') or {}
                    # Example keys: {'1d': 0.051, '3d': 0.085, ...}
                    for hk, cap_val in hcaps.items():
                        try:
                            h_int = int(str(hk).replace('d', '').strip())
                            self.models[f"{symbol}_{h_int}d_cap"] = float(cap_val)
                        except Exception as e:
                            logger.debug(f"Failed to parse horizon cap for {hk}: {e}")
                            continue
                    if hcaps:
                        logger.debug(f"Loaded horizon caps from manifest: {len(hcaps)} entries")
            except Exception as _e:
                logger.debug(f"Horizon caps load skipped: {_e}")
            
            # ⚡ META-LEARNERS yükle
            try:
                meta_file = f"{self.model_directory}/{symbol}_meta_learners.pkl"
                if os.path.exists(meta_file):
                    symbol_meta = joblib.load(meta_file) or {}
                    self.meta_learners.update(symbol_meta)
                    logger.debug(f"Meta-learners loaded: {len(symbol_meta)} models")
            except Exception as e:
                logger.debug(f"Meta-learner load failed: {e}")
            
            # ⚡ META-SCALERS yükle (Ridge için gerekli!)
            try:
                scalers_file = f"{self.model_directory}/{symbol}_meta_scalers.pkl"
                if os.path.exists(scalers_file):
                    symbol_scalers = joblib.load(scalers_file) or {}
                    self.scalers.update(symbol_scalers)
                    logger.debug(f"Meta-scalers loaded: {len(symbol_scalers)} scalers")
            except Exception as e:
                logger.debug(f"Meta-scaler load failed: {e}")
            
            # ✅ FIX: Load meta model orders (ensures prediction uses same order as training)
            # Try loading from manifest first (newer format), then fallback to pkl file
            try:
                manifest_path = f"{self.model_directory}/{symbol}_manifest.json"
                if os.path.exists(manifest_path):
                    with open(manifest_path, 'r') as mf:
                        manifest_obj = json.load(mf) or {}
                    manifest_meta_orders = manifest_obj.get('meta_model_orders', {})
                    if manifest_meta_orders:
                        if not hasattr(self, 'meta_model_orders'):
                            self.meta_model_orders = {}
                        # Convert manifest format to internal format
                        for hk, order_list in manifest_meta_orders.items():
                            order_key = f"{symbol}_{hk}_meta_model_order"
                            self.meta_model_orders[order_key] = order_list
                        logger.debug(f"Meta model orders loaded from manifest: {len(manifest_meta_orders)} orders")
            except Exception as e:
                logger.debug(f"Meta model orders load from manifest failed: {e}")
            
            # Fallback: Load from pkl file (older format)
            try:
                orders_file = f"{self.model_directory}/{symbol}_meta_model_orders.pkl"
                if os.path.exists(orders_file):
                    symbol_meta_orders = joblib.load(orders_file) or {}
                    if not hasattr(self, 'meta_model_orders'):
                        self.meta_model_orders = {}
                    self.meta_model_orders.update(symbol_meta_orders)
                    logger.debug(f"Meta model orders loaded from pkl: {len(symbol_meta_orders)} orders")
            except Exception as e:
                logger.debug(f"Meta-scaler load failed: {e}")
            
            # ⚡ CRITICAL FIX: Load horizon-specific feature columns!
            try:
                horizon_cols_file = f"{self.model_directory}/{symbol}_horizon_features.json"
                if os.path.exists(horizon_cols_file):
                    with open(horizon_cols_file, 'r') as rf:
                        horizon_features = json.load(rf) or {}

                    # Restore horizon-specific features to self.models
                    for h in self.prediction_horizons:
                        h_key = f"{h}d"
                        if h_key in horizon_features:
                            feature_key = f"{symbol}_{h}d_features"
                            self.models[feature_key] = horizon_features[h_key]

                    logger.debug(f"Horizon-specific features loaded: {len(horizon_features)} horizons")
            except Exception as e:
                logger.debug(f"Horizon-specific features load failed: {e}")
            
            return loaded_any
        except Exception as e:
            logger.warning(f"load_trained_models failed for {symbol}: {e}")
            return False
    
    def get_top_features(self, symbol, model_type='xgboost', top_n=20):
        """En önemli feature'ları döndür"""
        try:
            top_features = {}
            
            for horizon in self.prediction_horizons:
                # Normalize model_type mapping: 'xgboost'->'xgb', 'lightgbm'->'lgb', 'catboost'->'cat'
                mt = model_type.lower()
                if mt.startswith('xgboost') or mt == 'xgb':
                    short = 'xgb'
                elif mt.startswith('lightgbm') or mt == 'lgb':
                    short = 'lgb'
                elif mt.startswith('catboost') or mt == 'cat':
                    short = 'cat'
                else:
                    short = mt[:3]
                key = f"{symbol}_{horizon}d_{short}"
                if key in self.feature_importance:
                    importance = self.feature_importance[key]
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    top_features[f"{horizon}d"] = sorted_features[:top_n]
            
            return top_features
            
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}
    
    def get_system_info(self):
        """Enhanced ML system bilgileri"""
        return {
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
            'catboost_available': CATBOOST_AVAILABLE,
            'base_ml_available': BASE_ML_AVAILABLE,
            'models_trained': len(self.models),
            'prediction_horizons': self.prediction_horizons,
            'feature_count': len(getattr(self, 'feature_columns', [])),
            'performance_tracked': len(self.model_performance)
        }


# Global singleton instance
_enhanced_ml_system = None
_singleton_lock = threading.Lock()


def get_enhanced_ml_system():
    """
    Enhanced ML System singleton'ını döndür (thread-safe).
    
    Uses double-checked locking pattern to ensure thread safety while
    avoiding unnecessary locking after the instance is created.
    """
    global _enhanced_ml_system
    if _enhanced_ml_system is None:
        with _singleton_lock:
            # Double-check after acquiring lock (another thread may have created it)
            if _enhanced_ml_system is None:
                _enhanced_ml_system = EnhancedMLSystem()
    return _enhanced_ml_system


def clear_enhanced_ml_system():
    """
    Thread-safe singleton temizleme fonksiyonu.
    
    Singleton instance'ı thread-safe bir şekilde temizler.
    Test veya reset durumlarında kullanılır.
    """
    global _enhanced_ml_system
    with _singleton_lock:
        _enhanced_ml_system = None


if __name__ == "__main__":
    # Test
    enhanced_ml = get_enhanced_ml_system()
    info = enhanced_ml.get_system_info()

    print("🧠 Enhanced ML System Test:")
    print(f"📊 XGBoost: {info['xgboost_available']}")
    print(f"📊 LightGBM: {info['lightgbm_available']}")
    print(f"📊 CatBoost: {info['catboost_available']}")
    print(f"🎯 Prediction Horizons: {info['prediction_horizons']}")
    print("✅ Enhanced ML System ready!")
