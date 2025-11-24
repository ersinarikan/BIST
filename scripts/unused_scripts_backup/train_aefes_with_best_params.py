#!/usr/bin/env python3
"""
Train AEFES with best HPO parameters from test run
"""
import os
import sys
import json
import logging
from datetime import datetime

# Set environment
os.environ['PYTHONPATH'] = '/opt/bist-pattern'

# Load environment from systemd service if available
import subprocess
try:
    result = subprocess.run(['systemctl', 'show', 'bist-pattern.service', '--property=Environment', '--value'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0 and result.stdout.strip():
        env_vars = result.stdout.strip()
        env_vars = env_vars.strip('"').strip("'")
        for var in env_vars.split():
            if '=' in var:
                key, value = var.split('=', 1)
                value = value.strip('"').strip("'")
                os.environ[key] = value
except Exception:
    pass

# Ensure DATABASE_URL is set
if 'DATABASE_URL' not in os.environ:
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:5432/bist_pattern_db'

# Load best params from HPO test results
best_params_file = '/opt/bist-pattern/results/optuna_pilot_h7_20251031_231015.json'
if not os.path.exists(best_params_file):
    print(f"❌ Best params file not found: {best_params_file}")
    sys.exit(1)

with open(best_params_file, 'r') as f:
    results = json.load(f)

best_params = results.get('best_params', {})
print("=" * 80)
print("🔬 BEST HPO PARAMETERS FOR AEFES (7d horizon)")
print("=" * 80)
print(f"Best DirHit: {results.get('best_value', 0):.2f}%")
print("\nSetting environment variables...")

# Set XGBoost parameters
os.environ['OPTUNA_XGB_N_ESTIMATORS'] = str(best_params.get('xgb_n_estimators', 431))
os.environ['OPTUNA_XGB_MAX_DEPTH'] = str(best_params.get('xgb_max_depth', 8))
os.environ['OPTUNA_XGB_LEARNING_RATE'] = str(best_params.get('xgb_learning_rate', 0.1055))
os.environ['OPTUNA_XGB_SUBSAMPLE'] = str(best_params.get('xgb_subsample', 0.8395))
os.environ['OPTUNA_XGB_COLSAMPLE_BYTREE'] = str(best_params.get('xgb_colsample_bytree', 0.5780))
os.environ['OPTUNA_XGB_REG_ALPHA'] = str(best_params.get('xgb_reg_alpha', 1.1092e-05))
os.environ['OPTUNA_XGB_REG_LAMBDA'] = str(best_params.get('xgb_reg_lambda', 0.0002))
os.environ['OPTUNA_XGB_MIN_CHILD_WEIGHT'] = str(best_params.get('xgb_min_child_weight', 11))
os.environ['OPTUNA_XGB_GAMMA'] = str(best_params.get('xgb_gamma', 0.0101))

# Set LightGBM parameters
os.environ['OPTUNA_LGB_N_ESTIMATORS'] = str(best_params.get('lgb_n_estimators', 242))
os.environ['OPTUNA_LGB_MAX_DEPTH'] = str(best_params.get('lgb_max_depth', 3))
os.environ['OPTUNA_LGB_LEARNING_RATE'] = str(best_params.get('lgb_learning_rate', 0.1114))
os.environ['OPTUNA_LGB_NUM_LEAVES'] = str(best_params.get('lgb_num_leaves', 29))
os.environ['OPTUNA_LGB_SUBSAMPLE'] = str(best_params.get('lgb_subsample', 0.6849))
os.environ['OPTUNA_LGB_COLSAMPLE_BYTREE'] = str(best_params.get('lgb_colsample_bytree', 0.6727))
os.environ['OPTUNA_LGB_REG_ALPHA'] = str(best_params.get('lgb_reg_alpha', 1.5415e-05))
os.environ['OPTUNA_LGB_REG_LAMBDA'] = str(best_params.get('lgb_reg_lambda', 0.0154))

# Set CatBoost parameters
os.environ['OPTUNA_CAT_ITERATIONS'] = str(best_params.get('cat_iterations', 205))
os.environ['OPTUNA_CAT_DEPTH'] = str(best_params.get('cat_depth', 4))
os.environ['OPTUNA_CAT_LEARNING_RATE'] = str(best_params.get('cat_learning_rate', 0.0206))
os.environ['OPTUNA_CAT_L2_LEAF_REG'] = str(best_params.get('cat_l2_leaf_reg', 0.2444))
os.environ['OPTUNA_CAT_SUBSAMPLE'] = str(best_params.get('cat_subsample', 0.6558))
os.environ['OPTUNA_CAT_RSM'] = str(best_params.get('cat_rsm', 0.7169))

# Set adaptive and pattern weights
os.environ['ML_ADAPTIVE_K_7D'] = str(best_params.get('adaptive_k', 1.6931))
os.environ['ML_PATTERN_WEIGHT_SCALE_7D'] = str(best_params.get('pattern_weight', 1.0824))

# Override training settings
os.environ['FORCE_FULL_RETRAIN'] = '1'
os.environ['ML_MAX_MODEL_AGE_DAYS'] = '0'

# All improvements ON
os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
os.environ['ML_USE_REGIME_DETECTION'] = '1'
os.environ['ML_USE_ADAPTIVE_LEARNING'] = '1'
os.environ['ML_USE_DIRECTIONAL_LOSS'] = '1'
os.environ['ENABLE_SEED_BAGGING'] = '1'
os.environ['N_SEEDS'] = '3'
os.environ['STRICT_HORIZON_FEATURES'] = '1'
os.environ['ML_HORIZONS'] = '1,3,7,14,30'

# Enable TA-Lib patterns for all horizons
os.environ['ENABLE_TALIB_PATTERNS'] = '1'

# External features (can be enabled later if needed)
os.environ['ENABLE_EXTERNAL_FEATURES'] = '0'
os.environ['ENABLE_FINGPT_FEATURES'] = '0'
os.environ['ENABLE_YOLO_FEATURES'] = '0'

sys.path.insert(0, '/opt/bist-pattern')

from app import app  # noqa: E402
from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402
from pattern_detector import HybridPatternDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_aefes():
    """Train AEFES with best HPO parameters"""
    symbol = 'AEFES'
    logger.info("=" * 80)
    logger.info(f"🎯 Training {symbol} with BEST HPO Parameters")
    logger.info("=" * 80)
    logger.info(f"Best DirHit from HPO: {results.get('best_value', 0):.2f}%")
    logger.info(f"XGBoost: n_est={os.environ.get('OPTUNA_XGB_N_ESTIMATORS')}, max_d={os.environ.get('OPTUNA_XGB_MAX_DEPTH')}, lr={os.environ.get('OPTUNA_XGB_LEARNING_RATE')}")
    logger.info(f"LightGBM: n_est={os.environ.get('OPTUNA_LGB_N_ESTIMATORS')}, max_d={os.environ.get('OPTUNA_LGB_MAX_DEPTH')}, lr={os.environ.get('OPTUNA_LGB_LEARNING_RATE')}")
    logger.info(f"CatBoost: iter={os.environ.get('OPTUNA_CAT_ITERATIONS')}, depth={os.environ.get('OPTUNA_CAT_DEPTH')}, lr={os.environ.get('OPTUNA_CAT_LEARNING_RATE')}")
    logger.info("=" * 80)
    
    with app.app_context():
        det = HybridPatternDetector()
        ml = get_enhanced_ml_system()
        
        # Get stock data
        logger.info(f"📊 Fetching data for {symbol}...")
        df = det.get_stock_data(symbol, days=0)
        
        if df is None or len(df) < 50:
            logger.error(f"❌ {symbol}: Insufficient data ({len(df) if df is not None else 0} days)")
            return False
        
        logger.info(f"✅ {symbol}: Data loaded ({len(df)} days)")
        
        # Train enhanced models with best params
        logger.info(f"🎯 Training enhanced models for {symbol} with HPO best parameters...")
        try:
            result = ml.train_enhanced_models(symbol, df)
            if result:
                logger.info(f"✅ {symbol}: Training completed successfully with best HPO parameters")
                
                # Verify horizon_features.json
                import json
                model_dir = os.getenv('ML_MODEL_PATH', '/opt/bist-pattern/.cache/enhanced_ml_models')
                horizon_file = f"{model_dir}/{symbol}_horizon_features.json"
                
                if os.path.exists(horizon_file):
                    with open(horizon_file, 'r') as f:
                        data = json.load(f)
                    logger.info(f"✅ {symbol}: horizon_features.json saved with {len(data)} horizons: {list(data.keys())}")
                    for h, features in data.items():
                        logger.info(f"   {h}: {len(features)} features")
                else:
                    logger.warning(f"⚠️ {symbol}: horizon_features.json NOT found after training!")
                
                # Verify model files
                model_files = []
                for h in [1, 3, 7, 14, 30]:
                    for m in ['xgboost', 'lightgbm', 'catboost']:
                        fpath = f"{model_dir}/{symbol}_{h}d_{m}.pkl"
                        if os.path.exists(fpath):
                            model_files.append(f"{h}d_{m}")
                
                logger.info(f"✅ {symbol}: Model files saved: {len(model_files)} files")
                logger.info(f"   Files: {', '.join(model_files[:10])}{'...' if len(model_files) > 10 else ''}")
                
                return True
            else:
                logger.error(f"❌ {symbol}: Training failed")
                return False
        except Exception as e:
            logger.error(f"❌ {symbol}: Training error: {e}", exc_info=True)
            return False


if __name__ == '__main__':
    success = train_aefes()
    
    if success:
        logger.info("=" * 80)
        logger.info("✅ AEFES training completed successfully with best HPO parameters!")
        logger.info("=" * 80)
        logger.info("💡 Next step: Wait for one automation cycle and check predictions")
        sys.exit(0)
    else:
        logger.error("=" * 80)
        logger.error("❌ AEFES training failed!")
        logger.error("=" * 80)
        sys.exit(1)

