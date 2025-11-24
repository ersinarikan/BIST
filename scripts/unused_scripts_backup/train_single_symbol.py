#!/usr/bin/env python3
"""
Train a single symbol with all horizons - Fixes horizon_features.json issue
"""
import os
import sys
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
        # Remove quotes if present
        env_vars = env_vars.strip('"').strip("'")
        for var in env_vars.split():
            if '=' in var:
                key, value = var.split('=', 1)
                # Remove quotes from value if present
                value = value.strip('"').strip("'")
                os.environ[key] = value
except Exception:
    pass  # Fallback to defaults

# Ensure DATABASE_URL is set
if 'DATABASE_URL' not in os.environ:
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:5432/bist_pattern_db'

# Override training settings
os.environ['FORCE_FULL_RETRAIN'] = '1'
os.environ['ML_MAX_MODEL_AGE_DAYS'] = '0'

# All improvements ON
os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
os.environ['ML_USE_REGIME_DETECTION'] = '1'
os.environ['ML_USE_ADAPTIVE_LEARNING'] = '1'
os.environ['ML_USE_STACKED_SHORT'] = '1'
os.environ['ML_USE_DIRECTIONAL_LOSS'] = '1'
os.environ['ENABLE_SEED_BAGGING'] = '1'
os.environ['N_SEEDS'] = '3'
os.environ['STRICT_HORIZON_FEATURES'] = '1'
os.environ['ML_HORIZONS'] = '1,3,7,14,30'

sys.path.insert(0, '/opt/bist-pattern')

from app import app  # noqa: E402
from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402
from pattern_detector import HybridPatternDetector  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_symbol(symbol: str):
    """Train a single symbol"""
    logger.info(f"🚀 Starting training for {symbol}...")
    
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
        
        # Train enhanced models
        logger.info(f"🎯 Training enhanced models for {symbol}...")
        try:
            result = ml.train_enhanced_models(symbol, df)
            if result:
                logger.info(f"✅ {symbol}: Training completed successfully")
                
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
                
                return True
            else:
                logger.error(f"❌ {symbol}: Training failed")
                return False
        except Exception as e:
            logger.error(f"❌ {symbol}: Training error: {e}", exc_info=True)
            return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_single_symbol.py <SYMBOL>")
        print("Example: python train_single_symbol.py AGYO")
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    logger.info(f"=" * 80)
    logger.info(f"🎯 Single Symbol Training: {symbol}")
    logger.info(f"=" * 80)
    
    success = train_symbol(symbol)
    
    if success:
        logger.info(f"✅ {symbol}: Training completed successfully!")
        sys.exit(0)
    else:
        logger.error(f"❌ {symbol}: Training failed!")
        sys.exit(1)

