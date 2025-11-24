#!/usr/bin/env python3
"""
Train models with HPO best parameters and evaluate DirHit (all features ON)
This script:
1. Loads completed HPO results from JSON files
2. For each symbol/horizon pair:
   - Sets best parameters as environment variables
   - Trains model with all features ON (production-like)
   - Evaluates DirHit using walk-forward validation
   - Saves results to JSON and CSV
"""
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/opt/bist-pattern')
os.environ['PYTHONPATH'] = '/opt/bist-pattern'
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# Ensure DATABASE_URL is set
if 'DATABASE_URL' not in os.environ:
    os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@127.0.0.1:5432/bist_pattern_db'

# Import after path setup
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Import Flask app and modules (requires app context)
from app import app
from pattern_detector import HybridPatternDetector
from enhanced_ml_system import get_enhanced_ml_system

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DirHit calculation (same as optuna_hpo_pilot.py)
def compute_returns(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Compute forward returns."""
    c = df['close'].values.astype(float)
    fwd = np.empty_like(c)
    fwd[:] = np.nan
    if len(c) > horizon:
        future = c[horizon:]
        past = c[:-horizon]
        valid_mask = past > 0
        fwd[:-horizon][valid_mask] = (future[valid_mask] / past[valid_mask] - 1.0)
        fwd = np.where(np.isinf(fwd), np.nan, fwd)
    return fwd

def dirhit(y_true: np.ndarray, y_pred: np.ndarray, thr: float = 0.005) -> float:
    """Calculate directional hit rate."""
    yt = np.where(np.abs(y_true) < thr, 0, np.sign(y_true))
    yp = np.where(np.abs(y_pred) < thr, 0, np.sign(yp))
    m = ~np.isnan(yt) & ~np.isnan(yp)
    if m.sum() == 0:
        return 0.0
    correct = (yt[m] == yp[m]).sum()
    return 100.0 * float(correct) / float(m.sum())

def fetch_prices(engine, symbol: str, limit: int = 1200) -> pd.DataFrame:
    """Fetch stock prices from database."""
    query = text("""
        SELECT p.date, p.open_price, p.high_price, p.low_price, p.close_price, p.volume
        FROM stock_prices p
        JOIN stocks s ON s.id = p.stock_id
        WHERE s.symbol = :sym
        ORDER BY p.date ASC
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(query, {"sym": symbol}).fetchmany(limit)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            'date': r[0],
            'open': float(r[1]),
            'high': float(r[2]),
            'low': float(r[3]),
            'close': float(r[4]),
            'volume': float(r[5]) if r[5] is not None else 0.0,
        }
        for r in rows
    ])
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'close', 'volume']]

def load_hpo_params_from_json(json_file: str) -> dict:
    """Load HPO parameters from JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data.get('best_params', {})
    except Exception as e:
        logger.error(f"Failed to load HPO params from {json_file}: {e}")
        return {}

def set_hpo_params_as_env(params: dict, horizon: int):
    """Set HPO parameters as environment variables."""
    # XGBoost params
    for key, val in params.items():
        if key.startswith('xgb_'):
            env_key = f"OPTUNA_XGB_{key.replace('xgb_', '').upper()}"
            os.environ[env_key] = str(val)
        elif key.startswith('lgb_'):
            env_key = f"OPTUNA_LGB_{key.replace('lgb_', '').upper()}"
            os.environ[env_key] = str(val)
        elif key.startswith('cat_'):
            env_key = f"OPTUNA_CAT_{key.replace('cat_', '').upper()}"
            os.environ[env_key] = str(val)
        elif key == 'adaptive_k':
            os.environ[f'ML_ADAPTIVE_K_{horizon}D'] = str(val)
        elif key == 'pattern_weight':
            os.environ[f'ML_PATTERN_WEIGHT_SCALE_{horizon}D'] = str(val)

def train_and_evaluate(symbol: str, horizon: int, best_params: dict, engine, det: HybridPatternDetector) -> dict:
    """Train model with best params and evaluate DirHit."""
    try:
        # Fetch data using HybridPatternDetector (same as production)
        df = det.get_stock_data(symbol, days=0)
        if df is None or df.empty or len(df) < 60:
            logger.warning(f"{symbol} {horizon}d: Insufficient data")
            return None
        
        # Train/validation split (same as HPO)
        total_days = len(df)
        min_test_days = horizon + 10
        
        if total_days >= 240:
            split_idx = total_days - 120
        elif total_days >= 180:
            split_idx = int(total_days * 2 / 3)
        else:
            split_idx = max(1, int(total_days * 2 / 3))
        
        if total_days - split_idx < min_test_days:
            split_idx = total_days - min_test_days
            if split_idx < 1:
                logger.warning(f"{symbol} {horizon}d: Insufficient data for horizon")
                return None
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        if len(test_df) < min_test_days:
            logger.warning(f"{symbol} {horizon}d: Test set too small")
            return None
        
        logger.info(f"{symbol} {horizon}d: {total_days} days total, {len(train_df)} train, {len(test_df)} validation")
        
        # Set HPO parameters as environment variables
        set_hpo_params_as_env(best_params, horizon)
        
        # Enable ALL features for production-like training
        os.environ['ML_USE_ADAPTIVE_LEARNING'] = '1'
        os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
        os.environ['ML_USE_STACKED_SHORT'] = '1'
        os.environ['ENABLE_META_STACKING'] = '0'  # ML_USE_STACKED_SHORT already handles it
        os.environ['ML_USE_REGIME_DETECTION'] = '1'
        os.environ['ENABLE_SEED_BAGGING'] = '1'
        os.environ['N_SEEDS'] = '3'
        os.environ['ENABLE_TALIB_PATTERNS'] = '1'
        os.environ['ML_HORIZONS'] = str(horizon)
        
        # Train model
        ml = get_enhanced_ml_system()
        ok = ml.train_enhanced_models(symbol, train_df)
        
        if not ok:
            logger.warning(f"{symbol} {horizon}d: Training failed")
            return None
        
        # Evaluate on test set (walk-forward)
        y_true = compute_returns(test_df, horizon)
        preds = np.full(len(test_df), np.nan, dtype=float)
        
        for t in range(len(test_df) - horizon):
            try:
                cur = pd.concat([train_df, test_df.iloc[: t + 1]], axis=0).copy()
                p = ml.predict_enhanced(symbol, cur)
                
                if not isinstance(p, dict):
                    continue
                
                key = f"{horizon}d"
                obj = p.get(key)
                if isinstance(obj, dict):
                    pred_price = obj.get('ensemble_prediction')
                    try:
                        if isinstance(pred_price, (int, float)):
                            last_close = float(cur['close'].iloc[-1])
                            if last_close > 0:
                                preds[t] = float(pred_price) / last_close - 1.0
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"{symbol} {horizon}d: Prediction error at t={t}: {e}")
                continue
        
        # Calculate DirHit
        dh = dirhit(y_true, preds)
        
        return {
            'symbol': symbol,
            'horizon': horizon,
            'best_params': best_params,
            'dirhit': float(dh) if not np.isnan(dh) else None,
            'train_days': len(train_df),
            'test_days': len(test_df),
            'total_days': total_days,
        }
        
    except Exception as e:
        logger.error(f"{symbol} {horizon}d: Error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def main():
    """Main function."""
    # Load completed HPO symbols
    completed_file = '/opt/bist-pattern/results/completed_hpo_symbols.json'
    if not os.path.exists(completed_file):
        logger.error(f"Completed HPO symbols file not found: {completed_file}")
        return 1
    
    with open(completed_file, 'r') as f:
        completed = json.load(f)
    
    # Database connection
    db_url = os.getenv('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})
    
    # Results storage
    results = []
    
    # Process each symbol/horizon pair
    with app.app_context():
        for horizon_str, items in completed.items():
            horizon = int(horizon_str.replace('d', ''))
            
            for item in items:
                symbol = item['symbol']
                json_file = item['json_file']
                best_params = item['best_params']
                
                logger.info(f"Processing {symbol} {horizon_str}...")
                
                result = train_and_evaluate(symbol, horizon, best_params, engine, det)
                
                if result:
                    results.append(result)
                    logger.info(f"✅ {symbol} {horizon_str}: DirHit = {result['dirhit']:.2f}%")
                else:
                    logger.warning(f"❌ {symbol} {horizon_str}: Failed")
    
    # Save results
    output_dir = Path('/opt/bist-pattern/results')
    output_dir.mkdir(exist_ok=True)
    
    # JSON output
    json_file = output_dir / 'features_off_dirhits.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Saved {len(results)} results to {json_file}")
    
    # CSV output
    if results:
        df_results = pd.DataFrame(results)
        csv_file = output_dir / 'features_off_dirhits.csv'
        df_results.to_csv(csv_file, index=False)
        logger.info(f"✅ Saved CSV to {csv_file}")
        
        # Summary
        avg_dirhit = df_results['dirhit'].mean()
        logger.info(f"📊 Average DirHit: {avg_dirhit:.2f}%")
        logger.info(f"📊 Total symbols: {len(df_results)}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

