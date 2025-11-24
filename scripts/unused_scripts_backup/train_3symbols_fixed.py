#!/usr/bin/env python3
"""
Train 3 symbols with FIXED code (directional loss + sample weights)
"""
import os
import sys
import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

# Set environment
os.environ['PYTHONPATH'] = '/opt/bist-pattern'
os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db'
os.environ['FORCE_FULL_RETRAIN'] = '1'
os.environ['ML_MAX_MODEL_AGE_DAYS'] = '0'

# All improvements ON
os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
os.environ['ML_USE_REGIME_DETECTION'] = '1'
os.environ['ML_USE_ADAPTIVE_LEARNING'] = '1'
os.environ['ML_ADAPTIVE_DEADBAND_MODE'] = 'std'
os.environ['ML_ADAPTIVE_K_1D'] = '2.0'
os.environ['ML_ADAPTIVE_K_3D'] = '1.8'
os.environ['ML_ADAPTIVE_K_7D'] = '1.6'
os.environ['ML_PATTERN_WEIGHT_SCALE_1D'] = '1.2'
os.environ['ML_PATTERN_WEIGHT_SCALE_3D'] = '1.15'
os.environ['ML_PATTERN_WEIGHT_SCALE_7D'] = '1.1'
os.environ['ML_CAP_PCTL_3D'] = '92.5'
os.environ['ENABLE_EXTERNAL_FEATURES'] = '0'
os.environ['ENABLE_FINGPT_FEATURES'] = '0'
os.environ['ML_HORIZONS'] = '1,3,7,14,30'

# Directional loss ON (default is 1, but explicit)
os.environ['ML_USE_DIRECTIONAL_LOSS'] = '1'

sys.path.insert(0, '/opt/bist-pattern')

from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_stock_data(symbol: str, engine) -> pd.DataFrame:
    """Fetch stock data from database"""
    query = text("""
        SELECT p.date, p.open_price as open, p.high_price as high,
               p.low_price as low, p.close_price as close, p.volume
        FROM stock_prices p
        JOIN stocks s ON p.stock_id = s.id
        WHERE s.symbol = :symbol
        ORDER BY p.date
    """)
    
    df = pd.read_sql(query, engine, params={'symbol': symbol})
    if df.empty:
        return df
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def main():
    symbols = ['GARAN', 'AKBNK', 'EREGL']
    
    logger.info("=" + "=" * 79)
    logger.info("🚀 TRAINING 3 SYMBOLS WITH FIXED CODE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("✅ Fixes applied:")
    logger.info("  1. Directional loss preserved in final model")
    logger.info("  2. Sample weights used in final fit")
    logger.info("  3. Horizon features strict mode supported")
    logger.info("  4. Cron gate fixed (FORCE_FULL_RETRAIN=1)")
    logger.info("")
    logger.info(f"📊 Symbols: {', '.join(symbols)}")
    logger.info("📊 Horizons: 1d, 3d, 7d, 14d, 30d")
    logger.info("")
    
    # Create engine
    engine = create_engine(os.environ['DATABASE_URL'])
    
    # Get ML system
    ml = get_enhanced_ml_system()
    
    results = {}
    
    for i, symbol in enumerate(symbols, 1):
        logger.info("=" * 80)
        logger.info(f"[{i}/{len(symbols)}] TRAINING: {symbol}")
        logger.info("=" * 80)
        
        try:
            # Fetch data
            logger.info(f"📥 Fetching data for {symbol}...")
            df = fetch_stock_data(symbol, engine)
            
            if df.empty:
                logger.error(f"❌ No data for {symbol}")
                continue
            
            logger.info(f"✅ Fetched {len(df)} days of data")
            logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
            
            # Train
            logger.info(f"🔧 Training models for {symbol}...")
            start_time = datetime.now()
            
            result = ml.train_enhanced_models(symbol, df)
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            logger.info(f"✅ Training completed in {duration:.1f} minutes")
            
            # Extract metrics
            if result and 'metrics' in result:
                metrics = result['metrics']
                logger.info("")
                logger.info(f"📊 Metrics for {symbol}:")
                for horizon, m in metrics.items():
                    if isinstance(m, dict):
                        dirhit = m.get('dir_hit_pct', 0)
                        r2 = m.get('r2', 0)
                        logger.info(f"   {horizon}: DirHit={dirhit:.1f}%, R²={r2:.3f}")
                
                results[symbol] = metrics
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary
    logger.info("=" * 80)
    logger.info("📊 TRAINING SUMMARY")
    logger.info("=" * 80)
    
    if results:
        # Calculate averages per horizon
        horizon_stats = {}
        for symbol, metrics in results.items():
            for horizon, m in metrics.items():
                if isinstance(m, dict):
                    if horizon not in horizon_stats:
                        horizon_stats[horizon] = {'dirhit': [], 'r2': []}
                    horizon_stats[horizon]['dirhit'].append(m.get('dir_hit_pct', 0))
                    horizon_stats[horizon]['r2'].append(m.get('r2', 0))
        
        logger.info("")
        logger.info("Average metrics across 3 symbols:")
        for horizon in ['1d', '3d', '7d', '14d', '30d']:
            if horizon in horizon_stats:
                avg_dirhit = sum(horizon_stats[horizon]['dirhit']) / len(horizon_stats[horizon]['dirhit'])
                avg_r2 = sum(horizon_stats[horizon]['r2']) / len(horizon_stats[horizon]['r2'])
                logger.info(f"  {horizon}: DirHit={avg_dirhit:.1f}%, R²={avg_r2:.3f}")
    
    logger.info("")
    logger.info("✅ Training completed!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run walk-forward: python scripts/pilot_walkforward_3symbols_simple.py")
    logger.info("  2. Compare with previous results")
    logger.info("  3. If +10pp gain, proceed with full BIST30 retrain")
    logger.info("")


if __name__ == '__main__':
    main()
