#!/usr/bin/env python3
"""
BIST30 Detailed Training - Tüm iyileştirmeler + Detaylı logging
"""
import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, '/opt/bist-pattern')

# ⚡ CRITICAL: Set environment variables BEFORE importing
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
os.environ.setdefault('TRANSFORMERS_CACHE', '/opt/bist-pattern/.cache/huggingface')
os.environ.setdefault('HF_HOME', '/opt/bist-pattern/.cache/huggingface')

# ⚡ İYİLEŞTİRMELER - Environment variables
os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
os.environ['ML_USE_REGIME_DETECTION'] = '1'
os.environ['ML_ADAPTIVE_DEADBAND_MODE'] = 'std'
os.environ['ML_ADAPTIVE_K_1D'] = '2.0'
os.environ['ML_ADAPTIVE_K_3D'] = '1.8'
os.environ['ML_ADAPTIVE_K_7D'] = '1.6'
os.environ['ML_PATTERN_WEIGHT_SCALE_1D'] = '1.2'
os.environ['ML_PATTERN_WEIGHT_SCALE_3D'] = '1.15'
os.environ['ML_PATTERN_WEIGHT_SCALE_7D'] = '1.1'
os.environ['ML_CAP_PCTL_3D'] = '92.5'
os.environ['ENABLE_EXTERNAL_FEATURES'] = '1'
os.environ['ENABLE_FINGPT_FEATURES'] = '1'

from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
import pandas as pd  # noqa: E402

# Detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BIST30_SYMBOLS = [
    'AKBNK', 'ARCLK', 'ASELS', 'BIMAS', 'EKGYO', 'ENJSA', 'EREGL',
    'FROTO', 'GARAN', 'HEKTS', 'ISCTR', 'KCHOL', 'KOZAL', 'KOZAA',
    'KRDMD', 'PETKM', 'PGSUS', 'SAHOL', 'SASA', 'SISE', 'TAVHL',
    'TCELL', 'THYAO', 'TOASO', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK',
    'ODAS', 'SMRTG'
]


def fetch_stock_data(engine, symbol, days=0):
    """Fetch stock data from database"""
    query = text("""
        SELECT p.date, p.open_price, p.high_price, p.low_price, p.close_price, p.volume
        FROM stock_prices p
        JOIN stocks s ON s.id = p.stock_id
        WHERE s.symbol = :sym
        ORDER BY p.date DESC
    """)
    
    limit = days if days > 0 else 99999
    
    with engine.connect() as conn:
        rows = conn.execute(query, {"sym": symbol}).fetchmany(limit)
    
    if not rows:
        return None
    
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
    return df


def main():
    start_time = datetime.now()
    
    logger.info("=" * 100)
    logger.info("BIST30 DETAILED TRAINING - TÜM İYİLEŞTİRMELER + DETAYLI LOGGING")
    logger.info("=" * 100)
    logger.info(f"Semboller: {len(BIST30_SYMBOLS)} adet")
    logger.info("")
    
    # Environment variables check
    logger.info("🔧 ENVIRONMENT VARIABLES:")
    logger.info(f"  ML_USE_SMART_ENSEMBLE: {os.getenv('ML_USE_SMART_ENSEMBLE')}")
    logger.info(f"  ML_USE_REGIME_DETECTION: {os.getenv('ML_USE_REGIME_DETECTION')}")
    logger.info(f"  ML_ADAPTIVE_DEADBAND_MODE: {os.getenv('ML_ADAPTIVE_DEADBAND_MODE')}")
    logger.info(f"  ML_ADAPTIVE_K_1D: {os.getenv('ML_ADAPTIVE_K_1D')}")
    logger.info(f"  ML_ADAPTIVE_K_3D: {os.getenv('ML_ADAPTIVE_K_3D')}")
    logger.info(f"  ML_ADAPTIVE_K_7D: {os.getenv('ML_ADAPTIVE_K_7D')}")
    logger.info(f"  ML_PATTERN_WEIGHT_SCALE_1D: {os.getenv('ML_PATTERN_WEIGHT_SCALE_1D')}")
    logger.info(f"  ML_PATTERN_WEIGHT_SCALE_3D: {os.getenv('ML_PATTERN_WEIGHT_SCALE_3D')}")
    logger.info(f"  ML_PATTERN_WEIGHT_SCALE_7D: {os.getenv('ML_PATTERN_WEIGHT_SCALE_7D')}")
    logger.info(f"  ML_CAP_PCTL_3D: {os.getenv('ML_CAP_PCTL_3D')}")
    logger.info(f"  ENABLE_EXTERNAL_FEATURES: {os.getenv('ENABLE_EXTERNAL_FEATURES')}")
    logger.info(f"  ENABLE_FINGPT_FEATURES: {os.getenv('ENABLE_FINGPT_FEATURES')}")
    logger.info("=" * 100)
    logger.info("")
    
    # Database connection
    db_url = os.getenv('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    engine = create_engine(db_url)
    
    ml = get_enhanced_ml_system()
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    error_details = []
    
    for i, sym in enumerate(BIST30_SYMBOLS, 1):
        symbol_start = datetime.now()
        logger.info("")
        logger.info("=" * 100)
        logger.info(f"[{i}/{len(BIST30_SYMBOLS)}] 🔄 TRAINING: {sym}")
        logger.info("=" * 100)
        
        try:
            # Fetch data
            logger.info(f"📊 Fetching data for {sym}...")
            df = fetch_stock_data(engine, sym, days=0)
            
            if df is None or len(df) < 200:
                logger.warning(f"⚠️ Insufficient data: {len(df) if df is not None else 0} days (min: 200)")
                skip_count += 1
                continue
            
            logger.info(f"✅ Data fetched: {len(df)} days")
            logger.info(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
            logger.info(f"   Columns: {list(df.columns)}")
            logger.info("")
            
            # Train models
            logger.info(f"🧠 Starting model training for {sym}...")
            success = ml.train_enhanced_models(sym, df)
            
            if success:
                # Save models
                logger.info(f"💾 Saving models for {sym}...")
                ml.save_enhanced_models(sym)
                success_count += 1
                
                symbol_duration = (datetime.now() - symbol_start).total_seconds()
                logger.info(f"✅ SUCCESS: {sym} trained in {symbol_duration:.1f} seconds")
            else:
                fail_count += 1
                error_details.append(f"{sym}: Training returned False")
                logger.error(f"❌ FAILED: {sym} - Training returned False")
                
        except Exception as e:
            fail_count += 1
            error_msg = f"{sym}: {str(e)}"
            error_details.append(error_msg)
            logger.error(f"❌ ERROR: {sym}")
            logger.error(f"   Exception: {e}")
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
    
    # Summary
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 100)
    logger.info("📊 TRAINING SUMMARY")
    logger.info("=" * 100)
    logger.info(f"✅ Success: {success_count}/{len(BIST30_SYMBOLS)} ({success_count/len(BIST30_SYMBOLS)*100:.1f}%)")
    logger.info(f"❌ Failed:  {fail_count}/{len(BIST30_SYMBOLS)} ({fail_count/len(BIST30_SYMBOLS)*100:.1f}%)")
    logger.info(f"⚠️  Skipped: {skip_count}/{len(BIST30_SYMBOLS)} ({skip_count/len(BIST30_SYMBOLS)*100:.1f}%)")
    logger.info(f"⏱️  Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    logger.info(f"⏱️  Average per symbol: {total_duration/len(BIST30_SYMBOLS):.1f} seconds")
    
    if error_details:
        logger.info("")
        logger.info("❌ ERROR DETAILS:")
        for err in error_details:
            logger.info(f"   - {err}")
    
    logger.info("=" * 100)
    
    # Final check
    if success_count == len(BIST30_SYMBOLS):
        logger.info("🎉 ALL SYMBOLS TRAINED SUCCESSFULLY!")
    elif success_count > 0:
        logger.warning(f"⚠️ PARTIAL SUCCESS: {success_count}/{len(BIST30_SYMBOLS)} symbols trained")
    else:
        logger.error("🚨 COMPLETE FAILURE: No symbols trained successfully")


if __name__ == '__main__':
    main()

