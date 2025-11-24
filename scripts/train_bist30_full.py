#!/usr/bin/env python3
"""
BIST30 Full Retrain - Tüm iyileştirmeler dahil
Ensemble + Regime + Adaptive + Deadband + Pattern Weighting + Cap
"""
import os
import sys
import logging

sys.path.insert(0, '/opt/bist-pattern')

# Environment setup
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
os.environ.setdefault('TRANSFORMERS_CACHE', '/opt/bist-pattern/.cache/huggingface')
os.environ.setdefault('HF_HOME', '/opt/bist-pattern/.cache/huggingface')

from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_BIST30_SYMBOLS = [
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
    logger.info("=" * 80)
    logger.info("BIST30 FULL RETRAIN - TÜM İYİLEŞTİRMELER DAHİL")
    logger.info("=" * 80)
    # Optional env override for symbols (comma-separated)
    symbols_env = os.getenv('ML_SYMBOLS', '').strip()
    if symbols_env:
        symbols_list = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
    else:
        symbols_list = list(DEFAULT_BIST30_SYMBOLS)
    logger.info(f"Semboller: {len(symbols_list)} adet")
    logger.info(f"External Features: {os.getenv('ENABLE_EXTERNAL_FEATURES', '0')}")
    logger.info(f"FinGPT Features: {os.getenv('ENABLE_FINGPT_FEATURES', '0')}")
    logger.info(f"Smart Ensemble: {os.getenv('ML_USE_SMART_ENSEMBLE', '1')}")
    logger.info(f"Regime Detection: {os.getenv('ML_USE_REGIME_DETECTION', '1')}")
    logger.info(f"Adaptive Deadband: {os.getenv('ML_ADAPTIVE_DEADBAND_MODE', 'std')}")
    logger.info("=" * 80)
    logger.info("")
    
    # Database connection
    db_url = os.getenv('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})
    
    ml = get_enhanced_ml_system()
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, sym in enumerate(symbols_list, 1):
        logger.info(f"[{i}/{len(symbols_list)}] Training {sym}...")
        
        try:
            df = fetch_stock_data(engine, sym, days=0)  # Full history
            
            if df is None or len(df) < 200:
                logger.warning(f"  ⚠️ Insufficient data: {len(df) if df is not None else 0} days")
                skip_count += 1
                continue
            
            logger.info("  📊 Data: %d days", len(df))
            
            success = ml.train_enhanced_models(sym, df)
            
            if success:
                ml.save_enhanced_models(sym)
                success_count += 1
                logger.info("  ✅ Success")
            else:
                fail_count += 1
                logger.error("  ❌ Training failed")
                
        except Exception as e:
            fail_count += 1
            logger.error(f"  ❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("BIST30 RETRAIN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Success: {success_count}/{len(symbols_list)}")
    logger.info(f"❌ Failed:  {fail_count}/{len(symbols_list)}")
    logger.info(f"⚠️  Skipped: {skip_count}/{len(symbols_list)}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
