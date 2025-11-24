#!/usr/bin/env python3
"""
BIST30 Production Training - bulk_train_all.py mantığını kullanır
Detaylı loglama + sorulara cevap verebilecek format
"""
import os
import sys
import logging
from datetime import datetime
import json

sys.path.insert(0, '/opt/bist-pattern')

# ⚡ Environment variables BEFORE imports
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
os.environ.setdefault('TRANSFORMERS_CACHE', '/opt/bist-pattern/.cache/huggingface')
os.environ.setdefault('HF_HOME', '/opt/bist-pattern/.cache/huggingface')

# ⚡ İYİLEŞTİRMELER (bulk_train_all.py'den önce set edilmeli)
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
os.environ['ENABLE_EXTERNAL_FEATURES'] = '0'  # Geçmiş veri yok
os.environ['ENABLE_FINGPT_FEATURES'] = '0'    # Geçmiş veri yok
os.environ['FORCE_FULL_RETRAIN'] = '1'  # Training gate bypass
os.environ['ML_USE_ADAPTIVE_LEARNING'] = '1'  # ⚡ GERÇEK ADAPTIVE LEARNING!
os.environ['DATABASE_URL'] = 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db'  # Macro features

from enhanced_ml_system import get_enhanced_ml_system  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402
import pandas as pd  # noqa: E402

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
    logger.info("BIST30 PRODUCTION TRAINING")
    logger.info("=" * 100)
    logger.info(f"Başlangıç: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Semboller: {len(BIST30_SYMBOLS)} adet")
    logger.info("")
    
    # Environment check
    logger.info("🔧 İYİLEŞTİRMELER:")
    logger.info(f"  Smart Ensemble: {os.getenv('ML_USE_SMART_ENSEMBLE')}")
    logger.info(f"  Regime Detection: {os.getenv('ML_USE_REGIME_DETECTION')}")
    logger.info(f"  Adaptive Deadband: {os.getenv('ML_ADAPTIVE_DEADBAND_MODE')}")
    logger.info(f"  Adaptive K (1d/3d/7d): {os.getenv('ML_ADAPTIVE_K_1D')}/{os.getenv('ML_ADAPTIVE_K_3D')}/{os.getenv('ML_ADAPTIVE_K_7D')}")
    logger.info(f"  Pattern Weight (1d/3d/7d): {os.getenv('ML_PATTERN_WEIGHT_SCALE_1D')}/{os.getenv('ML_PATTERN_WEIGHT_SCALE_3D')}/{os.getenv('ML_PATTERN_WEIGHT_SCALE_7D')}")
    logger.info(f"  Cap (3d): {os.getenv('ML_CAP_PCTL_3D')} percentile")
    logger.info(f"  External Features: {os.getenv('ENABLE_EXTERNAL_FEATURES')} (geçmiş veri yok)")
    logger.info("=" * 100)
    logger.info("")
    
    # Database
    db_url = os.getenv('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})
    
    ml = get_enhanced_ml_system()
    
    results = []
    success_count = 0
    fail_count = 0
    skip_count = 0
    total_errors = []
    
    for i, sym in enumerate(BIST30_SYMBOLS, 1):
        symbol_start = datetime.now()
        logger.info("")
        logger.info(f"[{i}/{len(BIST30_SYMBOLS)}] 🔄 {sym}")
        logger.info("-" * 100)
        
        try:
            # Fetch data
            df = fetch_stock_data(engine, sym, days=0)
            
            if df is None or len(df) < 200:
                logger.warning(f"⚠️ Yetersiz veri: {len(df) if df is not None else 0} gün")
                skip_count += 1
                results.append({
                    'symbol': sym,
                    'status': 'skipped',
                    'reason': 'insufficient_data',
                    'data_days': len(df) if df is not None else 0
                })
                continue
            
            logger.info(f"📊 Veri: {len(df)} gün ({df.index.min().date()} - {df.index.max().date()})")
            
            # Train
            logger.info("🧠 Model eğitimi başlıyor...")
            success = ml.train_enhanced_models(sym, df)
            
            if success:
                ml.save_enhanced_models(sym)
                symbol_duration = (datetime.now() - symbol_start).total_seconds()
                success_count += 1
                
                results.append({
                    'symbol': sym,
                    'status': 'success',
                    'data_days': len(df),
                    'date_range': f"{df.index.min().date()} - {df.index.max().date()}",
                    'duration_seconds': round(symbol_duration, 1)
                })
                
                logger.info(f"✅ Başarılı ({symbol_duration:.1f}s)")
            else:
                fail_count += 1
                results.append({
                    'symbol': sym,
                    'status': 'failed',
                    'reason': 'training_returned_false',
                    'data_days': len(df)
                })
                logger.error("❌ Başarısız (training returned False)")
                total_errors.append(f"{sym}: training returned False")
                
        except Exception as e:
            fail_count += 1
            error_msg = str(e)
            results.append({
                'symbol': sym,
                'status': 'error',
                'error': error_msg,
                'data_days': len(df) if 'df' in locals() and df is not None else 0
            })
            logger.error(f"❌ Hata: {e}")
            total_errors.append(f"{sym}: {error_msg}")
    
    # Summary
    total_duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("")
    logger.info("=" * 100)
    logger.info("📊 EĞİTİM SONUÇLARI")
    logger.info("=" * 100)
    logger.info(f"✅ Başarılı: {success_count}/{len(BIST30_SYMBOLS)} ({success_count/len(BIST30_SYMBOLS)*100:.1f}%)")
    logger.info(f"❌ Başarısız: {fail_count}/{len(BIST30_SYMBOLS)}")
    logger.info(f"⚠️  Atlanan: {skip_count}/{len(BIST30_SYMBOLS)}")
    logger.info(f"⏱️  Toplam süre: {total_duration:.1f}s ({total_duration/60:.1f} dakika)")
    
    if success_count > 0:
        avg_duration = sum(r['duration_seconds'] for r in results if r['status'] == 'success') / success_count
        logger.info(f"⏱️  Ortalama (sembol başına): {avg_duration:.1f}s")
    
    logger.info("=" * 100)
    
    # Error summary
    if total_errors:
        logger.info("")
        logger.info("🚨 HATALAR:")
        for err in total_errors[:10]:  # İlk 10 hata
            logger.info(f"  - {err}")
        if len(total_errors) > 10:
            logger.info(f"  ... ve {len(total_errors) - 10} hata daha")
    
    # Save detailed results
    results_file = f'/opt/bist-pattern/logs/bist30_prod_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': total_duration,
            'success_count': success_count,
            'fail_count': fail_count,
            'skip_count': skip_count,
            'improvements': {
                'smart_ensemble': os.getenv('ML_USE_SMART_ENSEMBLE'),
                'regime_detection': os.getenv('ML_USE_REGIME_DETECTION'),
                'adaptive_deadband': os.getenv('ML_ADAPTIVE_DEADBAND_MODE'),
                'pattern_weighting': 'enabled',
                'cap_calibration': '3d only',
                'external_features': 'disabled (no historical data)'
            },
            'results': results,
            'errors': total_errors
        }, f, indent=2)
    
    logger.info(f"📄 Detaylı sonuçlar: {results_file}")
    logger.info("")
    
    if success_count == len(BIST30_SYMBOLS):
        logger.info("🎉 TÜM SEMBOLLER BAŞARIYLA EĞİTİLDİ!")
        return 0
    elif success_count > 0:
        logger.warning(f"⚠️ KISMİ BAŞARI: {success_count}/{len(BIST30_SYMBOLS)}")
        return 1
    else:
        logger.error("🚨 TAM BAŞARISIZLIK!")
        return 2


if __name__ == '__main__':
    sys.exit(main())
