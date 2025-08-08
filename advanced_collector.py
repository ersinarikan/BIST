import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from models import db, Stock, StockPrice
from app import app
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedBISTCollector:
    def __init__(self):
        self.session_requests = 0
        self.max_workers = 5  # Paralel işlem sayısı
        self.delay_range = (1, 3)  # Rastgele gecikme
        
    def get_bist_symbol(self, symbol):
        """BIST sembolünü Yahoo Finance formatına çevir"""
        return f"{symbol}.IS"
    
    def collect_single_stock(self, symbol, period="3mo"):
        """Tek hisse için veri topla"""
        try:
            yf_symbol = self.get_bist_symbol(symbol)
            stock = yf.Ticker(yf_symbol)
            
            # Veri çek
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"Veri yok: {symbol}")
                return {'symbol': symbol, 'success': False, 'records': 0}
            
            # DataFrame'i temizle
            hist.reset_index(inplace=True)
            
            # Veritabanına kaydet
            saved_records = self.save_stock_data(symbol, hist)
            
            # Rastgele gecikme
            delay = random.uniform(*self.delay_range)
            time.sleep(delay)
            
            return {'symbol': symbol, 'success': True, 'records': saved_records}
            
        except Exception as e:
            logger.error(f"Hata {symbol}: {e}")
            return {'symbol': symbol, 'success': False, 'error': str(e), 'records': 0}
    
    def save_stock_data(self, symbol, data):
        """Veriyi PostgreSQL'e kaydet"""
        try:
            with app.app_context():
                stock_obj = Stock.query.filter_by(symbol=symbol).first()
                if not stock_obj:
                    logger.warning(f"Hisse DB'de yok: {symbol}")
                    return 0
                
                saved_count = 0
                for _, row in data.iterrows():
                    # Mevcut veriyi kontrol et
                    existing = StockPrice.query.filter_by(
                        stock_id=stock_obj.id,
                        date=row['Date'].date()
                    ).first()
                    
                    if not existing:
                        price_record = StockPrice(
                            stock_id=stock_obj.id,
                            date=row['Date'].date(),
                            open_price=float(row['Open']),
                            high_price=float(row['High']),
                            low_price=float(row['Low']),
                            close_price=float(row['Close']),
                            volume=int(row['Volume']) if row['Volume'] > 0 else 0
                        )
                        db.session.add(price_record)
                        saved_count += 1
                
                if saved_count > 0:
                    db.session.commit()
                
                return saved_count
                
        except Exception as e:
            logger.error(f"Kaydetme hatası {symbol}: {e}")
            try:
                db.session.rollback()
            except Exception as rollback_error:
                logger.error(f"Rollback hatası {symbol}: {rollback_error}")
            return 0
    
    def collect_all_stocks_parallel(self, batch_size=50):
        """Tüm hisseleri paralel olarak topla"""
        
        # Veritabanından tüm hisseleri al
        with app.app_context():
            all_stocks = Stock.query.all()
            symbols = [stock.symbol for stock in all_stocks]
        
        logger.info(f"Toplam {len(symbols)} hisse için veri toplama başlıyor...")
        
        # Batch'lere böl
        batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        total_success = 0
        total_failed = 0
        total_records = 0
        
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Batch {batch_num}/{len(batches)} işleniyor ({len(batch)} hisse)...")
            
            # Paralel işleme
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Görevleri başlat
                future_to_symbol = {
                    executor.submit(self.collect_single_stock, symbol): symbol 
                    for symbol in batch
                }
                
                # Sonuçları topla
                batch_results = []
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        batch_results.append(result)
                        
                        if result['success']:
                            total_success += 1
                            total_records += result['records']
                            logger.info(f"✅ {symbol}: {result['records']} kayıt")
                        else:
                            total_failed += 1
                            logger.warning(f"❌ {symbol}: başarısız")
                            
                    except Exception as e:
                        total_failed += 1
                        logger.error(f"❌ {symbol}: {e}")
            
            # Batch'ler arası bekleme
            if batch_num < len(batches):
                logger.info(f"Batch {batch_num} tamamlandı. 30 saniye bekleniyor...")
                time.sleep(30)
        
        logger.info(f"""
        📊 VERİ TOPLAMA RAPORU:
        ✅ Başarılı: {total_success}
        ❌ Başarısız: {total_failed}
        📈 Toplam kayıt: {total_records}
        """)
        
        return {
            'success_count': total_success,
            'failed_count': total_failed,
            'total_records': total_records
        }
    
    def collect_priority_stocks(self):
        """Öncelikli hisseleri topla (hızlı test için)"""
        priority_symbols = [
            'THYAO', 'AKBNK', 'GARAN', 'EREGL', 'ASELS', 'TUPRS', 'SAHOL', 'VAKBN',
            'KOZAL', 'SISE', 'TCELL', 'MGROS', 'ARCELIK', 'BIMAS', 'KRDMD', 'TOASO',
            'PETKM', 'SASA', 'DOHOL', 'TKFEN', 'ENKAI', 'TAVHL', 'ISCTR', 'HALKB',
            'FROTO', 'OTKAR', 'PGSUS', 'DOAS', 'GUBRF', 'ULKER', 'TTKOM', 'AEFES'
        ]
        
        logger.info(f"Öncelikli {len(priority_symbols)} hisse için veri toplama...")
        
        total_success = 0
        total_failed = 0
        total_records = 0
        
        for symbol in priority_symbols:
            result = self.collect_single_stock(symbol, period="6mo")  # 6 ay veri
            
            if result['success']:
                total_success += 1
                total_records += result['records']
                logger.info(f"✅ {symbol}: {result['records']} kayıt")
            else:
                total_failed += 1
                logger.warning(f"❌ {symbol}: başarısız")
        
        logger.info(f"Öncelikli hisseler tamamlandı: {total_success} başarılı, {total_failed} başarısız, {total_records} kayıt")
        
        return {
            'success_count': total_success,
            'failed_count': total_failed,
            'total_records': total_records
        }

def main():
    collector = AdvancedBISTCollector()
    
    # Önce priority hisseleri topla
    logger.info("🚀 BIST Veri Toplama Sistemi Başlatılıyor...")
    result = collector.collect_priority_stocks()
    
    print(f"""
    📊 SONUÇ:
    ✅ Başarılı: {result['success_count']}
    ❌ Başarısız: {result['failed_count']}
    📈 Toplam Kayıt: {result['total_records']}
    """)

if __name__ == "__main__":
    main()
