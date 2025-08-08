"""
BIST Data Collector
Tüm BIST hisseleri için geçmiş veri toplama ve veritabanına kaydetme
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Local imports
try:
    from models import Stock, StockPrice, db
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

logger = logging.getLogger(__name__)

class BISTDataCollector:
    """BIST hisse verilerini toplayan sistem"""
    
    def __init__(self):
        self.bist_symbols = []
        self.collected_count = 0
        self.failed_count = 0
        self.lock = threading.Lock()
        
        # BIST 100 sembolleri (en aktif hisseler)
        self.bist100_symbols = [
            'THYAO', 'AKBNK', 'ISCTR', 'GARAN', 'SASA', 'TCELL', 'TUPRS', 'ARCLK',
            'BIMAS', 'HALKB', 'KCHOL', 'SAHOL', 'VAKBN', 'YKBNK', 'ASELS', 'KOZAL',
            'PGSUS', 'TAVHL', 'TKFEN', 'DOHOL', 'ENKAI', 'FROTO', 'GUBRF', 'KRDMD',
            'OTKAR', 'PETKM', 'SISE', 'TOASO', 'ULKER', 'VESTL', 'AEFES', 'AKSA',
            'ALARK', 'ANACM', 'ARDL', 'BANVT', 'BRYAT', 'CCOLA', 'CIMSA', 'DOAS',
            'ECILC', 'EGEEN', 'ENJSA', 'EREGL', 'GLYHO', 'GOLTS', 'GOODY', 'GOZDE',
            'ISGYO', 'ITTFH', 'KERVT', 'KLMSN', 'KONYA', 'KORDS', 'KOZAA', 'LOGO',
            'MAVI', 'MGROS', 'NTHOL', 'ODAS', 'OYAKC', 'PAPIL', 'PARSN', 'PINSU',
            'PRKAB', 'QUAGR', 'RTALB', 'SELEC', 'SKBNK', 'SOKM', 'TATGD', 'TMSN',
            'TRGYO', 'TSKB', 'TTKOM', 'TTRAK', 'TURSG', 'ULUKA', 'UZERB', 'YATAS',
            'ZOREN', 'ACIBD', 'ADEL', 'AGHOL', 'AHGAZ', 'AKSEN', 'ALBRK', 'ALFAS',
            'ALKIM', 'ALMAD', 'ANSGR', 'ASUZU', 'ATEKS', 'AVGYO', 'AVHOL', 'AVTUR',
            'BERA', 'BFREN', 'BILIM', 'BJKAS', 'BRISA', 'BRSAN', 'BSOKE', 'BTCIM',
            'BUCIM', 'CEMTS', 'CRDFA', 'CRFSA', 'CVKMD', 'CWENE', 'DERHL', 'DESPC',
            'DGKLB', 'DGSN', 'DMRGD', 'DOCO', 'DURDO', 'DYOBY', 'DZGYO', 'EGGUB'
        ]
        
        logger.info("📊 BIST Data Collector başlatıldı")
    
    def get_bist_symbols(self, use_api=True):
        """BIST sembollerini al"""
        try:
            if use_api:
                # Investing.com veya başka bir API'den güncel listeyi al
                # Şimdilik manuel liste kullanıyoruz
                self.bist_symbols = self.bist100_symbols.copy()
            else:
                self.bist_symbols = self.bist100_symbols.copy()
            
            logger.info(f"📋 {len(self.bist_symbols)} BIST hissesi listelendi")
            return self.bist_symbols
            
        except Exception as e:
            logger.error(f"BIST sembol listesi alma hatası: {e}")
            self.bist_symbols = self.bist100_symbols.copy()
            return self.bist_symbols
    
    def collect_symbol_data(self, symbol, period="2y", retry_count=3):
        """Tek bir hisse için veri topla"""
        try:
            ticker_symbol = f"{symbol}.IS"
            ticker = yf.Ticker(ticker_symbol)
            
            # Veri al
            for attempt in range(retry_count):
                try:
                    data = ticker.history(period=period)
                    if not data.empty:
                        break
                    else:
                        logger.warning(f"⚠️ {symbol} için veri boş (deneme {attempt + 1})")
                        time.sleep(1)
                except Exception as e:
                    logger.warning(f"⚠️ {symbol} veri alma hatası (deneme {attempt + 1}): {e}")
                    time.sleep(2)
            
            if data.empty:
                with self.lock:
                    self.failed_count += 1
                logger.error(f"❌ {symbol} için veri alınamadı")
                return None
            
            # Veriyi işle
            processed_data = []
            for date, row in data.iterrows():
                processed_data.append({
                    'symbol': symbol,
                    'date': date.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']) if row['Volume'] > 0 else 0
                })
            
            with self.lock:
                self.collected_count += 1
            
            logger.info(f"✅ {symbol}: {len(processed_data)} günlük veri toplandı")
            return processed_data
            
        except Exception as e:
            with self.lock:
                self.failed_count += 1
            logger.error(f"❌ {symbol} veri toplama hatası: {e}")
            return None
    
    def save_to_database(self, symbol_data):
        """Veriyi veritabanına kaydet"""
        try:
            if not DATABASE_AVAILABLE:
                logger.warning("Veritabanı mevcut değil, dosyaya kaydediliyor")
                return self.save_to_file(symbol_data)
            
            # Import app locally to avoid circular imports
            from app import app
            
            # Centralized app context management
            with app.app_context():
                symbol = symbol_data[0]['symbol']
                
                # Stock kaydını kontrol et/oluştur
                stock = Stock.query.filter_by(symbol=symbol).first()
                if not stock:
                    stock = Stock(
                        symbol=symbol,
                        name=f"{symbol} Hisse Senedi",
                        sector="Unknown",
                        created_at=datetime.now()
                    )
                    db.session.add(stock)
                    db.session.commit()
                
                # Mevcut fiyat verilerini kontrol et
                existing_dates = {
                    price.date for price in 
                    StockPrice.query.filter_by(stock_id=stock.id).all()
                }
                
                                # Yeni verileri ekle
                new_count = 0
                for data_point in symbol_data:
                    if data_point['date'] not in existing_dates:
                        price = StockPrice(
                            stock_id=stock.id,
                            date=data_point['date'],
                            open_price=data_point['open'],
                            high_price=data_point['high'],
                            low_price=data_point['low'],
                            close_price=data_point['close'],
                            volume=data_point['volume']
                        )
                        db.session.add(price)
                        new_count += 1

                # Batch commit - performans için tek seferde commit
                if new_count > 0:
                    db.session.commit()
                    logger.info(f"💾 {symbol}: {new_count} yeni veri veritabanına kaydedildi")
                else:
                    logger.info(f"ℹ️ {symbol}: Yeni veri bulunamadı")
                return True
                
        except Exception as e:
            logger.error(f"Veritabanı kaydetme hatası: {e}")
            try:
                db.session.rollback()
            except Exception as rollback_error:
                logger.error(f"Database rollback hatası: {rollback_error}")
            return False
    
    def save_to_file(self, symbol_data):
        """Veriyi CSV dosyasına kaydet (yedek)"""
        try:
            symbol = symbol_data[0]['symbol']
            filename = f"data/bist_{symbol}_data.csv"
            
            # Data klasörünü oluştur
            os.makedirs("data", exist_ok=True)
            
            df = pd.DataFrame(symbol_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"📄 {symbol}: Veriler {filename} dosyasına kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Dosya kaydetme hatası: {e}")
            return False
    
    def collect_all_data(self, max_workers=5, period="2y"):
        """Tüm BIST hisseleri için veri topla"""
        try:
            symbols = self.get_bist_symbols()
            total_symbols = len(symbols)
            
            logger.info(f"🚀 {total_symbols} hisse için veri toplama başlatılıyor...")
            
            # Reset counters
            self.collected_count = 0
            self.failed_count = 0
            
            start_time = datetime.now()
            
            # Paralel veri toplama
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_symbol = {
                    executor.submit(self.collect_symbol_data, symbol, period): symbol
                    for symbol in symbols
                }
                
                # Process results
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        symbol_data = future.result()
                        if symbol_data:
                            # Veritabanına kaydet
                            self.save_to_database(symbol_data)
                        
                        # Progress update
                        progress = (self.collected_count + self.failed_count) / total_symbols * 100
                        print(f"\r📊 İlerleme: {progress:.1f}% ({self.collected_count} başarılı, {self.failed_count} hata)", end="")
                        
                    except Exception as e:
                        logger.error(f"❌ {symbol} işleme hatası: {e}")
                        with self.lock:
                            self.failed_count += 1
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print()  # New line
            logger.info(f"🎯 Veri toplama tamamlandı!")
            logger.info(f"✅ Başarılı: {self.collected_count}")
            logger.info(f"❌ Başarısız: {self.failed_count}")
            logger.info(f"⏱️ Süre: {duration}")
            
            return {
                'success_count': self.collected_count,
                'failed_count': self.failed_count,
                'total_count': total_symbols,
                'duration': str(duration)
            }
            
        except Exception as e:
            logger.error(f"Toplu veri toplama hatası: {e}")
            return None
    
    def update_single_stock(self, symbol, days=30):
        """Tek bir hisse için güncel veri güncelle"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(f"{symbol}.IS")
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                processed_data = []
                for date, row in data.iterrows():
                    processed_data.append({
                        'symbol': symbol,
                        'date': date.date(),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']) if row['Volume'] > 0 else 0
                    })
                
                success = self.save_to_database(processed_data)
                logger.info(f"🔄 {symbol} güncellendi ({len(processed_data)} gün)")
                return success
            else:
                logger.warning(f"⚠️ {symbol} için güncel veri bulunamadı")
                return False
                
        except Exception as e:
            logger.error(f"❌ {symbol} güncelleme hatası: {e}")
            return False
    
    def get_collection_stats(self):
        """Toplanan veri istatistikleri"""
        try:
            if not DATABASE_AVAILABLE:
                return {'error': 'Veritabanı mevcut değil'}
            
            # Import app locally to avoid circular imports
            from app import app
            with app.app_context():
                total_stocks = Stock.query.count()
                total_prices = StockPrice.query.count()
                
                # En son veri tarihi
                latest_price = StockPrice.query.order_by(StockPrice.date.desc()).first()
                latest_date = latest_price.date if latest_price else None
                
                return {
                    'total_stocks': total_stocks,
                    'total_price_records': total_prices,
                    'latest_date': latest_date.isoformat() if latest_date else None,
                    'avg_records_per_stock': total_prices / total_stocks if total_stocks > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"İstatistik alma hatası: {e}")
            return {'error': str(e)}

# Global singleton instance
_data_collector = None

def get_data_collector():
    """Data Collector singleton'ını döndür"""
    global _data_collector
    if _data_collector is None:
        _data_collector = BISTDataCollector()
    return _data_collector

if __name__ == "__main__":
    # Test data collection
    collector = get_data_collector()
    
    print("🚀 BIST Data Collection Test başlatılıyor...")
    
    # İlk olarak 5 hisse ile test
    test_symbols = ['THYAO', 'AKBNK', 'GARAN', 'ISCTR', 'TUPRS']
    
    print(f"📊 Test: {test_symbols} için veri toplama...")
    
    for symbol in test_symbols:
        data = collector.collect_symbol_data(symbol, period="1y")
        if data:
            collector.save_to_database(data)
            print(f"✅ {symbol} tamamlandı")
        else:
            print(f"❌ {symbol} başarısız")
    
    # İstatistikleri göster
    stats = collector.get_collection_stats()
    print(f"\n📈 İstatistikler: {stats}")
    
    print("\n🎯 Test tamamlandı! Tüm BIST verilerini toplamak için:")
    print("python3 -c \"from data_collector import get_data_collector; get_data_collector().collect_all_data()\"")