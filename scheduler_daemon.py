#!/usr/bin/env python3
import schedule
import time
import logging
import threading
import signal
import sys
import shutil
import requests
from datetime import datetime
from advanced_collector import AdvancedBISTCollector

# ML System imports
try:
    from simple_enhanced_ml import get_simple_enhanced_ml
    from data_collector import get_data_collector
    ML_SYSTEMS_AVAILABLE = True
except ImportError as e:
    ML_SYSTEMS_AVAILABLE = False
    print(f"⚠️ ML Systems import error: {e}")

# Logging setup
import os

# Environment variable'dan log path'i al
log_path = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
log_file = os.path.join(log_path, 'scheduler.log')

# Log klasörünü oluştur
os.makedirs(log_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BISTSchedulerDaemon:
    def __init__(self):
        # PID file singleton pattern
        self.pid_file = '/opt/bist-pattern/scheduler_daemon.pid'
        self._check_singleton()
        
        self.collector = AdvancedBISTCollector()
        self.is_running = False
        self.scheduler_thread = None
        
        # WebSocket broadcasting
        self.websocket_url = "http://localhost:5000"
        
        # Write current PID to file
        self._write_pid()
        
        # Graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

    def _check_singleton(self):
        """PID file kontrolü ile singleton pattern"""
        try:
            if os.path.exists(self.pid_file):
                with open(self.pid_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Process'in çalışıp çalışmadığını kontrol et
                try:
                    os.kill(old_pid, 0)  # Signal 0 - sadece process existence kontrolü
                    logger.error(f"❌ Scheduler daemon zaten çalışıyor (PID: {old_pid})")
                    print(f"❌ Another scheduler daemon is already running with PID {old_pid}")
                    sys.exit(1)
                except OSError:
                    # Process çalışmıyor, PID file'ı temizle
                    logger.info(f"🧹 Eski PID file temizleniyor: {old_pid}")
                    os.remove(self.pid_file)
        except (ValueError, IOError) as e:
            logger.warning(f"⚠️ PID file okuma hatası: {e}")
    
    def _write_pid(self):
        """Mevcut PID'i dosyaya yaz"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.info(f"📝 PID yazıldı: {os.getpid()}")
        except IOError as e:
            logger.error(f"❌ PID yazma hatası: {e}")

    def _cleanup_pid(self):
        """PID file'ı temizle"""
        try:
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
                logger.info("🧹 PID file temizlendi")
        except IOError as e:
            logger.error(f"❌ PID temizleme hatası: {e}")

    def broadcast_log(self, level, message, category='scheduler'):
        """WebSocket üzerinden real-time log gönder"""
        try:
            # Flask app'e log mesajı gönder (internal API)
            # Bu mesaj app.py'deki broadcast_log fonksiyonunu tetikleyecek
            data = {
                'level': level,
                'message': message,
                'category': category
            }
            # Not: Bu basit bir HTTP request, production'da internal message queue kullanılabilir
            requests.post(f"{self.websocket_url}/api/internal/broadcast-log", 
                         json=data, timeout=2)
        except Exception as e:
            # Broadcast hataları sessizce geç
            pass
    
    def log_and_broadcast(self, level, message, category='scheduler'):
        """Hem log'a yaz hem WebSocket'e yayınla"""
        # Normal logging
        if level.upper() == 'INFO':
            logger.info(message)
        elif level.upper() == 'ERROR':
            logger.error(message)
        elif level.upper() == 'WARNING':
            logger.warning(message)
        else:
            logger.info(message)
        
        # Real-time broadcast
        self.broadcast_log(level, message, category)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.log_and_broadcast('INFO', f"Received signal {signum}, shutting down gracefully...")
        self.stop_scheduler()
        self._cleanup_pid()  # PID file'ı temizle
        sys.exit(0)

    def collect_priority_data(self):
        """Öncelikli hisseler için veri toplama"""
        logger.info("📊 Öncelikli hisse veri toplama başlatılıyor...")
        try:
            result = self.collector.collect_priority_stocks()
            logger.info(f"✅ Öncelikli toplama tamamlandı: {result['success_count']} başarılı, {result['total_records']} kayıt")
            return result
        except Exception as e:
            logger.error(f"❌ Öncelikli toplama hatası: {e}")
            return None

    def collect_all_data(self):
        """Tüm hisseler için veri toplama"""
        logger.info("📊 Tam veri toplama başlatılıyor...")
        try:
            result = self.collector.collect_all_stocks_parallel(batch_size=25)
            logger.info(f"✅ Tam toplama tamamlandı: {result['success_count']} başarılı, {result['total_records']} kayıt")
            return result
        except Exception as e:
            logger.error(f"❌ Tam toplama hatası: {e}")
            return None

    def check_model_performance(self, symbol):
        """Model performansını kontrol et (her zaman True döner - ileriye dönük feature)"""
        try:
            # Bu fonksiyon ileride model performans metrikleri geliştirildiğinde kullanılacak
            # Şimdilik her zaman yeniden eğitim yapalım
            return True
            
            # Gelecekteki implementasyon:
            # - Son tahminlerin doğruluğunu kontrol et
            # - Model drift detection
            # - Performance threshold kontrolü
            # return performance_score < self.performance_threshold
            
        except Exception as e:
            logger.error(f"❌ {symbol} performans kontrolü hatası: {e}")
            return True  # Hata durumunda eğitimi yap

    def auto_model_retraining(self):
        """Otomatik model yeniden eğitimi"""
        try:
            logger.info("🧠 Otomatik model eğitimi başlatılıyor...")
            
            if not ML_SYSTEMS_AVAILABLE:
                logger.error("❌ ML sistem modülleri mevcut değil")
                return False
            
            # Öncelikli hisseler (en aktif)
            priority_symbols = ['THYAO', 'AKBNK', 'GARAN', 'ISCTR', 'TUPRS']
            
            retrained_models = []
            failed_models = []
            
            for symbol in priority_symbols:
                try:
                    # Simple Enhanced ML eğitimi
                    simple_ml = get_simple_enhanced_ml()
                    
                    # Model performansını kontrol et (opsiyonel)
                    needs_retraining = self.check_model_performance(symbol)
                    
                    if needs_retraining:
                        logger.info(f"🔄 {symbol} için model eğitimi başlatılıyor...")
                        
                        success = simple_ml.train_simple_models(symbol)
                        
                        if success:
                            retrained_models.append(symbol)
                            logger.info(f"✅ {symbol} model eğitimi başarılı")
                        else:
                            failed_models.append(symbol)
                            logger.error(f"❌ {symbol} model eğitimi başarısız")
                    else:
                        logger.info(f"⏭️ {symbol} model eğitimine ihtiyaç yok")
                        
                except Exception as e:
                    failed_models.append(symbol)
                    logger.error(f"❌ {symbol} model eğitimi hatası: {e}")
            
            # Sonuçları logla
            logger.info(f"🎯 Model eğitimi tamamlandı:")
            logger.info(f"  ✅ Başarılı: {len(retrained_models)} - {retrained_models}")
            logger.info(f"  ❌ Başarısız: {len(failed_models)} - {failed_models}")
            
            # Stats'ları kaydet
            stats = {
                'timestamp': datetime.now().isoformat(),
                'retrained_models': retrained_models,
                'failed_models': failed_models,
                'success_rate': len(retrained_models) / len(priority_symbols) if priority_symbols else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Otomatik model eğitimi hatası: {e}")
            return False

    def system_health_check(self):
        """Sistem sağlık kontrolü"""
        try:
            logger.info("🔍 Sistem sağlık kontrolü başlatılıyor...")
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'systems': {},
                'overall_status': 'healthy'
            }
            
            # Data collection system check
            try:
                if ML_SYSTEMS_AVAILABLE:
                    collector = get_data_collector()
                    stats = collector.get_collection_stats()
                    health_status['systems']['data_collection'] = {
                        'status': 'healthy' if stats.get('total_stocks', 0) > 0 else 'warning',
                        'details': stats
                    }
                else:
                    health_status['systems']['data_collection'] = {
                        'status': 'warning',
                        'details': 'ML systems not available'
                    }
            except Exception as e:
                health_status['systems']['data_collection'] = {
                    'status': 'error',
                    'details': str(e)
                }
            
            # ML Systems check
            try:
                if ML_SYSTEMS_AVAILABLE:
                    simple_ml = get_simple_enhanced_ml()
                    info = simple_ml.get_system_info()
                    health_status['systems']['ml_prediction'] = {
                        'status': 'healthy',
                        'details': info
                    }
                else:
                    health_status['systems']['ml_prediction'] = {
                        'status': 'warning',
                        'details': 'ML systems not available'
                    }
            except Exception as e:
                health_status['systems']['ml_prediction'] = {
                    'status': 'error',
                    'details': str(e)
                }
            
            # Disk space check
            try:
                disk_usage = shutil.disk_usage('/')
                free_gb = disk_usage.free / (1024**3)
                health_status['systems']['disk_space'] = {
                    'status': 'healthy' if free_gb > 5 else 'warning',
                    'details': f'{free_gb:.1f} GB free'
                }
            except Exception as e:
                health_status['systems']['disk_space'] = {
                    'status': 'error',
                    'details': str(e)
                }
            
            # Overall status determination
            error_count = sum(1 for system in health_status['systems'].values() if system['status'] == 'error')
            warning_count = sum(1 for system in health_status['systems'].values() if system['status'] == 'warning')
            
            if error_count > 0:
                health_status['overall_status'] = 'error'
            elif warning_count > 0:
                health_status['overall_status'] = 'warning'
            
            # Log sonuçları
            status_emoji = {'healthy': '✅', 'warning': '⚠️', 'error': '❌'}
            overall_emoji = status_emoji.get(health_status['overall_status'], '❓')
            
            logger.info(f"{overall_emoji} Sistem sağlık durumu: {health_status['overall_status']}")
            
            for system_name, system_status in health_status['systems'].items():
                emoji = status_emoji.get(system_status['status'], '❓')
                logger.info(f"  {emoji} {system_name}: {system_status['status']}")
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Sistem sağlık kontrolü hatası: {e}")
            return {'overall_status': 'error', 'error': str(e)}
    
    def run_ai_analysis_batch(self):
        """Toplu AI analizi - Her 30 dakikada TÜM hisseler için 5 katmanlı analiz"""
        try:
            logger.info("🧠 Toplu AI analizi başlatılıyor - TÜM 606 HISSE...")
            
            # TÜM aktif hisseleri database'den al
            from models import Stock
            all_stocks = Stock.query.filter_by(is_active=True).all()
            all_symbols = [stock.symbol for stock in all_stocks]
            
            logger.info(f"📊 Analiz edilecek hisse sayısı: {len(all_symbols)}")
            
            analyzed_count = 0
            signal_count = 0
            failed_count = 0
            
            # Performans için batch processing (100'lük gruplar)
            batch_size = 100
            total_batches = (len(all_symbols) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(all_symbols))
                batch_symbols = all_symbols[start_idx:end_idx]
                
                logger.info(f"🔄 Batch {batch_num + 1}/{total_batches}: {len(batch_symbols)} hisse analiz ediliyor...")
                
                for symbol in batch_symbols:
                    try:
                        # Pattern detector'ı kullan
                        try:
                            from pattern_detector import HybridPatternDetector
                            detector = HybridPatternDetector()
                        except ImportError:
                            logger.warning("⚠️ Pattern detector import edilemedi")
                            failed_count += 1
                            continue
                        
                        # 5 katmanlı analiz yap
                        analysis_result = detector.analyze_stock(symbol)
                        
                        if analysis_result and analysis_result.get('status') == 'success':
                            analyzed_count += 1
                            
                            # Güçlü sinyal var mı kontrol et
                            overall_signal = analysis_result.get('overall_signal', {})
                            confidence = overall_signal.get('confidence', 0)
                            signal_type = overall_signal.get('signal', 'NEUTRAL')
                            
                            if confidence >= 0.6 and signal_type in ['BULLISH', 'BEARISH']:
                                signal_count += 1
                                logger.info(f"🎯 Sinyal: {symbol} - {signal_type} ({confidence:.1%})")
                                
                                # Kullanıcılara watchlist bazlı sinyal gönder
                                try:
                                    self.broadcast_signal_to_users(symbol, analysis_result)
                                except Exception as e:
                                    logger.warning(f"User signal broadcast hatası: {e}")
                                
                                # Simulation engine'e sinyal gönder (eğer aktif simulation varsa)
                                try:
                                    self.process_simulation_signal(symbol, analysis_result)
                                except Exception as e:
                                    logger.warning(f"Simulation signal hatası: {e}")
                        else:
                            failed_count += 1
                        
                    except Exception as e:
                        logger.error(f"❌ {symbol} analiz hatası: {e}")
                        failed_count += 1
                
                # Batch tamamlandı, kısa ara ver
                import time
                time.sleep(2)  # 2 saniye ara
            
            logger.info(f"✅ AI analizi tamamlandı: {analyzed_count}/{len(all_symbols)} başarılı, {signal_count} sinyal, {failed_count} hata")
            
            return {
                'analyzed': analyzed_count,
                'total': len(all_symbols),
                'failed': failed_count,
                'signals': signal_count,
                'timestamp': datetime.now().isoformat(),
                'duration_minutes': 30,
                'batch_size': batch_size
            }
            
        except Exception as e:
            logger.error(f"❌ Toplu AI analizi hatası: {e}")
            return False
    
    def process_simulation_signal(self, symbol: str, analysis_result: dict):
        """Aktif simulation'lara sinyal gönder"""
        try:
            import requests
            
            overall_signal = analysis_result.get('overall_signal', {})
            confidence = overall_signal.get('confidence', 0)
            signal_type = overall_signal.get('signal', 'NEUTRAL')
            
            # Minimum confidence kontrolü
            if confidence >= 0.6 and signal_type in ['BULLISH', 'BEARISH']:
                # Simulation engine'e sinyal gönder
                requests.post('http://localhost:5000/api/simulation/process-signal',
                             json={
                                 'symbol': symbol,
                                 'signal_data': analysis_result
                             }, timeout=2)
                logger.info(f"📡 Simulation sinyali gönderildi: {symbol} - {signal_type}")
                             
        except Exception as e:
            logger.warning(f"Simulation signal hatası: {e}")
    
    def broadcast_signal_to_users(self, symbol: str, analysis_result: dict):
        """Kullanıcılara watchlist bazlı sinyal gönder"""
        try:
            from models import Watchlist, User, Stock
            
            # Bu hisseyi watchlist'inde olan kullanıcıları bul
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                return
            
            watchlist_users = Watchlist.query.filter_by(
                stock_id=stock.id,
                alert_enabled=True
            ).all()
            
            if not watchlist_users:
                return
            
            overall_signal = analysis_result.get('overall_signal', {})
            confidence = overall_signal.get('confidence', 0)
            signal_type = overall_signal.get('signal', 'NEUTRAL')
            
            signal_data = {
                'symbol': symbol,
                'signal': signal_type,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'patterns': analysis_result.get('patterns', []),
                'current_price': analysis_result.get('current_price', 0)
            }
            
            # Her kullanıcı için personalized signal gönder
            for watchlist_item in watchlist_users:
                try:
                    user_id = watchlist_item.user_id
                    
                    # WebSocket ile kullanıcıya özel oda
                    import requests
                    requests.post('http://localhost:5000/api/internal/broadcast-user-signal',
                                 json={
                                     'user_id': user_id,
                                     'signal_data': signal_data
                                 }, timeout=2)
                    
                    logger.info(f"📡 User {user_id} için {symbol} sinyali gönderildi")
                    
                except Exception as e:
                    logger.warning(f"User {watchlist_item.user_id} signal hatası: {e}")
                    
        except Exception as e:
            logger.error(f"❌ User signal broadcast hatası: {e}")

    def run_scheduler(self):
        """Scheduler loop"""
        logger.info("🕒 Scheduler loop başlatıldı")
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Her dakika kontrol et
            except Exception as e:
                logger.error(f"Scheduler loop hatası: {e}")
                time.sleep(60)

    def start_scheduler(self):
        """Scheduler'ı başlat"""
        if self.is_running:
            logger.warning("⚠️ Scheduler zaten çalışıyor!")
            return

        logger.info("🚀 BIST Scheduler Daemon başlatılıyor...")

                    # Zamanlama kuralları - Real-time trading için optimize edildi
            schedule.every(15).minutes.do(self.collect_all_data)       # Her 15 dakikada TÜM hisseler
            schedule.every().day.at("09:30").do(self.collect_all_data)  # Borsa açılış - tüm hisseler
            schedule.every().day.at("12:00").do(self.collect_all_data)  # Öğle - tüm hisseler
            schedule.every().day.at("18:00").do(self.collect_all_data)  # Akşam - tüm hisseler
            schedule.every().sunday.at("02:00").do(self.collect_all_data)    # Hafta sonu bakım
            
            # AI Analysis jobs - 5 Katmanlı Analiz TÜM HİSSELER İÇİN
            schedule.every(30).minutes.do(self.run_ai_analysis_batch)  # Her 30 dakikada TÜM 606 hisse
            
            # ML Training jobs
            schedule.every().day.at("20:00").do(self.auto_model_retraining)  # Günlük model eğitimi
            
            # Health Check jobs
            schedule.every(15).minutes.do(self.system_health_check)  # Her 15 dakikada health check

        # İlk veriyi topla - TÜM hisseler
        logger.info("🔄 İlk veri toplama işlemi - TÜM 606 hisse...")
        self.collect_all_data()

        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ Scheduler başarıyla başlatıldı")
        
        # Ana thread'i canlı tut
        try:
            while self.is_running:
                time.sleep(10)
        except KeyboardInterrupt:
            self.stop_scheduler()

    def stop_scheduler(self):
        """Scheduler'ı durdur"""
        logger.info("🛑 Scheduler durduruluyor...")
        self.is_running = False
        schedule.clear()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        logger.info("✅ Scheduler durduruldu")

if __name__ == "__main__":
    daemon = BISTSchedulerDaemon()
    try:
        daemon.start_scheduler()
    except Exception as e:
        logger.error(f"Daemon başlatma hatası: {e}")
        sys.exit(1)
