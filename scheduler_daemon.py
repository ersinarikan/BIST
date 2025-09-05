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
import os
import pandas as pd

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
        
        # WebSocket/API base URL (from env or config)
        try:
            from config import config as _cfg
            default_api = getattr(_cfg['default'], 'BIST_API_URL', 'http://localhost:5000')
        except Exception:
            default_api = 'http://localhost:5000'
        self.websocket_url = os.getenv('BIST_API_URL', default_api)
        
        # Write current PID to file
        self._write_pid()
        
        # Graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

    def _get_stock_dataframe(self, symbol: str):
        """Fetch OHLCV dataframe for a stock from PostgreSQL (lower-case columns)."""
        try:
            from app import app
            with app.app_context():
                from models import Stock, StockPrice
                stock = Stock.query.filter_by(symbol=symbol).first()
                if not stock:
                    return None
                prices = StockPrice.query.filter_by(stock_id=stock.id)\
                    .order_by(StockPrice.date.asc()).all()
                if not prices:
                    return None
                rows = []
                for p in prices:
                    rows.append({
                        'date': p.date,
                        'open': float(p.open_price),
                        'high': float(p.high_price),
                        'low': float(p.low_price),
                        'close': float(p.close_price),
                        'volume': int(p.volume),
                    })
                df = pd.DataFrame(rows)
                if df.empty:
                    return None
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        except Exception as e:
            logging.getLogger(__name__).error(f"DF fetch error {symbol}: {e}")
            return None

    def run_bulk_predictions_all(self) -> dict | bool:
        """Train and generate 1/3/7/14/30d predictions for ALL active stocks.

        - Uses basic ML for all symbols by default (fast, on-the-fly training if needed)
        - If ENABLE_ENHANCED_ML=true, also trains enhanced models and adds ENH predictions
        - Persists results to /opt/bist-pattern/logs/ml_bulk_predictions.json
        """
        try:
            self.log_and_broadcast('INFO', '🤖 ML bulk predictions starting...', 'ml')
            from app import app
            with app.app_context():
                from models import Stock
                symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]

            # ML systems (lazy import)
            basic = None
            try:
                from ml_prediction_system import get_ml_prediction_system
                basic = get_ml_prediction_system()
            except Exception:
                basic = None

            use_enhanced = str(os.getenv('ENABLE_ENHANCED_ML', 'false')).lower() in ('1', 'true', 'yes')
            enhanced = None
            if use_enhanced:
                try:
                    from enhanced_ml_system import get_enhanced_ml_system
                    enhanced = get_enhanced_ml_system()
                except Exception:
                    enhanced = None

            results: dict = {'timestamp': datetime.now().isoformat(), 'predictions': {}}
            processed = 0
            for sym in symbols:
                try:
                    df = self.run_safe_get_df(sym)
                    if df is None or len(df) < 50:
                        continue
                    out_sym: dict = {}
                    # Basic predictions (always)
                    if basic is not None:
                        try:
                            preds = basic.predict_prices(sym, df, None) or {}
                            out_sym['basic'] = preds
                        except Exception:
                            pass
                    # Enhanced (optional)
                    if enhanced is not None and len(df) >= 200:
                        try:
                            enhanced.train_enhanced_models(sym, df)
                            ep = enhanced.predict_enhanced(sym, df) or {}
                            out_sym['enhanced'] = ep
                        except Exception:
                            pass
                    if out_sym:
                        results['predictions'][sym] = out_sym
                        processed += 1
                    # polite small sleep to reduce DB/CPU spikes
                    time.sleep(0.01)
                except Exception:
                    continue

            # Persist
            try:
                log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                os.makedirs(log_dir, exist_ok=True)
                fpath = os.path.join(log_dir, 'ml_bulk_predictions.json')
                import json
                with open(fpath, 'w') as wf:
                    json.dump(results, wf)
            except Exception:
                pass

            self.log_and_broadcast('SUCCESS', f'✅ ML bulk predictions completed: {processed} symbols', 'ml')
            return results
        except Exception as e:
            self.log_and_broadcast('ERROR', f'ML bulk predictions error: {e}', 'ml')
            return False

    # Safe wrapper for dataframe fetch to keep loop robust
    def run_safe_get_df(self, symbol: str):
        try:
            return self._get_stock_dataframe(symbol)
        except Exception:
            return None

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
            headers = {}
            try:
                from config import config
                token = config['default'].INTERNAL_API_TOKEN
                if token:
                    headers['X-Internal-Token'] = token
            except Exception:
                pass
            requests.post(f"{self.websocket_url}/api/internal/broadcast-log", 
                          json=data, headers=headers, timeout=2)
        except Exception as e:
            # Broadcast hataları sessizce geç
            pass

    def _update_pipeline_status(self, phase: str, state: str, details: dict | None = None):
        """Persist pipeline phase status to a JSON file for dashboard/REST usage."""
        try:
            import json
            from datetime import datetime
            status_path = os.path.join(os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs'), 'pipeline_status.json')
            payload = {
                'timestamp': datetime.now().isoformat(),
                'phase': phase,
                'state': state,
                'details': details or {}
            }
            # Append-friendly structure
            data = {'history': []}
            if os.path.exists(status_path):
                try:
                    with open(status_path, 'r') as rf:
                        data = json.load(rf) or data
                except Exception:
                    data = {'history': []}
            data.setdefault('history', []).append(payload)
            # Keep last 200 entries
            data['history'] = data['history'][-200:]
            with open(status_path, 'w') as wf:
                json.dump(data, wf)
        except Exception:
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
        self.log_and_broadcast('INFO', '📊 Tam veri toplama başlatılıyor...', 'collector')
        try:
            self._update_pipeline_status('data_collection', 'start', {})
            # batch_size=None => collector config'ten (COLLECTOR_BATCH_SIZE) gelsin
            result = self.collector.collect_all_stocks_parallel(batch_size=None)
            logger.info(f"✅ Tam toplama tamamlandı: {result['success_count']} başarılı, {result['total_records']} kayıt")
            try:
                self._update_pipeline_status('data_collection', 'end', result or {})
            except Exception:
                pass
            try:
                self.log_and_broadcast('SUCCESS', f"✅ Veri toplama tamamlandı: {result.get('success_count',0)} başarılı, {result.get('total_records',0)} kayıt", 'collector')
            except Exception:
                pass
            return result
        except Exception as e:
            logger.error(f"❌ Tam toplama hatası: {e}")
            self._update_pipeline_status('data_collection', 'error', {'error': str(e)})
            try:
                self.log_and_broadcast('ERROR', f"❌ Veri toplama hatası: {e}", 'collector')
            except Exception:
                pass
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
            
            # Write status to a file for the API to read
            try:
                import json
                status_path = os.path.join(os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs'), 'health_status.json')
                with open(status_path, 'w') as f:
                    json.dump(health_status, f)
            except Exception as e:
                logger.error(f"❌ Sağlık durumu dosyaya yazılamadı: {e}")

            return health_status
            
        except Exception as e:
            logger.error(f"❌ Sistem sağlık kontrolü hatası: {e}")
            return {'overall_status': 'error', 'error': str(e)}
    
    def run_ai_analysis_batch(self):
        """Toplu AI analizi - Her 30 dakikada TÜM hisseler için 5 katmanlı analiz"""
        try:
            self.log_and_broadcast('INFO', "🧠 Toplu AI analizi başlatılıyor - TÜM hisseler...", 'ai_analysis')
            self._update_pipeline_status('ai_analysis', 'start', {})
            
            # TÜM aktif hisseleri database'den al (Flask app context gerekli)
            from app import app
            with app.app_context():
                from models import Stock
                all_stocks = Stock.query.filter_by(is_active=True).all()
            all_symbols = [stock.symbol for stock in all_stocks]
            
            self.log_and_broadcast('INFO', f"📊 Analiz edilecek hisse sayısı: {len(all_symbols)}", 'ai_analysis')
            
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
                
                self.log_and_broadcast('INFO', f"🔄 Batch {batch_num + 1}/{total_batches}: {len(batch_symbols)} hisse analiz ediliyor...", 'ai_analysis')
                
                for symbol in batch_symbols:
                    try:
                        # Pattern detector'ı kullan
                        try:
                            from pattern_detector import HybridPatternDetector
                            detector = HybridPatternDetector()
                        except ImportError:
                            self.log_and_broadcast('WARNING', "⚠️ Pattern detector import edilemedi", 'ai_analysis')
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
                                self.log_and_broadcast('INFO', f"🎯 Sinyal: {symbol} - {signal_type} ({confidence:.0%})", 'ai_analysis')
                                
                                # Kullanıcılara watchlist bazlı sinyal gönder
                                try:
                                    self.broadcast_signal_to_users(symbol, analysis_result)
                                except Exception as e:
                                    self.log_and_broadcast('WARNING', f"User signal broadcast hatası: {e}", 'ai_analysis')
                                
                                # Simulation engine'e sinyal gönder (eğer aktif simulation varsa)
                                try:
                                    self.process_simulation_signal(symbol, analysis_result)
                                except Exception as e:
                                    self.log_and_broadcast('WARNING', f"Simulation signal hatası: {e}", 'ai_analysis')
                        else:
                            failed_count += 1
                        
                    except Exception as e:
                        self.log_and_broadcast('ERROR', f"❌ {symbol} analiz hatası: {e}", 'ai_analysis')
                        failed_count += 1
                
                # Batch tamamlandı, kısa ara ver
                import time
                time.sleep(2)  # 2 saniye ara
            
            summary = {'analyzed': analyzed_count, 'total': len(all_symbols), 'signals': signal_count, 'failed': failed_count}
            self._update_pipeline_status('ai_analysis', 'end', summary)
            self.log_and_broadcast('SUCCESS', f"✅ AI analizi tamamlandı: {analyzed_count}/{len(all_symbols)} başarılı, {signal_count} sinyal, {failed_count} hata", 'ai_analysis')
            
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
            self._update_pipeline_status('ai_analysis', 'error', {'error': str(e)})
            self.log_and_broadcast('ERROR', f"❌ Toplu AI analizi hatası: {e}", 'ai_analysis')
            return False
    
    def process_simulation_signal(self, symbol: str, analysis_result: dict):
        """Aktif simulation'lara sinyal gönder"""
        try:
            import requests
            # Aktif session'lar için app context gerekli
            from app import app
            with app.app_context():
                from models import SimulationSession, Watchlist, Stock
                active_sessions = SimulationSession.query.filter_by(status='active').all()

            overall_signal = analysis_result.get('overall_signal', {})
            confidence = overall_signal.get('confidence', 0)
            signal_type = overall_signal.get('signal', 'NEUTRAL')
            
            # Minimum confidence kontrolü
            if confidence >= 0.6 and signal_type in ['BULLISH', 'BEARISH']:
                # Her aktif session için simulation engine'e sinyal gönder
                headers = {}
                try:
                    from config import config
                    token = config['default'].INTERNAL_API_TOKEN
                    if token:
                        headers['X-Internal-Token'] = token
                except Exception:
                    pass
                for session in (active_sessions or []):
                    # Watchlist filtresi: kullanıcı watchlist'inde yoksa atla
                    try:
                        with app.app_context():
                            stock_obj = Stock.query.filter_by(symbol=symbol).first()
                            user_watch = Watchlist.query.filter_by(user_id=session.user_id, stock_id=stock_obj.id).first() if stock_obj else None
                        if user_watch is None:
                            # Kullanıcının watchlist'i yoksa veya sembol ekli değilse bu sinyali atla
                            continue
                    except Exception:
                        pass
                    requests.post('http://localhost:5000/api/simulation/process-signal',
                                  json={
                                      'session_id': session.id,
                                      'symbol': symbol,
                                      'signal_data': analysis_result
                                  }, headers=headers, timeout=2)
                logger.info(f"📡 Simulation sinyali gönderildi: {symbol} - {signal_type} (sessions: {len(active_sessions or [])})")
                             
        except Exception as e:
            logger.warning(f"Simulation signal hatası: {e}")
    
    def broadcast_signal_to_users(self, symbol: str, analysis_result: dict):
        """Kullanıcılara watchlist bazlı sinyal gönder"""
        try:
            from app import app
            with app.app_context():
                from models import Watchlist, Stock
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
                    headers = {}
                    try:
                        from config import config
                        token = config['default'].INTERNAL_API_TOKEN
                        if token:
                            headers['X-Internal-Token'] = token
                    except Exception:
                        pass
                    requests.post('http://localhost:5000/api/internal/broadcast-user-signal',
                                  json={
                                      'user_id': user_id,
                                      'signal_data': signal_data
                                  }, headers=headers, timeout=2)
                    
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

        # Zamanlama kuralları veya sürekli döngü modu
        PIPELINE_MODE = os.getenv('PIPELINE_MODE', 'SCHEDULED').upper()
        # Sık periyotlarda: öncelikli semboller → ardından analiz (hafif pipeline)
        def run_pipeline_priority():
            try:
                self.collect_priority_data()
            finally:
                try:
                    self.run_ai_analysis_batch()
                except Exception:
                    pass
            # Bulk ML predictions after AI analysis
            try:
                self.run_bulk_predictions_all()
            except Exception:
                pass
        
        # Gün içinde 3 kez ve pazar gecesi: tam koleksiyon → ardından analiz (tam pipeline)
        def run_pipeline_full():
            try:
                self.collect_all_data()
            finally:
                try:
                    self.run_ai_analysis_batch()
                except Exception:
                    pass
                # Bulk ML predictions after AI analysis
                try:
                    self.run_bulk_predictions_all()
                except Exception:
                    pass
        
        if PIPELINE_MODE == 'CONTINUOUS_FULL':
            self.log_and_broadcast('INFO', '🌀 Continuous full pipeline mode', 'scheduler')
        else:
            # Öncelik pipeline'ı 30 dakikada bir
            schedule.every(30).minutes.do(run_pipeline_priority)
            # Tam pipeline belirli saatlerde
            schedule.every().day.at("09:30").do(run_pipeline_full)
            schedule.every().day.at("12:00").do(run_pipeline_full)
            schedule.every().day.at("18:00").do(run_pipeline_full)
            schedule.every().sunday.at("02:00").do(run_pipeline_full)
        
        # ML Training jobs
        schedule.every().day.at("20:00").do(self.auto_model_retraining)  # Günlük model eğitimi
        
        # Health Check jobs
        schedule.every(15).minutes.do(self.system_health_check)  # Her 15 dakikada health check

        # İlk başlangıç: mod'a göre tetikleme
        if PIPELINE_MODE == 'CONTINUOUS_FULL':
            # RUNNING flag'i thread başlamadan önce set edilmeli; aksi halde döngü hiç çalışmaz
            self.is_running = True
            def run_continuous_full():
                sleep_seconds = int(float(os.getenv('FULL_CYCLE_SLEEP_SECONDS', '0')))
                while self.is_running:
                    try:
                        run_pipeline_full()
                    except Exception as e:
                        self.log_and_broadcast('ERROR', f'Continuous pipeline error: {e}', 'scheduler')
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
            # Ayrı bir pipeline thread'i tutalım; schedule thread'i ile karışmasın
            self.pipeline_thread = threading.Thread(target=run_continuous_full, daemon=True)
            self.pipeline_thread.start()
        else:
            logger.info("🔄 İlk pipeline: Öncelikli hisseler + analiz")
            try:
                run_pipeline_priority()
            except Exception:
                pass

        # Schedule çalıştıran thread
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
