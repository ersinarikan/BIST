"""
BIST Automated Data Pipeline Scheduler
Günlük otomatik veri toplama, model eğitimi ve sistem izleme
"""

import schedule
import time
import logging
import threading
from datetime import datetime, timedelta
import os
import json
import pandas as pd
# Local imports
try:
    from data_collector import get_data_collector
    from ml_prediction_system import get_ml_prediction_system
    from simple_enhanced_ml import get_simple_enhanced_ml
    from enhanced_ml_system import get_enhanced_ml_system
    from alert_system import get_alert_system
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    SYSTEMS_AVAILABLE = False
    print(f"⚠️ System import error: {e}")

# Email imports (optional, don't break SYSTEMS_AVAILABLE)
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError as e:
    EMAIL_AVAILABLE = False
    print(f"⚠️ Email system not available: {e}")

logger = logging.getLogger(__name__)

class AutomatedDataPipeline:
    """Otomatik veri pipeline'ı ve sistem yönetimi"""
    
    def __init__(self):
        self.is_running = False
        self.last_run_stats = {}
        self.scheduler_thread = None
        self.performance_threshold = 0.7  # %70 altındaki modeller yeniden eğitilir
        # Watchdog: thread ölürse yeniden başlatma için throttle timestamp
        self.last_watchdog_restart_ts = 0.0
        # Idle watchdog: en son etkinlik zaman damgası ve kullanıcı kaynaklı durdurma bayrağı
        self.last_activity_ts = 0.0
        self.user_stopped = False
        self._idle_watchdog_thread = None
        
        # Email settings (opsiyonel)
        self.email_enabled = False
        self.email_settings = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'to_emails': []
        }
        
        logger.info("🤖 Automated Data Pipeline başlatıldı")

    def _get_stock_dataframe(self, symbol: str):
        """PostgreSQL'den bir hissenin OHLCV verisini DataFrame olarak getir (index=date)."""
        try:
            from app import app as flask_app
            from models import Stock, StockPrice
            with flask_app.app_context():
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
                        'volume': int(p.volume)
                    })
                df = pd.DataFrame(rows)
                if df.empty:
                    return None
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        except Exception as e:
            logger.error(f"DF fetch error {symbol}: {e}")
            return None

    def run_bulk_predictions_all(self) -> dict | bool:
        """Tüm aktif hisseler için 1/3/7/14/30 günlük tahminleri üret ve kaydet.

        - Temel ML her zaman çalışır (hızlı)
        - ENV: ENABLE_ENHANCED_ML=True ise Enhanced ML de eğitim+tahmin yapar
        - Sonuçlar: /opt/bist-pattern/logs/ml_bulk_predictions.json
        """
        try:
            # UI'ya bilgi amaçlı yayın
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('INFO', '🤖 ML bulk predictions starting...', 'ml')
            except Exception:
                pass

            # Semboller
            from app import app as flask_app
            with flask_app.app_context():
                from models import Stock
                symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]

            # ML sistemleri
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
                    df = self._get_stock_dataframe(sym)
                    if df is None or len(df) < 50:
                        continue
                    out_sym: dict = {}
                    if basic is not None:
                        try:
                            preds = basic.predict_prices(sym, df, None) or {}
                            out_sym['basic'] = preds
                        except Exception:
                            pass
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
                except Exception:
                    continue

            # Kaydet
            try:
                log_dir = '/opt/bist-pattern/logs'
                os.makedirs(log_dir, exist_ok=True)
                fp = os.path.join(log_dir, 'ml_bulk_predictions.json')
                with open(fp, 'w') as wf:
                    json.dump(results, wf)
            except Exception:
                pass

            # UI yayın
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('SUCCESS', f'✅ ML bulk predictions completed: {processed} symbols', 'ml')
            except Exception:
                pass
            return results
        except Exception as e:
            logger.error(f"ML bulk predictions error: {e}")
            return False
    
    def daily_data_collection(self):
        """Günlük veri toplama görevi"""
        try:
            logger.info("📅 Günlük veri toplama başlatılıyor...")
            
            if not SYSTEMS_AVAILABLE:
                logger.error("❌ Sistem modülleri mevcut değil")
                return False
            
            # Import app locally to avoid circular imports
            from app import app
            with app.app_context():
                collector = get_data_collector()
                
                # Güncel veri toplama (son 7 gün)
                symbols = collector.get_bist_symbols()
                updated_count = 0
                failed_count = 0
                
                for symbol in symbols[:20]:  # İlk 20 hisse ile başla
                    try:
                        success = collector.update_single_stock(symbol, days=7)
                        if success:
                            updated_count += 1
                        else:
                            failed_count += 1
                        
                        # Rate limiting
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"❌ {symbol} günlük güncelleme hatası: {e}")
                        failed_count += 1
                
                # İstatistikleri kaydet
                stats = {
                    'date': datetime.now().isoformat(),
                    'updated_stocks': updated_count,
                    'failed_stocks': failed_count,
                    'total_processed': updated_count + failed_count
                }
                
                self.last_run_stats['data_collection'] = stats
                logger.info(f"✅ Günlük veri toplama tamamlandı: {updated_count} başarılı, {failed_count} hata")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Günlük veri toplama hatası: {e}")
            return False
    
    def weekly_full_collection(self):
        """Haftalık tam veri toplama"""
        try:
            logger.info("📅 Haftalık tam veri toplama başlatılıyor...")
            
            if not SYSTEMS_AVAILABLE:
                logger.error("❌ Sistem modülleri mevcut değil")
                return False
            
            # Import app locally to avoid circular imports
            from app import app
            with app.app_context():
                collector = get_data_collector()
                
                # Tam veri toplama (son 1 ay)
                result = collector.collect_all_data(max_workers=3, period="1mo")
                
                if result:
                    self.last_run_stats['weekly_collection'] = result
                    logger.info(f"✅ Haftalık tam veri toplama tamamlandı: {result}")
                    return True
                else:
                    logger.error("❌ Haftalık veri toplama başarısız")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Haftalık veri toplama hatası: {e}")
            return False
    
    # MIGRATED TO DAEMON: auto_model_retraining, check_model_performance
    # These functions are now handled by scheduler_daemon.py
    
    # MIGRATED TO DAEMON: system_health_check  
    # This function is now handled by scheduler_daemon.py
    
    def send_status_email(self, subject, content):
        """Durum raporu email gönder (opsiyonel)"""
        try:
            if not EMAIL_AVAILABLE:
                logger.warning("📧 Email system not available, skipping email")
                return True
                
            if not self.email_enabled or not self.email_settings['to_emails']:
                return True  # Email devre dışı
            
            msg = MIMEMultipart()
            msg['From'] = self.email_settings['username']
            msg['To'] = ', '.join(self.email_settings['to_emails'])
            msg['Subject'] = f"BIST AI System - {subject}"
            
            msg.attach(MIMEText(content, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(self.email_settings['smtp_server'], self.email_settings['smtp_port'])
            server.starttls()
            server.login(self.email_settings['username'], self.email_settings['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_settings['username'], self.email_settings['to_emails'], text)
            server.quit()
            
            logger.info(f"📧 Status email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Email gönderme hatası: {e}")
            return False
    
    def daily_status_report(self):
        """Günlük durum raporu"""
        try:
            logger.info("📊 Günlük durum raporu oluşturuluyor...")
            
            # SAFE: Skip health check to avoid Flask context issues
            # health_status = self.system_health_check()  # This kills the thread!
            health_status = {'overall_status': 'unknown', 'systems': {}}
            
            # Rapor oluştur
            report = f"""
🤖 BIST AI System Daily Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}

📊 System Health: {health_status.get('overall_status', 'unknown')}

📈 Data Collection Stats:
{json.dumps(self.last_run_stats.get('data_collection', {}), indent=2)}

🧠 Model Training Stats:
{json.dumps(self.last_run_stats.get('model_retraining', {}), indent=2)}

🔍 Health Check Results:
{json.dumps(health_status.get('systems', {}), indent=2)}

---
BIST Automated Data Pipeline
            """
            
            logger.info("📋 Günlük rapor oluşturuldu")
            
            # Email gönder (eğer aktifse)
            if health_status.get('overall_status') in ['warning', 'error']:
                self.send_status_email("System Alert", report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Günlük rapor hatası: {e}")
            return None
    
    def system_health_check(self):
        """Basit sağlık kontrolü: DB istatistikleri ve disk boş alanını yaz."""
        try:
            logger.info("🔍 Sağlık kontrolü (internal)")
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'systems': {},
                'overall_status': 'healthy'
            }
            # Data collection stats
            try:
                from app import app as flask_app
                with flask_app.app_context():
                    stats = get_data_collector().get_collection_stats()
                health_status['systems']['data_collection'] = {
                    'status': 'healthy' if (isinstance(stats, dict) and stats.get('total_stocks', 0) > 0) else 'warning',
                    'details': stats
                }
            except Exception as e:
                health_status['systems']['data_collection'] = {'status': 'error', 'details': str(e)}
            # Disk space
            try:
                import shutil
                free_gb = shutil.disk_usage('/').free / (1024**3)
                health_status['systems']['disk_space'] = {
                    'status': 'healthy' if free_gb > 5 else 'warning',
                    'details': f"{free_gb:.1f} GB free"
                }
            except Exception as e:
                health_status['systems']['disk_space'] = {'status': 'error', 'details': str(e)}
            # Overall
            if any(s.get('status') == 'error' for s in health_status['systems'].values()):
                health_status['overall_status'] = 'error'
            elif any(s.get('status') == 'warning' for s in health_status['systems'].values()):
                health_status['overall_status'] = 'warning'
            # Persist JSON (for dashboard)
            try:
                import json, os
                path = '/opt/bist-pattern/logs/health_status.json'
                os.makedirs('/opt/bist-pattern/logs', exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(health_status, f)
            except Exception:
                pass
            # Broadcast (optional)
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('INFO', f"Health: {health_status['overall_status']}", 'health')
            except Exception:
                pass
            return health_status
        except Exception as e:
            logger.error(f"❌ Internal health check hatası: {e}")
            return {'overall_status': 'error', 'error': str(e)}
    def _merge_predictions_file(self, symbol: str, out_sym: dict) -> bool:
        """`ml_bulk_predictions.json` dosyasına sembol bazlı tahmini birleştirerek yazar."""
        try:
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            os.makedirs(log_dir, exist_ok=True)
            fpath = os.path.join(log_dir, 'ml_bulk_predictions.json')
            import json
            data = {'timestamp': datetime.now().isoformat(), 'predictions': {}}
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r') as rf:
                        prev = json.load(rf) or {}
                        if isinstance(prev, dict):
                            data['predictions'] = prev.get('predictions') or {}
                except Exception:
                    pass
            data['predictions'][symbol] = out_sym or {}
            with open(fpath, 'w') as wf:
                json.dump(data, wf)
            return True
        except Exception as _err:
            logger.warning(f"Predictions merge error for {symbol}: {_err}")
            return False

    def run_incremental_cycle(self) -> dict:
        """Sembol bazlı döngü: her sembol için tek tek veri toplama → analiz → tahmin.

        Dış servis yükünü azaltmak ve CPU/bellek kullanımını yaymak için tam toplama yerine
        sembol bazında ardışık çalışır.
        """
        stats = {'processed': 0, 'analyzed': 0, 'predicted': 0}
        try:
            # Hazırlık
            from pattern_detector import HybridPatternDetector
            det = HybridPatternDetector()
            col = None
            try:
                col = get_data_collector()
            except Exception:
                col = None
            # Semboller
            from app import app as flask_app
            with flask_app.app_context():
                from models import Stock
                symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
            # ML sistemleri
            basic = None
            try:
                from ml_prediction_system import get_ml_prediction_system
                basic = get_ml_prediction_system()
            except Exception:
                basic = None
            use_enhanced = str(os.getenv('ENABLE_ENHANCED_ML', 'false')).lower() in ('1','true','yes')
            enhanced = None
            if use_enhanced:
                try:
                    from enhanced_ml_system import get_enhanced_ml_system
                    enhanced = get_enhanced_ml_system()
                except Exception:
                    enhanced = None

            # Uyku ayarları
            try:
                symbol_sleep = float(os.getenv('SYMBOL_SLEEP_SECONDS', '0.3'))
            except Exception:
                symbol_sleep = 0.3

            for sym in symbols:
                try:
                    # 1) Veri güncelle (hafif)
                    if col is not None:
                        try:
                            col.update_single_stock(sym, days=7)
                        except Exception:
                            pass
                    # 2) Analiz
                    try:
                        det.analyze_stock(sym)
                        stats['analyzed'] += 1
                    except Exception:
                        pass
                    # 3) Tahmin ve dosyaya yaz (birleştirerek)
                    out_sym: dict = {}
                    df = self._get_stock_dataframe(sym)
                    if df is not None and len(df) >= 50:
                        # Basic
                        if basic is not None:
                            try:
                                preds = basic.predict_prices(sym, df, None) or {}
                                out_sym['basic'] = preds
                            except Exception:
                                pass
                        # Enhanced
                        if enhanced is not None and len(df) >= 200:
                            try:
                                enhanced.train_enhanced_models(sym, df)
                                ep = enhanced.predict_enhanced(sym, df) or {}
                                out_sym['enhanced'] = ep
                            except Exception:
                                pass
                    if out_sym:
                        if self._merge_predictions_file(sym, out_sym):
                            stats['predicted'] += 1
                    stats['processed'] += 1
                except Exception:
                    pass
                # Dış servislere nazik ol
                try:
                    time.sleep(symbol_sleep)
                except Exception:
                    pass
            return stats
        except Exception as e:
            logger.error(f"❌ Incremental cycle error: {e}")
            return stats
    def setup_schedule(self):
        """Zamanlama ayarları"""
        try:
            logger.info("⏰ Scheduled tasks ayarlanıyor...")
            
            # Clear existing jobs first
            schedule.clear()
            
            # Only configure when PIPELINE_MODE explicitly SCHEDULED
            if os.getenv('PIPELINE_MODE', 'CONTINUOUS_FULL').upper() != 'SCHEDULED':
                logger.info("🛑 PIPELINE_MODE != SCHEDULED → schedule jobs are skipped (continuous mode)")
                return True
            
            # MINIMAL TEST: NO JOBS AT ALL (test schedule library itself)
            # def simple_heartbeat():
            #     logger.info("💓 Scheduler heartbeat - thread alive")
            #     return True
            
            # NO JOBS - Pure schedule.run_pending() test
            # schedule.every(2).minutes.do(simple_heartbeat).tag('heartbeat')
            
            # STEP 1: En basit job'dan başla - daily_status_report
            schedule.every().day.at("08:00").do(self.daily_status_report).tag('daily')
            
            # Diğer complex job'lar geçici olarak devre dışı
            # schedule.every().day.at("06:00").do(self.daily_data_collection).tag('daily')
            # schedule.every().day.at("07:00").do(self.auto_model_retraining).tag('daily') 
            # schedule.every().monday.at("05:00").do(self.weekly_full_collection).tag('weekly')
            # schedule.every(6).hours.do(self.system_health_check).tag('health')
            
            # İç pipeline işleri (dashboard kontrollü)
            # 30 dakikada bir: öncelikli toplama → AI analizi
            schedule.every(30).minutes.do(self.run_priority_pipeline).tag('priority_pipeline')
            # Gün içinde 3 kez + pazar gecesi: tam toplama → AI analizi
            schedule.every().day.at("09:30").do(self.run_full_pipeline).tag('full_pipeline')
            schedule.every().day.at("12:00").do(self.run_full_pipeline).tag('full_pipeline')
            schedule.every().day.at("18:00").do(self.run_full_pipeline).tag('full_pipeline')
            schedule.every().sunday.at("02:00").do(self.run_full_pipeline).tag('weekly_full')
            # 15 dakikada bir health check
            schedule.every(15).minutes.do(self.system_health_check).tag('health')

            # Test için - her 2 dakikada bir health check (opsiyonel)
            if os.getenv('BIST_DEBUG', '').lower() == 'true':
                schedule.every(2).minutes.do(self.system_health_check).tag('debug')
                logger.info("🔧 Debug mode: Health check every 2 minutes")
            
            job_count = len(schedule.jobs)
            logger.info(f"✅ Scheduled tasks kuruldu ({job_count} job):")
            logger.info("  🔄 Her 30 dk - Öncelikli toplama + AI analiz")
            logger.info("  📅 09:30/12:00/18:00 - Tam toplama + AI analiz")
            logger.info("  🕑 Pazar 02:00 - Haftalık tam toplama + AI analiz")
            logger.info("  🔍 Her 15 dk - Health check")
            
            return job_count > 0
            
        except Exception as e:
            logger.error(f"❌ Schedule kurulum hatası: {e}")
            return False
    
    def start_scheduler(self):
        """Scheduler'ı başlat"""
        try:
            if self.is_running:
                logger.warning("⚠️ Scheduler zaten çalışıyor")
                return False
            
            logger.info("🚀 Automated Data Pipeline başlatılıyor...")
            
            # Tek mod: CONTINUOUS_FULL (sadeleştirildi)
            mode = 'CONTINUOUS_FULL'

            # Yardımcı: pipeline history'ye kayıt ekle
            def _append_pipeline_history(phase: str, state: str, details: dict = None):
                try:
                    log_dir = '/opt/bist-pattern/logs'
                    os.makedirs(log_dir, exist_ok=True)
                    status_file = os.path.join(log_dir, 'pipeline_status.json')
                    payload = {'history': []}
                    try:
                        if os.path.exists(status_file):
                            with open(status_file, 'r') as f:
                                payload = json.load(f) or {'history': []}
                    except Exception:
                        payload = {'history': []}
                    entry = {
                        'phase': phase,
                        'state': state,
                        'timestamp': datetime.now().isoformat(),
                        'details': details or {}
                    }
                    payload.setdefault('history', []).append(entry)
                    # Keep last 200
                    payload['history'] = payload['history'][-200:]
                    with open(status_file, 'w') as f:
                        json.dump(payload, f)
                except Exception as _hist_err:
                    logger.warning(f"Pipeline history write failed: {_hist_err}")

            # Temizlik: Önce mevcut schedule işlerini temizle (mode ne olursa olsun)
            try:
                schedule.clear()
            except Exception:
                pass

            self.is_running = True
            self.user_stopped = False
            self.last_activity_ts = time.time()

            if mode == 'CONTINUOUS_FULL':
                logger.info("🔁 Mode: CONTINUOUS_FULL - Incremental (sembol bazlı) döngü")

                def run_continuous_full_loop():
                    try:
                        loop_idx = 0
                        while self.is_running:
                            loop_idx += 1
                            # heartbeat: etkinlik güncelle
                            self.last_activity_ts = time.time()
                            try:
                                from app import app as flask_app
                                if hasattr(flask_app, 'broadcast_log'):
                                    flask_app.broadcast_log('INFO', f'Cycle {loop_idx}: Incremental cycle starting', 'collector')
                            except Exception:
                                pass

                            # Incremental: sembol bazlı toplama → analiz → tahmin
                            _append_pipeline_history('incremental_cycle', 'start', {'cycle': loop_idx})
                            try:
                                inc = self.run_incremental_cycle()
                                _append_pipeline_history('incremental_cycle', 'end', {'cycle': loop_idx, **(inc or {})})
                                self.last_activity_ts = time.time()
                            except Exception as e:
                                _append_pipeline_history('incremental_cycle', 'error', {'error': str(e)})
                                logger.error(f"Incremental cycle error: {e}")

                            # 4) Bekle (5 dakika)
                            try:
                                from app import app as flask_app
                                if hasattr(flask_app, 'broadcast_log'):
                                    flask_app.broadcast_log('INFO', 'Sleeping 300s before next cycle', 'scheduler')
                            except Exception:
                                pass
                            for _ in range(300):
                                if not self.is_running:
                                    break
                                # heartbeat: uykuda da etkinlik güncelle (panelde idle zannedilmesin)
                                if _ % 30 == 0:
                                    self.last_activity_ts = time.time()
                                time.sleep(1)
                        logger.info("⏹️ Continuous loop stopped")
                    except Exception as e:
                        logger.error(f"❌ Continuous loop critical error: {e}")
                        # is_running bayrağını kapatmayalım ki watchdog devreye girebilsin
                        # Böylece UI "STOPPED" göstermeden otomatik restart yapılır
                        try:
                            from app import app as flask_app
                            if hasattr(flask_app, 'broadcast_log'):
                                flask_app.broadcast_log('ERROR', f'Continuous loop crashed: {e}', 'scheduler')
                        except Exception:
                            pass

                self.scheduler_thread = threading.Thread(target=run_continuous_full_loop, daemon=False)
                self.scheduler_thread.start()
                logger.info("✅ Continuous automation loop started")
                # Idle watchdog (tek sefer başlatılacak)
                def _idle_monitor():
                    max_idle = int(float(os.getenv('MAX_IDLE_SECONDS', '900')))  # 15 dk varsayılan
                    while True:
                        try:
                            now = time.time()
                            if self.is_running and (now - float(self.last_activity_ts or 0.0)) > max_idle:
                                if not self.user_stopped:
                                    logger.warning(f"⏰ Idle watchdog: {int(now - self.last_activity_ts)}s hareketsizlik. Restart ediliyor...")
                                    try:
                                        self.is_running = False
                                        try:
                                            schedule.clear()
                                        except Exception:
                                            pass
                                        time.sleep(0.2)
                                        self.start_scheduler()
                                    except Exception as _idle_err:
                                        logger.error(f"Idle watchdog restart failed: {_idle_err}")
                        except Exception:
                            pass
                        time.sleep(30)
                if self._idle_watchdog_thread is None or not self._idle_watchdog_thread.is_alive():
                    self._idle_watchdog_thread = threading.Thread(target=_idle_monitor, daemon=True)
                    self._idle_watchdog_thread.start()
                return True

            # Scheduled mod kaldırıldı
            logger.info("ℹ️ Scheduled mode is removed; running continuous loop only")
            return True
            
        except Exception as e:
            logger.error(f"❌ Scheduler başlatma hatası: {e}")
            self.is_running = False
            return False
    
    def stop_scheduler(self):
        """Scheduler'ı durdur"""
        try:
            if not self.is_running:
                logger.warning("⚠️ Scheduler zaten durmuş")
                return True
            
            logger.info("🛑 Automated Data Pipeline durduruluyor...")
            
            # History: clear file on explicit stop (requested behavior)
            try:
                log_dir = '/opt/bist-pattern/logs'
                os.makedirs(log_dir, exist_ok=True)
                status_file = os.path.join(log_dir, 'pipeline_status.json')
                with open(status_file, 'w') as f:
                    json.dump({'history': []}, f)
            except Exception:
                pass

            self.is_running = False
            self.user_stopped = True
            schedule.clear()
            
            # Thread'in bitmesini bekle
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            logger.info("✅ Automated Data Pipeline durduruldu")
            return True
            
        except Exception as e:
            logger.error(f"❌ Scheduler durdurma hatası: {e}")
            return False
    
    def get_scheduler_status(self):
        """Scheduler durumu"""
        try:
            # Thread health check
            thread_alive = self.scheduler_thread.is_alive() if self.scheduler_thread else False
            
            # Debug: Thread death detection (NO auto-restart)
            if self.is_running and not thread_alive:
                logger.error("❌ CRITICAL: Scheduler thread öldü! Root cause araştırılmalı.")
                logger.error("🔍 Thread alive: False, is_running: True - Bu durumun sebebi bulunmalı")
                # Otomatik yeniden başlatma (watchdog) - varsayılan açık
                try:
                    enabled = str(os.getenv('ENABLE_SCHEDULER_WATCHDOG', 'true')).lower() in ('1', 'true', 'yes')
                except Exception:
                    enabled = True
                if enabled:
                    try:
                        now = time.time()
                        # 30 sn'den sık restart etme
                        if now - float(self.last_watchdog_restart_ts or 0.0) > 30.0:
                            logger.warning("🛠️ Watchdog: Scheduler thread dead, restarting...")
                            self.last_watchdog_restart_ts = now
                            def _do_restart():
                                try:
                                    # Güvenli sıfırlama ve yeniden başlat
                                    self.is_running = False
                                    try:
                                        schedule.clear()
                                    except Exception:
                                        pass
                                    # Kısa gecikme ile yeniden başlat
                                    time.sleep(0.2)
                                    self.start_scheduler()
                                except Exception as e:
                                    logger.error(f"Watchdog restart failed: {e}")
                            threading.Thread(target=_do_restart, daemon=True).start()
                    except Exception as _wd_err:
                        logger.error(f"Watchdog error: {_wd_err}")
            
            # schedule.jobs içi boş olsa bile (pure loop modunda) UI'da 1 iş gösterelim
            try:
                scheduled_jobs_count = len(schedule.jobs)
            except Exception:
                scheduled_jobs_count = 0
            if self.is_running and scheduled_jobs_count == 0:
                scheduled_jobs_count = 1

            status = {
                'is_running': self.is_running,
                'thread_alive': thread_alive,
                'scheduled_jobs': scheduled_jobs_count,
                'last_run_stats': self.last_run_stats,
                'next_runs': []
            }
            
            # Sonraki çalışma zamanları
            for job in schedule.jobs:
                try:
                    next_run = job.next_run
                    status['next_runs'].append({
                        'job': str(job.job_func.__name__),
                        'next_run': next_run.isoformat() if next_run else None
                    })
                except:
                    pass
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Status alma hatası: {e}")
            return {'error': str(e)}
    
    def run_manual_task(self, task_name):
        """Manuel görev çalıştırma"""
        try:
            logger.info(f"🔧 Manuel görev çalıştırılıyor: {task_name}")
            
            # Available tasks (some migrated to daemon)
            task_map = {
                'data_collection': self.daily_data_collection,
                'status_report': self.daily_status_report,
                'weekly_collection': self.weekly_full_collection,
                'bulk_predictions': self.run_bulk_predictions_all,
            }
            
            # Migrated to daemon tasks 
            migrated_tasks = ['model_retraining', 'health_check']
            if task_name in migrated_tasks:
                logger.warning(f"⚠️ {task_name} migrated to scheduler_daemon.py")
                return {"status": "migrated", "message": f"{task_name} is now handled by daemon"}
            
            if task_name not in task_map:
                logger.error(f"❌ Bilinmeyen görev: {task_name}")
                return False
            
            result = task_map[task_name]()
            logger.info(f"✅ Manuel görev tamamlandı: {task_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Manuel görev hatası: {e}")
            return False

    def run_priority_pipeline(self):
        """Öncelikli toplama → AI analizi"""
        try:
            from advanced_collector import AdvancedBISTCollector
            from pattern_detector import HybridPatternDetector
            logger.info("🚀 Öncelikli pipeline başlıyor: veri toplama")
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('INFO', '🔄 Öncelikli veri toplama başlıyor', 'collector')
            except Exception:
                pass
            collector = AdvancedBISTCollector()
            col_res = collector.collect_priority_stocks()
            logger.info(f"✅ Öncelikli toplama bitti: {col_res}")
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('SUCCESS', f"✅ Öncelikli toplama bitti: {col_res}", 'collector')
            except Exception:
                pass
            # AI analizi
            logger.info("🧠 AI analizi başlıyor (öncelikli)")
            det = HybridPatternDetector()
            analyzed = 0
            try:
                from app import app as flask_app
                with flask_app.app_context():
                    from models import Stock
                    # Öncelikli semboller veya aktiflerden ilk 100
                    priority = getattr(__import__('config').config['default'], 'PRIORITY_SYMBOLS', [])
                    symbols = priority or [s.symbol for s in Stock.query.filter_by(is_active=True).limit(100).all()]
            except Exception:
                symbols = []
            for sym in symbols[:100]:
                try:
                    det.analyze_stock(sym)
                    analyzed += 1
                except Exception:
                    continue
            logger.info(f"🎯 AI analizi tamamlandı: {analyzed} hisse")
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('SUCCESS', f"🎯 AI analizi tamamlandı: {analyzed} hisse", 'ai_analysis')
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"❌ Öncelikli pipeline hatası: {e}")
            return False

    def run_full_pipeline(self):
        """Tam toplama → AI analizi"""
        try:
            from advanced_collector import AdvancedBISTCollector
            from pattern_detector import HybridPatternDetector
            logger.info("🚀 Tam pipeline başlıyor: veri toplama")
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('INFO', '📊 Tam veri toplama başlıyor', 'collector')
            except Exception:
                pass
            collector = AdvancedBISTCollector()
            res = collector.collect_all_stocks_parallel()
            logger.info(f"✅ Tam toplama bitti: {res}")
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('SUCCESS', f"✅ Tam toplama bitti: {res}", 'collector')
            except Exception:
                pass
            # AI analizi
            logger.info("🧠 AI analizi başlıyor (tam)")
            det = HybridPatternDetector()
            analyzed = 0
            try:
                from app import app as flask_app
                with flask_app.app_context():
                    from models import Stock
                    symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
            except Exception:
                symbols = []
            for sym in symbols[:600]:
                try:
                    det.analyze_stock(sym)
                    analyzed += 1
                except Exception:
                    continue
            logger.info(f"🎯 AI analizi tamamlandı: {analyzed} hisse")
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('SUCCESS', f"🎯 AI analizi tamamlandı: {analyzed} hisse", 'ai_analysis')
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"❌ Tam pipeline hatası: {e}")
            return False
# Global singleton instance
_automated_pipeline = None

def get_automated_pipeline():
    """Automated Pipeline singleton'ını döndür"""
    global _automated_pipeline
    if _automated_pipeline is None:
        _automated_pipeline = AutomatedDataPipeline()
    return _automated_pipeline

if __name__ == "__main__":
    # Test run
    pipeline = get_automated_pipeline()
    
    print("🚀 Automated Data Pipeline Test başlatılıyor...")
    
    # Manuel görev testleri
    # Health check migrated to daemon; skip direct call here to avoid missing method errors
    print("\n📊 Health Check Test: (skipped - handled by scheduler_daemon.py)")
    
    print("\n📈 Data Collection Test:")
    data_result = pipeline.daily_data_collection()
    print(f"Result: {data_result}")
    
    print("\n📋 Status Report Test:")
    report = pipeline.daily_status_report()
    if report:
        print("Report generated successfully")
    
    print("\n⏰ Scheduler Start Test:")
    if pipeline.start_scheduler():
        print("✅ Scheduler başlatıldı")
        
        # 30 saniye bekle
        print("⏳ 30 saniye test...")
        time.sleep(30)
        
        # Status kontrol
        status = pipeline.get_scheduler_status()
        print(f"📊 Scheduler Status: {status}")
        
        # Durdur
        pipeline.stop_scheduler()
        print("🛑 Scheduler durduruldu")
    else:
        print("❌ Scheduler başlatılamadı")
    
    print("\n🎯 Test tamamlandı!")
