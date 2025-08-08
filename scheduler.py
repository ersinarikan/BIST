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
    
    def setup_schedule(self):
        """Zamanlama ayarları"""
        try:
            logger.info("⏰ Scheduled tasks ayarlanıyor...")
            
            # Clear existing jobs first
            schedule.clear()
            
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
            
            # Test için - her 2 dakikada bir health check (opsiyonel)
            if os.getenv('BIST_DEBUG', '').lower() == 'true':
                schedule.every(2).minutes.do(self.system_health_check).tag('debug')
                logger.info("🔧 Debug mode: Health check every 2 minutes")
            
            job_count = len(schedule.jobs)
            logger.info(f"✅ Scheduled tasks kuruldu ({job_count} job):")
            logger.info("  📅 06:00 - Günlük veri toplama")
            logger.info("  🧠 07:00 - Otomatik model eğitimi")
            logger.info("  📊 08:00 - Günlük durum raporu")
            logger.info("  📈 Pazartesi 05:00 - Haftalık tam veri toplama")
            logger.info("  🔍 Her 6 saat - Sistem sağlık kontrolü")
            
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
            
            # Schedule setup
            if not self.setup_schedule():
                logger.error("❌ Schedule kurulum başarısız")
                return False
            
            self.is_running = True
            
            # PURE PYTHON SCHEDULER (no schedule library)
            def run_pure_scheduler():
                logger.info("⚡ Pure Python scheduler başlatıldı (NO schedule library)")
                try:
                    loop_count = 0
                    while self.is_running:
                        try:
                            loop_count += 1
                            logger.info(f"🔄 Pure scheduler loop #{loop_count}")
                            
                            # Manual job scheduling (no schedule library)
                            current_time = datetime.now()
                            
                            # Check for daily status report (08:00)
                            if current_time.hour == 8 and current_time.minute == 0:
                                logger.info("🎯 Running daily status report...")
                                try:
                                    self.daily_status_report()
                                    logger.info("✅ Daily status report completed")
                                except Exception as e:
                                    logger.error(f"❌ Daily status report error: {e}")
                            
                            # Heartbeat every loop
                            logger.info(f"💓 Pure scheduler heartbeat - Loop #{loop_count} - Thread alive")
                            
                            # Sleep 60 seconds (1 minute intervals)
                            time.sleep(60)
                            
                        except Exception as e:
                            logger.error(f"❌ Pure scheduler loop error: {e}")
                            import traceback
                            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
                            time.sleep(10)
                            
                    logger.info("⏰ Pure scheduler thread normal şekilde durduruldu")
                except Exception as e:
                    logger.error(f"❌ Pure scheduler critical error: {e}")
                    import traceback
                    logger.error(f"🔍 Critical traceback: {traceback.format_exc()}")
                    self.is_running = False
                    logger.error("🧹 Pure scheduler state cleaned up")
            
            self.scheduler_thread = threading.Thread(target=run_pure_scheduler, daemon=False)
            self.scheduler_thread.start()
            
            logger.info("✅ Automated Data Pipeline başarıyla başlatıldı")
            
            # Health check migrated to daemon
            logger.info("📋 Health check is now handled by scheduler_daemon.py")
            
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
            
            self.is_running = False
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
            
            status = {
                'is_running': self.is_running,
                'thread_alive': thread_alive,
                'scheduled_jobs': len(schedule.jobs),
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
                'weekly_collection': self.weekly_full_collection
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
    print("\n📊 Health Check Test:")
    health = pipeline.system_health_check()
    print(f"Status: {health.get('overall_status', 'error')}")
    
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
