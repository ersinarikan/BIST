import schedule
import time
import threading
from datetime import datetime
from advanced_collector import AdvancedBISTCollector
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BISTSchedulerService:
    def __init__(self):
        self.collector = AdvancedBISTCollector()
        self.is_running = False
        self.scheduler_thread = None
        
    def daily_data_collection(self):
        """Günlük veri toplama görevi"""
        logger.info("📅 Günlük veri toplama başlatılıyor...")
        try:
            result = self.collector.collect_priority_stocks()
            logger.info(f"Günlük toplama tamamlandı: {result['success_count']} başarılı, {result['total_records']} kayıt")
        except Exception as e:
            logger.error(f"Günlük toplama hatası: {e}")
    
    def weekly_full_collection(self):
        """Haftalık tam veri toplama"""
        logger.info("📅 Haftalık tam veri toplama başlatılıyor...")
        try:
            result = self.collector.collect_all_stocks_parallel(batch_size=30)
            logger.info(f"Haftalık toplama tamamlandı: {result['success_count']} başarılı, {result['total_records']} kayıt")
        except Exception as e:
            logger.error(f"Haftalık toplama hatası: {e}")
    
    def start_scheduler(self):
        """Scheduler'ı başlat"""
        if self.is_running:
            logger.warning("Scheduler zaten çalışıyor!")
            return
            
        logger.info("🕒 BIST Scheduler başlatılıyor...")
        
        # Zamanlama kuralları
        schedule.every().day.at("09:00").do(self.daily_data_collection)  # Her sabah 9:00
        schedule.every().day.at("18:00").do(self.daily_data_collection)  # Her akşam 18:00
        schedule.every().sunday.at("10:00").do(self.weekly_full_collection)  # Pazar 10:00
        
        self.is_running = True
        
        # Ayrı thread'de çalıştır
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Her dakika kontrol et
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ Scheduler başlatıldı!")
        logger.info("📋 Zamanlamalar:")
        logger.info("  - Günlük veri: 09:00 ve 18:00")
        logger.info("  - Haftalık tam veri: Pazar 10:00")
    
    def stop_scheduler(self):
        """Scheduler'ı durdur"""
        self.is_running = False
        schedule.clear()
        logger.info("🛑 Scheduler durduruldu!")
    
    def get_status(self):
        """Scheduler durumunu al"""
        return {
            'is_running': self.is_running,
            'next_jobs': [str(job) for job in schedule.jobs],
            'job_count': len(schedule.jobs)
        }

def main():
    """Test için çalıştır"""
    scheduler_service = BISTSchedulerService()
    
    print("🚀 BIST Scheduler Test Modu")
    print("1. Scheduler başlatılıyor...")
    scheduler_service.start_scheduler()
    
    print("2. Test veri toplama...")
    scheduler_service.daily_data_collection()
    
    print("3. Durum kontrolü...")
    status = scheduler_service.get_status()
    print(f"Çalışıyor: {status['is_running']}")
    print(f"Toplam görev: {status['job_count']}")
    
    print("✅ Test tamamlandı!")

if __name__ == "__main__":
    main()
