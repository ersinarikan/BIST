#!/usr/bin/env python3
"""
AI Systems Integration Script
Mevcut Windows BIST klasöründeki AI sistemlerini Ubuntu production'a kopyalar
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def copy_ai_files():
    """AI dosyalarını Windows BIST klasöründen Ubuntu'ya kopyala"""
    
    # Kaynak ve hedef klasörler
    source_dir = "../BIST"  # Windows BIST klasörü
    target_dir = "."        # Ubuntu production klasörü
    
    # Kopyalanacak AI dosyaları
    ai_files = [
        "advanced_patterns.py",
        "pattern_detector.py", 
        "visual_pattern_detector.py",
        "fingpt_analyzer.py",
        "alert_system.py",
        "yolov8n.pt"
    ]
    
    logger.info("🤖 AI sistemleri Ubuntu production'a kopyalanıyor...")
    
    for filename in ai_files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, target_path)
                logger.info(f"✅ {filename} kopyalandı")
            except Exception as e:
                logger.error(f"❌ {filename} kopyalanamadı: {e}")
        else:
            logger.warning(f"⚠️ {filename} bulunamadı: {source_path}")
    
    logger.info("🎯 AI sistemleri entegrasyon tamamlandı!")

if __name__ == "__main__":
    copy_ai_files()
