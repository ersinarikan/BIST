"""
BIST Visual Pattern Detection with YOLOv8
Grafik görüntülerinde pattern detection için computer vision
"""

import matplotlib
matplotlib.use('Agg')  # GUI backend kullanma

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import logging
import os
from PIL import Image

# YOLOv8 için
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 (ultralytics) not available. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)

class VisualPatternSystem:
    """YOLOv8 tabanlı görsel pattern detection sistemi"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.chart_cache = {}
        
        # YOLOv8 modelini yüklemeyi dene
        if YOLO_AVAILABLE:
            try:
                # Önceden eğitilmiş model yolu (eğer varsa)
                model_path = "models/stock_patterns_yolo.pt"
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                    self.model_loaded = True
                    logger.info("Özel YOLOv8 stock pattern modeli yüklendi")
                else:
                    # Genel YOLOv8 modeli
                    self.model = YOLO('yolov8n.pt')  # Nano model (hızlı)
                    logger.info("Genel YOLOv8 modeli yüklendi - pattern detection sınırlı")
            except Exception as e:
                logger.error(f"YOLOv8 model yükleme hatası: {e}")
    
    def generate_chart_image(self, data, symbol, width=800, height=600):
        """Hisse verisi için grafik görüntüsü oluştur"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(width/100, height/100), 
                                              gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Ana fiyat grafiği
            ax1.plot(data.index, data['close'], linewidth=2, color='#2E86C1', label='Kapanış')
            ax1.plot(data.index, data['high'], linewidth=1, color='#28B463', alpha=0.7, label='Yüksek')
            ax1.plot(data.index, data['low'], linewidth=1, color='#E74C3C', alpha=0.7, label='Düşük')
            
            # Moving averages
            if len(data) >= 20:
                sma_20 = data['close'].rolling(20).mean()
                ax1.plot(data.index, sma_20, linewidth=1, color='#F39C12', alpha=0.8, label='SMA 20')
            
            if len(data) >= 50:
                sma_50 = data['close'].rolling(50).mean()
                ax1.plot(data.index, sma_50, linewidth=1, color='#8E44AD', alpha=0.8, label='SMA 50')
            
            ax1.set_title(f'{symbol} - Fiyat Analizi', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Volume grafiği
            colors = ['#E74C3C' if data['close'].iloc[i] < data['open'].iloc[i] else '#28B463' 
                     for i in range(len(data))]
            ax2.bar(data.index, data.get('volume', [0]*len(data)), color=colors, alpha=0.6)
            ax2.set_title('Hacim', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # RSI grafiği
            if len(data) >= 14:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                ax3.plot(data.index, rsi, color='#9B59B6', linewidth=2)
                ax3.axhline(y=70, color='#E74C3C', linestyle='--', alpha=0.7)
                ax3.axhline(y=30, color='#28B463', linestyle='--', alpha=0.7)
                ax3.set_ylim(0, 100)
                ax3.set_title('RSI (14)', fontsize=12)
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Görüntüyü byte array'e çevir
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            
            # PIL Image'e çevir
            img = Image.open(img_buffer)
            
            plt.close(fig)
            
            return img
            
        except Exception as e:
            logger.error(f"Grafik oluşturma hatası: {e}")
            return None
    
    def detect_visual_patterns(self, image):
        """YOLOv8 ile görsel pattern detection"""
        try:
            if not self.model_loaded or not YOLO_AVAILABLE:
                return {
                    'patterns': [],
                    'confidence': 0.0,
                    'status': 'model_unavailable',
                    'message': 'YOLOv8 modeli yüklü değil'
                }
            
            # YOLO prediction
            results = self.model(image)
            
            detected_patterns = []
            total_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Confidence threshold
                        conf = float(box.conf[0])
                        if conf > 0.5:  # %50'den yüksek güven
                            
                            # Class ID'den pattern ismi
                            class_id = int(box.cls[0])
                            pattern_name = self.get_pattern_name(class_id)
                            
                            # Bounding box koordinatları
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            pattern_info = {
                                'pattern': pattern_name,
                                'confidence': conf,
                                'bbox': {
                                    'x1': int(x1), 'y1': int(y1),
                                    'x2': int(x2), 'y2': int(y2)
                                },
                                'area': (x2-x1) * (y2-y1)
                            }
                            
                            detected_patterns.append(pattern_info)
                            total_confidence += conf
            
            avg_confidence = total_confidence / len(detected_patterns) if detected_patterns else 0.0
            
            return {
                'patterns': detected_patterns,
                'confidence': avg_confidence,
                'pattern_count': len(detected_patterns),
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Visual pattern detection hatası: {e}")
            return {
                'patterns': [],
                'confidence': 0.0,
                'status': 'error',
                'message': str(e)
            }
    
    def get_pattern_name(self, class_id):
        """Class ID'den pattern ismini döndür"""
        # Özel model için pattern class'ları
        pattern_classes = {
            0: 'HEAD_AND_SHOULDERS',
            1: 'INVERSE_HEAD_AND_SHOULDERS',
            2: 'DOUBLE_TOP',
            3: 'DOUBLE_BOTTOM',
            4: 'TRIANGLE_ASCENDING',
            5: 'TRIANGLE_DESCENDING',
            6: 'TRIANGLE_SYMMETRICAL',
            7: 'WEDGE_RISING',
            8: 'WEDGE_FALLING',
            9: 'FLAG_BULLISH',
            10: 'FLAG_BEARISH',
            11: 'PENNANT',
            12: 'CUP_AND_HANDLE',
            13: 'CHANNEL_UP',
            14: 'CHANNEL_DOWN',
            15: 'SUPPORT_LEVEL',
            16: 'RESISTANCE_LEVEL'
        }
        
        return pattern_classes.get(class_id, f'UNKNOWN_PATTERN_{class_id}')
    
    def analyze_stock_visual(self, symbol, data):
        """Hisse için görsel analiz yap"""
        try:
            # Grafik oluştur
            chart_image = self.generate_chart_image(data, symbol)
            if chart_image is None:
                return {
                    'status': 'error',
                    'message': 'Grafik oluşturulamadı'
                }
            
            # Visual pattern detection
            visual_result = self.detect_visual_patterns(chart_image)
            
            # Sonuçları birleştir
            result = {
                'symbol': symbol,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'visual_analysis': visual_result,
                'data_points': len(data)
            }
            
            # Chart'ı base64 string'e çevir (isteğe bağlı)
            if hasattr(chart_image, 'save'):
                img_buffer = io.BytesIO()
                chart_image.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                result['chart_image'] = f"data:image/png;base64,{img_base64}"
            
            return result
            
        except Exception as e:
            logger.error(f"Visual analiz hatası: {e}")
            return {
                'symbol': symbol,
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_info(self):
        """Sistem bilgilerini döndür"""
        return {
            'yolo_available': YOLO_AVAILABLE,
            'model_loaded': self.model_loaded,
            'model_type': 'custom' if self.model_loaded and os.path.exists("models/stock_patterns_yolo.pt") else 'general',
            'cache_size': len(self.chart_cache)
        }

# Global singleton instance
_visual_system_instance = None

def get_visual_pattern_system():
    """Visual pattern system singleton'ını döndür"""
    global _visual_system_instance
    if _visual_system_instance is None:
        _visual_system_instance = VisualPatternSystem()
    return _visual_system_instance

if __name__ == "__main__":
    # Test için basit veri
    test_data = pd.DataFrame({
        'open': np.random.rand(50) * 100 + 100,
        'high': np.random.rand(50) * 100 + 110,
        'low': np.random.rand(50) * 100 + 90,
        'close': np.random.rand(50) * 100 + 100,
        'volume': np.random.rand(50) * 1000000
    })
    
    visual_system = get_visual_pattern_system()
    result = visual_system.analyze_stock_visual("TEST", test_data)
    print("Visual Pattern System Test:")
    print(f"Status: {result['status']}")
    if 'visual_analysis' in result:
        print(f"Pattern Count: {result['visual_analysis'].get('pattern_count', 0)}")
