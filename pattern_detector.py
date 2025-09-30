"""
Hibrit Pattern Detection Sistemi
TA-Lib + YOLOv8 + FinGPT kombinasyonu ile kesin formasyon tespiti
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging
import os
import math
import time
import json
try:
    import fcntl  # Posix lock
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore
from models import Stock, StockPrice
from app import app
from config import config
from bist_pattern.utils.debug_utils import ddebug as _ddebug
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# Gelişmiş pattern detection sistemi
try:
    from advanced_patterns import AdvancedPatternDetector
    ADVANCED_PATTERNS_AVAILABLE = True
except ImportError:
    ADVANCED_PATTERNS_AVAILABLE = False
    logger.warning("⚠️ Advanced patterns modülü yüklenemedi")

# Visual pattern detection sistemi (now using async version)
VISUAL_PATTERNS_AVAILABLE = True  # Always available with async implementation

# ML Prediction sistemi
try:
    from ml_prediction_system import get_ml_prediction_system
    ML_PREDICTION_AVAILABLE = True
except ImportError:
    ML_PREDICTION_AVAILABLE = False
    logger.warning("⚠️ ML Prediction modülü yüklenemedi")

# Enhanced ML Prediction sistemi (opsiyonel)
try:
    from enhanced_ml_system import get_enhanced_ml_system
    ENHANCED_ML_AVAILABLE = True
except ImportError:
    ENHANCED_ML_AVAILABLE = False  # Import failed
    logger.warning("⚠️ Enhanced ML Prediction modülü yüklenemedi")


class HybridPatternDetector:

    def __init__(self):
        # Cache sistemi (TTL + boyut sınırı) - tamamen environment-driven
        self.cache = {}
        try:
            self.cache_ttl = int(os.getenv('PATTERN_RESULT_CACHE_TTL', str(getattr(config['default'], 'PATTERN_CACHE_TTL', 300))))
        except Exception:
            self.cache_ttl = 300
        try:
            self.result_cache_max_size = int(os.getenv('PATTERN_RESULT_CACHE_MAX_SIZE', '200'))
        except Exception:
            self.result_cache_max_size = 200
        # DataFrame kısa süreli cache (DB yükünü azaltmak için)
        try:
            self.data_cache_ttl = int(os.getenv('PATTERN_DATA_CACHE_TTL', '60'))
        except Exception:
            self.data_cache_ttl = 60
        try:
            self.df_cache_max_size = int(os.getenv('PATTERN_DF_CACHE_MAX_SIZE', '512'))
        except Exception:
            self.df_cache_max_size = 512
        self._df_cache: dict[str, dict] = {}
        self._bulk_write_ts: dict[str, float] = {}
        
        # Gelişmiş pattern detector
        if ADVANCED_PATTERNS_AVAILABLE:
            self.advanced_detector = AdvancedPatternDetector()
        else:
            self.advanced_detector = None
            
        # Visual pattern detector - SIMPLE SYNC VERSION
        self.visual_detector = None
        self._yolo_model = None
        yolo_enabled = getattr(config['default'], 'ENABLE_YOLO', True)
        logger.info(f"🔍 YOLO config check: ENABLE_YOLO={yolo_enabled}")
        
        if yolo_enabled:
            try:
                # Test YOLO availability
                model_path = os.getenv('YOLO_MODEL_PATH', '/opt/bist-pattern/yolo/patterns_all_v7_rectblend.pt')
                if os.path.exists(model_path):
                    self.visual_detector = True  # Simple flag
                    logger.info(f"✅ SYNC YOLO enabled: {model_path}")
                else:
                    logger.error(f"❌ YOLO model not found: {model_path}")
            except Exception as e:
                logger.error(f"❌ YOLO init failed: {e}")
        else:
            logger.info("🔍 YOLO disabled by config")
            
        # ML Prediction system
        if ML_PREDICTION_AVAILABLE:
            self.ml_predictor = get_ml_prediction_system()
        else:
            self.ml_predictor = None
        
        # Enhanced ML Prediction system (optional)
        if ENHANCED_ML_AVAILABLE:
            try:
                self.enhanced_ml = get_enhanced_ml_system()
                logger.info("🚀 Enhanced ML system initialized")
            except Exception as e:
                self.enhanced_ml = None
                logger.warning(f"Enhanced ML init failed: {e}")
        else:
            self.enhanced_ml = None

        # FinGPT / FinBERT (optional sentiment; gated by ENABLE_FINGPT)
        self.fingpt = None
        self.fingpt_available = False
        if config['default'].ENABLE_FINGPT:
            try:
                from fingpt_analyzer import get_fingpt_analyzer  # type: ignore
                self.fingpt = get_fingpt_analyzer()
                self.fingpt_available = True
            except Exception:
                self.fingpt = None
                self.fingpt_available = False
        # Async RSS News Provider (non-blocking)
        try:
            from rss_news_async import get_async_rss_news_provider
            self._async_rss_provider = get_async_rss_news_provider()
            logger.info("📰 Async RSS News Provider initialized")
        except Exception as e:
            self._async_rss_provider = None
            logger.warning(f"Async RSS News Provider init failed: {e}")
            
        logger.info("🤖 Hybrid Pattern Detector başlatıldı")
    
    def _cleanup_cache(self):
        """Cache temizliği ve boyut kesmesi (LRU-vari) - Thread safe"""
        try:
            current_time = datetime.now().timestamp()
            expired_keys = []
            
            # Thread-safe cache iteration
            cache_items = list(self.cache.items())
            for key, entry in cache_items:
                if isinstance(entry, dict) and 'timestamp' in entry:
                    if current_time - float(entry['timestamp']) > float(self.cache_ttl):
                        expired_keys.append(key)
                else:
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                self.cache.pop(key, None)
            removed = len(expired_keys)

            # Boyut sınırını aşarsa en eski kayıtları at
            if len(self.cache) > self.result_cache_max_size:
                try:
                    items = sorted(self.cache.items(), key=lambda kv: float(kv[1].get('timestamp', 0)))
                except Exception:
                    items = list(self.cache.items())
                to_remove = max(0, len(self.cache) - self.result_cache_max_size)
                for i in range(to_remove):
                    self.cache.pop(items[i][0], None)
                removed += to_remove

            if removed:
                logger.info(f"Cache cleanup: removed={removed} size={len(self.cache)} ttl={self.cache_ttl}s max={self.result_cache_max_size}")
                
            # Schedule automatic cleanup every 10 minutes
            if not hasattr(self, '_cleanup_scheduled'):
                import threading
            
                def schedule_cleanup():
                    time.sleep(600)  # 10 minutes
                    self._cleanup_cache()
                    self._cleanup_scheduled = False
                
                if not getattr(self, '_cleanup_scheduled', False):
                    self._cleanup_scheduled = True
                    cleanup_thread = threading.Thread(target=schedule_cleanup, daemon=True)
                    cleanup_thread.start()
                    
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

    def _prune_df_cache(self) -> None:
        """DataFrame cache boyutunu sınırla"""
        try:
            if len(self._df_cache) <= self.df_cache_max_size:
                return
            try:
                items = sorted(self._df_cache.items(), key=lambda kv: float(kv[1].get('ts', 0)))
            except Exception:
                items = list(self._df_cache.items())
            to_remove = max(0, len(self._df_cache) - self.df_cache_max_size)
            for i in range(to_remove):
                self._df_cache.pop(items[i][0], None)
            logger.info(f"DF cache prune: removed={to_remove} size={len(self._df_cache)} max={self.df_cache_max_size}")
        except Exception:
            pass
    
    def get_visual_signal(self, pattern_name):
        """Visual pattern'den sinyal türünü belirle"""
        bearish_patterns = [
            'HEAD_AND_SHOULDERS', 'DOUBLE_TOP', 'TRIANGLE_DESCENDING',
            'WEDGE_RISING', 'FLAG_BEARISH', 'CHANNEL_DOWN', 'RESISTANCE_LEVEL'
        ]
        
        bullish_patterns = [
            'INVERSE_HEAD_AND_SHOULDERS', 'DOUBLE_BOTTOM', 'TRIANGLE_ASCENDING',
            'WEDGE_FALLING', 'FLAG_BULLISH', 'CUP_AND_HANDLE', 'CHANNEL_UP', 'SUPPORT_LEVEL'
        ]
        
        if pattern_name in bearish_patterns:
            return 'BEARISH'
        elif pattern_name in bullish_patterns:
            return 'BULLISH'
        else:
            return 'NEUTRAL'
    
    def get_stock_data(self, symbol, days=None):
        """PostgreSQL'den hisse verilerini al"""
        try:
            # Kısa süreli cache: aynı sembol için tekrar DB'yi yormayalım
            now_ts = time.time()
            cached = self._df_cache.get(symbol)
            if cached and (now_ts - float(cached.get('ts', 0))) < float(self.data_cache_ttl):
                df_cached = cached.get('df')
                if df_cached is not None:
                    return df_cached
            # Default days from config
            try:
                default_days = int(getattr(config['default'], 'PATTERN_DATA_DAYS', 365))
            except Exception:
                default_days = 365
            days = days or default_days
            with app.app_context():
                # Stock ID'yi bul
                stock = Stock.query.filter_by(symbol=symbol.upper()).first()
                if not stock:
                    logger.warning(f"Hisse bulunamadı: {symbol}")
                    # Yahoo Finance fallback dene
                    yahoo_data = self._try_yahoo_finance_fallback(symbol, days)
                    if yahoo_data is not None:
                        return yahoo_data
                    return None
                
                # Son N günlük veriyi al
                prices = StockPrice.query.filter_by(stock_id=stock.id)\
                            .order_by(StockPrice.date.desc())\
                            .limit(days).all()
            
            if not prices:
                logger.warning(f"Fiyat verisi bulunamadı: {symbol}")
                # Yahoo Finance fallback dene
                yahoo_data = self._try_yahoo_finance_fallback(symbol, days)
                if yahoo_data is not None:
                    return yahoo_data
                return None
            
            # DataFrame'e çevir
            data = []
            for price in reversed(prices):  # Tarihe göre sırala
                data.append({
                    'date': price.date,
                    'open': float(price.open_price),
                    'high': float(price.high_price),
                    'low': float(price.low_price),
                    'close': float(price.close_price),
                    'volume': int(price.volume)
                })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            # Cache'e koy
            try:
                self._df_cache[symbol] = {'df': df, 'ts': now_ts}
                self._prune_df_cache()
            except Exception:
                pass
            
            return df
            
        except Exception as e:
            logger.error(f"Veri alma hatası {symbol}: {e}")
            # Son çare Yahoo Finance fallback
            yahoo_data = self._try_yahoo_finance_fallback(symbol, days)
            if yahoo_data is not None:
                return yahoo_data
            return None
    
    def _try_yahoo_finance_fallback(self, symbol, days=None):
        """
        Yahoo Finance fallback sistemi (enhanced curl_cffi ile)
        PostgreSQL'den veri alınamadığında kullanılır
        """
        if not config['default'].ENABLE_YAHOO_FALLBACK:
            _ddebug(f"{symbol}: Yahoo Finance fallback devre dışı", logger)
            return None
        
        try:
            logger.info(f"🌐 {symbol}: Yahoo Finance fallback başlatılıyor...")
            
            # BIST sembolünü sanitize et ve Yahoo Finance formatına çevir
            try:
                from bist_pattern.utils.symbols import sanitize_symbol, to_yf_symbol
                clean = sanitize_symbol(symbol)
                yf_symbol = to_yf_symbol(clean)
            except Exception:
                yf_symbol = (symbol or '').upper()
                if not yf_symbol.endswith('.IS'):
                    yf_symbol = f"{yf_symbol}.IS"
            
            # Period hesaplama
            days = days or config['default'].PATTERN_DATA_DAYS
            if days <= 30:
                period = '1mo'
            elif days <= 90:
                period = '3mo'
            elif days <= 180:
                period = '6mo'
            elif days <= 365:
                period = '1y'
            else:
                period = '2y'
            
            # Enhanced Yahoo Finance kullan (SYNC, asyncio çatışmasını önlemek için)
            if config['default'].YF_ENHANCED_ENABLED:
                try:
                    from yahoo_finance_enhanced import get_enhanced_yahoo_finance_wrapper
                    wrapper = get_enhanced_yahoo_finance_wrapper()

                    # Use thread-safe sync method to avoid 'event loop is running' errors in gevent/gunicorn
                    try:
                        result = wrapper.fetch_data_sync(
                            symbol, yf_symbol, period,
                            timeout=min(float(getattr(config['default'], 'YF_FALLBACK_TIMEOUT', 30.0)), 60.0)
                        )
                    except Exception as e:
                        logger.error(f"Enhanced Yahoo sync fetch error {symbol}: {e}")
                        result = {'success': False, 'error': str(e)}
                    
                    if result['success']:
                        data = result['data']
                        logger.info(f"✅ {symbol}: Enhanced Yahoo Finance fallback başarılı! "
                                  f"({len(data)} satır, method: {result.get('method')})")
                        
                        # DataFrame'i uygun formata çevir
                        processed_data = data.copy()
                        processed_data.columns = [col.lower() for col in processed_data.columns]
                        
                        # Cache'e kaydet
                        try:
                            now_ts = time.time()
                            self._df_cache[symbol] = {'df': processed_data, 'ts': now_ts}
                            self._prune_df_cache()
                        except Exception:
                            pass
                        
                        return processed_data
                    else:
                        logger.warning(f"⚠️ {symbol}: Enhanced Yahoo Finance fallback failed: {result.get('error')}")
                        
                except Exception as e:
                    logger.error(f"❌ {symbol}: Enhanced Yahoo Finance fallback error: {e}")
            
            # Standart fallback (curl_cffi'li gevent native)
            try:
                from yfinance_gevent_native import get_native_yfinance_wrapper
                native_wrapper = get_native_yfinance_wrapper()
                
                result = native_wrapper.fetch_data_native_async(
                    symbol, yf_symbol, period, 
                    timeout=min(config['default'].YF_FALLBACK_TIMEOUT, 30.0)
                )
                
                if result['success']:
                    data = result['data']
                    logger.info(f"✅ {symbol}: Native Yahoo Finance fallback başarılı! ({len(data)} satır)")
                    
                    # DataFrame'i uygun formata çevir
                    processed_data = data.copy()
                    processed_data.columns = [col.lower() for col in processed_data.columns]
                    
                    # Cache'e kaydet
                    try:
                        now_ts = time.time()
                        self._df_cache[symbol] = {'df': processed_data, 'ts': now_ts}
                        self._prune_df_cache()
                    except Exception:
                        pass
                    
                    return processed_data
                else:
                    logger.warning(f"⚠️ {symbol}: Native Yahoo Finance fallback failed: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Native Yahoo Finance fallback error: {e}")
            
            logger.warning(f"💀 {symbol}: Tüm Yahoo Finance fallback yöntemleri başarısız")
            return None
            
        except Exception as e:
            logger.error(f"💥 {symbol}: Yahoo Finance fallback critical error: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Teknik indikatörleri hesapla"""
        try:
            if len(data) < 20:
                return {}
            
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = float(data['close'].rolling(20).mean().iloc[-1])
            indicators['sma_50'] = float(data['close'].rolling(50).mean().iloc[-1]) if len(data) >= 50 else None
            indicators['ema_12'] = float(data['close'].ewm(span=12).mean().iloc[-1])
            indicators['ema_26'] = float(data['close'].ewm(span=26).mean().iloc[-1])
            
            # RSI calculation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = float((100 - (100 / (1 + rs))).iloc[-1])
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            macd_signal = macd_line.ewm(span=9).mean()
            indicators['macd'] = float(macd_line.iloc[-1])
            indicators['macd_signal'] = float(macd_signal.iloc[-1])
            indicators['macd_histogram'] = float((macd_line - macd_signal).iloc[-1])
            
            # Bollinger Bands
            sma_20 = data['close'].rolling(20).mean()
            std_20 = data['close'].rolling(20).std()
            bb_upper = float((sma_20 + (std_20 * 2)).iloc[-1])
            bb_lower = float((sma_20 - (std_20 * 2)).iloc[-1])
            indicators['bb_upper'] = bb_upper
            indicators['bb_lower'] = bb_lower
            indicators['bb_position'] = float((data['close'].iloc[-1] - bb_lower) / (bb_upper - bb_lower))
            
            # Support/Resistance levels
            high_max = data['high'].rolling(20).max()
            low_min = data['low'].rolling(20).min()
            indicators['resistance'] = float(high_max.iloc[-1])
            indicators['support'] = float(low_min.iloc[-1])
            
            return indicators
            
        except Exception as e:
            logger.error(f"Teknik indikatör hesaplama hatası: {e}")
            return {}
    
    def detect_basic_patterns(self, data):
        """Temel pattern detection"""
        try:
            patterns = []
            
            if len(data) < 20:
                return patterns
            
            highs = data['high'].values  # noqa: F841
            lows = data['low'].values  # noqa: F841
            closes = data['close'].values
            
            # Trend detection
            sma_5 = data['close'].rolling(5).mean()
            sma_20 = data['close'].rolling(20).mean()
            
            # current_trend bilgisi şu an kullanılmıyor; ileride rapora eklenebilir
            _current_trend = "BULLISH" if sma_5.iloc[-1] > sma_20.iloc[-1] else "BEARISH"  # noqa: F841
            
            # Price action patterns
            current_price = closes[-1]
            prev_price = closes[-2] if len(closes) > 1 else current_price
            
            price_change = (current_price - prev_price) / prev_price * 100
            
            # Volume analysis
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Basic pattern detection
            if abs(price_change) > 3 and volume_ratio > 1.5:  # Strong move with volume
                pattern_type = "BREAKOUT_UP" if price_change > 0 else "BREAKDOWN"
                # Varsayılan vurgulama aralığı: son 20 bar
                start_idx = max(0, len(data) - 20)
                end_idx = len(data) - 1
                patterns.append({
                    'pattern': pattern_type,
                    'signal': 'BULLISH' if price_change > 0 else 'BEARISH',
                    'confidence': min(0.8, 0.5 + abs(price_change) / 10),
                    'strength': min(100, 50 + abs(price_change) * 10),
                    'price_change': price_change,
                    'volume_ratio': volume_ratio,
                    'source': 'BASIC',
                    'range': {
                        'start_index': int(start_idx),
                        'end_index': int(end_idx)
                    }
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Basic pattern detection hatası: {e}")
            return []
    
    def analyze_stock(self, symbol):
        """Hisse analizi yap"""
        try:
            try:
                # Progress broadcast: analysis start (best-effort)
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('INFO', f'🧠 AI analiz başlıyor: {symbol}', 'ai_analysis')  # type: ignore[attr-defined]
            except Exception:
                pass
            # Cache kontrolü - TTL ile
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            current_time = datetime.now().timestamp()
            
            # Cache'de var mı ve TTL süresi geçmemiş mi?
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                if isinstance(cache_entry, dict) and 'timestamp' in cache_entry:
                    if current_time - cache_entry['timestamp'] < self.cache_ttl:
                        logger.info(f"Cache hit for {symbol}")
                        return cache_entry['data']
                    else:
                        # TTL süresi geçmiş, cache'den sil
                        del self.cache[cache_key]
                        logger.info(f"Cache expired for {symbol}")
                elif isinstance(cache_entry, dict) and 'timestamp' not in cache_entry:
                    # Eski format cache, direkt kullan ama sonra güncellenecek
                    logger.info(f"Legacy cache hit for {symbol}")
                    return cache_entry
            
            # Veri al
            data = self.get_stock_data(symbol)
            if data is None or len(data) < 10:
                return {
                    'symbol': symbol,
                    'status': 'insufficient_data',
                    'message': 'Yeterli veri bulunamadı'
                }
            
            # Teknik indikatörler
            indicators = self.calculate_technical_indicators(data)
            
            # ==========================================
            # PATTERN DETECTION WITH VALIDATION PIPELINE
            # ==========================================
            # Collect patterns from all sources, then validate
            basic_patterns = []
            advanced_patterns = []
            yolo_patterns_raw = []
            
            # Stage 1: Basic TA patterns
            basic_patterns = self.detect_basic_patterns(data)
            
            # Stage 2: Advanced TA patterns (if available)
            if self.advanced_detector and ADVANCED_PATTERNS_AVAILABLE:
                try:
                    adv_raw = self.advanced_detector.analyze_all_patterns(data)
                    # Normalize and ensure range info
                    for ap in (adv_raw or []):
                        try:
                            if isinstance(ap, dict):
                                if 'source' not in ap:
                                    ap['source'] = 'ADVANCED_TA'
                                rng = ap.get('range') if isinstance(ap.get('range'), dict) else None
                                if not rng:
                                    ap['range'] = {
                                        'start_index': int(max(0, len(data) - 30)),
                                        'end_index': int(len(data) - 1)
                                    }
                                advanced_patterns.append(ap)
                            else:
                                advanced_patterns.append(ap)
                        except Exception:
                            advanced_patterns.append(ap)
                except Exception as e:
                    logger.error(f"Advanced pattern analysis hatası: {e}")
            
            # Visual pattern analysis - ASYNC NON-BLOCKING VERSION
            if self.visual_detector and config['default'].ENABLE_YOLO:
                try:
                    import threading
                    from concurrent.futures import ThreadPoolExecutor
                    
                    # Initialize thread pool if not exists (environment-driven)
                    if not hasattr(self, '_visual_thread_pool'):
                        try:
                            max_workers = int(os.getenv('VISUAL_THREAD_POOL_WORKERS', '1'))
                        except Exception:
                            max_workers = 1
                        self._visual_thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="YOLO")
                        self._visual_results = {}
                        self._visual_lock = threading.Lock()
                    
                    def _async_yolo_analysis():
                        """Background YOLO analysis"""
                        try:
                            from ultralytics import YOLO
                            from PIL import Image
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            import io
                            
                            model_path = os.getenv('YOLO_MODEL_PATH', '/opt/bist-pattern/yolo/patterns_all_v7_rectblend.pt')
                            # CRITICAL FIX: Raised from 0.12/0.22 to 0.45 for realistic pattern detection
                            # Lower values caused excessive false positives in production
                            min_conf = float(os.getenv('YOLO_MIN_CONF', '0.45'))
                            
                            # Load model (cached)
                            if not hasattr(self, '_yolo_model') or self._yolo_model is None:
                                self._yolo_model = YOLO(model_path)
                            
                            # Create lightweight chart
                            if len(data) >= 20:
                                fig, ax = plt.subplots(figsize=(6, 3))  # Very small
                                recent_data = data.tail(30)  # Only 30 days
                                ax.plot(recent_data['close'], linewidth=1, color='blue')
                                ax.axis('off')  # No labels for speed
                                
                                buf = io.BytesIO()
                                plt.savefig(buf, format='png', dpi=50, bbox_inches='tight')  # Very low DPI
                                buf.seek(0)
                                img = Image.open(buf)
                                plt.close(fig)
                                
                                # Fast YOLO prediction
                                results = self._yolo_model.predict(img, conf=min_conf, verbose=False, imgsz=320)  # Small image
                                
                                visual_patterns = []
                                if results and len(results) > 0:
                                    result = results[0]
                                    if hasattr(result, 'boxes') and result.boxes is not None:
                                        boxes = result.boxes
                                        if hasattr(boxes, 'cls') and hasattr(boxes, 'conf'):
                                            names = getattr(result, 'names', {}) or getattr(self._yolo_model, 'names', {})
                                            num_detections = len(boxes.cls) if hasattr(boxes, 'cls') else 0
                                            
                                            for i in range(min(num_detections, 3)):  # Max 3 patterns
                                                try:
                                                    conf = float(boxes.conf[i])
                                                    cls_idx = int(boxes.cls[i])
                                                    if conf >= min_conf:
                                                        pattern_name = str(names.get(cls_idx, f'pattern_{cls_idx}'))
                                                        visual_patterns.append({
                                                            'pattern': pattern_name,
                                                            'confidence': conf,
                                                            'signal': self.get_visual_signal(pattern_name),
                                                            'strength': int(conf * 100),
                                                            'source': 'VISUAL_YOLO',
                                                            'details': {'detection_index': i},
                                                            'range': {'start_index': max(0, len(data) - 30), 'end_index': len(data) - 1}
                                                        })
                                                except Exception:
                                                    continue
                                
                                # Store result in thread-safe way
                                with self._visual_lock:
                                    self._visual_results[symbol] = {
                                        'patterns': visual_patterns,
                                        'timestamp': time.time(),
                                        'count': len(visual_patterns)
                                    }
                                
                                return len(visual_patterns)
                            
                        except Exception as e:
                            logger.warning(f"Background YOLO error for {symbol}: {e}")
                            return 0
                    
                    # Submit to background thread (non-blocking)
                    self._visual_thread_pool.submit(_async_yolo_analysis)
                    
                    # Check for immediate cached result (from previous analysis)
                    cached_result = None
                    try:
                        with self._visual_lock:
                            cached = self._visual_results.get(symbol)
                            if cached and (time.time() - cached['timestamp']) < 300:  # 5 min cache
                                cached_result = cached
                    except Exception:
                        pass
                    
                    # Stage 3: Collect YOLO patterns for validation
                    if cached_result:
                        cached_patterns = cached_result.get('patterns', [])
                        yolo_patterns_raw.extend(cached_patterns)
                        logger.info(f"📸 Cached YOLO patterns for {symbol}: {len(cached_patterns)}")
                    else:
                        logger.info(f"🔄 YOLO analysis queued for {symbol} (background)")
                    
                except Exception as e:
                    logger.error(f"Async YOLO setup error for {symbol}: {e}")
            
            # ==========================================
            # VALIDATION PIPELINE: Multi-stage pattern validation
            # ==========================================
            patterns = []
            # ✅ Pattern Validation with standalone ADVANCED/YOLO support
            validation_enabled = str(os.getenv('ENABLE_PATTERN_VALIDATION', 'True')).lower() == 'true'
            
            if validation_enabled:
                try:
                    from bist_pattern.core.pattern_validator import get_pattern_validator
                    validator = get_pattern_validator()
                    
                    # Validate patterns through 3-stage pipeline
                    validated_patterns, validation_stats = validator.validate_patterns(
                        basic_patterns=basic_patterns,
                        advanced_patterns=advanced_patterns,
                        yolo_patterns=yolo_patterns_raw,
                        data=data
                    )
                    
                    patterns = validated_patterns
                    
                    logger.info(
                        f"✅ Pattern Validation {symbol}: "
                        f"{validation_stats['validated']}/{validation_stats['total_basic']} validated "
                        f"(rejected: {validation_stats['rejected']}, "
                        f"avg score: {validation_stats.get('avg_validation_score', 0):.2f})"
                    )
                    
                except Exception as e:
                    logger.error(f"Pattern validation error for {symbol}: {e}")
                    # Fallback: use basic patterns with reduced confidence
                    patterns = basic_patterns + advanced_patterns
                    for p in patterns:
                        p['confidence'] = p.get('confidence', 0.5) * 0.8
            else:
                # Validation disabled: use all patterns without filtering
                patterns = basic_patterns + advanced_patterns + yolo_patterns_raw
                logger.debug(f"Pattern validation disabled for {symbol}")
            
            # ML predictions: coordinated (Basic + Enhanced) in one place
            try:
                from bist_pattern.core.ml_coordinator import get_ml_coordinator
                mlc = get_ml_coordinator()
                coord = mlc.predict_with_coordination(symbol, data)
                # Map results into pattern signals for UI consistency
                current_px = float(data['close'].iloc[-1])
                
                # Calibration helper: squash extreme deltas smoothly
                def _calibrate_delta(delta: float) -> float:
                    try:
                        tau = float(os.getenv('DELTA_CAL_TAU', '0.08'))
                    except Exception:
                        tau = 0.08
                    try:
                        return float(math.tanh(delta / max(1e-9, tau)) * tau)
                    except Exception:
                        return float(delta)
                
                # Extract raw predictions for response payload
                if not isinstance(coord, dict):
                    coord = {}
                ml_predictions = coord.get('basic', {}) if isinstance(coord.get('basic'), dict) else coord.get('basic', {}) or {}
                enhanced_predictions = coord.get('enhanced', {}) if isinstance(coord.get('enhanced'), dict) else coord.get('enhanced', {}) or {}
                
                def _emit(hkey: str, pred_px: float, source: str, base_w: float = 0.6, reliability: float | None = None):
                    try:
                        if not isinstance(pred_px, (int, float)) or current_px <= 0:
                            return
                        raw_delta = (float(pred_px) - current_px) / current_px
                        delta_pct = _calibrate_delta(raw_delta)
                        # Reliability defaults
                        rel = reliability
                        if not isinstance(rel, (int, float)):
                            try:
                                rel = float(os.getenv('BASIC_RELIABILITY', '0.6')) if source.upper() in ('ML_PREDICTOR', 'ML') else 0.65
                            except Exception:
                                rel = 0.6
                        rel = max(0.0, min(1.0, float(rel)))
                        if abs(delta_pct) < 0.003:
                            return
                        conf = max(0.25, min(0.95, base_w * rel * min(1.0, abs(delta_pct) / 0.05)))
                        # Canonicalize source names for downstream counters
                        src = 'ML_PREDICTOR' if source.upper() in ('ML', 'ML_PREDICTOR') else (
                            'ENHANCED_ML' if source.upper() in ('ENH', 'ENHANCED_ML') else source
                        )
                        patterns.append({
                            'pattern': f'{src}_{hkey.upper()}',
                            'signal': 'BULLISH' if delta_pct > 0 else 'BEARISH',
                            'confidence': conf,
                            'strength': int(conf * 100),
                            'source': src,
                            'delta_pct': float(delta_pct)
                        })
                    except Exception:
                        pass
                # Basic ML predictions: ignore metadata keys like 'timestamp'/'model'
                basic = ml_predictions if isinstance(ml_predictions, dict) else {}
                basic_count = 0
                for hk, pobj in basic.items():
                    try:
                        if not isinstance(pobj, dict):
                            # Skip non-dict values such as 'timestamp' or 'model'
                            continue
                        px = pobj.get('price') or pobj.get('prediction') or pobj.get('target')
                        if not isinstance(px, (int, float)):
                            continue
                        # Basic reliability from env (no per-horizon metric)
                        try:
                            basic_rel = float(os.getenv('BASIC_RELIABILITY', '0.6'))
                        except Exception:
                            basic_rel = 0.6
                        _emit(hk, float(px), 'ML_PREDICTOR', reliability=basic_rel)
                        basic_count += 1
                    except Exception:
                        continue
                
                # Debug: Log basic ML status
                if basic_count == 0 and basic:
                    _ddebug(f"⚠️ Basic ML data available but no valid predictions for {symbol}: {list(basic.keys())}", logger)

                # Enhanced ML predictions: map ensemble structure to a numeric prediction
                enh = enhanced_predictions if isinstance(enhanced_predictions, dict) else {}
                enh_count = 0
                for hk, pobj in enh.items():
                    try:
                        pred_val = None
                        rel_hint = None
                        if isinstance(pobj, dict):
                            pred_val = (
                                pobj.get('ensemble_prediction')
                                or pobj.get('prediction')
                                or pobj.get('price')
                                or pobj.get('target')
                            )
                            # CRITICAL FIX: Proper R² to confidence conversion
                            # R² can be negative (model worse than baseline)
                            # Previous linear mapping was incorrect
                            try:
                                raw_conf = pobj.get('confidence')
                                if isinstance(raw_conf, (int, float)):
                                    r2 = float(raw_conf)
                                    if r2 < 0:
                                        # Negative R² = model is worse than baseline → no confidence
                                        rel_hint = 0.0
                                    elif r2 < 0.5:
                                        # Weak model: use R² directly as confidence
                                        rel_hint = r2
                                    else:
                                        # Good model: scale 0.5-1.0 R² → 0.5-0.95 confidence
                                        # Cap at 0.95 to remain conservative
                                        rel_hint = 0.5 + (r2 - 0.5) * 0.9
                                        rel_hint = min(0.95, rel_hint)
                            except Exception:
                                rel_hint = None
                        elif isinstance(pobj, (int, float)):
                            pred_val = float(pobj)
                        if not isinstance(pred_val, (int, float)):
                            continue
                        _emit(hk, float(pred_val), 'ENHANCED_ML', base_w=0.7, reliability=rel_hint)
                        enh_count += 1
                    except Exception:
                        continue
                
                # Debug: Log enhanced ML status
                if enh_count == 0 and enh:
                    _ddebug(f"⚠️ Enhanced ML data available but no valid predictions for {symbol}: {list(enh.keys())}", logger)
                elif enh_count > 0:
                    _ddebug(f"✅ Enhanced ML predictions for {symbol}: {enh_count} horizons", logger)
                    
            except Exception as e:
                logger.error(f"Coordinated ML prediction integration hatası {symbol}: {e}")

            # FinGPT sentiment (optional) - integrate as additional signal
            try:
                if config['default'].ENABLE_FINGPT and getattr(self, 'fingpt_available', False) and self.fingpt is not None:
                    news_texts = []
                    try:
                        # Use async RSS news provider for non-blocking news fetching
                        if hasattr(self, '_async_rss_provider') and self._async_rss_provider:
                            news_texts = self._async_rss_provider.get_recent_news_async(symbol) or []
                            _ddebug(f"📰 Got {len(news_texts)} news items for {symbol}")
                    except Exception:
                        news_texts = []
                    if news_texts:
                        sent_res = self.fingpt.analyze_stock_news(symbol, news_texts)
                        # Convert sentiment to trading direction
                        sig = self.fingpt.get_sentiment_signal(sent_res)
                        conf = float(sent_res.get('confidence', 0.0) or 0.0)
                        if sig in ('BULLISH', 'BEARISH') and conf > 0:
                            patterns.append({
                                'pattern': 'FINGPT_SENTIMENT',
                                'signal': sig,
                                'confidence': max(0.3, min(0.9, conf)),
                                'strength': int(max(0.3, min(0.9, conf)) * 100),
                                'source': 'FINGPT',
                                'news_count': int(sent_res.get('news_count', 0) or 0)
                            })
            except Exception as e:
                logger.error(f"FinGPT sentiment integration hatası {symbol}: {e}")

            # Overall signal generation
            overall_signal = self.generate_overall_signal(indicators, patterns)
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # ML unified schema build (basic + enhanced → per-horizon)
            ml_unified = {}
            try:
                
                def _norm_basic(basic_map):
                    out = {}
                    if isinstance(basic_map, dict):
                        for h, v in basic_map.items():
                            try:
                                price = None
                                if isinstance(v, (int, float)):
                                    price = float(v)
                                elif isinstance(v, dict):
                                    for k in ('price', 'prediction', 'target', 'value', 'y'):
                                        if isinstance(v.get(k), (int, float)):
                                            price = float(v[k])
                                            break
                                if price is not None:
                                    try:
                                        basic_rel = float(os.getenv('BASIC_RELIABILITY', '0.6'))
                                    except Exception:
                                        basic_rel = 0.6
                                    out[str(h).lower()] = {'source': 'basic', 'price': price, 'reliability': float(max(0.0, min(1.0, basic_rel)))}
                            except Exception:
                                continue
                    return out
                
                def _norm_enh(enh_map):
                    out = {}
                    if isinstance(enh_map, dict):
                        for h, v in enh_map.items():
                            try:
                                price = None
                                rel = None
                                if isinstance(v, (int, float)):
                                    price = float(v)
                                elif isinstance(v, dict):
                                    for k in ('ensemble_prediction', 'prediction', 'price', 'target'):
                                        if isinstance(v.get(k), (int, float)):
                                            price = float(v[k])
                                            break
                                    try:
                                        raw_conf = v.get('confidence')
                                        if isinstance(raw_conf, (int, float)):
                                            # Same R² conversion as above
                                            r2 = float(raw_conf)
                                            if r2 < 0:
                                                rel = 0.0
                                            elif r2 < 0.5:
                                                rel = r2
                                            else:
                                                rel = min(0.95, 0.5 + (r2 - 0.5) * 0.9)
                                    except Exception:
                                        rel = None
                                if price is not None:
                                    if not isinstance(rel, (int, float)):
                                        rel = 0.65
                                    out[str(h).lower()] = {'source': 'enhanced', 'price': price, 'reliability': float(max(0.0, min(1.0, rel)))}
                            except Exception:
                                continue
                    return out
                bmap = _norm_basic(ml_predictions)
                emap = _norm_enh(enhanced_predictions)
                # Merge per horizon and compute delta/confidence
                horizons = set(bmap.keys()) | set(emap.keys())
                # Policy: enhanced-first if available (env-driven)
                try:
                    enhanced_first = (os.getenv('ENHANCED_FIRST', '1').lower() in ('1', 'true', 'yes', 'on'))
                except Exception:
                    enhanced_first = True
                for h in horizons:
                    cur = {}
                    if h in bmap:
                        cur['basic'] = bmap[h]
                    if h in emap:
                        cur['enhanced'] = emap[h]
                    # Pick best: enhanced-first if available; otherwise by calibrated |delta| × reliability
                    entries = []
                    for src in ('basic', 'enhanced'):
                        if src in cur and isinstance(cur[src], dict):
                            price = cur[src].get('price')
                            rel = cur[src].get('reliability')
                            if isinstance(price, (int, float)) and current_px > 0:
                                delta = (price - current_px) / current_px
                                cdelta = _calibrate_delta(delta)
                                try:
                                    base_w = 0.7 if src == 'enhanced' else 0.6
                                    rr = float(rel) if isinstance(rel, (int, float)) else (0.65 if src == 'enhanced' else 0.6)
                                    conf = max(0.25, min(0.95, base_w * rr * min(1.0, abs(cdelta) / 0.05)))
                                except Exception:
                                    conf = max(0.25, min(0.95, abs(cdelta) / 0.05))
                                cur[src]['delta_pct'] = float(cdelta)
                                cur[src]['confidence'] = conf
                                score = abs(cdelta) * float(max(0.0, min(1.0, rel))) if isinstance(rel, (int, float)) else abs(cdelta) * (0.65 if src == 'enhanced' else 0.6)
                                entries.append((score, src))
                    # Decide best
                    if enhanced_first and ('enhanced' in cur):
                        cur['best'] = 'enhanced'
                    elif enhanced_first and ('enhanced' not in cur) and ('basic' in cur):
                        cur['best'] = 'basic'
                    elif entries:
                        best_src = sorted(entries, key=lambda x: x[0], reverse=True)[0][1]
                        cur['best'] = best_src
                    ml_unified[h] = cur
            except Exception:
                ml_unified = {}

            result = {
                'symbol': symbol,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'current_price': float(data['close'].iloc[-1]),
                'indicators': convert_numpy_types(indicators),
                'patterns': convert_numpy_types(patterns),
                'overall_signal': convert_numpy_types(overall_signal),
                'data_points': int(len(data)),
                'ml_predictions': ml_predictions or {},
                'enhanced_predictions': enhanced_predictions or {},
                'ml_unified': ml_unified
            }
            # Optional: persist enhanced predictions into ml_bulk_predictions.json per symbol (during cycle)
            try:
                truthy = ('1', 'true', 'yes', 'on')
                if str(os.getenv('WRITE_ENHANCED_DURING_CYCLE', '0')).lower() in truthy:
                    # Throttle to reduce IO
                    now_ts = time.time()
                    try:
                        throttle_seconds = int(os.getenv('ENH_WRITE_THROTTLE_SECONDS', '900'))
                    except Exception:
                        throttle_seconds = 900
                    last_ts = float(self._bulk_write_ts.get(symbol, 0.0))

                    # If we don't already have enhanced in this pass, try to compute it explicitly
                    cur_enhanced = enhanced_predictions if isinstance(enhanced_predictions, dict) else {}
                    if not cur_enhanced:
                        try:
                            from bist_pattern.core.ml_coordinator import get_ml_coordinator as _get_mlc
                            mlc_for_write = _get_mlc()
                            if getattr(mlc_for_write, 'enhanced_ml', None) and mlc_for_write.enhanced_ml.has_trained_models(symbol):
                                try:
                                    mlc_for_write.enhanced_ml.load_trained_models(symbol)
                                except Exception:
                                    pass
                                # Use same data frame
                                cur_enhanced = mlc_for_write.enhanced_ml.predict_enhanced(symbol, data) or {}
                        except Exception:
                            cur_enhanced = {}

                    if (now_ts - last_ts) >= throttle_seconds and (cur_enhanced or ml_predictions):
                        log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                        os.makedirs(log_dir, exist_ok=True)
                        path = os.path.join(log_dir, 'ml_bulk_predictions.json')
                        obj = {'predictions': {}}
                        try:
                            if os.path.exists(path):
                                with open(path, 'r') as rf:
                                    obj = json.load(rf) or {'predictions': {}}
                        except Exception:
                            obj = {'predictions': {}}

                        preds = obj.setdefault('predictions', {})
                        ent = preds.setdefault(symbol, {})
                        if isinstance(cur_enhanced, dict) and cur_enhanced:
                            ent['enhanced'] = cur_enhanced
                        elif isinstance(enhanced_predictions, dict) and enhanced_predictions:
                            ent['enhanced'] = enhanced_predictions
                        elif isinstance(ml_predictions, dict) and ml_predictions:
                            ent.setdefault('basic', ml_predictions)

                        # Atomic write with optional advisory lock
                        tmp_path = path + '.tmp'
                        try:
                            payload = json.dumps(obj)
                            if fcntl is not None:
                                lock_path = path + '.lock'
                                with open(lock_path, 'a') as lf:
                                    try:
                                        fcntl.flock(lf, fcntl.LOCK_EX)
                                    except Exception:
                                        pass
                                    with open(tmp_path, 'w') as wf:
                                        wf.write(payload)
                                        try:
                                            wf.flush()
                                            os.fsync(wf.fileno())
                                        except Exception:
                                            pass
                                    try:
                                        os.replace(tmp_path, path)
                                    finally:
                                        try:
                                            fcntl.flock(lf, fcntl.LOCK_UN)
                                        except Exception:
                                            pass
                            else:
                                with open(tmp_path, 'w') as wf:
                                    wf.write(payload)
                                    try:
                                        wf.flush()
                                        os.fsync(wf.fileno())
                                    except Exception:
                                        pass
                                os.replace(tmp_path, path)
                            self._bulk_write_ts[symbol] = now_ts
                            _ddebug(f"✅ Enhanced merge attempt: sym={symbol} wrote_enh={bool(ent.get('enhanced'))}", logger)
                        except Exception as e:
                            try:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except Exception:
                                pass
                            logger.error(f"Error writing bulk predictions for {symbol}: {e}")
                    else:
                        # Skip with reason for diagnostics (best-effort)
                        _ddebug(
                            f"⏭️ Skip write: sym={symbol} since={int(now_ts-last_ts)}s (<{throttle_seconds}) has_enh={bool(cur_enhanced)} has_basic={bool(ml_predictions)}",
                            logger
                        )
            except Exception as e:
                logger.error(f"Cycle-time enhanced write error {symbol}: {e}")
            # Broadcast compact component summary for live diagnostics
            try:
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    src_counts = {'BASIC': 0, 'VISUAL_YOLO': 0, 'ML_PREDICTOR': 0, 'ENHANCED_ML': 0, 'FINGPT': 0}
                    for p in (patterns or []):
                        s = (p.get('source') or '').upper()
                        # Backward-compatible aliasing
                        if s == 'ML':
                            s = 'ML_PREDICTOR'
                        elif s == 'ENH':
                            s = 'ENHANCED_ML'
                        if s in src_counts:
                            src_counts[s] += 1
                    total_p = len(patterns or [])
                    ml_on = bool(self.ml_predictor and ML_PREDICTION_AVAILABLE)
                    enh_on = bool(getattr(self, 'enhanced_ml', None) and ENHANCED_ML_AVAILABLE)
                    vis_on = bool(self.visual_detector)
                    adv_on = bool(self.advanced_detector and ADVANCED_PATTERNS_AVAILABLE)
                    fg_on = bool(getattr(self, 'fingpt_available', False) and self.fingpt is not None)
                    msg = (
                        f"🧩 AI components: data={len(data)} patt={total_p} "
                        f"[basic {src_counts['BASIC']}, visual {src_counts['VISUAL_YOLO']}, "
                        f"ml {src_counts['ML_PREDICTOR']}, enh {src_counts['ENHANCED_ML']}, fingpt {src_counts['FINGPT']}] "
                        f"features: ML={int(ml_on)} ENH={int(enh_on)} VIS={int(vis_on)} ADV={int(adv_on)} FG={int(fg_on)}"
                    )
                    flask_app.broadcast_log('INFO', msg, 'ai_analysis')  # type: ignore[attr-defined]
            except Exception:
                pass
            # Persist compact last-signal snapshot for dashboards (non-breaking)
            try:
                log_dir = '/opt/bist-pattern/logs'
                os.makedirs(log_dir, exist_ok=True)
                snap_path = os.path.join(log_dir, 'signals_last.json')
                snap = {}
                try:
                    if os.path.exists(snap_path):
                        with open(snap_path, 'r') as rf:
                            snap = json.load(rf) or {}
                except Exception:
                    snap = {}
                # Include VISUAL_YOLO evidence (top up to 3 by confidence) for UI
                visual_evidence = []
                try:
                    vis = [p for p in (result.get('patterns') or []) if (p.get('source') == 'VISUAL_YOLO')]
                    vis_sorted = sorted(vis, key=lambda p: float(p.get('confidence', 0.0)), reverse=True)
                    for p in vis_sorted[:3]:
                        visual_evidence.append({'pattern': p.get('pattern'), 'confidence': float(p.get('confidence', 0.0))})
                except Exception:
                    visual_evidence = []

                snap[symbol] = {
                    'timestamp': result['timestamp'],
                    'signal': result['overall_signal'].get('signal', 'NEUTRAL'),
                    'confidence': result['overall_signal'].get('confidence', 0.0),
                    'strength': result['overall_signal'].get('strength', 0),
                    'visual': visual_evidence,
                }
                with open(snap_path, 'w') as wf:
                    json.dump(snap, wf)
            except Exception:
                pass
            try:
                # Progress broadcast: analysis end (best-effort)
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    sig = result['overall_signal'].get('signal', '?')
                    conf = int(round(float(result['overall_signal'].get('confidence', 0)) * 100))
                    flask_app.broadcast_log('SUCCESS', f'🎯 {symbol} AI: {sig} (%{conf})', 'ai_analysis')  # type: ignore[attr-defined]
                
                # ⭐ EKLENEN: User signal broadcast for live signals (with VISUAL_YOLO evidence)
                try:
                    import requests
                    
                    # Internal API ile user signal gönder (tüm kullanıcılara)
                    # VISUAL evidence: include only VISUAL_YOLO patterns (top 3 by confidence)
                    visual_evidence = []
                    try:
                        vis = [p for p in (result.get('patterns') or []) if (p.get('source') == 'VISUAL_YOLO')]
                        vis_sorted = sorted(vis, key=lambda p: float(p.get('confidence', 0.0)), reverse=True)
                        for p in vis_sorted[:3]:
                            visual_evidence.append({
                                'pattern': p.get('pattern'),
                                'confidence': float(p.get('confidence', 0.0))
                            })
                    except Exception:
                        visual_evidence = []

                    signal_data = {
                        'symbol': symbol,
                        'overall_signal': result.get('overall_signal', {}),
                        'patterns': result.get('patterns', []),
                        'visual': visual_evidence,
                        'current_price': result.get('current_price', 0),
                        'timestamp': result.get('timestamp')
                    }
                    
                    # Test user_id = 1 (gerçek implementasyonda watchlist'ten alınacak)
                    payload = {
                        'user_id': 1,
                        'signal_data': signal_data
                    }
                    
                    token = flask_app.config.get('INTERNAL_API_TOKEN')
                    if not token:
                        logger.warning("INTERNAL_API_TOKEN not configured - skipping live signal broadcast")
                        return
                    headers = {
                        'Content-Type': 'application/json',
                        'X-Internal-Token': token
                    }
                    
                    # localhost:5000/api/internal/broadcast-user-signal
                    resp = requests.post(
                        'http://localhost:5000/api/internal/broadcast-user-signal',
                        json=payload,
                        headers=headers,
                        timeout=5
                    )
                    
                    if resp.status_code == 200:
                        logger.info(f"🔔 Live signal sent for {symbol}")
                    else:
                        logger.warning(f"⚠️ Live signal failed for {symbol}: {resp.status_code}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Live signal broadcast error for {symbol}: {e}")
                    
            except Exception:
                pass
            
            # Cache'e TTL ile kaydet
            self.cache[cache_key] = {
                'data': result,
                'timestamp': current_time
            }
            
            # Cache cleanup - 100'den fazla entry varsa eski olanları temizle
            if len(self.cache) > self.result_cache_max_size:
                self._cleanup_cache()
            
            # Opsiyonel: overall sinyali canlı loga yaz (dashboard için)
            try:
                from app import app as flask_app
                overall = (result or {}).get('overall_signal') or {}
                if overall and hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('INFO', f"{symbol}: {overall.get('signal', '?')} ({overall.get('confidence', 0):.2f})", 'ai_analysis')  # type: ignore[attr-defined]
            except Exception:
                pass

            return result
            
        except Exception as e:
            logger.error(f"Stock analysis hatası {symbol}: {e}")
            return {
                'symbol': symbol,
                'status': 'error',
                'message': str(e)
            }

    def get_basic_predictions(self, symbol: str, data):
        """Temel ML tahminlerini (1/3/7/14/30g) fiyat formatında döndür."""
        try:
            if not (self.ml_predictor and ML_PREDICTION_AVAILABLE):
                return {}
            preds = self.ml_predictor.predict_prices(symbol, data, None) or {}
            out: dict = {}
            for key, val in preds.items():
                v = None
                if isinstance(val, (int, float)):
                    v = float(val)
                elif isinstance(val, dict):
                    for cand in ('price', 'prediction', 'target', 'value', 'y'):
                        if cand in val and isinstance(val[cand], (int, float)):
                            v = float(val[cand])
                            break
                if v is not None:
                    out[str(key).lower()] = v
            return out
        except Exception as e:
            logger.error(f"Basic predictions error {symbol}: {e}")
            return {}

    def get_enhanced_predictions(self, symbol: str, data):
        """Gelişmiş ML tahminlerini (1/3/7/14/30g) fiyat formatında döndür."""
        try:
            if not (hasattr(self, 'enhanced_ml') and self.enhanced_ml):
                return {}
            
            # Use direct prediction method (returns dict of horizon predictions)
            result = self.enhanced_ml.predict_enhanced(symbol, data)
            if result and isinstance(result, dict):
                # Convert to expected format: {horizon: prediction_value}
                formatted_predictions = {}
                for horizon, pred_data in result.items():
                    if isinstance(pred_data, dict) and 'ensemble_prediction' in pred_data:
                        formatted_predictions[horizon] = float(pred_data['ensemble_prediction'])
                    elif isinstance(pred_data, (int, float)):
                        formatted_predictions[horizon] = float(pred_data)
                
                _ddebug(f"✅ Enhanced ML predictions for {symbol}: {len(formatted_predictions)} horizons")
                return formatted_predictions
            else:
                _ddebug(f"❌ Enhanced ML no result for {symbol}")
                return {}
                
        except Exception as e:
            logger.error(f"Enhanced predictions error {symbol}: {e}")
            return {}
    
    def generate_overall_signal(self, indicators, patterns):
        """Genel sinyal üret"""
        try:
            signals = []
            
            # Technical indicator signals
            if indicators.get('rsi'):
                rsi = indicators['rsi']
                if rsi < 30:
                    signals.append(('BULLISH', 0.7, 'RSI Oversold'))
                elif rsi > 70:
                    signals.append(('BEARISH', 0.7, 'RSI Overbought'))
            
            if indicators.get('macd_histogram'):
                if indicators['macd_histogram'] > 0:
                    signals.append(('BULLISH', 0.6, 'MACD Positive'))
                else:
                    signals.append(('BEARISH', 0.6, 'MACD Negative'))
            
            if indicators.get('bb_position'):
                bb_pos = indicators['bb_position']
                if bb_pos < 0.2:
                    signals.append(('BULLISH', 0.5, 'Near Lower Bollinger'))
                elif bb_pos > 0.8:
                    signals.append(('BEARISH', 0.5, 'Near Upper Bollinger'))
            
            # Pattern signals
            for pattern in patterns:
                if pattern.get('signal'):
                    confidence = pattern.get('confidence', 0.5)
                    signals.append((pattern['signal'], confidence, pattern['pattern']))
            
            # WEIGHTED CONSENSUS: Quality-based (not just count)
            # Uses confidence × reliability for ML/Enhanced predictions
            try:
                import os  # type: ignore
                try:
                    # Weighted threshold (environment-driven)
                    # Default 2.0 = requires strong combined confidence
                    weighted_threshold = float(os.getenv('WEIGHTED_CONSENSUS_THRESHOLD', '2.0'))
                except Exception:
                    weighted_threshold = 2.0
                
                # Collect ML/Enhanced patterns with weights
                weighted_bull = 0.0
                weighted_bear = 0.0
                
                for p in patterns:
                    if not isinstance(p.get('pattern'), str):
                        continue
                    
                    # Only ML and Enhanced ML patterns
                    if not (p['pattern'].startswith('ML_') or p['pattern'].startswith('ENH_')):
                        continue
                    
                    signal = p.get('signal', '')
                    if signal not in ('BULLISH', 'BEARISH'):
                        continue
                    
                    # Weight = confidence × reliability × validation_score
                    confidence = float(p.get('confidence', 0.5))
                    reliability = float(p.get('reliability', 0.6))  # From ML system
                    validation_score = float(p.get('validation_score', 1.0))  # From pattern validator
                    
                    weight = confidence * reliability * validation_score
                    
                    if signal == 'BULLISH':
                        weighted_bull += weight
                    else:
                        weighted_bear += weight
                
                # Consensus signal if weighted difference exceeds threshold
                weighted_diff = abs(weighted_bull - weighted_bear)
                
                if weighted_diff > weighted_threshold:
                    # Consensus confidence proportional to weight difference
                    # Cap at 0.9 for humility
                    consensus_conf = min(0.9, weighted_diff / 5.0)
                    
                    if weighted_bull > weighted_bear:
                        signals.append(('BULLISH', consensus_conf, 'WEIGHTED_CONSENSUS'))
                        logger.debug(f"🎯 Weighted Consensus: BULLISH (weight: {weighted_bull:.2f} vs {weighted_bear:.2f})")
                    else:
                        signals.append(('BEARISH', consensus_conf, 'WEIGHTED_CONSENSUS'))
                        logger.debug(f"🎯 Weighted Consensus: BEARISH (weight: {weighted_bear:.2f} vs {weighted_bull:.2f})")
                        
            except Exception as e:
                logger.warning(f"Weighted consensus calculation error: {e}")
                pass

            # Volatility-aware weighting (using Bollinger band width proxy)
            try:
                upper = indicators.get('bb_upper')
                lower = indicators.get('bb_lower')
                import os  # type: ignore
                try:
                    vol_high = float(os.getenv('VOL_HIGH_THRESHOLD', '0.10'))
                except Exception:
                    vol_high = 0.10
                try:
                    vol_low = float(os.getenv('VOL_LOW_THRESHOLD', '0.03'))
                except Exception:
                    vol_low = 0.03
                weight_scale = 1.0
                if isinstance(upper, (int, float)) and isinstance(lower, (int, float)) and (upper + lower) != 0:
                    vol_ratio = (upper - lower) / max(1e-9, (upper + lower) / 2.0)
                    if vol_ratio > vol_high:  # high volatility → require stronger evidence
                        weight_scale = 0.85
                    elif vol_ratio < vol_low:  # very low volatility → small signals get a boost
                        weight_scale = 1.10
                if weight_scale != 1.0 and signals:
                    signals = [(s, max(0.0, min(1.0, c * weight_scale)), r) for (s, c, r) in signals]
            except Exception:
                pass

            # Overall calculation
            if not signals:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.5,
                    'strength': 50,
                    'reasoning': 'Yeterli sinyal bulunamadı'
                }
            
            # Weight calculation
            bullish_weight = sum(conf for sig, conf, _ in signals if sig == 'BULLISH')
            bearish_weight = sum(conf for sig, conf, _ in signals if sig == 'BEARISH')
            total_weight = bullish_weight + bearish_weight
            
            if total_weight == 0:
                overall_signal = 'NEUTRAL'
                confidence = 0.5
            elif bullish_weight > bearish_weight:
                overall_signal = 'BULLISH'
                confidence = bullish_weight / total_weight
            else:
                overall_signal = 'BEARISH'
                confidence = bearish_weight / total_weight
            
            return {
                'signal': overall_signal,
                'confidence': confidence,
                'strength': int(confidence * 100),
                'reasoning': f"{len(signals)} sinyal analiz edildi",
                'signals': [{'signal': s, 'confidence': c, 'source': r} for s, c, r in signals]
            }
            
        except Exception as e:
            logger.error(f"Signal generation hatası: {e}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.5,
                'strength': 50,
                'reasoning': 'Signal hesaplama hatası'
            }
    
    def analyze_multiple_stocks(self, symbols):
        """Birden fazla hisse analiz et"""
        results = {}
        
        for symbol in symbols:
            try:
                result = self.analyze_stock(symbol)
                results[symbol] = result
                logger.info(f"✅ {symbol}: {result.get('overall_signal', {}).get('signal', 'N/A')}")
            except Exception as e:
                logger.error(f"❌ {symbol} analiz hatası: {e}")
                results[symbol] = {
                    'symbol': symbol,
                    'status': 'error',
                    'message': str(e)
                }
        
        return results
    
    def get_pattern_summary(self, symbols=None):
        """Pattern özeti çıkar"""
        if symbols is None:
            # En aktif hisseler
            symbols = ['THYAO', 'AKBNK', 'GARAN', 'EREGL', 'ASELS', 'VAKBN']
        
        results = self.analyze_multiple_stocks(symbols)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'analyzed_stocks': len(symbols),
            'bullish_signals': 0,
            'bearish_signals': 0,
            'neutral_signals': 0,
            'strong_patterns': [],
            'stock_details': results
        }
        
        for symbol, data in results.items():
            if data.get('status') == 'success':
                signal = data.get('overall_signal', {}).get('signal', 'NEUTRAL')
                if signal == 'BULLISH':
                    summary['bullish_signals'] += 1
                elif signal == 'BEARISH':
                    summary['bearish_signals'] += 1
                else:
                    summary['neutral_signals'] += 1
                
                # Strong patterns
                patterns = data.get('patterns', [])
                for pattern in patterns:
                    if pattern.get('confidence', 0) > 0.7:
                        summary['strong_patterns'].append({
                            'symbol': symbol,
                            'pattern': pattern['pattern'],
                            'signal': pattern.get('signal'),
                            'confidence': pattern.get('confidence')
                        })
        
        return summary
