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
from typing import Any, Callable
try:
    import fcntl  # Posix lock
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore
from models import Stock, StockPrice
from bist_pattern.utils.debug_utils import ddebug as _ddebug
from bist_pattern.core.config_manager import ConfigManager
from bist_pattern.utils.error_handler import ErrorHandler
# ✅ FIX: Removed 'from app import app' to prevent circular dependency
# Use current_app (from flask) inside functions when needed
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# Gelişmiş pattern detection sistemi
try:
    from advanced_patterns import AdvancedPatternDetector
    ADVANCED_PATTERNS_AVAILABLE = True
except ImportError:
    ADVANCED_PATTERNS_AVAILABLE = False
    AdvancedPatternDetector = None  # type: ignore[assignment]
    logger.warning("⚠️ Advanced patterns modülü yüklenemedi")

# Visual pattern detection sistemi (now using async version)
VISUAL_PATTERNS_AVAILABLE = True  # Always available with async implementation

# ML Prediction sistemi
try:
    from ml_prediction_system import get_ml_prediction_system as _real_get_ml_prediction_system
    get_ml_prediction_system: Callable[[], Any] = _real_get_ml_prediction_system  # type: ignore[assignment]
    ML_PREDICTION_AVAILABLE = True
except ImportError:
    ML_PREDICTION_AVAILABLE = False

    def _fallback_get_ml_prediction_system() -> Any:
        return None
    get_ml_prediction_system = _fallback_get_ml_prediction_system

    logger.warning("⚠️ ML Prediction modülü yüklenemedi")

# Enhanced ML Prediction sistemi (opsiyonel)
try:
    from enhanced_ml_system import get_enhanced_ml_system as _real_get_enhanced_ml_system
    get_enhanced_ml_system: Callable[[], Any] = _real_get_enhanced_ml_system  # type: ignore[assignment]
    ENHANCED_ML_AVAILABLE = True
except ImportError:
    ENHANCED_ML_AVAILABLE = False  # Import failed

    def _fallback_get_enhanced_ml_system() -> Any:
        return None
    get_enhanced_ml_system = _fallback_get_enhanced_ml_system
    logger.warning("⚠️ Enhanced ML Prediction modülü yüklenemedi")


class HybridPatternDetector:

    def __init__(self):
        # Cache sistemi (TTL + boyut sınırı) - tamamen environment-driven
        self.cache = {}
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            self.cache_ttl = int(ConfigManager.get('PATTERN_RESULT_CACHE_TTL', 
                ConfigManager.get('PATTERN_CACHE_TTL', 300)))
        except Exception as e:
            ErrorHandler.handle(e, 'pattern_detector_init', level='warning')
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
        if ADVANCED_PATTERNS_AVAILABLE and AdvancedPatternDetector is not None:
            self.advanced_detector = AdvancedPatternDetector()
        else:
            self.advanced_detector = None
            
        # Visual pattern detector - SIMPLE SYNC VERSION
        self.visual_detector = None
        self._yolo_model = None
        # ✅ FIX: Use ConfigManager for consistent config access
        yolo_enabled = ConfigManager.get('ENABLE_YOLO', True)
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
        # ✅ FIX: Use ConfigManager for consistent config access
        # Default None to allow config.py default (True) to be used
        enable_fingpt = ConfigManager.get('ENABLE_FINGPT', None)
        if enable_fingpt is None:
            # Fallback to config.py default (True)
            try:
                from config import config
                enable_fingpt = getattr(config['default'], 'ENABLE_FINGPT', True)
            except Exception:
                enable_fingpt = True  # Default to True if config.py unavailable
        if enable_fingpt:
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
    
    def _calculate_pattern_agreement(self, patterns, ml_signal, ml_confidence):
        """
        🎯 NEW: Calculate pattern agreement score for ML Primary + Pattern Confirmation
        
        Logic:
        - Count how many patterns agree with ML signal
        - Apply boost/penalty based on agreement count
        - Handle conflicts (opposite signals)
        
        Returns:
            float: Agreement score (-0.20 to +0.15)
        """
        try:
            # Known pattern aliases for proper de-duplication
            PATTERN_ALIASES = {
                'DOJI': {'DOJI', 'DOJI_STAR'},
                'HAMMER': {'HAMMER'},
                'INVERTED_HAMMER': {'INVERTED_HAMMER'},
                'SHOOTING_STAR': {'SHOOTING_STAR'},
                'ENGULFING': {'ENGULFING', 'ENGULFING_BULLISH', 'ENGULFING_BEARISH'},
                'HARAMI': {'HARAMI', 'HARAMI_BULLISH', 'HARAMI_BEARISH'},
                'MORNING_STAR': {'MORNING_STAR', 'MORNING_DOJI_STAR'},
                'EVENING_STAR': {'EVENING_STAR', 'EVENING_DOJI_STAR'},
                'PIERCING': {'PIERCING', 'PIERCING_LINE'},
                'DARK_CLOUD': {'DARK_CLOUD', 'DARK_CLOUD_COVER'},
                'THREE_WHITE_SOLDIERS': {'THREE_WHITE_SOLDIERS', '3_WHITE_SOLDIERS'},
                'THREE_BLACK_CROWS': {'THREE_BLACK_CROWS', '3_BLACK_CROWS'},
            }
            
            def get_pattern_family(pattern_name):
                """Get pattern family for de-duplication"""
                pattern_upper = pattern_name.upper()
                for family, aliases in PATTERN_ALIASES.items():
                    if pattern_upper in aliases or any(alias in pattern_upper for alias in aliases):
                        return family
                return pattern_name  # Return original if no match
            
            # Count patterns by signal type (excluding ML patterns)
            bullish_patterns = []
            bearish_patterns = []
            total_conf = 0.0
            
            for p in (patterns or []):
                try:
                    src = str(p.get('source', '')).upper()
                    # Skip ML patterns (we only want external validation)
                    if src in ('ML_PREDICTOR', 'ENHANCED_ML', 'ML', 'ENH'):
                        continue
                    
                    sig = str(p.get('signal', '')).upper()
                    conf = float(p.get('confidence', 0.5))
                    
                    # Only count meaningful confidence
                    if conf < 0.25:
                        continue
                    
                    # De-duplication: Check for similar patterns using family matching
                    pattern_name = str(p.get('pattern', '')).upper()
                    pattern_family = get_pattern_family(pattern_name)
                    is_duplicate = False
                    
                    if sig == 'BULLISH':
                        for existing in bullish_patterns:
                            if pattern_family == existing:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            bullish_patterns.append(pattern_family)
                            total_conf += conf
                    elif sig == 'BEARISH':
                        for existing in bearish_patterns:
                            if pattern_family == existing:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            bearish_patterns.append(pattern_family)
                            total_conf += conf
                except Exception:
                    continue
            
            # Calculate agreement
            bull_count = len(bullish_patterns)
            bear_count = len(bearish_patterns)
            total_count = bull_count + bear_count
            
            if total_count == 0:
                # No patterns found → slight penalty
                return -0.10
            
            # Determine agreement with ML signal
            if ml_signal == 'BULLISH':
                agreeing = bull_count
                conflicting = bear_count
            else:  # BEARISH
                agreeing = bear_count
                conflicting = bull_count
            
            # Conflict detection: if opposite signals dominate
            if conflicting > agreeing:
                # Strong conflict → penalty
                return -0.20
            
            # Agreement boost based on count
            if agreeing >= 3:
                # 3+ patterns agree → strong boost
                return 0.15
            elif agreeing == 2:
                # 2 patterns agree → moderate boost
                return 0.08
            elif agreeing == 1:
                # 1 pattern agrees → slight boost
                return 0.03
            else:
                # No agreement → slight penalty
                return -0.10
                
        except Exception as e:
            logger.debug(f"Pattern agreement calculation error: {e}")
            return 0.0
    
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
                # ✅ FIX: Use ConfigManager for consistent config access
                default_days = int(ConfigManager.get('PATTERN_DATA_DAYS', 365))
            except Exception as e:
                ErrorHandler.handle(e, 'get_stock_data', level='debug')
                default_days = 365
            # If days is None, use default; if days <= 0, fetch full history (no limit)
            try:
                use_days = default_days if days is None else int(days)
            except Exception:
                use_days = default_days
            try:
                from flask import current_app
                with current_app.app_context():
                    # Stock ID'yi bul
                    stock = Stock.query.filter_by(symbol=symbol.upper()).first()
                    if not stock:
                        logger.warning(f"Hisse bulunamadı: {symbol}")
                        # Yahoo Finance fallback dene
                        yahoo_data = self._try_yahoo_finance_fallback(symbol, (use_days if use_days > 0 else None))
                        if yahoo_data is not None:
                            return yahoo_data
                        return None
                    
                    # Son N günlük veriyi al (use_days <= 0 ise limitsiz)
                    query = StockPrice.query.filter_by(stock_id=stock.id)\
                                .order_by(StockPrice.date.desc())
                    if use_days > 0:
                        query = query.limit(use_days)
                    prices = query.all()
                    
                    if not prices:
                        logger.warning(f"Fiyat verisi bulunamadı: {symbol}")
                        # Yahoo Finance fallback dene
                        yahoo_data = self._try_yahoo_finance_fallback(symbol, (use_days if use_days > 0 else None))
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
                    
                    # ✅ HİBRİT YAKLAŞIM: Duplicate date kontrolü (aynı tarihli kayıtlar varsa temizle)
                    if df.index.duplicated().any():
                        duplicate_count = df.index.duplicated().sum()
                        logger.warning(f"{symbol}: {duplicate_count} duplicate date found, dropping (keep='last')")
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.sort_index()
                    
                    # Cache'e koy
                    try:
                        self._df_cache[symbol] = {'df': df, 'ts': now_ts}
                        self._prune_df_cache()
                    except Exception:
                        pass
                    
                    return df
            except RuntimeError:
                # No app context, create temporary app context
                from app import create_app
                temp_app = create_app()
                with temp_app.app_context():
                    # Stock ID'yi bul
                    stock = Stock.query.filter_by(symbol=symbol.upper()).first()
                    if not stock:
                        logger.warning(f"Hisse bulunamadı: {symbol}")
                        # Yahoo Finance fallback dene
                        yahoo_data = self._try_yahoo_finance_fallback(symbol, (use_days if use_days > 0 else None))
                        if yahoo_data is not None:
                            return yahoo_data
                        return None
                    
                    # Son N günlük veriyi al (use_days <= 0 ise limitsiz)
                    query = StockPrice.query.filter_by(stock_id=stock.id)\
                                .order_by(StockPrice.date.desc())
                    if use_days > 0:
                        query = query.limit(use_days)
                    prices = query.all()
                    
                    if not prices:
                        logger.warning(f"Fiyat verisi bulunamadı: {symbol}")
                        # Yahoo Finance fallback dene
                        yahoo_data = self._try_yahoo_finance_fallback(symbol, (use_days if use_days > 0 else None))
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
                    
                    # ✅ HİBRİT YAKLAŞIM: Duplicate date kontrolü (aynı tarihli kayıtlar varsa temizle)
                    if df.index.duplicated().any():
                        duplicate_count = df.index.duplicated().sum()
                        logger.warning(f"{symbol}: {duplicate_count} duplicate date found, dropping (keep='last')")
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.sort_index()
                    
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
        # ✅ FIX: Use ConfigManager for consistent config access
        if not ConfigManager.get('ENABLE_YAHOO_FALLBACK', False):
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
            # ✅ FIX: Use ConfigManager for consistent config access
            days = days or ConfigManager.get('PATTERN_DATA_DAYS', 365)
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
            # ✅ FIX: Use ConfigManager for consistent config access
            if ConfigManager.get('YF_ENHANCED_ENABLED', False):
                try:
                    from yahoo_finance_enhanced import get_enhanced_yahoo_finance_wrapper
                    wrapper = get_enhanced_yahoo_finance_wrapper()

                    # Use thread-safe sync method to avoid 'event loop is running' errors in gevent/gunicorn
                    try:
                        result = wrapper.fetch_data_sync(
                            symbol, yf_symbol, period,
                            # ✅ FIX: Use ConfigManager for consistent config access
                            timeout=min(float(ConfigManager.get('YF_FALLBACK_TIMEOUT', 30.0)), 60.0)
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
                    # ✅ FIX: Use ConfigManager for consistent config access
                    timeout=min(float(ConfigManager.get('YF_FALLBACK_TIMEOUT', 30.0)), 30.0)
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
            # PROBE: trace analyze_stock callers (pid/argv) with symbol
            # ⚡ SILENT: Only log if not in training mode to reduce log noise
            try:
                import os as _os
                import sys as _sys
                _argv = ' '.join(_sys.argv) if hasattr(_sys, 'argv') else ''
                # Only log if not in training mode (DISABLE_ML_PREDICTION_DURING_TRAINING not set)
                if not os.getenv('DISABLE_ML_PREDICTION_DURING_TRAINING', '0').lower() in ('1', 'true', 'yes', 'on'):
                    logger.warning(f"PROBE analyze_stock pid={_os.getpid()} argv={_argv} symbol={symbol}")
            except Exception:
                pass
            try:
                # Progress broadcast: analysis start (best-effort)
                from flask import current_app
                try:
                    if hasattr(current_app, 'broadcast_log'):
                        current_app.broadcast_log('INFO', f'🧠 AI analiz başlıyor: {symbol}', 'ai_analysis')  # type: ignore[attr-defined]
                except RuntimeError:
                    # No app context, skip broadcast
                    pass
            except Exception:
                pass
            # ✅ FIX: Cache key symbol-based (not minute-based!)
            # Automation results should be reused by user requests
            cache_key = symbol  # Simple, effective!
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
            
            # Persisted calibration override (soft toggle)
            calib_override = None
            try:
                import json as _json
                base_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                cpath = os.path.join(base_dir, 'calibration_state.json')
                if os.path.exists(cpath):
                    with open(cpath, 'r') as cf:
                        calib_override = _json.load(cf) or {}
            except Exception:
                calib_override = None

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
            # PROBE: before advanced TA step to align with TA-Lib logs
            # ⚡ SILENT: Only log if not in training mode to reduce log noise
            try:
                import os as _os
                import sys as _sys
                _argv = ' '.join(_sys.argv) if hasattr(_sys, 'argv') else ''
                # Only log if not in training mode (DISABLE_ML_PREDICTION_DURING_TRAINING not set)
                if not os.getenv('DISABLE_ML_PREDICTION_DURING_TRAINING', '0').lower() in ('1', 'true', 'yes', 'on'):
                    logger.warning(f"PROBE before_advanced_ta pid={_os.getpid()} argv={_argv} symbol={symbol}")
            except Exception:
                pass
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
            # ✅ FIX: Use ConfigManager for consistent config access
            if self.visual_detector and ConfigManager.get('ENABLE_YOLO', True):
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
                    future = self._visual_thread_pool.submit(_async_yolo_analysis)
                    
                    # Check for immediate cached result (from previous analysis)
                    cached_result = None
                    try:
                        with self._visual_lock:
                            cached = self._visual_results.get(symbol)
                            if cached and (time.time() - cached['timestamp']) < 300:  # 5 min cache
                                cached_result = cached
                    except Exception:
                        pass
                    
                    # If no cache, wait briefly for fresh result (2 sec timeout)
                    if not cached_result:
                        try:
                            import concurrent.futures
                            future.result(timeout=2.0)  # Wait up to 2 seconds for YOLO
                            # After completion, check result immediately
                            with self._visual_lock:
                                fresh = self._visual_results.get(symbol)
                                if fresh:
                                    cached_result = fresh
                        except concurrent.futures.TimeoutError:
                            logger.debug(f"⏱️ YOLO timeout for {symbol}, will use cache next cycle")
                        except Exception as e:
                            logger.debug(f"⚠️ YOLO wait error for {symbol}: {e}")
                    
                    # Stage 3: Collect YOLO patterns for validation
                    if cached_result:
                        cached_patterns = cached_result.get('patterns', [])
                        yolo_patterns_raw.extend(cached_patterns)
                        if len(cached_patterns) > 0:
                            logger.info(f"📸 YOLO patterns for {symbol}: {len(cached_patterns)} patterns detected")
                            for p in cached_patterns[:3]:  # Log first 3 patterns
                                logger.info(f"   → {p.get('pattern', 'unknown')} ({p.get('signal', 'NEUTRAL')}, conf={p.get('confidence', 0.0):.2f})")
                        else:
                            logger.debug(f"📸 YOLO analysis for {symbol}: 0 patterns (model may be too strict or chart unclear)")
                    else:
                        logger.debug(f"🔄 YOLO analysis queued for {symbol} (background)")
                    
                except Exception as e:
                    logger.error(f"Async YOLO setup error for {symbol}: {e}")
            
            # ==========================================
            # VALIDATION PIPELINE: Multi-stage pattern validation
            # ==========================================
            patterns = []
            # ✅ FIX: Initialize FinGPT patterns list before validation (will be populated now)
            fingpt_patterns = []
            
            # FinGPT sentiment (optional) - integrate as additional signal
            # ✅ FIX: Run FinGPT analysis BEFORE validation pipeline so patterns are available for validation checks
            try:
                # ✅ FIX: Use ConfigManager for consistent config access
                # Check if FinGPT is enabled and available (robust bool parsing)
                raw_flag = ConfigManager.get('ENABLE_FINGPT', None)
                if raw_flag is None:
                    try:
                        from config import config
                        raw_flag = getattr(config['default'], 'ENABLE_FINGPT', True)
                    except Exception:
                        raw_flag = True
                # Normalize to boolean
                if isinstance(raw_flag, bool):
                    enable_fingpt = raw_flag
                elif isinstance(raw_flag, (int, float)):
                    enable_fingpt = bool(int(raw_flag))
                else:
                    val = str(raw_flag).strip().lower()
                    if val in ('1', 'true', 'yes', 'on'):
                        enable_fingpt = True
                    elif val in ('0', 'false', 'no', 'off', ''):
                        enable_fingpt = False
                    else:
                        # Default to True to preserve previous behavior if value is ambiguous
                        enable_fingpt = True
                if enable_fingpt and getattr(self, 'fingpt_available', False) and self.fingpt is not None:
                    news_texts = []
                    try:
                        # Use async RSS news provider for non-blocking news fetching
                        if hasattr(self, '_async_rss_provider') and self._async_rss_provider:
                            news_texts = self._async_rss_provider.get_recent_news_async(symbol) or []
                            if news_texts:
                                logger.info(f"📰 Got {len(news_texts)} news items for {symbol}")
                            else:
                                # ✅ FIX: Log at INFO level so we can see why no badge appears
                                logger.info(f"📰 {symbol}: No news items found from RSS (sentiment badge will not appear)")
                        else:
                            logger.debug(f"📰 {symbol}: RSS provider not available")
                    except Exception as e:
                        logger.warning(f"📰 RSS news fetch failed for {symbol}: {e}")
                        news_texts = []
                    if news_texts:
                        sent_res = self.fingpt.analyze_stock_news(symbol, news_texts)
                        # ✅ Guard against None/invalid responses from FinGPT
                        sig = 'NEUTRAL'
                        conf = 0.0
                        news_count = 0
                        if isinstance(sent_res, dict):
                            # Convert sentiment to trading direction
                            sig = self.fingpt.get_sentiment_signal(sent_res)
                            try:
                                conf = float(sent_res.get('confidence', 0.0) or 0.0)
                            except Exception:
                                conf = 0.0
                            try:
                                news_count = int(sent_res.get('news_count', 0) or 0)
                            except Exception:
                                news_count = 0
                        else:
                            logger.debug(f"📰 FinGPT returned invalid result for {symbol}: {type(sent_res)}")
                        if sig in ('BULLISH', 'BEARISH') and conf > 0:
                            logger.info(f"✅ FinGPT sentiment {symbol}: {sig} (conf={conf:.2f}, news={news_count})")
                            # ✅ FIX: Include news texts for tooltip display (limit to 200 chars each)
                            news_items_preview = []
                            for news_text in news_texts[:5]:  # Max 5 news items
                                preview = str(news_text)[:200]  # Limit to 200 chars
                                if len(str(news_text)) > 200:
                                    preview += "..."
                                news_items_preview.append(preview)
                            
                            fingpt_patterns.append({
                                'pattern': 'FINGPT_SENTIMENT',
                                'signal': sig,
                                'confidence': max(0.3, min(0.9, conf)),
                                'strength': int(max(0.3, min(0.9, conf)) * 100),
                                'source': 'FINGPT',
                                'news_count': news_count,
                                'news_items': news_items_preview  # ✅ FIX: Add news items for tooltip
                            })
                        else:
                            logger.debug(f"📰 FinGPT sentiment {symbol}: {sig} (conf={conf:.2f}, news={news_count}) - below threshold")
                    else:
                        logger.debug(f"📰 FinGPT skipped for {symbol}: no news items")
            except Exception as e:
                logger.error(f"FinGPT sentiment integration hatası {symbol}: {e}")
            
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
                    # ✅ FIX: Add FinGPT patterns after validation (they don't go through validation pipeline)
                    if fingpt_patterns:
                        patterns.extend(fingpt_patterns)
                    
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
                    # ✅ FIX: Add FinGPT patterns even in fallback
                    if fingpt_patterns:
                        patterns.extend(fingpt_patterns)
            else:
                # Validation disabled: use all patterns without filtering
                patterns = basic_patterns + advanced_patterns + yolo_patterns_raw
                # ✅ FIX: Add FinGPT patterns even when validation is disabled
                if fingpt_patterns:
                    patterns.extend(fingpt_patterns)
                logger.debug(f"Pattern validation disabled for {symbol}")
            
            # ML predictions: coordinated (Basic + Enhanced) in one place
            # Initialize variables first to avoid UnboundLocalError
            ml_predictions = {}
            enhanced_predictions = {}
            
            # ⚡ CRITICAL: Skip ML prediction during training to prevent horizon features not found errors
            # Training script'inde prediction yapmamalıyız çünkü model henüz eğitilmemiş
            # WRITE_ENHANCED_DURING_CYCLE=0 kontrolü yeterli değil, çünkü prediction yapılıyor ama yazılmıyor
            # DISABLE_ML_PREDICTION_DURING_TRAINING kontrolü ekliyoruz
            skip_ml_prediction = (
                os.getenv('DISABLE_ML_PREDICTION_DURING_TRAINING', '0').lower() in ('1', 'true', 'yes', 'on')
                or os.getenv('WRITE_ENHANCED_DURING_CYCLE', '0').lower() not in ('1', 'true', 'yes', 'on')
            )
            
            try:
                if not skip_ml_prediction:
                    from bist_pattern.core.ml_coordinator import get_ml_coordinator
                    mlc = get_ml_coordinator()
                    
                    # Extract FinGPT sentiment score for ML prediction adjustment
                    sentiment_score = None
                    try:
                        fingpt_patterns = [p for p in patterns if p.get('source') == 'FINGPT']
                        if fingpt_patterns:
                            # Use confidence as sentiment score (0-1 range)
                            # BULLISH: high confidence = bullish score (>0.5)
                            # BEARISH: high confidence = bearish score (<0.5)
                            fg = fingpt_patterns[0]
                            conf = fg.get('confidence', 0.5)
                            if fg.get('signal') == 'BULLISH':
                                sentiment_score = 0.5 + (conf * 0.5)  # Map to 0.5-1.0 range
                            elif fg.get('signal') == 'BEARISH':
                                sentiment_score = 0.5 - (conf * 0.5)  # Map to 0.0-0.5 range
                            logger.debug(f"FinGPT sentiment for ML: {sentiment_score:.2f} (signal={fg.get('signal')}, conf={conf:.2f})")
                    except Exception as se:
                        logger.debug(f"Sentiment extraction error: {se}")
                    
                    coord = mlc.predict_with_coordination(symbol, data, sentiment_score=sentiment_score)
                    # Extract raw predictions for response payload
                    if not isinstance(coord, dict):
                        coord = {}
                    ml_predictions = coord.get('basic', {}) if isinstance(coord.get('basic'), dict) else coord.get('basic', {}) or {}
                    enhanced_predictions = coord.get('enhanced', {}) if isinstance(coord.get('enhanced'), dict) else coord.get('enhanced', {}) or {}
                else:
                    # Skip ML prediction during training
                    # ⚡ SILENT: Don't log every skip during training to reduce log noise
                    # logger.debug(f"🔒 ML prediction skipped for {symbol} (training mode)")
                    ml_predictions = {}
                    enhanced_predictions = {}
                
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
                # Regime score: use realized volatility to modulate weights (higher vol → favor enhanced)
                try:
                    recent_ret = data['close'].pct_change().tail(20)
                    vol20 = float(recent_ret.std()) if len(recent_ret) > 5 else 0.0
                    recent_ret60 = data['close'].pct_change().tail(60)
                    vol60 = float(recent_ret60.std()) if len(recent_ret60) > 5 else 0.0
                    regime = min(1.0, max(0.0, (vol20 / max(1e-6, vol60)) if vol60 > 0 else vol20 / 0.05))
                except Exception:
                    regime = 0.5

                # Aggregate evidence from patterns and sentiment
                # Pre-compute visual confirmation and sentiment gating thresholds
                try:
                    enable_yolo_confirm = (os.getenv('ENABLE_YOLO_CONFIRM', '1').lower() in ('1', 'true', 'yes', 'on'))
                except Exception:
                    enable_yolo_confirm = True
                try:
                    yolo_confirm_mult = float(os.getenv('YOLO_CONFIRM_MULT', '1.5'))
                except Exception:
                    yolo_confirm_mult = 1.5
                try:
                    yolo_min_conf_ev = float(os.getenv('YOLO_MIN_CONF_EVID', '0.25'))
                except Exception:
                    yolo_min_conf_ev = 0.25
                try:
                    # ⚡ IMPROVED: Lowered threshold from 0.65 to 0.50 for better Turkish model compatibility
                    # Turkish BERT produces lower confidence scores than English FinBERT
                    # 0.50 threshold allows 45% more sentiment data to impact ML predictions
                    fingpt_min_conf = float(os.getenv('FINGPT_MIN_CONF', '0.50'))
                except Exception:
                    fingpt_min_conf = 0.50
                try:
                    fingpt_min_news = int(os.getenv('FINGPT_MIN_NEWS', '1'))
                except Exception:
                    fingpt_min_news = 1

                # Visual confirmation flags (recent YOLO detections)
                visual_bullish = False
                visual_bearish = False
                try:
                    for _vp in (patterns or []):
                        if str(_vp.get('source', '')).upper() == 'VISUAL_YOLO':
                            try:
                                if float(_vp.get('confidence', 0.0)) < yolo_min_conf_ev:
                                    continue
                            except Exception:
                                continue
                            _sig = str(_vp.get('signal', '')).upper()
                            if _sig == 'BULLISH':
                                visual_bullish = True
                            elif _sig == 'BEARISH':
                                visual_bearish = True
                except Exception:
                    visual_bullish = visual_bearish = False

                def _agg_evidence(h_days: int):
                    try:
                        pat_score = 0.0
                        pat_w = 0.0
                        sent_score = 0.0
                        sent_w = 0.0
                        for p in (patterns or []):
                            try:
                                src = str(p.get('source', '')).upper()
                                sig = str(p.get('signal', '')).upper()
                                sgn = 1.0 if sig == 'BULLISH' else (-1.0 if sig == 'BEARISH' else 0.0)
                                confp = float(p.get('confidence', (p.get('strength', 50)/100.0)))
                                confp = max(0.0, min(1.0, confp))
                                # ✅ FIX: Horizon weighting with smooth exponential decay instead of sharp steps
                                # Exponential decay: h_w = 1.0 * exp(-decay_rate * (h_days - 1))
                                # Adjust decay_rate to match approximate values: h_w(3)≈0.95, h_w(7)≈0.8, h_w(14)≈0.6
                                # Use decay_rate ≈ 0.03 for smooth transitions
                                decay_rate = 0.03
                                h_w = max(0.5, 1.0 * math.exp(-decay_rate * max(0, h_days - 1)))
                                # ✅ FIX: Source weighting - YOLO and Advanced TA get boost, FinGPT gets slight boost
                                src_w = 1.1 if src in ('VISUAL_YOLO', 'ADVANCED_TA') else (1.05 if src == 'FINGPT' else 1.0)
                                # FinGPT gating: require sufficient confidence and news count
                                if src == 'FINGPT':
                                    try:
                                        nnews = int(p.get('news_count', 0) or 0)
                                    except Exception:
                                        nnews = 0
                                    if (confp >= fingpt_min_conf) and (nnews >= fingpt_min_news):
                                        w = confp * h_w * src_w
                                        sent_score += sgn * w
                                        sent_w += w
                                    else:
                                        # Ignore weak/insufficient sentiment
                                        pass
                                else:
                                    w = confp * h_w * src_w
                                    # YOLO confirmation: if TA pattern direction matches YOLO, amplify
                                    if enable_yolo_confirm and src in ('ADVANCED_TA', 'BASIC'):
                                        if (sgn > 0 and visual_bullish) or (sgn < 0 and visual_bearish):
                                            w *= yolo_confirm_mult
                                    pat_score += sgn * w
                                    pat_w += w
                            except Exception:
                                continue
                        pat_val = (pat_score / pat_w) if pat_w > 0 else 0.0
                        sent_val = (sent_score / sent_w) if sent_w > 0 else 0.0
                        return max(-1.0, min(1.0, pat_val)), max(-1.0, min(1.0, sent_val))
                    except Exception:
                        return 0.0, 0.0

                # --- Helper: 1D directional booster (lightweight, on-cycle) ---
                def _compute_1d_booster_prob(df):
                    try:
                        try:
                            from sklearn.linear_model import LogisticRegression
                        except Exception:
                            return None
                        import numpy as np  # local import
                        import pandas as pd  # local import

                        if not isinstance(df, (pd.DataFrame,)) or len(df) < 140:
                            return None
                        # Use last ~360 bars to keep fit light
                        d = df[['open', 'high', 'low', 'close', 'volume']].copy().tail(360)
                        close = d['close']
                        high = d['high']
                        low = d['low']
                        volume = d['volume'].astype(float)

                        feats = pd.DataFrame(index=d.index)
                        # overnight proxy
                        feats['overnight'] = (d['open'] / close.shift(1) - 1.0) * 100.0
                        feats['ret1'] = close.pct_change(1) * 100.0
                        feats['mom3'] = ((close / close.rolling(3).mean()) - 1.0) * 100.0
                        feats['rv5'] = np.log(close).diff().rolling(5).std() * np.sqrt(252) * 100.0
                        # RSI(3)
                        delta = close.diff()
                        up = delta.clip(lower=0).rolling(3).mean()
                        down = (-delta.clip(upper=0)).rolling(3).mean()
                        rs = up / (down + 1e-9)
                        feats['rsi3'] = 100.0 - (100.0 / (1.0 + rs))
                        feats['gap'] = ((d['open'] - close.shift(1)) / (close.shift(1) + 1e-9)) * 100.0
                        feats['tail_up'] = ((high - close) / (close + 1e-9)) * 100.0
                        feats['tail_dn'] = ((close - low) / (close + 1e-9)) * 100.0
                        tr1 = (high - low).abs()
                        tr2 = (high - close.shift(1)).abs()
                        tr3 = (low - close.shift(1)).abs()
                        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                        atr5 = tr.rolling(5).mean()
                        feats['atr5_n'] = (atr5 / (close + 1e-9)) * 100.0
                        ma20 = close.rolling(20).mean()
                        sd20 = close.rolling(20).std()
                        feats['boll_pos'] = (close - ma20) / (sd20 + 1e-9)
                        ll14 = low.rolling(14).min()
                        hh14 = high.rolling(14).max()
                        feats['stoch_k'] = ((close - ll14) / ((hh14 - ll14) + 1e-9)) * 100.0
                        ema10 = close.ewm(span=10, adjust=False).mean()
                        feats['ema10_slope'] = ema10.pct_change(1) * 100.0
                        vol_ma = volume.rolling(20).mean()
                        vol_sd = volume.rolling(20).std()
                        feats['vol_z'] = (volume - vol_ma) / (vol_sd + 1e-9)
                        # day of week one-hot
                        dow = feats.index.dayofweek
                        for dval in range(5):
                            feats[f'dow_{dval}'] = (dow == dval).astype(int)

                        # target (t+1 up?)
                        y = (close.shift(-1) > close).astype(float)
                        mask = feats.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & y.notna()
                        feats = feats.loc[mask]
                        y = y.loc[mask]
                        if len(feats) < 120:
                            return None
                        # Train on all but last row; predict last row prob for up move
                        X_train = feats.iloc[:-1]
                        y_train = y.iloc[:-1]
                        X_last = feats.iloc[[-1]]
                        if len(np.unique(y_train)) < 2:
                            return None
                        clf = LogisticRegression(max_iter=400, solver='liblinear')
                        clf.fit(X_train.values, y_train.values)
                        p_up = float(clf.predict_proba(X_last.values)[0, 1])
                        return p_up
                    except Exception:
                        return None

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
                                    # Horizon-specific calibration: longer horizons need stronger move for same confidence
                                    h_days = int(str(h).replace('d', '') or 7)
                                    move_scale = 0.05 * max(1.0, h_days / 7.0)
                                    # Regime-weighted base (more volatile → prioritize enhanced)
                                    base_w = (0.6 + 0.2 * regime) if src == 'enhanced' else (0.65 - 0.15 * regime)
                                    rr = float(rel) if isinstance(rel, (int, float)) else (0.65 if src == 'enhanced' else 0.6)
                                    conf = max(0.25, min(0.95, base_w * rr * min(1.0, abs(cdelta) / move_scale)))
                                except Exception:
                                    conf = max(0.25, min(0.95, abs(cdelta) / 0.05))
                                
                                # Evidence-based confidence adjustment (applied to base model confidence)
                                try:
                                    h_days = int(str(h).replace('d', '') or 7)
                                except Exception:
                                    h_days = 7
                                # Gate by env: PATTERN_SENTI_META (default on)
                                enable_meta = True
                                try:
                                    enable_meta = (os.getenv('PATTERN_SENTI_META', '1').lower() in ('1', 'true', 'yes', 'on'))
                                except Exception:
                                    enable_meta = True
                                pat_s, sent_s = _agg_evidence(h_days)
                                
                                # 🎯 NEW: ML Primary + Pattern Confirmation System
                                # Instead of fixed weights, use agreement-based boost/penalty
                                # ✅ FIX: Initialize w_pat, w_sent, signed_adj for all code paths
                                w_pat = 0.0
                                w_sent = 0.0
                                signed_adj = 0.0
                                
                                try:
                                    use_new_validation = (os.getenv('USE_PATTERN_AGREEMENT', '1').lower() in ('1', 'true', 'yes', 'on'))
                                except Exception:
                                    use_new_validation = True
                                
                                if use_new_validation:
                                    # Calculate pattern agreement score (extracted)
                                    try:
                                        from bist_pattern.pattern.validation import calculate_pattern_agreement
                                        pattern_agreement = calculate_pattern_agreement(
                                            patterns=patterns,
                                            ml_signal='BULLISH' if cdelta > 0 else 'BEARISH',
                                            ml_confidence=conf
                                        )
                                    except Exception:
                                        pattern_agreement = 0.0
                                    evidence_boost = pattern_agreement  # Already clamped in function
                                    # ✅ FIX: Calculate signed_adj for delta tilt even when using new validation
                                    try:
                                        _ps = self._load_param_store() or {}
                                        _hkey = str(h)
                                        _wmap = (_ps.get('weights') or {}).get(_hkey) if isinstance(_ps, dict) else None
                                        if isinstance(_wmap, dict):
                                            w_pat = float(_wmap.get('w_pat', 0.0)) or 0.0
                                            w_sent = float(_wmap.get('w_sent', 0.0)) or 0.0
                                        else:
                                            raise KeyError('no weights')
                                    except Exception:
                                        if h_days <= 1:
                                            w_pat, w_sent = 0.12, 0.10
                                        elif h_days <= 3:
                                            w_pat, w_sent = 0.10, 0.08
                                        elif h_days <= 7:
                                            w_pat, w_sent = 0.06, 0.05
                                        elif h_days <= 14:
                                            w_pat, w_sent = 0.04, 0.03
                                        else:
                                            w_pat, w_sent = 0.03, 0.02
                                    signed_adj = (w_pat * pat_s + w_sent * sent_s) if enable_meta else 0.0
                                else:
                                    # OLD SYSTEM: Fixed weights
                                    try:
                                        _ps = self._load_param_store() or {}
                                        _hkey = str(h)
                                        _wmap = (_ps.get('weights') or {}).get(_hkey) if isinstance(_ps, dict) else None
                                        if isinstance(_wmap, dict):
                                            w_pat = float(_wmap.get('w_pat', 0.0)) or 0.0
                                            w_sent = float(_wmap.get('w_sent', 0.0)) or 0.0
                                        else:
                                            raise KeyError('no weights')
                                    except Exception:
                                        if h_days <= 1:
                                            w_pat, w_sent = 0.12, 0.10
                                        elif h_days <= 3:
                                            w_pat, w_sent = 0.10, 0.08
                                        elif h_days <= 7:
                                            w_pat, w_sent = 0.06, 0.05
                                        elif h_days <= 14:
                                            w_pat, w_sent = 0.04, 0.03
                                        else:
                                            w_pat, w_sent = 0.03, 0.02
                                    signed_adj = (w_pat * pat_s + w_sent * sent_s) if enable_meta else 0.0
                                    evidence_boost = max(-0.15, min(0.15, signed_adj))  # Clamped evidence contribution

                                # Optional: 1D directional booster (confidence alignment)
                                booster_adj = 0.0
                                booster_p = None
                                try:
                                    enable_booster = (os.getenv('ENABLE_1D_BOOSTER', '1').lower() in ('1', 'true', 'yes', 'on'))
                                except Exception:
                                    enable_booster = True
                                if enable_booster and h_days <= 1:
                                    booster_p = _compute_1d_booster_prob(data)
                                    if isinstance(booster_p, float):
                                        # alignment: if booster agrees with direction, increase confidence up to ~0.08
                                        agree = (cdelta >= 0 and booster_p >= 0.5) or (cdelta < 0 and booster_p < 0.5)
                                        strength = abs(booster_p - 0.5) * 2.0  # [0..1]
                                        booster_adj = (0.08 * strength) * (1.0 if agree else -1.0)

                                # ⚡ ONLINE CONFIDENCE ADJUSTMENT: Gerçekleşmelere bakarak confidence'ı düzelt
                                # Modeli yeniden eğitmeden, son 30-60 gün içinde benzer confidence seviyesinde
                                # yapılan tahminlerin gerçekleşme oranına (dir_hit rate) göre confidence'ı ayarlar.
                                try:
                                    online_adj = self._get_empirical_confidence_adjustment(str(h), conf)
                                except Exception:
                                    online_adj = 0.0
                                
                                # Add evidence boost directly to base model confidence
                                conf_after_meta = max(0.25, min(0.95, conf + evidence_boost))
                                conf_after_meta_and_booster = max(0.25, min(0.95, conf_after_meta + booster_adj))
                                
                                # ⚡ Apply online adjustment AFTER evidence/booster (so it adjusts the final confidence)
                                # This ensures the adjustment is based on the model's actual performance, not just raw confidence
                                conf_final = max(0.25, min(0.95, conf_after_meta_and_booster + online_adj))

                                # Small horizon-aware delta tilt using evidence alignment (strictly bounded)
                                try:
                                    try:
                                        enable_delta_tilt = (os.getenv('ENABLE_DELTA_TILT', '1').lower() in ('1', 'true', 'yes', 'on'))
                                    except Exception:
                                        enable_delta_tilt = True
                                    thr_map = {'1d': 0.008, '3d': 0.021, '7d': 0.03, '14d': 0.03, '30d': 0.025}
                                    alpha_map = {'1d': 0.25, '3d': 0.15, '7d': 0.10, '14d': 0.08, '30d': 0.08}
                                    h_key = str(h)
                                    base_thr = float(thr_map.get(h_key, 0.03))
                                    alpha = float(alpha_map.get(h_key, 0.08))
                                    # Normalize signed_adj to [-1,1] scale via conf clip used (0.15 window)
                                    mag = min(1.0, max(0.0, abs(signed_adj) / 0.15 if 0.15 > 0 else 0.0))
                                    sgn = 1.0 if signed_adj >= 0 else -1.0
                                    # ✅ FIX: Smooth agreement instead of binary
                                    # Calculate alignment strength: how well evidence aligns with prediction
                                    alignment_strength = (cdelta * sgn) / max(abs(cdelta), 1e-8) if abs(cdelta) > 1e-8 else 0.0
                                    # Sigmoid function for smooth agreement (k=5 for sharp but smooth transition)
                                    k = 5.0
                                    agreement_score = 1.0 / (1.0 + np.exp(-k * alignment_strength))
                                    # Convert to agree factor (0.0 = disagree, 1.0 = agree) with smooth transition
                                    agree_factor = agreement_score
                                    # Primary tilt follows evidence with smooth agreement
                                    base_tilt = (alpha * base_thr * mag) if enable_delta_tilt else 0.0
                                    # Smooth interpolation: full agree → +base_tilt, full disagree → -0.5*base_tilt
                                    tilt_ev = base_tilt * (2.0 * agree_factor - 1.0) if agree_factor >= 0.5 else -0.5 * base_tilt * (1.0 - agree_factor * 2.0)
                                    # Booster tilt (lighter than confidence impact)
                                    tilt_boost = 0.0
                                    if isinstance(booster_p, float):
                                        bmag = abs(booster_p - 0.5) * 2.0
                                        # ✅ FIX: Smooth booster agreement instead of binary
                                        booster_alignment = (cdelta * (booster_p - 0.5)) / max(abs(cdelta), 1e-8) if abs(cdelta) > 1e-8 else 0.0
                                        booster_agreement = 1.0 / (1.0 + np.exp(-k * booster_alignment))
                                        bagree_factor = booster_agreement
                                        # Smooth interpolation for booster agreement
                                        tilt_boost = ((0.5 * alpha) * base_thr * bmag * (2.0 * bagree_factor - 1.0)) if enable_delta_tilt else 0.0
                                    cdelta_tilted = cdelta + tilt_ev + tilt_boost
                                    # Safety clamp
                                    if cdelta_tilted > 0.5:
                                        cdelta_tilted = 0.5
                                    if cdelta_tilted < -0.5:
                                        cdelta_tilted = -0.5
                                    delta_contrib = (cdelta_tilted - cdelta)
                                except Exception:
                                    cdelta_tilted = cdelta
                                    delta_contrib = 0.0

                                cur[src]['delta_pct'] = float(cdelta_tilted)
                                cur[src]['confidence'] = conf_final
                                cur[src]['evidence'] = {
                                    'pattern_score': float(pat_s),
                                    'sentiment_score': float(sent_s),
                                    'contrib_conf': float(conf_final - conf),
                                    'w_pat': float(w_pat),
                                    'w_sent': float(w_sent),
                                }
                                if booster_p is not None:
                                    cur[src]['evidence']['booster_prob'] = float(booster_p)
                                    cur[src]['evidence']['contrib_booster'] = float(booster_adj)
                                if isinstance(delta_contrib, float) and abs(delta_contrib) > 0:
                                    cur[src]['evidence']['contrib_delta'] = float(delta_contrib)
                                # Regime-aware score for best-of selection
                                eff_rel = float(max(0.0, min(1.0, rel))) if isinstance(rel, (int, float)) else (0.65 if src == 'enhanced' else 0.6)
                                score = abs(cdelta) * eff_rel * (1.0 + (0.15 if (src == 'enhanced' and regime >= 0.6) else 0.0))
                                entries.append((score, src))
                    # Decide best
                    if enhanced_first and ('enhanced' in cur):
                        cur['best'] = 'enhanced'
                    elif enhanced_first and ('enhanced' not in cur) and ('basic' in cur):
                        cur['best'] = 'basic'
                    elif entries:
                        best_src = sorted(entries, key=lambda x: x[0], reverse=True)[0][1]
                        cur['best'] = best_src
                    
                    # Apply param_store thresholds (delta_thr, conf_thr for gating small signals)
                    try:
                        store = self._load_param_store() or {}
                        ps_h = (store.get('horizons') or {}).get(str(h), {})
                        th = ps_h.get('thresholds') or {}
                        # --- Bandit A/B: optionally use challenger thresholds for a stable subset of symbols ---
                        ab_label = 'ab:prod'
                        try:
                            import hashlib
                            bcfg = ((store.get('bandit') or {}).get('horizons') or {}).get(str(h), {})
                            traffic = float(bcfg.get('traffic', 0.10))
                            chall = bcfg.get('challenger') or None
                            # Stable assignment by symbol×horizon hash
                            hv = f"{symbol.upper()}|{str(h)}".encode()
                            hv_int = int(hashlib.sha1(hv).hexdigest()[:8], 16)
                            frac = (hv_int % 1000) / 1000.0
                            if chall and frac < max(0.0, min(1.0, traffic)):
                                th = chall
                                ab_label = 'ab:chall'
                        except Exception:
                            ab_label = 'ab:prod'
                        # ✅ FIX: Use sensible defaults if thresholds are missing (0.0 would disable thresholds)
                        # Default to conservative thresholds if not found in param_store
                        delta_thr = float(th.get('delta_thr') or ConfigManager.get('DEFAULT_DELTA_THR', 0.03))
                        conf_thr = float(th.get('conf_thr') or ConfigManager.get('DEFAULT_CONF_THR', 0.65))
                        # If best exists, gate tiny signals by thresholds
                        # ⚡ REMOVED: Isotonic bypass check (isotonic calibration removed from prediction)
                        try:
                            if isinstance(calib_override, dict) and ('penalty_factor' in calib_override) and calib_override.get('penalty_factor') is not None:
                                penalty_factor = float(calib_override.get('penalty_factor'))
                            else:
                                penalty_factor = float(os.getenv('THRESHOLD_PENALTY_FACTOR', '0.95'))
                        except Exception:
                            penalty_factor = 0.95
                        
                        # Apply threshold penalty if signal is below thresholds
                        if 'best' in cur and isinstance(cur.get(cur['best']), dict):
                            be = cur[cur['best']]
                            try:
                                if abs(float(be.get('delta_pct', 0.0))) < delta_thr or float(be.get('confidence', 0.0)) < conf_thr:
                                    # Demote confidence smoothly (softer penalty temporarily)
                                    be['confidence'] = max(0.25, float(be.get('confidence', 0.0)) * penalty_factor)
                            except Exception:
                                pass
                            # Tag param version used for A/B attribution
                            try:
                                pv = str((store or {}).get('generated_at') or '')
                                be['param_version'] = (pv + '|' + ab_label) if pv else ab_label
                            except Exception:
                                pass
                        # ⚡ REMOVED: Isotonic calibration (removed from both training and prediction for consistency)
                    except Exception:
                        pass
                    
                    # ✅ FIX: Add current horizon to ml_unified dictionary (inside loop scope)
                    # Only add if there's actual data
                    if cur:  # Only add if there's actual data
                        ml_unified[h] = cur
            except Exception as e:
                # ✅ FIX: Log exception to diagnose ml_unified creation failures
                import traceback
                logger.error(f"❌ {symbol}: ml_unified creation failed: {e}")
                logger.debug(f"   Traceback:\n{traceback.format_exc()}")
                ml_unified = {}

            # ✅ FIX: Debug log to verify FinGPT and YOLO patterns are in the list before caching
            fingpt_count = len([p for p in patterns if p.get('source') == 'FINGPT'])
            yolo_count = len([p for p in patterns if 'YOLO' in str(p.get('source', ''))])
            if fingpt_count > 0 or yolo_count > 0:
                logger.debug(f"📊 {symbol}: Patterns before cache - FinGPT: {fingpt_count}, YOLO: {yolo_count}, Total: {len(patterns)}")
            
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
                'ml_unified': ml_unified or {}  # ✅ FIX: Ensure ml_unified is always a dict
            }
            # --- Feedback logging: write one row per horizon into predictions_log (best-effort)
            try:
                from models import PredictionsLog, Stock  # type: ignore
                
                # Training/detect-only context gate: skip DB writes in training or when explicitly disabled
                try:
                    import os as _os
                    import sys as _sys
                    _argv = ' '.join(_sys.argv) if hasattr(_sys, 'argv') else ''
                    _skip_predlog = (
                        ('bulk_train_all.py' in _argv)
                        or (_os.getenv('DISABLE_PREDICTIONS_LOG', '0').lower() in ('1', 'true', 'yes', 'on'))
                        or (_os.getenv('DETECT_ONLY', '0').lower() in ('1', 'true', 'yes', 'on'))
                    )
                except Exception:
                    _skip_predlog = False
                if _skip_predlog:
                    logger.debug(f"🛑 Skipping PredictionsLog write for {symbol} (training/detect-only context)")
                    raise RuntimeError("predlog_skipped")

                # Debug logging for prediction tracking
                logger.debug(f"🔍 Prediction logging for {symbol}:")
                logger.debug(f"  ml_predictions: {len(ml_predictions) if ml_predictions else 0} horizons")
                logger.debug(f"  enhanced_predictions: {len(enhanced_predictions) if enhanced_predictions else 0} horizons")
                logger.debug(f"  ml_unified: {len(ml_unified) if ml_unified else 0} horizons")
                
                # Resolve stock_id quickly (optional)
                # Note: We're already inside app.app_context() from analyze_stock()
                stock_id = None
                try:
                    st = Stock.query.filter_by(symbol=symbol.upper()).first()
                    stock_id = getattr(st, 'id', None)
                except Exception:
                    stock_id = None
                # Fill missing horizons from enhanced/basic predictions (even if ml_unified partially filled)
                try:
                    expected_horizons = ['1d', '3d', '7d', '14d', '30d']
                    missing_horizons = [h for h in expected_horizons if h not in ml_unified]
                    
                    if missing_horizons and (isinstance(enhanced_predictions, dict) or isinstance(ml_predictions, dict)):
                        cur_price = float(data['close'].iloc[-1])
                        
                        for hk in missing_horizons:
                            enh_obj = enhanced_predictions.get(hk) if isinstance(enhanced_predictions, dict) else None
                            bas_obj = ml_predictions.get(hk) if isinstance(ml_predictions, dict) else None
                            
                            enh_price = None
                            enh_conf = None
                            if isinstance(enh_obj, dict):
                                # accept ensemble_prediction or price
                                v = enh_obj.get('ensemble_prediction') if 'ensemble_prediction' in enh_obj else enh_obj.get('price')
                                if isinstance(v, (int, float)):
                                    enh_price = float(v)
                                if isinstance(enh_obj.get('confidence'), (int, float)):
                                    enh_conf = float(enh_obj['confidence'])
                            
                            bas_price = None
                            if isinstance(bas_obj, dict):
                                v = bas_obj.get('price')
                                if isinstance(v, (int, float)):
                                    bas_price = float(v)
                            
                            # Add minimal evidence for fallback horizons
                            h_days = int(str(hk).replace('d', '') or 7)
                            if h_days <= 1:
                                w_pat, w_sent = 0.12, 0.10
                            elif h_days <= 3:
                                w_pat, w_sent = 0.10, 0.08
                            elif h_days <= 7:
                                w_pat, w_sent = 0.06, 0.05
                            elif h_days <= 14:
                                w_pat, w_sent = 0.04, 0.03
                            else:
                                w_pat, w_sent = 0.03, 0.02
                            
                            entry: dict[str, dict | str] = {}
                            if enh_price is not None:
                                entry['enhanced'] = {
                                    'price': enh_price,
                                    'confidence': enh_conf,
                                    'delta_pct': (enh_price - cur_price) / cur_price if cur_price else None,
                                    'evidence': {
                                        'pattern_score': 0.0,  # Fallback: neutral
                                        'sentiment_score': 0.0,  # Fallback: neutral
                                        'w_pat': float(w_pat),
                                        'w_sent': float(w_sent),
                                        'contrib_conf': 0.0,  # No adjustment in fallback
                                        'source': 'fallback'  # Tag to identify fallback entries
                                    }
                                }
                            if bas_price is not None:
                                entry['basic'] = {
                                    'price': bas_price,
                                    'confidence': None,
                                    'delta_pct': (bas_price - cur_price) / cur_price if cur_price else None,
                                    'evidence': {
                                        'pattern_score': 0.0,
                                        'sentiment_score': 0.0,
                                        'w_pat': float(w_pat),
                                        'w_sent': float(w_sent),
                                        'contrib_conf': 0.0,
                                        'source': 'fallback'
                                    }
                                }
                            
                            if entry:
                                entry['best'] = 'enhanced' if 'enhanced' in entry else 'basic'
                                ml_unified[str(hk)] = entry
                        
                        if len(ml_unified) > 0:
                            logger.debug(f"✅ Filled {len(missing_horizons)} missing horizons for {symbol}, total now: {len(ml_unified)}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fill missing horizons for {symbol}: {e}")
                    ml_unified = ml_unified or {}
                
                # Check if we have anything to log
                if not ml_unified or len(ml_unified) == 0:
                    logger.warning(f"⚠️ {symbol}: ml_unified is EMPTY - no predictions will be logged!")
                    logger.debug("  This means no ML models produced predictions for this symbol")
                    raise ValueError("ml_unified empty - skipping prediction logging")

                # Iterate horizons present in ml_unified
                for hkey, entry in (ml_unified or {}).items():
                    try:
                        if not isinstance(entry, dict):
                            continue
                        best = entry.get('best')
                        src_entry = entry.get(best) if best in ('basic', 'enhanced') else None
                        pred_px = None
                        conf = None
                        delta = None
                        pscore = None
                        sscore = None
                        if isinstance(src_entry, dict):
                            pred_px = src_entry.get('price')
                            conf = src_entry.get('confidence')
                            delta = src_entry.get('delta_pct')
                            try:
                                ev = src_entry.get('evidence') or {}
                                pscore = float(ev.get('pattern_score')) if isinstance(ev.get('pattern_score'), (int, float)) else None
                                sscore = float(ev.get('sentiment_score')) if isinstance(ev.get('sentiment_score'), (int, float)) else None
                            except Exception:
                                pscore = sscore = None
                        # ⚡ FIX: param_version from src_entry (contains ab:chall or ab:prod tag)
                        # First try to get from src_entry (set during threshold application)
                        param_version = None
                        if isinstance(src_entry, dict):
                            param_version = src_entry.get('param_version')
                        # Fallback: get from param_store if not in src_entry
                        if not param_version:
                            try:
                                _ps = self._load_param_store() or {}
                                pv_base = str(_ps.get('generated_at') or '') or None
                                # Default to 'ab:prod' if no tag is present
                                param_version = (pv_base + '|ab:prod') if pv_base else 'ab:prod'
                            except Exception:
                                param_version = 'ab:prod'  # Default tag
                        log = PredictionsLog(
                            stock_id=stock_id,
                            symbol=symbol.upper(),
                            horizon=str(hkey),
                            ts_pred=datetime.utcnow(),
                            price_now=float(data['close'].iloc[-1]),
                            pred_price=float(pred_px) if isinstance(pred_px, (int, float)) else None,
                            delta_pred=float(delta) if isinstance(delta, (int, float)) else None,
                            model=(best or None),
                            unified_best=(best or None),
                            confidence=float(conf) if isinstance(conf, (int, float)) else None,
                            pat_score=pscore,
                            sent_score=sscore,
                            visual_bullish=any(
                                (
                                    p.get('source') == 'VISUAL_YOLO'
                                    and str(p.get('signal', '')).upper() == 'BULLISH'
                                )
                                for p in (patterns or [])
                            ),
                            visual_bearish=any(
                                (
                                    p.get('source') == 'VISUAL_YOLO'
                                    and str(p.get('signal', '')).upper() == 'BEARISH'
                                )
                                for p in (patterns or [])
                            ),
                            param_version=param_version,
                        )
                        # Note: Already in app.app_context(), no need for nested context
                        try:
                            from bist_pattern.core.db_manager import DBManager
                            DBManager.add(log)
                        except Exception as e:
                            # ✅ FIX: Use ErrorHandler for consistent error handling
                            ErrorHandler.handle(e, f'predictions_log_add_{symbol}_{hkey}', level='error')
                        # Commit outside the loop to batch multiple inserts
                    except Exception as e:
                        logger.debug(f"⚠️ Skipped {symbol} {hkey}: {e}")
                        continue
                
                # Commit all predictions for this symbol (already in app.app_context())
                try:
                    from bist_pattern.core.db_manager import DBManager
                    DBManager.commit()
                    logger.info(f"✅ {symbol}: Wrote {len(ml_unified)} predictions to PredictionsLog")
                except Exception as e:
                    # ✅ FIX: Use ErrorHandler and DBManager for consistent error handling
                    ErrorHandler.handle_database_error(e, f'predictions_log_commit_{symbol}', 'commit')
                    try:
                        DBManager.rollback()
                    except Exception as rollback_e:
                        ErrorHandler.handle(rollback_e, f'predictions_log_rollback_{symbol}', level='error')
            except Exception as e:
                # Log the failure but don't break analysis
                if str(e) == 'predlog_skipped':
                    logger.debug(f"PredictionsLog write skipped for {symbol}")
                else:
                    logger.warning(f"⚠️ Prediction logging failed for {symbol}: {e}")
                    if "ml_unified empty" in str(e):
                        logger.debug("  → This is expected if no ML models are available")
                    else:
                        logger.error(f"  → Unexpected error: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
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
                        # ✅ FIX: Always write enhanced if available
                        if isinstance(cur_enhanced, dict) and cur_enhanced:
                            ent['enhanced'] = cur_enhanced
                        elif isinstance(enhanced_predictions, dict) and enhanced_predictions:
                            ent['enhanced'] = enhanced_predictions
                        # ✅ FIX: Always write basic if available (even if enhanced exists)
                        # This allows fallback to basic for horizons not in enhanced
                        if isinstance(ml_predictions, dict) and ml_predictions:
                            ent['basic'] = ml_predictions

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
                from flask import current_app
                try:
                    flask_app = current_app
                except RuntimeError:
                    flask_app = None
                if flask_app and hasattr(flask_app, 'broadcast_log'):
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

                # Include FinGPT sentiment evidence (compact) for UI counters
                fingpt_evidence = []
                try:
                    fg = [p for p in (result.get('patterns') or []) if (str(p.get('source', '')).upper() == 'FINGPT')]
                    fg_sorted = sorted(fg, key=lambda p: float(p.get('confidence', 0.0)), reverse=True)
                    for p in fg_sorted[:2]:
                        fingpt_evidence.append({
                            'source': 'FINGPT',
                            'pattern': p.get('pattern') or 'FINGPT_SENTIMENT',
                            'signal': p.get('signal'),
                            'confidence': float(p.get('confidence', 0.0) or 0.0),
                            'news_count': int(p.get('news_count', 0) or 0)
                        })
                except Exception:
                    fingpt_evidence = []

                snap[symbol] = {
                    'timestamp': result['timestamp'],
                    'signal': result['overall_signal'].get('signal', 'NEUTRAL'),
                    'confidence': result['overall_signal'].get('confidence', 0.0),
                    'strength': result['overall_signal'].get('strength', 0),
                    'visual': visual_evidence,
                    'evidence': fingpt_evidence,
                }
                with open(snap_path, 'w') as wf:
                    json.dump(snap, wf)
            except Exception:
                pass

            try:
                from flask import current_app
                try:
                    flask_app = current_app
                except RuntimeError:
                    flask_app = None
                if flask_app and hasattr(flask_app, 'broadcast_log'):
                    sig = result['overall_signal'].get('signal', '?')
                    conf = int(round(float(result['overall_signal'].get('confidence', 0)) * 100))
                    # ✅ FIX: Açıklayıcı log mesajı - Pattern Consensus Confidence
                    sig_tr = 'YÜKSELİŞ' if sig == 'BULLISH' else 'DÜŞÜŞ' if sig == 'BEARISH' else 'NÖTR'
                    flask_app.broadcast_log('SUCCESS', f'🎯 {symbol} Pattern Consensus: {sig_tr} (güven=%{conf})', 'ai_analysis')  # type: ignore[attr-defined]
                
                # ⭐ EKLENEN: User signal broadcast (VISUAL_YOLO evidence dahil) – ayrı modülde, eğitim bağlamında gate'li
                try:
                    from bist_pattern.core.broadcaster import broadcast_user_signal
                    broadcast_user_signal(symbol, result, flask_app)
                except Exception as e:
                    logger.warning(f"⚠️ Live signal broadcast error for {symbol}: {e}")
                    
            except Exception:
                pass
            
            # Cache'e TTL ile kaydet
            self.cache[cache_key] = {
                'data': result,
                'timestamp': current_time
            }

            # Persist file cache for cross-process reuse (used by batch API)
            try:
                import os as _os
                import json as _json
                log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                file_cache_dir = _os.path.join(log_dir, 'pattern_cache')
                try:
                    _os.makedirs(file_cache_dir, exist_ok=True)
                except Exception:
                    pass
                fpath = _os.path.join(file_cache_dir, f'{symbol}.json')
                with open(fpath, 'w') as wf:
                    _json.dump(result, wf)
            except Exception:
                # Best-effort only; ignore persistence errors to avoid breaking analysis
                pass
            
            # Cache cleanup - 100'den fazla entry varsa eski olanları temizle
            if len(self.cache) > self.result_cache_max_size:
                self._cleanup_cache()
            
            # Opsiyonel: overall sinyali canlı loga yaz (dashboard için)
            try:
                from flask import current_app
                try:
                    flask_app = current_app
                except RuntimeError:
                    flask_app = None
                overall = (result or {}).get('overall_signal') or {}
                if overall and flask_app and hasattr(flask_app, 'broadcast_log'):
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
                conf = 0.4  # Basic ML default confidence (lower than enhanced 0.6-0.7)
                if isinstance(val, (int, float)):
                    v = float(val)
                elif isinstance(val, dict):
                    for cand in ('price', 'prediction', 'target', 'value', 'y'):
                        if cand in val and isinstance(val[cand], (int, float)):
                            v = float(val[cand])
                            break
                    # Try to extract confidence if available
                    if 'confidence' in val and isinstance(val['confidence'], (int, float)):
                        conf = float(val['confidence'])
                if v is not None:
                    out[str(key).lower()] = {'price': v, 'confidence': conf}
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
                # Convert to expected format: {horizon: {price, confidence, ...}}
                formatted_predictions = {}
                for horizon, pred_data in result.items():
                    if isinstance(pred_data, dict) and 'ensemble_prediction' in pred_data:
                        # ✅ FIX: Keep confidence and other important fields
                        formatted_predictions[horizon] = {
                            'price': float(pred_data['ensemble_prediction']),
                            'confidence': float(pred_data.get('confidence', 0.5)) if pred_data.get('confidence') is not None else 0.5
                        }
                    elif isinstance(pred_data, (int, float)):
                        formatted_predictions[horizon] = {'price': float(pred_data), 'confidence': 0.5}
                
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

    def _load_param_store(self):
        """Load param_store.json once (best-effort)."""
        try:
            if hasattr(self, '_param_store') and isinstance(self._param_store, dict):
                return self._param_store
            import json as _json
            import os as _os
            path = _os.path.join(_os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs'), 'param_store.json')
            if not _os.path.exists(path):
                self._param_store = {}
                return self._param_store
            with open(path, 'r') as rf:
                self._param_store = _json.load(rf) or {}
            return self._param_store
        except Exception:
            self._param_store = {}
            return self._param_store
    
    def _get_empirical_confidence_adjustment(self, horizon: str, model_confidence: float) -> float:
        """
        ⚡ ONLINE CONFIDENCE ADJUSTMENT: Gerçekleşmelere bakarak confidence'ı düzelt
        
        Modeli yeniden eğitmeden, son 30-60 gün içinde BENZER confidence seviyesinde
        yapılan tahminlerin gerçekleşme oranına (dir_hit rate) göre confidence'ı ayarlar.
        
        Args:
            horizon: Horizon string (e.g., '1d', '7d')
            model_confidence: Model'in tahmin ettiği confidence (0.0-1.0)
        
        Returns:
            float: Confidence adjustment (-0.15 to +0.15)
        
        Logic:
            - Son 30-60 gün içinde ±0.05 band içinde confidence'a sahip tahminler bulunur
            - Bu tahminlerin dir_hit rate'i hesaplanır
            - Eğer dir_hit rate < model_confidence → Düzeltme = negatif (confidence düşer)
            - Eğer dir_hit rate > model_confidence → Düzeltme = pozitif (confidence yükselir)
            - Adjustment = (dir_hit_rate - model_confidence) * alpha (alpha=0.5 yumuşak düzeltme)
        """
        try:
            # ✅ FIX: Check calibration_state.json bypass first (admin dashboard compatibility)
            # If calibration is bypassed, skip online adjustment
            try:
                import json as _json
                base_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                cpath = os.path.join(base_dir, 'calibration_state.json')
                if os.path.exists(cpath):
                    with open(cpath, 'r') as cf:
                        calib_state = _json.load(cf) or {}
                        bypass = calib_state.get('bypass', True)  # Default: bypass=True (safe)
                        if bypass:
                            # Calibration bypassed → skip online adjustment
                            return 0.0
                        
                        # ✅ NEW: Check if this specific horizon has sufficient data
                        # If horizon is in skipped_horizons (from param_store.json), skip adjustment
                        param_path = os.path.join(base_dir, 'param_store.json')
                        if os.path.exists(param_path):
                            try:
                                with open(param_path, 'r') as pf:
                                    param_store = _json.load(pf) or {}
                                    skipped_horizons = param_store.get('skipped_horizons', [])
                                    # Check if this horizon is in skipped list
                                    for skipped in skipped_horizons:
                                        if skipped.get('horizon') == horizon:
                                            # This horizon doesn't have enough data → skip adjustment
                                            logger.debug(f"⚠️ Online adjustment skipped for {horizon}: insufficient data ({skipped.get('reason', 'not_enough_samples')})")
                                            return 0.0
                            except Exception:
                                pass  # If param_store can't be read, continue (may not exist yet)
            except Exception:
                # If we can't read calibration_state, assume bypass=True (safe default)
                pass
            
            # Gate by environment variable (default: ON)
            enable_online_calib = ConfigManager.get('ENABLE_ONLINE_CONFIDENCE_ADJUSTMENT', '1').lower() in ('1', 'true', 'yes')
            if not enable_online_calib:
                return 0.0
            
            # Minimum samples required (default: 10)
            min_samples = int(ConfigManager.get('ONLINE_CALIB_MIN_SAMPLES', '10'))
            
            # Window days (default: 45)
            window_days = int(ConfigManager.get('ONLINE_CALIB_WINDOW_DAYS', '45'))
            
            # Confidence band width (default: ±0.05)
            conf_band = float(ConfigManager.get('ONLINE_CALIB_CONF_BAND', '0.05'))
            
            # Adjustment strength (default: 0.5 = yumuşak)
            alpha = float(ConfigManager.get('ONLINE_CALIB_ALPHA', '0.5'))
            
            # Check if we're in Flask context (for DB access)
            try:
                from flask import current_app
                if not current_app:
                    return 0.0
            except Exception:
                # Not in Flask context (e.g., during training) → skip
                return 0.0
            
            with current_app.app_context():
                from models import db, PredictionsLog, OutcomesLog
                from datetime import datetime, timedelta
                
                # Get cutoff date
                cutoff = datetime.utcnow() - timedelta(days=window_days)
                
                # Query: Son window_days içinde, aynı horizon'da, benzer confidence'a sahip tahminler
                conf_low = max(0.25, model_confidence - conf_band)
                conf_high = min(0.95, model_confidence + conf_band)
                
                q = (
                    db.session.query(PredictionsLog, OutcomesLog)
                    .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                    .filter(PredictionsLog.horizon == horizon)
                    .filter(PredictionsLog.ts_pred >= cutoff)
                    .filter(PredictionsLog.confidence >= conf_low)
                    .filter(PredictionsLog.confidence <= conf_high)
                    .filter(OutcomesLog.dir_hit.isnot(None))  # Only completed outcomes
                )
                
                results = q.all()
                
                if len(results) < min_samples:
                    # Not enough data → no adjustment
                    return 0.0
                
                # Calculate dir_hit rate
                hits = sum(1 for _, o in results if o.dir_hit is True)
                total = len(results)
                dir_hit_rate = hits / total if total > 0 else 0.5
                
                # Calculate adjustment: (dir_hit_rate - model_confidence) * alpha
                # This moves model_confidence towards dir_hit_rate
                adjustment = (dir_hit_rate - model_confidence) * alpha
                
                # Clamp adjustment to reasonable bounds
                adjustment = max(-0.15, min(0.15, adjustment))
                
                return adjustment
                
        except Exception as e:
            # Silently fail (don't break prediction if adjustment fails)
            logger.debug(f"⚠️ Online confidence adjustment failed: {e}")
            return 0.0
    
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
