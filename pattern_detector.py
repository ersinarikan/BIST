"""
Hibrit Pattern Detection Sistemi
TA-Lib + YOLOv8 + FinGPT kombinasyonu ile kesin formasyon tespiti
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
from models import db, Stock, StockPrice
from app import app
from config import config
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Gelişmiş pattern detection sistemi
try:
    from advanced_patterns import AdvancedPatternDetector
    ADVANCED_PATTERNS_AVAILABLE = True
except ImportError:
    ADVANCED_PATTERNS_AVAILABLE = False
    logger.warning("⚠️ Advanced patterns modülü yüklenemedi")

# Visual pattern detection sistemi
try:
    from visual_pattern_detector import get_visual_pattern_system
    VISUAL_PATTERNS_AVAILABLE = True
except ImportError:
    VISUAL_PATTERNS_AVAILABLE = False
    logger.warning("⚠️ Visual patterns modülü yüklenemedi")

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
    ENHANCED_ML_AVAILABLE = False
    logger.warning("⚠️ Enhanced ML Prediction modülü yüklenemedi")

class HybridPatternDetector:
    def __init__(self):
        # Cache sistemi (5 dakika TTL)
        self.cache = {}
        try:
            self.cache_ttl = int(getattr(config['default'], 'PATTERN_CACHE_TTL', 300))
        except Exception:
            self.cache_ttl = 300
        
        # Gelişmiş pattern detector
        if ADVANCED_PATTERNS_AVAILABLE:
            self.advanced_detector = AdvancedPatternDetector()
        else:
            self.advanced_detector = None
            
        # Visual pattern detector
        if VISUAL_PATTERNS_AVAILABLE:
            self.visual_detector = get_visual_pattern_system()
        else:
            self.visual_detector = None
            
        # ML Prediction system
        if ML_PREDICTION_AVAILABLE:
            self.ml_predictor = get_ml_prediction_system()
        else:
            self.ml_predictor = None
        
        # Enhanced ML Prediction system (optional)
        if ENHANCED_ML_AVAILABLE:
            try:
                self.enhanced_ml = get_enhanced_ml_system()
            except Exception:
                self.enhanced_ml = None
        else:
            self.enhanced_ml = None
            
        logger.info("🤖 Hybrid Pattern Detector başlatıldı")
    
    def _cleanup_cache(self):
        """Eski cache entry'lerini temizle"""
        try:
            current_time = datetime.now().timestamp()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if isinstance(entry, dict) and 'timestamp' in entry:
                    if current_time - entry['timestamp'] > self.cache_ttl:
                        expired_keys.append(key)
                elif isinstance(entry, dict) and 'timestamp' not in entry:
                    # Eski format cache'leri de temizle
                    expired_keys.append(key)
            
            # Expired keys'leri sil
            for key in expired_keys:
                del self.cache[key]
            
            logger.info(f"Cache cleanup: {len(expired_keys)} expired entries removed")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
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
                    return None
                
                # Son N günlük veriyi al
                prices = StockPrice.query.filter_by(stock_id=stock.id)\
                            .order_by(StockPrice.date.desc())\
                            .limit(days).all()
            
            if not prices:
                logger.warning(f"Fiyat verisi bulunamadı: {symbol}")
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
            
            return df
            
        except Exception as e:
            logger.error(f"Veri alma hatası {symbol}: {e}")
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
            
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            # Trend detection
            sma_5 = data['close'].rolling(5).mean()
            sma_20 = data['close'].rolling(20).mean()
            
            current_trend = "BULLISH" if sma_5.iloc[-1] > sma_20.iloc[-1] else "BEARISH"
            
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
                patterns.append({
                    'pattern': pattern_type,
                    'signal': 'BULLISH' if price_change > 0 else 'BEARISH',
                    'confidence': min(0.8, 0.5 + abs(price_change) / 10),
                    'strength': min(100, 50 + abs(price_change) * 10),
                    'price_change': price_change,
                    'volume_ratio': volume_ratio
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
                    flask_app.broadcast_log('INFO', f'🧠 AI analiz başlıyor: {symbol}', 'ai_analysis')
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
            
            # Pattern detection
            patterns = []
            
            # Basic patterns
            basic_patterns = self.detect_basic_patterns(data)
            patterns.extend(basic_patterns)
            
            # Advanced patterns (if available)
            if self.advanced_detector and ADVANCED_PATTERNS_AVAILABLE:
                try:
                    advanced_patterns = self.advanced_detector.analyze_all_patterns(data)
                    patterns.extend(advanced_patterns)
                except Exception as e:
                    logger.error(f"Advanced pattern analysis hatası: {e}")
            
            # Visual pattern analysis (if available)
            if self.visual_detector and VISUAL_PATTERNS_AVAILABLE:
                try:
                    visual_result = self.visual_detector.analyze_stock_visual(symbol, data)
                    if visual_result['status'] == 'success':
                        visual_analysis = visual_result.get('visual_analysis', {})
                        detected_patterns = visual_analysis.get('patterns', [])
                        
                        for visual_pattern in detected_patterns:
                            pattern_info = {
                                'pattern': visual_pattern['pattern'],
                                'confidence': visual_pattern['confidence'],
                                'signal': self.get_visual_signal(visual_pattern['pattern']),
                                'strength': visual_pattern['confidence'] * 100,
                                'source': 'VISUAL_YOLO',
                                'bbox': visual_pattern.get('bbox'),
                                'area': visual_pattern.get('area')
                            }
                            patterns.append(pattern_info)
                        
                        logger.info(f"Visual patterns: {len(detected_patterns)} adet")
                except Exception as e:
                    logger.error(f"Visual pattern analysis hatası: {e}")
            
            # ML predictions (optional) and integrate into patterns as signals
            ml_predictions = None
            enhanced_predictions = None
            try:
                if self.ml_predictor and ML_PREDICTION_AVAILABLE:
                    ml_predictions = self.ml_predictor.predict_prices(symbol, data, None) or {}
                    current_px = float(data['close'].iloc[-1])
                    horizon_weights = {
                        '1d': 0.70,
                        '3d': 0.60,
                        '7d': 0.55,
                        '14d': 0.50,
                        '30d': 0.45,
                    }
                    for h_key, base_w in horizon_weights.items():
                        pred_obj = ml_predictions.get(h_key) or {}
                        pred_px = pred_obj.get('price') or pred_obj.get('prediction') or pred_obj.get('target')
                        if isinstance(pred_px, (int, float)) and current_px > 0:
                            delta_pct = (float(pred_px) - current_px) / current_px
                            if abs(delta_pct) >= 0.003:  # 0.3% altı gürültü say
                                conf_scale = min(1.0, abs(delta_pct) / 0.05)  # ~%5 hareketle tavan
                                confidence = max(0.3, min(0.9, base_w * conf_scale))
                                patterns.append({
                                    'pattern': f'ML_{h_key.upper()}',
                                    'signal': 'BULLISH' if delta_pct > 0 else 'BEARISH',
                                    'confidence': confidence,
                                    'strength': int(confidence * 100),
                                    'source': 'ML_PREDICTOR',
                                    'delta_pct': float(delta_pct)
                                })
            except Exception as e:
                logger.error(f"ML prediction integration hatası {symbol}: {e}")

            # Enhanced ML predictions (optional) and integrate as stronger signals
            try:
                if hasattr(self, 'enhanced_ml') and self.enhanced_ml and ENHANCED_ML_AVAILABLE:
                    enh = self.enhanced_ml.predict_enhanced(symbol, data)
                    if isinstance(enh, dict) and enh:
                        enhanced_predictions = enh
                        current_px = float(data['close'].iloc[-1])
                        # enhanced_ml.predict_enhanced returns a dict keyed by horizons directly
                        pred_map = enh if isinstance(enh, dict) else {}
                        for h_key, pred_obj in pred_map.items():
                            pred_px = pred_obj.get('ensemble_prediction')
                            conf_val = float(pred_obj.get('confidence', 0) or 0)
                            if isinstance(pred_px, (int, float)) and current_px > 0:
                                delta_pct = (float(pred_px) - current_px) / current_px
                                if abs(delta_pct) >= 0.003:
                                    confidence = max(0.35, min(0.95, conf_val))
                                    patterns.append({
                                        'pattern': f'ENH_{h_key.upper()}',
                                        'signal': 'BULLISH' if delta_pct > 0 else 'BEARISH',
                                        'confidence': confidence,
                                        'strength': int(confidence * 100),
                                        'source': 'ENHANCED_ML',
                                        'delta_pct': float(delta_pct)
                                    })
            except Exception as e:
                logger.error(f"Enhanced ML prediction integration hatası {symbol}: {e}")

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
                'enhanced_predictions': enhanced_predictions or {}
            }
            try:
                # Progress broadcast: analysis end (best-effort)
                from app import app as flask_app
                if hasattr(flask_app, 'broadcast_log'):
                    sig = result['overall_signal'].get('signal','?')
                    conf = int(round(float(result['overall_signal'].get('confidence',0))*100))
                    flask_app.broadcast_log('SUCCESS', f'🎯 {symbol} AI: {sig} (%{conf})', 'ai_analysis')
            except Exception:
                pass
            
            # Cache'e TTL ile kaydet
            self.cache[cache_key] = {
                'data': result,
                'timestamp': current_time
            }
            
            # Cache cleanup - 100'den fazla entry varsa eski olanları temizle
            if len(self.cache) > 100:
                self._cleanup_cache()
            
            # Opsiyonel: overall sinyali canlı loga yaz (dashboard için)
            try:
                from app import app as flask_app
                overall = (result or {}).get('overall_signal') or {}
                if overall and hasattr(flask_app, 'broadcast_log'):
                    flask_app.broadcast_log('INFO', f"{symbol}: {overall.get('signal','?')} ({overall.get('confidence',0):.2f})", 'ai_analysis')
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
