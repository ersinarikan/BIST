#!/usr/bin/env python3
"""
BIST Pattern Detection - Technical Pattern Analyzer
Teknik formasyonları tespit eden AI modülü
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import talib
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalPatternAnalyzer:
    """Teknik formasyonları tespit eden sınıf"""
    
    def __init__(self):
        self.patterns = {
            'triangle': 'Üçgen Formasyonu',
            'head_shoulders': 'Baş-Omuz Formasyonu', 
            'double_top': 'Çift Tepe',
            'double_bottom': 'Çift Dip',
            'support_resistance': 'Destek-Direnç',
            'trend_lines': 'Trend Çizgileri',
            'flag_pennant': 'Bayrak-Flama',
            'cup_handle': 'Fincan-Kulp'
        }
        
    def analyze_stock_patterns(self, symbol: str, price_data: pd.DataFrame) -> Dict:
        """Hisse için tüm formasyonları analiz et"""
        
        if len(price_data) < 50:
            return {"error": "Yeterli veri yok", "required": 50, "available": len(price_data)}
        
        # Prepare data
        high = price_data['high'].values
        low = price_data['low'].values
        close = price_data['close'].values
        volume = price_data['volume'].values
        dates = price_data.index
        
        results = {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            'data_points': len(price_data),
            'patterns_found': [],
            'technical_indicators': {},
            'signals': [],
            'confidence_score': 0.0
        }
        
        try:
            # 1. Trend Analysis
            trend_analysis = self._analyze_trend(close, dates)
            results['trend_analysis'] = trend_analysis
            
            # 2. Support & Resistance Levels
            support_resistance = self._find_support_resistance(high, low, close)
            results['support_resistance'] = support_resistance
            
            # 3. Pattern Detection
            patterns = self._detect_patterns(high, low, close, volume, dates)
            results['patterns_found'] = patterns
            
            # 4. Technical Indicators
            indicators = self._calculate_indicators(high, low, close, volume)
            results['technical_indicators'] = indicators
            
            # 5. Generate Signals
            signals = self._generate_signals(close, patterns, indicators)
            results['signals'] = signals
            
            # 6. Calculate Confidence Score
            results['confidence_score'] = self._calculate_confidence(patterns, indicators, signals)
            
            logger.info(f"✅ {symbol}: {len(patterns)} formasyon, {len(signals)} sinyal tespit edildi")
            
        except Exception as e:
            logger.error(f"❌ {symbol} analiz hatası: {e}")
            results['error'] = str(e)
            
        return results
    
    def _analyze_trend(self, close: np.array, dates) -> Dict:
        """Trend analizi"""
        
        # Simple trend calculation
        short_ma = np.mean(close[-20:])  # 20-day MA
        long_ma = np.mean(close[-50:])   # 50-day MA
        
        # Trend direction
        trend_direction = "Yükseliş" if short_ma > long_ma else "Düşüş"
        
        # Trend strength (slope)
        if len(close) >= 20:
            x = np.arange(20)
            slope = np.polyfit(x, close[-20:], 1)[0]
            trend_strength = abs(slope) / np.mean(close[-20:]) * 100
        else:
            trend_strength = 0
            
        return {
            'direction': trend_direction,
            'strength': round(trend_strength, 2),
            'short_ma': round(short_ma, 2),
            'long_ma': round(long_ma, 2),
            'current_price': round(close[-1], 2)
        }
    
    def _find_support_resistance(self, high: np.array, low: np.array, close: np.array) -> Dict:
        """Destek ve direnç seviyelerini bul"""
        
        # Local maxima and minima
        resistance_levels = []
        support_levels = []
        
        window = 5
        for i in range(window, len(high) - window):
            # Resistance (local maxima)
            if all(high[i] >= high[i-j] for j in range(1, window+1)) and \
               all(high[i] >= high[i+j] for j in range(1, window+1)):
                resistance_levels.append(high[i])
                
            # Support (local minima)  
            if all(low[i] <= low[i-j] for j in range(1, window+1)) and \
               all(low[i] <= low[i+j] for j in range(1, window+1)):
                support_levels.append(low[i])
        
        # Get significant levels
        current_price = close[-1]
        
        # Filter and sort
        resistance_levels = sorted([r for r in resistance_levels if r > current_price])[-3:]
        support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]
        
        return {
            'resistance_levels': [round(r, 2) for r in resistance_levels],
            'support_levels': [round(s, 2) for s in support_levels],
            'current_price': round(current_price, 2)
        }
    
    def _detect_patterns(self, high: np.array, low: np.array, close: np.array, 
                        volume: np.array, dates) -> List[Dict]:
        """Formasyonları tespit et"""
        
        patterns = []
        
        # 1. Double Top Pattern
        double_top = self._detect_double_top(high, close)
        if double_top:
            patterns.append({
                'type': 'double_top',
                'name': 'Çift Tepe',
                'signal': 'SELL',
                'confidence': double_top['confidence'],
                'description': 'Düşüş sinyali - Çift tepe formasyonu tespit edildi'
            })
        
        # 2. Double Bottom Pattern
        double_bottom = self._detect_double_bottom(low, close)
        if double_bottom:
            patterns.append({
                'type': 'double_bottom', 
                'name': 'Çift Dip',
                'signal': 'BUY',
                'confidence': double_bottom['confidence'],
                'description': 'Yükseliş sinyali - Çift dip formasyonu tespit edildi'
            })
        
        # 3. Triangle Pattern
        triangle = self._detect_triangle(high, low, close)
        if triangle:
            patterns.append({
                'type': 'triangle',
                'name': 'Üçgen Formasyonu',
                'signal': triangle['signal'],
                'confidence': triangle['confidence'],
                'description': f'{triangle["triangle_type"]} üçgen formasyonu'
            })
        
        # 4. Head and Shoulders
        head_shoulders = self._detect_head_shoulders(high, low, close)
        if head_shoulders:
            patterns.append({
                'type': 'head_shoulders',
                'name': 'Baş-Omuz',
                'signal': 'SELL',
                'confidence': head_shoulders['confidence'],
                'description': 'Güçlü düşüş sinyali - Baş-omuz formasyonu'
            })
            
        return patterns
    
    def _detect_double_top(self, high: np.array, close: np.array) -> Optional[Dict]:
        """Çift tepe formasyonu tespit et"""
        
        if len(high) < 20:
            return None
            
        # Son 20-50 bardaki en yüksek noktaları bul
        recent_highs = high[-50:]
        peaks = []
        
        window = 3
        for i in range(window, len(recent_highs) - window):
            if all(recent_highs[i] >= recent_highs[i-j] for j in range(1, window+1)) and \
               all(recent_highs[i] >= recent_highs[i+j] for j in range(1, window+1)):
                peaks.append(recent_highs[i])
        
        if len(peaks) >= 2:
            # En yüksek iki tepeyi al
            peaks = sorted(peaks, reverse=True)[:2]
            
            # Tepeler birbirine yakın mı? (%5 tolerance)
            if abs(peaks[0] - peaks[1]) / peaks[0] < 0.05:
                confidence = 0.7 + (0.3 * (1 - abs(peaks[0] - peaks[1]) / peaks[0] / 0.05))
                return {
                    'confidence': min(confidence, 0.95),
                    'peaks': peaks
                }
        
        return None
    
    def _detect_double_bottom(self, low: np.array, close: np.array) -> Optional[Dict]:
        """Çift dip formasyonu tespit et"""
        
        if len(low) < 20:
            return None
            
        # Son 20-50 bardaki en düşük noktaları bul
        recent_lows = low[-50:]
        troughs = []
        
        window = 3
        for i in range(window, len(recent_lows) - window):
            if all(recent_lows[i] <= recent_lows[i-j] for j in range(1, window+1)) and \
               all(recent_lows[i] <= recent_lows[i+j] for j in range(1, window+1)):
                troughs.append(recent_lows[i])
        
        if len(troughs) >= 2:
            # En düşük iki dibi al
            troughs = sorted(troughs)[:2]
            
            # Dipler birbirine yakın mı? (%5 tolerance)
            if abs(troughs[0] - troughs[1]) / troughs[0] < 0.05:
                confidence = 0.7 + (0.3 * (1 - abs(troughs[0] - troughs[1]) / troughs[0] / 0.05))
                return {
                    'confidence': min(confidence, 0.95),
                    'troughs': troughs
                }
        
        return None
    
    def _detect_triangle(self, high: np.array, low: np.array, close: np.array) -> Optional[Dict]:
        """Üçgen formasyonu tespit et"""
        
        if len(close) < 30:
            return None
        
        # Son 30 barı analiz et
        recent_high = high[-30:]
        recent_low = low[-30:]
        recent_close = close[-30:]
        
        # Trend çizgileri
        x = np.arange(30)
        
        try:
            # Üst trend çizgisi (yüksekler)
            high_slope = np.polyfit(x, recent_high, 1)[0]
            
            # Alt trend çizgisi (düşükler)  
            low_slope = np.polyfit(x, recent_low, 1)[0]
            
            # Üçgen türünü belirle
            if high_slope < 0 and low_slope > 0:
                triangle_type = "Simetrik Üçgen"
                signal = "WAIT"  # Kırılım bekle
                confidence = 0.6
            elif high_slope < 0 and abs(low_slope) < 0.1:
                triangle_type = "Azalan Üçgen"
                signal = "SELL"
                confidence = 0.7
            elif abs(high_slope) < 0.1 and low_slope > 0:
                triangle_type = "Yükselen Üçgen"
                signal = "BUY"
                confidence = 0.7
            else:
                return None
            
            return {
                'triangle_type': triangle_type,
                'signal': signal,
                'confidence': confidence,
                'high_slope': high_slope,
                'low_slope': low_slope
            }
            
        except:
            return None
    
    def _detect_head_shoulders(self, high: np.array, low: np.array, close: np.array) -> Optional[Dict]:
        """Baş-omuz formasyonu tespit et"""
        
        if len(high) < 30:
            return None
        
        # Son 30-50 barı analiz et
        recent_highs = high[-50:]
        peaks = []
        
        window = 2
        for i in range(window, len(recent_highs) - window):
            if all(recent_highs[i] >= recent_highs[i-j] for j in range(1, window+1)) and \
               all(recent_highs[i] >= recent_highs[i+j] for j in range(1, window+1)):
                peaks.append((i, recent_highs[i]))
        
        if len(peaks) >= 3:
            # Son 3 tepeyi al
            peaks = peaks[-3:]
            
            # Baş-omuz paterni: orta tepe en yüksek olmalı
            left_shoulder = peaks[0][1]
            head = peaks[1][1] 
            right_shoulder = peaks[2][1]
            
            # Baş, omuzlardan yüksek mi?
            if head > left_shoulder and head > right_shoulder:
                # Omuzlar benzer seviyede mi? (%10 tolerance)
                shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                
                if shoulder_diff < 0.10:
                    confidence = 0.8 - (shoulder_diff * 2)  # Omuzlar ne kadar eşitse o kadar güvenilir
                    
                    return {
                        'confidence': confidence,
                        'left_shoulder': left_shoulder,
                        'head': head,
                        'right_shoulder': right_shoulder
                    }
        
        return None
    
    def _calculate_indicators(self, high: np.array, low: np.array, close: np.array, volume: np.array) -> Dict:
        """Teknik indikatörleri hesapla"""
        
        indicators = {}
        
        try:
            # RSI
            if len(close) >= 14:
                rsi = talib.RSI(close, timeperiod=14)
                indicators['rsi'] = {
                    'current': round(rsi[-1], 2),
                    'signal': 'OVERSOLD' if rsi[-1] < 30 else 'OVERBOUGHT' if rsi[-1] > 70 else 'NEUTRAL'
                }
            
            # MACD
            if len(close) >= 26:
                macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['macd'] = {
                    'macd': round(macd[-1], 4),
                    'signal': round(macdsignal[-1], 4),
                    'histogram': round(macdhist[-1], 4),
                    'trend': 'BULLISH' if macd[-1] > macdsignal[-1] else 'BEARISH'
                }
            
            # Bollinger Bands
            if len(close) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                
                indicators['bollinger_bands'] = {
                    'upper': round(bb_upper[-1], 2),
                    'middle': round(bb_middle[-1], 2),
                    'lower': round(bb_lower[-1], 2),
                    'position': round(bb_position, 2),
                    'signal': 'OVERBOUGHT' if bb_position > 0.8 else 'OVERSOLD' if bb_position < 0.2 else 'NEUTRAL'
                }
            
            # Moving Averages
            if len(close) >= 50:
                ma20 = talib.SMA(close, timeperiod=20)
                ma50 = talib.SMA(close, timeperiod=50)
                
                indicators['moving_averages'] = {
                    'ma20': round(ma20[-1], 2),
                    'ma50': round(ma50[-1], 2),
                    'trend': 'BULLISH' if ma20[-1] > ma50[-1] else 'BEARISH',
                    'price_vs_ma20': round((close[-1] - ma20[-1]) / ma20[-1] * 100, 2)
                }
                
        except Exception as e:
            logger.error(f"İndikatör hesaplama hatası: {e}")
            
        return indicators
    
    def _generate_signals(self, close: np.array, patterns: List[Dict], indicators: Dict) -> List[Dict]:
        """Al/sat sinyallerini üret"""
        
        signals = []
        current_price = close[-1]
        
        # Pattern-based signals
        for pattern in patterns:
            if pattern['signal'] in ['BUY', 'SELL']:
                signals.append({
                    'type': 'PATTERN',
                    'signal': pattern['signal'],
                    'reason': pattern['description'],
                    'confidence': pattern['confidence'],
                    'source': pattern['name']
                })
        
        # Indicator-based signals
        if 'rsi' in indicators:
            rsi_value = indicators['rsi']['current']
            if rsi_value < 30:
                signals.append({
                    'type': 'TECHNICAL',
                    'signal': 'BUY',
                    'reason': f'RSI aşırı satım bölgesinde ({rsi_value})',
                    'confidence': 0.6,
                    'source': 'RSI'
                })
            elif rsi_value > 70:
                signals.append({
                    'type': 'TECHNICAL',
                    'signal': 'SELL', 
                    'reason': f'RSI aşırı alım bölgesinde ({rsi_value})',
                    'confidence': 0.6,
                    'source': 'RSI'
                })
        
        if 'macd' in indicators:
            macd_trend = indicators['macd']['trend']
            if macd_trend == 'BULLISH':
                signals.append({
                    'type': 'TECHNICAL',
                    'signal': 'BUY',
                    'reason': 'MACD yükseliş sinyali veriyor',
                    'confidence': 0.5,
                    'source': 'MACD'
                })
        
        return signals
    
    def _calculate_confidence(self, patterns: List[Dict], indicators: Dict, signals: List[Dict]) -> float:
        """Genel güven skoru hesapla"""
        
        if not signals:
            return 0.0
        
        # Sinyal yoğunluğuna göre güven
        buy_signals = len([s for s in signals if s['signal'] == 'BUY'])
        sell_signals = len([s for s in signals if s['signal'] == 'SELL'])
        
        # Çelişkili sinyaller güveni düşürür
        if buy_signals > 0 and sell_signals > 0:
            confidence = 0.3
        else:
            confidence = min(0.8, (buy_signals + sell_signals) * 0.2)
        
        # Pattern güveni ekle
        if patterns:
            pattern_confidence = np.mean([p['confidence'] for p in patterns])
            confidence = (confidence + pattern_confidence) / 2
        
        return round(confidence, 2)

# Test function
def test_pattern_analyzer():
    """Pattern analyzer test fonksiyonu"""
    
    # Dummy data oluştur
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Trend + noise
    trend = np.linspace(100, 120, 100)
    noise = np.random.normal(0, 2, 100)
    close_prices = trend + noise
    
    # OHLCV data
    data = pd.DataFrame({
        'open': close_prices * 0.98,
        'high': close_prices * 1.02,
        'low': close_prices * 0.97,
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    # Analiz et
    analyzer = TechnicalPatternAnalyzer()
    result = analyzer.analyze_stock_patterns('TEST', data)
    
    print("📊 Pattern Analysis Test Results:")
    print(f"Patterns Found: {len(result['patterns_found'])}")
    print(f"Signals Generated: {len(result['signals'])}")
    print(f"Confidence Score: {result['confidence_score']}")
    
    return result

if __name__ == "__main__":
    test_pattern_analyzer()
