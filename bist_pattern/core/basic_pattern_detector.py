"""
Basic Pattern Detector
Fast technical analysis patterns using TA-Lib and simple heuristics
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Try to import TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("⚠️ TA-Lib not available, using simple alternatives")


class BasicPatternDetector:
    """
    Fast basic pattern detector using TA-Lib and simple heuristics
    
    Focus on speed and reliability over complexity
    Patterns detected:
    - Moving Average crossovers
    - RSI divergences  
    - MACD signals
    - Support/Resistance breaks
    - Simple reversal patterns
    """
    
    def __init__(self):
        self.min_data_length = 20
        logger.info("📊 Basic Pattern Detector initialized")
    
    def detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect basic technical patterns
        
        Returns:
            List of pattern dictionaries with standardized format
        """
        if data is None or len(data) < self.min_data_length:
            return []
        
        patterns = []
        
        try:
            # Normalize column names
            df = self._normalize_dataframe(data)
            
            # Moving Average patterns
            patterns.extend(self._detect_ma_patterns(df))
            
            # RSI patterns
            patterns.extend(self._detect_rsi_patterns(df))
            
            # MACD patterns
            patterns.extend(self._detect_macd_patterns(df))
            
            # Support/Resistance patterns
            patterns.extend(self._detect_support_resistance(df))
            
            # Volume patterns
            patterns.extend(self._detect_volume_patterns(df))
            
            # Filter and sort by confidence
            patterns = [p for p in patterns if p.get('confidence', 0) > 0.3]
            patterns.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            return patterns[:8]  # Limit to top 8 patterns
            
        except Exception as e:
            logger.error(f"Basic pattern detection error: {e}")
            return []
    
    def _normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame column names"""
        df = data.copy()
        
        # Common column name mappings
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume',
            'Adj Close': 'close'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df[new_name] = df[old_name]
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                return pd.DataFrame()  # Return empty if missing required columns
        
        return df
    
    def _detect_ma_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect moving average crossover patterns"""
        patterns = []
        
        try:
            close = df['close'].values
            
            if TALIB_AVAILABLE:
                ma_fast = talib.SMA(close, timeperiod=10)
                ma_slow = talib.SMA(close, timeperiod=20)
            else:
                ma_fast = df['close'].rolling(10).mean().values
                ma_slow = df['close'].rolling(20).mean().values
            
            # Check for recent crossover (last 3 periods)
            for i in range(max(0, len(close) - 3), len(close)):
                if i < 20 or np.isnan(ma_fast[i]) or np.isnan(ma_slow[i]):
                    continue
                
                # Golden Cross (bullish)
                if (ma_fast[i] > ma_slow[i] and 
                        i > 0 and ma_fast[i-1] <= ma_slow[i-1]):
                    
                    confidence = min(0.8, abs(ma_fast[i] - ma_slow[i]) / ma_slow[i] * 10)
                    patterns.append({
                        'pattern': 'GOLDEN_CROSS',
                        'signal': 'BULLISH',
                        'confidence': confidence,
                        'strength': confidence * 100,
                        'source': 'BASIC_TA',
                        'description': '10-day MA crossed above 20-day MA',
                        'location': i
                    })
                
                # Death Cross (bearish)
                elif (ma_fast[i] < ma_slow[i] and 
                      i > 0 and ma_fast[i-1] >= ma_slow[i-1]):
                    
                    confidence = min(0.8, abs(ma_fast[i] - ma_slow[i]) / ma_slow[i] * 10)
                    patterns.append({
                        'pattern': 'DEATH_CROSS',
                        'signal': 'BEARISH',
                        'confidence': confidence,
                        'strength': confidence * 100,
                        'source': 'BASIC_TA',
                        'description': '10-day MA crossed below 20-day MA',
                        'location': i
                    })
        
        except Exception as e:
            logger.warning(f"MA pattern detection error: {e}")
        
        return patterns
    
    def _detect_rsi_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect RSI-based patterns"""
        patterns = []
        
        try:
            close = df['close'].values
            
            if TALIB_AVAILABLE:
                rsi = talib.RSI(close, timeperiod=14)
            else:
                rsi = self._calculate_rsi_simple(close, 14)
            
            if len(rsi) < 14:
                return patterns
            
            current_rsi = rsi[-1]
            
            # RSI Oversold bounce
            if current_rsi < 30 and len(rsi) > 1 and rsi[-2] < rsi[-1]:
                confidence = (30 - current_rsi) / 30 * 0.7 + 0.3
                patterns.append({
                    'pattern': 'RSI_OVERSOLD_BOUNCE',
                    'signal': 'BULLISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': f'RSI oversold bounce ({current_rsi:.1f})',
                    'rsi_value': current_rsi
                })
            
            # RSI Overbought reversal
            elif current_rsi > 70 and len(rsi) > 1 and rsi[-2] > rsi[-1]:
                confidence = (current_rsi - 70) / 30 * 0.7 + 0.3
                patterns.append({
                    'pattern': 'RSI_OVERBOUGHT_REVERSAL',
                    'signal': 'BEARISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': f'RSI overbought reversal ({current_rsi:.1f})',
                    'rsi_value': current_rsi
                })
        
        except Exception as e:
            logger.warning(f"RSI pattern detection error: {e}")
        
        return patterns
    
    def _calculate_rsi_simple(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Simple RSI calculation when TA-Lib not available"""
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with NaN to match input length
        result = np.full(len(prices), np.nan)
        result[period:] = rsi
        
        return result
    
    def _detect_macd_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect MACD signal patterns"""
        patterns = []
        
        try:
            close = df['close'].values
            
            if TALIB_AVAILABLE:
                macd, signal, histogram = talib.MACD(close)
            else:
                macd, signal, histogram = self._calculate_macd_simple(close)
            
            if len(histogram) < 2:
                return patterns
            
            # MACD bullish crossover
            if (histogram[-1] > 0 and len(histogram) > 1 and histogram[-2] <= 0):
                confidence = min(0.75, abs(histogram[-1]) / (abs(macd[-1]) + 1e-10))
                patterns.append({
                    'pattern': 'MACD_BULLISH_CROSSOVER',
                    'signal': 'BULLISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': 'MACD crossed above signal line',
                    'macd_value': macd[-1] if len(macd) > 0 else 0
                })
            
            # MACD bearish crossover
            elif (histogram[-1] < 0 and len(histogram) > 1 and histogram[-2] >= 0):
                confidence = min(0.75, abs(histogram[-1]) / (abs(macd[-1]) + 1e-10))
                patterns.append({
                    'pattern': 'MACD_BEARISH_CROSSOVER',
                    'signal': 'BEARISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': 'MACD crossed below signal line',
                    'macd_value': macd[-1] if len(macd) > 0 else 0
                })
        
        except Exception as e:
            logger.warning(f"MACD pattern detection error: {e}")
        
        return patterns
    
    def _calculate_macd_simple(self, prices: np.ndarray):
        """Simple MACD calculation when TA-Lib not available"""
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        signal = self._calculate_ema(macd, 9)
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Simple EMA calculation"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _detect_support_resistance(self, df: pd.DataFrame) -> List[Dict]:
        """Detect support and resistance level breaks"""
        patterns = []
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            if len(close) < 20:
                return patterns
            
            # Simple support/resistance detection
            recent_high = np.max(high[-10:])
            recent_low = np.min(low[-10:])
            current_price = close[-1]
            
            # Resistance break
            if current_price > recent_high * 1.02:  # 2% above recent high
                confidence = min(0.8, (current_price - recent_high) / recent_high * 5)
                patterns.append({
                    'pattern': 'RESISTANCE_BREAK',
                    'signal': 'BULLISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': f'Price broke above resistance at {recent_high:.2f}',
                    'resistance_level': recent_high
                })
            
            # Support break
            elif current_price < recent_low * 0.98:  # 2% below recent low
                confidence = min(0.8, (recent_low - current_price) / recent_low * 5)
                patterns.append({
                    'pattern': 'SUPPORT_BREAK',
                    'signal': 'BEARISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': f'Price broke below support at {recent_low:.2f}',
                    'support_level': recent_low
                })
        
        except Exception as e:
            logger.warning(f"Support/Resistance detection error: {e}")
        
        return patterns
    
    def _detect_volume_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume-based patterns"""
        patterns = []
        
        try:
            if 'volume' not in df.columns:
                return patterns
            
            volume = df['volume'].values
            close = df['close'].values
            
            if len(volume) < 10:
                return patterns
            
            # Volume spike with price increase
            avg_volume = np.mean(volume[-10:-1])  # Exclude current day
            current_volume = volume[-1]
            price_change = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            
            if current_volume > avg_volume * 2 and price_change > 0.03:  # 2x volume + 3% price increase
                confidence = min(0.7, (current_volume / avg_volume - 1) * 0.1 + price_change * 5)
                patterns.append({
                    'pattern': 'VOLUME_SPIKE_BULLISH',
                    'signal': 'BULLISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': f'Volume spike with price increase ({current_volume/avg_volume:.1f}x volume)',
                    'volume_ratio': current_volume / avg_volume
                })
            
            elif current_volume > avg_volume * 2 and price_change < -0.03:  # 2x volume + 3% price decrease
                confidence = min(0.7, (current_volume / avg_volume - 1) * 0.1 + abs(price_change) * 5)
                patterns.append({
                    'pattern': 'VOLUME_SPIKE_BEARISH',
                    'signal': 'BEARISH',
                    'confidence': confidence,
                    'strength': confidence * 100,
                    'source': 'BASIC_TA',
                    'description': f'Volume spike with price decrease ({current_volume/avg_volume:.1f}x volume)',
                    'volume_ratio': current_volume / avg_volume
                })
        
        except Exception as e:
            logger.warning(f"Volume pattern detection error: {e}")
        
        return patterns
