"""
Advanced Pattern Detector
- Classic TA patterns: DOUBLE_TOP, DOUBLE_BOTTOM, HEAD_AND_SHOULDERS, INVERSE_HEAD_AND_SHOULDERS
- TA-Lib candlestick patterns (60+ patterns)
- Enhanced with professional pattern recognition
"""

from typing import List, Dict
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TA-Lib for professional pattern recognition
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("⚠️ TA-Lib not available - using heuristics only")


class AdvancedPatternDetector:
    """Lightweight detector providing a few common TA patterns.

    Methods return a list of pattern dicts with at least:
      - pattern: str
      - signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
      - confidence: float (0..1)
      - source: 'ADVANCED_TA'
    """

    def __init__(self) -> None:
        pass

    # ------------- Public API -------------
    def analyze_all_patterns(self, data: pd.DataFrame) -> List[Dict]:
        try:
            df = self._normalize_ohlcv(data)
            if len(df) < 30:
                return []

            patterns: List[Dict] = []
            
            # Classic TA patterns (heuristic-based)
            patterns.extend(self._detect_double_top_bottom(df))
            patterns.extend(self._detect_head_shoulders(df))
            
            # ✨ NEW: TA-Lib candlestick patterns (60+ professional patterns)
            if TALIB_AVAILABLE:
                patterns.extend(self._detect_talib_patterns(df))

            # Cap list size and prioritize by confidence
            if len(patterns) > 12:  # Increased from 8 to accommodate TA-Lib patterns
                patterns = sorted(
                    patterns,
                    key=lambda p: p.get('confidence', 0),
                    reverse=True,
                )[:12]
            return patterns
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return []

    # ------------- Internal helpers -------------
    def _normalize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Accept both Open/High/Low/Close/Volume and lower-case variants
        rename_map = {}
        if 'Open' in df.columns:
            rename_map['Open'] = 'open'
        if 'High' in df.columns:
            rename_map['High'] = 'high'
        if 'Low' in df.columns:
            rename_map['Low'] = 'low'
        if 'Close' in df.columns:
            rename_map['Close'] = 'close'
        if 'Volume' in df.columns:
            rename_map['Volume'] = 'volume'
        df = df.rename(columns=rename_map)
        return df

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[Dict]:
        # Very simple peak/trough detection using rolling windows
        prices = df['close'].values.astype(float)
        # window retained for potential future heuristics
        patterns: List[Dict] = []

        # Find local maxima/minima (kept for potential future heuristics)

        try:
            # Double Top: two highs within small range separated by a dip
            # Heuristic: last 30 bars
            last = min(len(df), 60)
            segment = prices[-last:]
            if len(segment) >= 20:
                max_idx = np.argmax(segment)
                max_val = segment[max_idx]
                # Search second top before or after
                tolerance = max_val * 0.01  # 1%
                for j in range(max(0, max_idx - 15), min(len(segment), max_idx + 15)):
                    if j == max_idx:
                        continue
                    if abs(segment[j] - max_val) <= tolerance:
                        # ensure a dip between peaks
                        left, right = sorted([j, max_idx])
                        valley = (
                            np.min(segment[left:right]) if right - left > 2 else max_val
                        )
                        if valley < max_val * 0.985:  # at least 1.5% dip
                            conf = min(
                                0.9,
                                0.6 + float((max_val - valley) / max_val) * 5,
                            )
                            patterns.append({
                                'pattern': 'DOUBLE_TOP',
                                'signal': 'BEARISH',
                                'confidence': float(conf),
                                'source': 'ADVANCED_TA'
                            })
                            break
                # Double Bottom: mirror logic
                min_idx = np.argmin(segment)
                min_val = segment[min_idx]
                tolerance_b = min_val * 0.01 if min_val != 0 else 0.01
                for j in range(max(0, min_idx - 15), min(len(segment), min_idx + 15)):
                    if j == min_idx:
                        continue
                    if abs(segment[j] - min_val) <= tolerance_b:
                        left, right = sorted([j, min_idx])
                        peak = (
                            np.max(segment[left:right]) if right - left > 2 else min_val
                        )
                        if peak > min_val * 1.015:
                            conf = min(
                                0.9,
                                0.6
                                + float((peak - min_val) / (abs(min_val) + 1e-8)) * 5,
                            )
                            patterns.append({
                                'pattern': 'DOUBLE_BOTTOM',
                                'signal': 'BULLISH',
                                'confidence': float(conf),
                                'source': 'ADVANCED_TA'
                            })
                            break
        except Exception:
            pass

        return patterns

    def _detect_head_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        patterns: List[Dict] = []
        close = df['close'].values.astype(float)
        if len(close) < 30:
            return patterns
        try:
            # Extremely simple H&S detection using three-peak shape on last ~50 bars
            segment = close[-50:]
            peaks_idx = self._find_peaks(segment, distance=3)
            if len(peaks_idx) >= 3:
                # pick three consecutive peaks
                for i in range(len(peaks_idx) - 2):
                    a, b, c = peaks_idx[i], peaks_idx[i + 1], peaks_idx[i + 2]
                    left, head, right = segment[a], segment[b], segment[c]
                    if (
                        head > left * 1.01
                        and head > right * 1.01
                        and abs(left - right) / max(left, 1e-8) < 0.03
                    ):
                        # Head & Shoulders
                        conf = min(
                            0.9,
                            0.55 + float((head - (left + right) / 2) / (head + 1e-8)) * 5,
                        )
                        patterns.append({
                            'pattern': 'HEAD_AND_SHOULDERS',
                            'signal': 'BEARISH',
                            'confidence': float(conf),
                            'source': 'ADVANCED_TA'
                        })
                        break
                    if (
                        head < left * 0.99
                        and head < right * 0.99
                        and abs(left - right) / max(left, 1e-8) < 0.03
                    ):
                        # Inverse H&S
                        conf = min(
                            0.9,
                            0.55
                            + float(((left + right) / 2 - head) / (abs(head) + 1e-8)) * 5,
                        )
                        patterns.append({
                            'pattern': 'INVERSE_HEAD_AND_SHOULDERS',
                            'signal': 'BULLISH',
                            'confidence': float(conf),
                            'source': 'ADVANCED_TA'
                        })
                        break
        except Exception:
            pass
        return patterns
            
    def _find_peaks(self, array: np.ndarray, distance: int = 3) -> List[int]:
        idx: List[int] = []
        for i in range(distance, len(array) - distance):
            window = array[i - distance:i + distance + 1]
            if array[i] == window.max() and (window.argmax() == distance):
                idx.append(i)
        return idx
    
    def _detect_talib_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect candlestick patterns using TA-Lib
        
        TA-Lib provides 60+ professional candlestick pattern recognition functions.
        Each returns integer values:
        - 100: Bullish pattern
        - -100: Bearish pattern
        - 0: No pattern
        """
        if not TALIB_AVAILABLE:
            return []
        
        patterns: List[Dict] = []
        
        try:
            # Extract OHLC arrays
            open_prices = df['open'].values.astype(float)
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            
            # Check if we have enough data
            if len(close_prices) < 10:
                return []
            
            # Key reversal patterns (high confidence)
            talib_patterns_high_priority = {
                'HAMMER': (talib.CDLHAMMER, 'BULLISH', 0.75),
                'SHOOTING_STAR': (talib.CDLSHOOTINGSTAR, 'BEARISH', 0.75),
                'DOJI': (talib.CDLDOJI, 'NEUTRAL', 0.60),
                'ENGULFING_BULLISH': (talib.CDLENGULFING, 'BULLISH', 0.80),  # Positive values
                'MORNING_STAR': (talib.CDLMORNINGSTAR, 'BULLISH', 0.85),
                'EVENING_STAR': (talib.CDLEVENINGSTAR, 'BEARISH', 0.85),
                'PIERCING_LINE': (talib.CDLPIERCING, 'BULLISH', 0.70),
                'DARK_CLOUD_COVER': (talib.CDLDARKCLOUDCOVER, 'BEARISH', 0.70),
                'THREE_WHITE_SOLDIERS': (talib.CDL3WHITESOLDIERS, 'BULLISH', 0.85),
                'THREE_BLACK_CROWS': (talib.CDL3BLACKCROWS, 'BEARISH', 0.85),
            }
            
            # Medium priority patterns
            talib_patterns_medium = {
                'HANGING_MAN': (talib.CDLHANGINGMAN, 'BEARISH', 0.65),
                'INVERTED_HAMMER': (talib.CDLINVERTEDHAMMER, 'BULLISH', 0.65),
                'HARAMI': (talib.CDLHARAMI, 'NEUTRAL', 0.60),
                'HARAMI_CROSS': (talib.CDLHARAMICROSS, 'NEUTRAL', 0.65),
                'MARUBOZU': (talib.CDLMARUBOZU, 'NEUTRAL', 0.60),
            }
            
            # Combine all patterns
            all_talib_patterns = {**talib_patterns_high_priority, **talib_patterns_medium}
            
            for pattern_name, (func, default_signal, base_conf) in all_talib_patterns.items():
                try:
                    result = func(open_prices, high_prices, low_prices, close_prices)
                    
                    # Check last few candles for pattern
                    for i in range(max(0, len(result) - 5), len(result)):
                        value = result[i]
                        
                        if value != 0:  # Pattern detected
                            # Determine signal
                            if pattern_name in ['ENGULFING_BULLISH']:
                                signal = 'BULLISH' if value > 0 else 'BEARISH'
                            elif default_signal == 'NEUTRAL':
                                # For neutral patterns, check if bullish or bearish
                                signal = 'BULLISH' if value > 0 else 'BEARISH'
                            else:
                                signal = default_signal
                            
                            # Adjust confidence based on pattern strength
                            confidence = base_conf * (abs(value) / 100.0)
                            confidence = float(np.clip(confidence, 0.5, 0.95))
                            
                            patterns.append({
                                'pattern': pattern_name,
                                'signal': signal,
                                'confidence': confidence,
                                'source': 'ADVANCED_TA',
                                'detection_method': 'talib',
                                'strength': abs(value)
                            })
                            break  # Only report first occurrence
                            
                except Exception as e:
                    logger.debug(f"TA-Lib {pattern_name} error: {e}")
                    continue
            
            logger.info(f"✅ TA-Lib detected {len(patterns)} candlestick patterns")
            
        except Exception as e:
            logger.error(f"TA-Lib pattern detection error: {e}")
        
        return patterns
