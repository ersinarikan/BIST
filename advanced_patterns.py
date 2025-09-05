"""
Advanced Pattern Detector
- Simple heuristics for classic patterns: DOUBLE_TOP, DOUBLE_BOTTOM, HEAD_AND_SHOULDERS, INVERSE_HEAD_AND_SHOULDERS
- Designed to be lightweight and safe in preprod
"""

from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd


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
            patterns.extend(self._detect_double_top_bottom(df))
            patterns.extend(self._detect_head_shoulders(df))

            # Cap list size
            if len(patterns) > 8:
                patterns = sorted(patterns, key=lambda p: p.get('confidence', 0), reverse=True)[:8]
            return patterns
        except Exception:
            return []

    # ------------- Internal helpers -------------
    def _normalize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        # Accept both Open/High/Low/Close/Volume and lower-case variants
        rename_map = {}
        if 'Open' in df.columns: rename_map['Open'] = 'open'
        if 'High' in df.columns: rename_map['High'] = 'high'
        if 'Low' in df.columns: rename_map['Low'] = 'low'
        if 'Close' in df.columns: rename_map['Close'] = 'close'
        if 'Volume' in df.columns: rename_map['Volume'] = 'volume'
        df = df.rename(columns=rename_map)
        return df

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> List[Dict]:
        # Very simple peak/trough detection using rolling windows
        prices = df['close'].values.astype(float)
        window = 5
        patterns: List[Dict] = []

        # Find local maxima/minima
        highs = (df['high'].rolling(window).max()).values
        lows = (df['low'].rolling(window).min()).values

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
                        valley = np.min(segment[left:right]) if right - left > 2 else max_val
                        if valley < max_val * 0.985:  # at least 1.5% dip
                            conf = min(0.9, 0.6 + float((max_val - valley) / max_val) * 5)
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
                        peak = np.max(segment[left:right]) if right - left > 2 else min_val
                        if peak > min_val * 1.015:
                            conf = min(0.9, 0.6 + float((peak - min_val) / (abs(min_val) + 1e-8)) * 5)
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
                    if head > left * 1.01 and head > right * 1.01 and abs(left - right) / max(left, 1e-8) < 0.03:
                        # Head & Shoulders
                        conf = min(0.9, 0.55 + float((head - (left + right) / 2) / (head + 1e-8)) * 5)
                        patterns.append({
                            'pattern': 'HEAD_AND_SHOULDERS',
                            'signal': 'BEARISH',
                            'confidence': float(conf),
                            'source': 'ADVANCED_TA'
                        })
                        break
                    if head < left * 0.99 and head < right * 0.99 and abs(left - right) / max(left, 1e-8) < 0.03:
                        # Inverse H&S
                        conf = min(0.9, 0.55 + float(((left + right) / 2 - head) / (abs(head) + 1e-8)) * 5)
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
