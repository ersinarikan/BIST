"""
Gelişmiş Teknik Analiz Formasyonları
Head & Shoulders, Cup & Handle, Double Top/Bottom, Fibonacci
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.stats import linregress
import logging

logger = logging.getLogger(__name__)

class AdvancedPatternDetector:
    def __init__(self):
        self.min_data_points = 30  # Minimum veri noktası
        
    def detect_head_and_shoulders(self, highs, lows, closes, dates=None):
        """Head and Shoulders (OBO) pattern detection"""
        if len(highs) < self.min_data_points:
            return None
            
        try:
            # Peak detection (tepeler)
            peaks, peak_properties = find_peaks(highs, distance=5, prominence=np.std(highs)*0.5)
            
            if len(peaks) < 3:
                return None
            
            # Son 3 önemli tepe
            recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            peak_values = highs[recent_peaks]
            peak_dates = recent_peaks
            
            # OBO pattern kontrolleri
            left_shoulder = peak_values[0]
            head = peak_values[1] 
            right_shoulder = peak_values[2]
            
            # Head en yüksek olmalı
            if head <= max(left_shoulder, right_shoulder):
                return None
                
            # Shoulder'lar birbirine yakın olmalı (±5% tolerance)
            shoulder_diff = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
            if shoulder_diff > 0.15:  # %15'ten fazla fark varsa geçersiz
                return None
            
            # Head, shoulder'lardan en az %8 yüksek olmalı
            min_shoulder = min(left_shoulder, right_shoulder)
            head_advantage = (head - min_shoulder) / min_shoulder
            if head_advantage < 0.08:
                return None
            
            # Neckline (boyun çizgisi) hesaplama
            # İki shoulder arasındaki minimum değerleri bul
            left_valley_idx = np.argmin(lows[recent_peaks[0]:recent_peaks[1]]) + recent_peaks[0]
            right_valley_idx = np.argmin(lows[recent_peaks[1]:recent_peaks[2]]) + recent_peaks[1]
            
            neckline_level = (lows[left_valley_idx] + lows[right_valley_idx]) / 2
            current_price = closes[-1]
            
            # Pattern strength hesaplama
            pattern_height = head - neckline_level
            price_position = (current_price - neckline_level) / pattern_height
            
            # Sinyal belirleme
            signal = "BEARISH"  # OBO genelde düşüş sinyali
            confidence = min(0.9, 0.6 + head_advantage)  # %60-90 arası güven
            
            # Neckline kırılımı kontrolü
            if current_price < neckline_level * 0.98:  # %2 tolerance ile kırılım
                signal_strength = 85 + (head_advantage * 100)
            else:
                signal_strength = 65 + (head_advantage * 50)
            
            return {
                'pattern': 'HEAD_AND_SHOULDERS',
                'signal': signal,
                'confidence': confidence,
                'strength': min(100, signal_strength),
                'neckline': neckline_level,
                'head_price': head,
                'left_shoulder': left_shoulder,
                'right_shoulder': right_shoulder,
                'current_position': price_position,
                'target_price': neckline_level - pattern_height,  # Projeksiyon
                'stop_loss': head * 1.02,
                'pattern_points': {
                    'peaks': recent_peaks.tolist(),
                    'valleys': [left_valley_idx, right_valley_idx]
                }
            }
            
        except Exception as e:
            logger.error(f"Head & Shoulders detection error: {e}")
            return None
    
    def detect_double_top(self, highs, lows, closes):
        """Double Top (Çift Tepe) pattern detection"""
        if len(highs) < self.min_data_points:
            return None
            
        try:
            peaks, _ = find_peaks(highs, distance=10, prominence=np.std(highs)*0.3)
            
            if len(peaks) < 2:
                return None
            
            # Son iki tepe
            recent_peaks = peaks[-2:]
            peak1_price = highs[recent_peaks[0]]
            peak2_price = highs[recent_peaks[1]]
            
            # Tepeler birbirine yakın olmalı (±3% tolerance)
            price_diff = abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
            if price_diff > 0.05:
                return None
            
            # Aradaki valle (dip)
            valley_start = recent_peaks[0]
            valley_end = recent_peaks[1]
            valley_idx = np.argmin(lows[valley_start:valley_end]) + valley_start
            valley_price = lows[valley_idx]
            
            # Valle, tepelerden en az %5 düşük olmalı
            min_peak = min(peak1_price, peak2_price)
            valley_depth = (min_peak - valley_price) / min_peak
            if valley_depth < 0.05:
                return None
            
            current_price = closes[-1]
            resistance_level = (peak1_price + peak2_price) / 2
            
            return {
                'pattern': 'DOUBLE_TOP',
                'signal': 'BEARISH',
                'confidence': 0.7 + (valley_depth * 2),
                'strength': 70 + (valley_depth * 200),
                'resistance': resistance_level,
                'support': valley_price,
                'target_price': valley_price - (resistance_level - valley_price),
                'stop_loss': resistance_level * 1.02,
                'pattern_points': {
                    'peaks': recent_peaks.tolist(),
                    'valley': valley_idx
                }
            }
            
        except Exception as e:
            logger.error(f"Double Top detection error: {e}")
            return None
            
    def detect_cup_and_handle(self, highs, lows, closes):
        """Cup and Handle pattern detection"""
        if len(closes) < 50:  # Cup için daha fazla veri gerek
            return None
            
        try:
            # Cup detection (U şekli)
            mid_point = len(closes) // 2
            left_high = np.max(highs[:mid_point//2])
            right_high = np.max(highs[-mid_point//2:])
            cup_bottom = np.min(lows[mid_point//2:-mid_point//2])
            
            # Cup depth (en az %12 olmalı)
            cup_high = (left_high + right_high) / 2
            cup_depth = (cup_high - cup_bottom) / cup_high
            if cup_depth < 0.12:
                return None
            
            # Handle detection (son %25'lik kısım)
            handle_start = int(len(closes) * 0.75)
            handle_data = closes[handle_start:]
            
            if len(handle_data) < 10:
                return None
            
            handle_high = np.max(handle_data)
            handle_low = np.min(handle_data)
            handle_depth = (handle_high - handle_low) / handle_high
            
            # Handle depth %1-8 arası olmalı
            if handle_depth < 0.01 or handle_depth > 0.08:
                return None
            
            current_price = closes[-1]
            breakout_level = handle_high * 1.01  # %1 breakout
            
            signal = "BULLISH" if current_price > breakout_level else "NEUTRAL"
            
            return {
                'pattern': 'CUP_AND_HANDLE',
                'signal': signal,
                'confidence': 0.8 if signal == "BULLISH" else 0.6,
                'strength': 80 if signal == "BULLISH" else 60,
                'cup_depth': cup_depth,
                'handle_depth': handle_depth,
                'breakout_level': breakout_level,
                'target_price': breakout_level + (cup_high - cup_bottom),
                'stop_loss': handle_low * 0.98
            }
            
        except Exception as e:
            logger.error(f"Cup and Handle detection error: {e}")
            return None
    
    def detect_ascending_triangle(self, highs, lows, closes):
        """Ascending Triangle (Yükselen Üçgen) pattern"""
        if len(closes) < self.min_data_points:
            return None
            
        try:
            # Resistance line (yatay direnc)
            peaks, _ = find_peaks(highs, distance=5)
            if len(peaks) < 3:
                return None
            
            recent_peaks = peaks[-3:]
            peak_prices = highs[recent_peaks]
            
            # Peaks arasındaki varyasyon %2'den az olmalı (yatay direnç)
            peak_std = np.std(peak_prices) / np.mean(peak_prices)
            if peak_std > 0.02:
                return None
            
            # Support line (yükselen destek)
            valleys, _ = find_peaks(-lows, distance=5)
            if len(valleys) < 2:
                return None
            
            recent_valleys = valleys[-2:]
            valley_prices = lows[recent_valleys]
            valley_trend = (valley_prices[-1] - valley_prices[0]) / valley_prices[0]
            
            # Destek çizgisi yükseliyorsa
            if valley_trend < 0.02:  # En az %2 yükseliş
                return None
            
            resistance_level = np.mean(peak_prices)
            current_price = closes[-1]
            
            # Breakout kontrolü
            breakout = current_price > resistance_level * 1.005  # %0.5 breakout
            
            return {
                'pattern': 'ASCENDING_TRIANGLE',
                'signal': 'BULLISH' if breakout else 'NEUTRAL',
                'confidence': 0.75 if breakout else 0.6,
                'strength': 75 if breakout else 60,
                'resistance': resistance_level,
                'support_trend': valley_trend,
                'target_price': resistance_level * (1 + valley_trend),
                'stop_loss': recent_valleys[-1] * 0.98
            }
            
        except Exception as e:
            logger.error(f"Ascending Triangle detection error: {e}")
            return None
    
    def fibonacci_retracement(self, highs, lows, trend_direction="up"):
        """Fibonacci Retracement levels"""
        try:
            if trend_direction == "up":
                swing_low = np.min(lows)
                swing_high = np.max(highs)
            else:
                swing_high = np.min(lows)
                swing_low = np.max(highs)
            
            diff = swing_high - swing_low
            
            fib_levels = {
                '0%': swing_high,
                '23.6%': swing_high - (0.236 * diff),
                '38.2%': swing_high - (0.382 * diff),
                '50%': swing_high - (0.5 * diff),
                '61.8%': swing_high - (0.618 * diff),
                '78.6%': swing_high - (0.786 * diff),
                '100%': swing_low
            }
            
            return fib_levels
            
        except Exception as e:
            logger.error(f"Fibonacci calculation error: {e}")
            return None
    
    def analyze_all_patterns(self, data):
        """Tüm pattern'leri analiz et"""
        if len(data) < self.min_data_points:
            return []
        
        try:
            highs = np.array(data['high'])
            lows = np.array(data['low'])
            closes = np.array(data['close'])
            
            patterns = []
            
            # Head & Shoulders
            hs_pattern = self.detect_head_and_shoulders(highs, lows, closes)
            if hs_pattern:
                patterns.append(hs_pattern)
            
            # Double Top
            dt_pattern = self.detect_double_top(highs, lows, closes)
            if dt_pattern:
                patterns.append(dt_pattern)
            
            # Cup & Handle
            ch_pattern = self.detect_cup_and_handle(highs, lows, closes)
            if ch_pattern:
                patterns.append(ch_pattern)
            
            # Ascending Triangle
            at_pattern = self.detect_ascending_triangle(highs, lows, closes)
            if at_pattern:
                patterns.append(at_pattern)
            
            # Fibonacci levels
            fib_up = self.fibonacci_retracement(highs, lows, "up")
            fib_down = self.fibonacci_retracement(highs, lows, "down")
            
            if fib_up:
                patterns.append({
                    'pattern': 'FIBONACCI_UP',
                    'levels': fib_up,
                    'current_price': closes[-1]
                })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return []
