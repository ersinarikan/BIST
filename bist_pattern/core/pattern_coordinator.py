"""
Pattern Detection Coordinator
Optimized pattern detection with smart strategy and resource management
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import threading
try:
    import gevent.lock
    GEVENT_AVAILABLE = True
except ImportError:
    GEVENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class PatternDetectionMode(Enum):
    """Pattern detection complexity modes"""
    FAST = "fast"          # Only basic patterns (< 100ms)
    STANDARD = "standard"  # Basic + Advanced patterns (< 500ms)
    COMPREHENSIVE = "comprehensive"  # All pattern types (< 2s)


class PatternPriority(Enum):
    """Pattern type priorities for resource allocation"""
    HIGH = 1      # Basic TA patterns - always run
    MEDIUM = 2    # Advanced TA patterns - run for important stocks
    LOW = 3       # Visual YOLO patterns - run selectively


class PatternDetectionCoordinator:
    """
    Optimized pattern detection coordinator
    
    Features:
    - Smart resource allocation based on stock importance
    - Performance-aware pattern selection
    - Intelligent caching with TTL
    - Conflict resolution between different detector results
    - Lazy loading of expensive detectors
    """
    
    def __init__(self):
        self.cache = {}
        # Environment-driven cache configuration
        try:
            self.cache_ttl = int(os.getenv('PATTERN_COORDINATOR_CACHE_TTL', '300'))
        except Exception:
            self.cache_ttl = 300
            
        self.performance_stats = {}
        # Use Gevent-compatible lock if available, fallback to threading
        if GEVENT_AVAILABLE:
            self.lock = gevent.lock.RLock()
        else:
            self.lock = threading.RLock()
        # Note: Keep lock operations minimal to avoid blocking
        
        # Pattern detectors - lazy loaded
        self._basic_detector = None
        self._advanced_detector = None
        self._visual_detector = None
        
        # Performance thresholds (milliseconds) - environment configurable
        try:
            fast_threshold = int(os.getenv('PATTERN_FAST_THRESHOLD_MS', '100'))
            standard_threshold = int(os.getenv('PATTERN_STANDARD_THRESHOLD_MS', '500'))
            comprehensive_threshold = int(os.getenv('PATTERN_COMPREHENSIVE_THRESHOLD_MS', '2000'))
        except Exception:
            fast_threshold, standard_threshold, comprehensive_threshold = 100, 500, 2000
            
        self.performance_thresholds = {
            PatternDetectionMode.FAST: fast_threshold,
            PatternDetectionMode.STANDARD: standard_threshold,
            PatternDetectionMode.COMPREHENSIVE: comprehensive_threshold
        }
        
        # Stock importance cache for smarter detection strategy
        self.stock_importance = {}
        
        logger.info("🎯 Pattern Detection Coordinator initialized")
    
    def _get_basic_detector(self):
        """Lazy load basic pattern detector"""
        # PHASE 1 RESTORE: Basic pattern detection (TA-Lib calculations)
        if self._basic_detector is None:
            from .basic_pattern_detector import BasicPatternDetector
            self._basic_detector = BasicPatternDetector()
        return self._basic_detector
    
    def _get_advanced_detector(self):
        """Lazy load advanced pattern detector"""
        if self._advanced_detector is None:
            try:
                from advanced_patterns import AdvancedPatternDetector
                self._advanced_detector = AdvancedPatternDetector()
            except ImportError:
                self._advanced_detector = None
        return self._advanced_detector
    
    def _get_visual_detector(self):
        """Lazy load visual pattern detector"""
        if self._visual_detector is None:
            try:
                from config import config
                if config['default'].ENABLE_YOLO:
                    # Use async visual pattern system for non-blocking operation
                    from visual_pattern_async import get_async_visual_pattern_system
                    self._visual_detector = get_async_visual_pattern_system()
            except ImportError:
                self._visual_detector = None
        return self._visual_detector
    
    def _get_cache_key(self, symbol: str, mode: PatternDetectionMode) -> str:
        """Generate cache key for pattern results"""
        return f"patterns_{symbol}_{mode.value}"
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        
        cached_time = cache_entry.get('timestamp')
        if not cached_time:
            return False
        
        try:
            cache_age = (datetime.now() - datetime.fromisoformat(cached_time)).total_seconds()
            return cache_age < self.cache_ttl
        except Exception:
            return False
    
    def _determine_detection_mode(self, symbol: str, data_length: int) -> PatternDetectionMode:
        """
        Intelligently determine detection mode based on:
        - Stock importance (volume, market cap, recent activity)
        - Data quality/length
        - System load
        """
        # Stock importance factors
        importance_score = self.stock_importance.get(symbol, 0.5)
        
        # Data quality factor
        data_quality = min(1.0, data_length / 200.0)  # Normalize to 200 days
        
        # Combine factors
        combined_score = (importance_score * 0.7) + (data_quality * 0.3)
        
        if combined_score >= 0.4:
            return PatternDetectionMode.COMPREHENSIVE
        elif combined_score >= 0.2:
            return PatternDetectionMode.STANDARD
        else:
            return PatternDetectionMode.FAST
    
    def _update_stock_importance(self, symbol: str, volume: float = None, 
                                price_change: float = None, recent_patterns: int = None):
        """Update stock importance score for smarter detection"""
        base_score = 0.5
        
        # Volume factor (higher volume = more important)
        if volume and volume > 1000000:  # 1M+ volume
            base_score += 0.2
        
        # Price volatility factor
        if price_change and abs(price_change) > 0.05:  # 5%+ change
            base_score += 0.2
        
        # Recent pattern activity
        if recent_patterns and recent_patterns > 2:
            base_score += 0.1
        
        self.stock_importance[symbol] = min(1.0, base_score)
    
    def analyze_patterns(self, symbol: str, data, 
                        mode: Optional[PatternDetectionMode] = None,
                        force_refresh: bool = False) -> Dict[str, Any]:
        """
        Coordinated pattern analysis with smart resource allocation
        
        Args:
            symbol: Stock symbol
            data: Price data DataFrame
            mode: Detection mode (auto-determined if None)
            force_refresh: Bypass cache
            
        Returns:
            Structured pattern analysis results
        """
        start_time = time.time()
        
        try:
            # Auto-determine mode if not specified
            if mode is None:
                # Update importance score using available volume/price info
                try:
                    vol_val = None
                    chg_val = None
                    if data is not None:
                        # Volume: 20-bar average if present
                        vol_series = getattr(data, 'columns', []) and ('volume' in data.columns)
                        if vol_series:
                            try:
                                vol_avg = data['volume'].tail(20).mean()
                                if vol_avg is not None:
                                    vol_val = float(vol_avg)
                            except Exception:
                                vol_val = None
                        # Price change: last close vs previous
                        cls_series = getattr(data, 'columns', []) and ('close' in data.columns)
                        if cls_series and len(data) >= 2:
                            try:
                                last = float(data['close'].iloc[-1])
                                prev = float(data['close'].iloc[-2])
                                if prev:
                                    chg_val = (last - prev) / prev
                            except Exception:
                                chg_val = None
                    self._update_stock_importance(symbol, volume=vol_val, price_change=chg_val)
                except Exception:
                    pass
                mode = self._determine_detection_mode(symbol, len(data) if data is not None else 0)
            
            # Check cache first
            cache_key = self._get_cache_key(symbol, mode)
            if not force_refresh and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if self._is_cache_valid(cached_result):
                    return cached_result['data']
            
            # Initialize results structure
            results = {
                'symbol': symbol,
                'analysis_mode': mode.value,
                'timestamp': datetime.now().isoformat(),
                'patterns': [],
                'technical_indicators': {},
                'summary': {
                    'total_patterns': 0,
                    'bullish_patterns': 0,
                    'bearish_patterns': 0,
                    'confidence_avg': 0.0
                },
                'performance': {
                    'detection_time_ms': 0,
                    'detectors_used': []
                }
            }
            
            if data is None or len(data) < 10:
                return results
            
            # Basic patterns (always run - fast)
            basic_detector = self._get_basic_detector()
            if basic_detector:
                try:
                    basic_patterns = basic_detector.detect_patterns(data)
                    results['patterns'].extend(basic_patterns)
                    results['performance']['detectors_used'].append('basic')
                except Exception as e:
                    logger.warning(f"Basic pattern detection failed for {symbol}: {e}")
            
            # Advanced patterns (standard+ mode)
            if mode in [PatternDetectionMode.STANDARD, PatternDetectionMode.COMPREHENSIVE]:
                advanced_detector = self._get_advanced_detector()
                if advanced_detector:
                    try:
                        advanced_patterns = advanced_detector.analyze_all_patterns(data)
                        results['patterns'].extend(advanced_patterns)
                        results['performance']['detectors_used'].append('advanced')
                    except Exception as e:
                        logger.warning(f"Advanced pattern detection failed for {symbol}: {e}")
            
            # Visual patterns (comprehensive mode only)
            if mode == PatternDetectionMode.COMPREHENSIVE:
                visual_detector = self._get_visual_detector()
                if visual_detector:
                    try:
                        # Use async visual analysis for non-blocking operation
                        if hasattr(visual_detector, 'request_visual_analysis_async'):
                            request_id = visual_detector.request_visual_analysis_async(symbol, data)
                            visual_result = visual_detector.get_visual_analysis_result(request_id)
                        else:
                            visual_result = visual_detector.analyze_stock_visual(symbol, data)
                        if visual_result and visual_result.get('status') == 'success':
                            visual_patterns = self._process_visual_patterns(
                                visual_result.get('visual_analysis', {})
                            )
                            results['patterns'].extend(visual_patterns)
                            results['performance']['detectors_used'].append('visual')
                    except Exception as e:
                        logger.warning(f"Visual pattern detection failed for {symbol}: {e}")
            
            # Process and merge results
            results = self._process_pattern_results(results)
            
            # Performance tracking
            detection_time_ms = int((time.time() - start_time) * 1000)
            results['performance']['detection_time_ms'] = detection_time_ms
            
            # Update performance stats
            with self.lock:  # RESTORED: Minimal lock for stats update
                if symbol not in self.performance_stats:
                    self.performance_stats[symbol] = []
                self.performance_stats[symbol].append({
                    'mode': mode.value,
                    'time_ms': detection_time_ms,
                    'patterns_found': len(results['patterns']),
                    'timestamp': datetime.now().isoformat()
                })
                # Keep only last 10 entries per symbol
                self.performance_stats[symbol] = self.performance_stats[symbol][-10:]
            
            # Cache results
            self.cache[cache_key] = {
                'data': results,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log performance if slow
            if detection_time_ms > self.performance_thresholds[mode]:
                logger.warning(
                    f"Slow pattern detection: {symbol} took {detection_time_ms}ms "
                    f"(threshold: {self.performance_thresholds[mode]}ms)"
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Pattern analysis coordinator error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'patterns': []
            }
    
    def _process_visual_patterns(self, visual_analysis: Dict) -> List[Dict]:
        """Process visual pattern results into standard format"""
        patterns = []
        detected_patterns = visual_analysis.get('patterns', [])
        
        for visual_pattern in detected_patterns:
            pattern_info = {
                'pattern': visual_pattern.get('pattern', 'UNKNOWN'),
                'confidence': visual_pattern.get('confidence', 0.0),
                'signal': self._get_visual_signal(visual_pattern.get('pattern')),
                'strength': visual_pattern.get('confidence', 0.0) * 100,
                'source': 'VISUAL_YOLO',
                'bbox': visual_pattern.get('bbox'),
                'area': visual_pattern.get('area')
            }
            patterns.append(pattern_info)
        
        return patterns
    
    def _get_visual_signal(self, pattern_name: str) -> str:
        """Map visual pattern names to trading signals"""
        bullish_patterns = {
            'ascending_triangle', 'cup_and_handle', 'double_bottom',
            'inverse_head_shoulders', 'bullish_flag', 'rising_wedge'
        }
        bearish_patterns = {
            'descending_triangle', 'double_top', 'head_shoulders',
            'bearish_flag', 'falling_wedge'
        }
        
        pattern_lower = pattern_name.lower()
        if any(p in pattern_lower for p in bullish_patterns):
            return 'BULLISH'
        elif any(p in pattern_lower for p in bearish_patterns):
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _process_pattern_results(self, results: Dict) -> Dict:
        """Process and merge pattern results with conflict resolution"""
        patterns = results['patterns']
        
        if not patterns:
            return results
        
        # Calculate summary statistics
        total_patterns = len(patterns)
        bullish_count = sum(1 for p in patterns if p.get('signal') == 'BULLISH')
        bearish_count = sum(1 for p in patterns if p.get('signal') == 'BEARISH')
        
        confidences = [p.get('confidence', 0.0) for p in patterns if 'confidence' in p]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Sort patterns by confidence
        patterns.sort(key=lambda p: p.get('confidence', 0.0), reverse=True)
        
        # Update summary
        results['summary'].update({
            'total_patterns': total_patterns,
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'confidence_avg': round(avg_confidence, 3)
        })
        
        # Limit patterns to top 10 for performance
        results['patterns'] = patterns[:10]
        
        return results
    
    def get_performance_stats(self, symbol: Optional[str] = None) -> Dict:
        """Get performance statistics"""
        with self.lock:
            if symbol:
                return {
                    'symbol': symbol,
                    'stats': self.performance_stats.get(symbol, [])
                }
            else:
                total_analyses = sum(len(stats) for stats in self.performance_stats.values())
                avg_time = 0
                if total_analyses > 0:
                    total_time = sum(
                        stat['time_ms'] 
                        for stats in self.performance_stats.values() 
                        for stat in stats
                    )
                    avg_time = total_time / total_analyses
                
                return {
                    'total_symbols': len(self.performance_stats),
                    'total_analyses': total_analyses,
                    'average_time_ms': round(avg_time, 1),
                    'cache_entries': len(self.cache)
                }
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear pattern analysis cache"""
        if symbol:
            # Clear specific symbol cache
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"patterns_{symbol}_")]
            for key in keys_to_remove:
                del self.cache[key]
        else:
            # Clear all cache
            self.cache.clear()
        
        logger.info(f"Pattern cache cleared for {symbol if symbol else 'all symbols'}")


# Global singleton
_pattern_coordinator = None


def get_pattern_coordinator() -> PatternDetectionCoordinator:
    """Get pattern detection coordinator singleton"""
    global _pattern_coordinator
    if _pattern_coordinator is None:
        _pattern_coordinator = PatternDetectionCoordinator()
    return _pattern_coordinator
