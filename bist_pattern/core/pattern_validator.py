"""
Pattern Validation System
3-stage validation: Basic TA → Advanced TA → YOLO Visual

Purpose: Reduce false positives by validating patterns across multiple detection methods
Strategy: Each stage adds confidence score, requiring multi-method agreement
"""

import os
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PatternValidator:
    """
    Multi-stage pattern validation system
    
    Validation Flow:
    1. Basic TA detects potential pattern (baseline confidence: 0.3)
    2. Advanced TA confirms pattern structure (+0.3 confidence)
    3. YOLO provides visual evidence (+0.4 confidence)
    
    Final confidence range: 0.3-1.0 based on validation stages passed
    """
    
    def __init__(self):
        # Environment-driven thresholds
        try:
            self.min_validation_confidence = float(os.getenv('PATTERN_MIN_VALIDATION_CONF', '0.5'))
        except Exception as e:
            logger.debug(f"Failed to get PATTERN_MIN_VALIDATION_CONF, using 0.5: {e}")
            self.min_validation_confidence = 0.5
        
        # Standalone pattern minimum confidence (for ADVANCED/YOLO without BASIC match)
        try:
            self.standalone_min_conf = float(os.getenv('PATTERN_STANDALONE_MIN_CONF', '0.55'))
        except Exception as e:
            logger.debug(f"Failed to get PATTERN_STANDALONE_MIN_CONF, using 0.55: {e}")
            self.standalone_min_conf = 0.55
        
        try:
            self.basic_weight = float(os.getenv('PATTERN_BASIC_WEIGHT', '0.3'))
        except Exception as e:
            logger.debug(f"Failed to get PATTERN_BASIC_WEIGHT, using 0.3: {e}")
            self.basic_weight = 0.3
            
        try:
            self.advanced_weight = float(os.getenv('PATTERN_ADVANCED_WEIGHT', '0.3'))
        except Exception as e:
            logger.debug(f"Failed to get PATTERN_ADVANCED_WEIGHT, using 0.3: {e}")
            self.advanced_weight = 0.3
            
        try:
            self.yolo_weight = float(os.getenv('PATTERN_YOLO_WEIGHT', '0.4'))
        except Exception as e:
            logger.debug(f"Failed to get PATTERN_YOLO_WEIGHT, using 0.4: {e}")
            self.yolo_weight = 0.4
        
        # Pattern matching similarity threshold
        try:
            self.pattern_match_threshold = float(os.getenv('PATTERN_MATCH_THRESHOLD', '0.7'))
        except Exception as e:
            logger.debug(f"Failed to get PATTERN_MATCH_THRESHOLD, using 0.7: {e}")
            self.pattern_match_threshold = 0.7
        
        logger.info(f"🔍 Pattern Validator initialized: min_conf={self.min_validation_confidence}, standalone={self.standalone_min_conf}")
    
    def validate_patterns(
        self, 
        basic_patterns: List[Dict[str, Any]], 
        advanced_patterns: List[Dict[str, Any]], 
        yolo_patterns: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate patterns through multi-stage confirmation
        
        Args:
            basic_patterns: Basic TA patterns (RSI, MACD, BB)
            advanced_patterns: Advanced TA patterns (complex formations)
            yolo_patterns: YOLO visual patterns
            data: Stock price data for context
            
        Returns:
            Tuple of (validated_patterns, validation_stats)
        """
        validated_patterns = []
        stats = {
            'total_basic': len(basic_patterns),
            'total_advanced': len(advanced_patterns),
            'total_yolo': len(yolo_patterns),
            'validated': 0,
            'rejected': 0,
            'validation_scores': []
        }
        
        try:
            # Process each basic pattern through validation pipeline
            for basic_p in basic_patterns:
                validation_result = self._validate_single_pattern(
                    basic_p, advanced_patterns, yolo_patterns, data
                )
                
                if validation_result['passed']:
                    # Add validated pattern with enhanced confidence
                    validated_p = basic_p.copy()
                    validated_p['original_confidence'] = basic_p.get('confidence', 0.5)
                    validated_p['confidence'] = validation_result['final_confidence']
                    validated_p['validation_stages'] = validation_result['stages_passed']
                    validated_p['validation_score'] = validation_result['validation_score']
                    
                    validated_patterns.append(validated_p)
                    stats['validated'] += 1
                    stats['validation_scores'].append(validation_result['validation_score'])
                else:
                    stats['rejected'] += 1
                    logger.debug(
                        f"❌ Pattern rejected: {basic_p.get('pattern')} "
                        f"(score: {validation_result['validation_score']:.2f} < "
                        f"{self.min_validation_confidence})"
                    )
            
            # Add standalone ADVANCED patterns (don't need BASIC match)
            # These are high-quality pattern formations (HEAD_AND_SHOULDERS, etc.)
            for adv_p in advanced_patterns:
                if not self._has_basic_match(adv_p, basic_patterns):
                    adv_conf = adv_p.get('confidence', 0.6)
                    
                    # YOLO confirmation boosts confidence further
                    yolo_conf = self._get_yolo_confirmation(adv_p, yolo_patterns)
                    
                    # Accept standalone ADVANCED pattern if confidence meets threshold
                    if adv_conf >= self.standalone_min_conf:
                        validated_p = adv_p.copy()
                        
                        # Boost confidence if YOLO confirms
                        if yolo_conf > 0:
                            validated_p['confidence'] = min(0.95, adv_conf * (1 + yolo_conf * 0.4))
                            validated_p['validation_stages'] = ['ADVANCED', 'YOLO']
                            validated_p['validation_score'] = self.advanced_weight + yolo_conf * self.yolo_weight
                        else:
                            validated_p['confidence'] = adv_conf
                            validated_p['validation_stages'] = ['ADVANCED']
                            validated_p['validation_score'] = self.advanced_weight
                        
                        validated_patterns.append(validated_p)
                        stats['validated'] += 1
                        logger.debug(f"✅ Standalone ADVANCED accepted: {adv_p.get('pattern')} (conf: {validated_p['confidence']:.2f})")
            
            # Add standalone YOLO patterns (visual evidence is valuable)
            for yolo_p in yolo_patterns:
                # Check if not already matched with BASIC or ADVANCED
                if not self._has_basic_match(yolo_p, basic_patterns) and \
                   not self._has_basic_match(yolo_p, advanced_patterns):
                    yolo_conf = yolo_p.get('confidence', 0.5)
                    
                    # Accept standalone YOLO pattern if confidence meets threshold
                    if yolo_conf >= self.standalone_min_conf:
                        validated_p = yolo_p.copy()
                        validated_p['confidence'] = yolo_conf
                        validated_p['validation_stages'] = ['YOLO']
                        validated_p['validation_score'] = self.yolo_weight
                        validated_patterns.append(validated_p)
                        stats['validated'] += 1
                        logger.debug(f"✅ Standalone YOLO accepted: {yolo_p.get('pattern')} (conf: {yolo_conf:.2f})")
            
            # Calculate average validation score
            if stats['validation_scores']:
                stats['avg_validation_score'] = float(np.mean(stats['validation_scores']))
            else:
                stats['avg_validation_score'] = 0.0
            
            logger.info(
                f"✅ Pattern Validation: {stats['validated']}/{stats['total_basic']} validated "
                f"(avg score: {stats['avg_validation_score']:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Pattern validation error: {e}")
            # Fallback: return basic patterns with reduced confidence
            validated_patterns = [
                {**p, 'confidence': p.get('confidence', 0.5) * 0.7} 
                for p in basic_patterns
            ]
        
        return validated_patterns, stats
    
    def _validate_single_pattern(
        self, 
        basic_pattern: Dict[str, Any],
        advanced_patterns: List[Dict[str, Any]],
        yolo_patterns: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate single pattern through multi-stage pipeline
        
        Returns:
            {
                'passed': bool,
                'final_confidence': float,
                'validation_score': float,
                'stages_passed': List[str]
            }
        """
        stages_passed = ['BASIC']  # Always starts with basic
        validation_score = self.basic_weight
        
        # Stage 2: Advanced TA confirmation
        advanced_conf = self._get_advanced_confirmation(basic_pattern, advanced_patterns)
        if advanced_conf > 0:
            stages_passed.append('ADVANCED')
            validation_score += self.advanced_weight * advanced_conf
        
        # Stage 3: YOLO visual evidence
        yolo_conf = self._get_yolo_confirmation(basic_pattern, yolo_patterns)
        if yolo_conf > 0:
            stages_passed.append('YOLO')
            validation_score += self.yolo_weight * yolo_conf
        
        # Final confidence calculation
        original_conf = basic_pattern.get('confidence', 0.5)
        
        # Boost original confidence by validation score
        # Full validation (all 3 stages) can boost up to 1.5x
        boost_factor = 1.0 + (validation_score * 0.5)
        final_confidence = min(0.95, original_conf * boost_factor)
        
        # Validation passes if score meets minimum threshold
        passed = validation_score >= self.min_validation_confidence
        
        return {
            'passed': passed,
            'final_confidence': final_confidence,
            'validation_score': validation_score,
            'stages_passed': stages_passed
        }
    
    def _get_advanced_confirmation(
        self, 
        basic_pattern: Dict[str, Any], 
        advanced_patterns: List[Dict[str, Any]]
    ) -> float:
        """
        Check if advanced TA confirms basic pattern
        
        Returns:
            Confidence multiplier [0-1] based on pattern similarity
        """
        if not advanced_patterns:
            return 0.0
        
        basic_signal = basic_pattern.get('signal', '').upper()
        basic_name = basic_pattern.get('pattern', '').upper()
        
        max_confirmation = 0.0
        
        for adv_p in advanced_patterns:
            adv_signal = adv_p.get('signal', '').upper()
            adv_name = adv_p.get('pattern', '').upper()
            
            # Signal must match (BULLISH/BEARISH)
            if adv_signal != basic_signal:
                continue
            
            # Check pattern similarity
            similarity = self._calculate_pattern_similarity(basic_name, adv_name)
            
            if similarity > self.pattern_match_threshold:
                # Confirmed! Use advanced pattern's confidence as multiplier
                adv_conf = adv_p.get('confidence', 0.6)
                confirmation = min(1.0, similarity * adv_conf)
                max_confirmation = max(max_confirmation, confirmation)
        
        return max_confirmation
    
    def _get_yolo_confirmation(
        self, 
        pattern: Dict[str, Any], 
        yolo_patterns: List[Dict[str, Any]]
    ) -> float:
        """
        Check if YOLO visual detection confirms pattern
        
        Returns:
            Confidence multiplier [0-1] based on visual evidence
        """
        if not yolo_patterns:
            return 0.0
        
        pattern_signal = pattern.get('signal', '').upper()
        pattern_name = pattern.get('pattern', '').upper()
        
        max_confirmation = 0.0
        
        for yolo_p in yolo_patterns:
            yolo_signal = yolo_p.get('signal', '').upper()
            yolo_name = yolo_p.get('pattern', '').upper()
            
            # Signal must match
            if yolo_signal != pattern_signal:
                continue
            
            # YOLO pattern names might be different, check similarity
            similarity = self._calculate_pattern_similarity(pattern_name, yolo_name)
            
            # YOLO is visual evidence, even partial match is valuable
            if similarity > 0.5:  # Lower threshold for YOLO
                yolo_conf = yolo_p.get('confidence', 0.5)
                confirmation = min(1.0, similarity * yolo_conf)
                max_confirmation = max(max_confirmation, confirmation)
        
        return max_confirmation
    
    def _calculate_pattern_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two pattern names
        
        Uses simple token overlap + signal keywords
        
        Returns:
            Similarity score [0-1]
        """
        if not name1 or not name2:
            return 0.0
        
        # Normalize
        n1 = str(name1).upper()
        n2 = str(name2).upper()
        
        # Exact match
        if n1 == n2:
            return 1.0
        
        # Token-based similarity
        tokens1 = set(n1.replace('_', ' ').split())
        tokens2 = set(n2.replace('_', ' ').split())
        
        # Remove common words
        stop_words = {'PATTERN', 'SIGNAL', 'ML', 'ENH', 'BASIC', 'ADVANCED', 'VISUAL', 'YOLO'}
        tokens1 = tokens1 - stop_words
        tokens2 = tokens2 - stop_words
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        # Boost if key pattern terms match
        key_terms = {
            'HEAD', 'SHOULDERS', 'TRIANGLE', 'WEDGE', 'FLAG', 'PENNANT',
            'DOUBLE', 'TOP', 'BOTTOM', 'CHANNEL', 'BREAKOUT', 'REVERSAL'
        }
        
        key_match = bool((tokens1 & key_terms) & (tokens2 & key_terms))
        if key_match:
            similarity = min(1.0, similarity * 1.3)
        
        return float(similarity)
    
    def _has_basic_match(
        self, 
        advanced_pattern: Dict[str, Any], 
        basic_patterns: List[Dict[str, Any]]
    ) -> bool:
        """Check if advanced pattern already matched with basic pattern"""
        for basic_p in basic_patterns:
            similarity = self._calculate_pattern_similarity(
                basic_p.get('pattern', ''),
                advanced_pattern.get('pattern', '')
            )
            if similarity > self.pattern_match_threshold:
                return True
        return False


# Global singleton
_pattern_validator = None


def get_pattern_validator() -> PatternValidator:
    """Get or create pattern validator singleton"""
    global _pattern_validator
    if _pattern_validator is None:
        _pattern_validator = PatternValidator()
    return _pattern_validator
