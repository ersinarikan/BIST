import os
import sys
import logging
import math


logger = logging.getLogger(__name__)


def _in_training_context() -> bool:
    try:
        argv = ' '.join(sys.argv) if hasattr(sys, 'argv') else ''
    except Exception as e:
        logger.debug(f"Failed to get sys.argv: {e}")
        argv = ''
    if 'bulk_train_all.py' in argv:
        return True
    try:
        if os.getenv('DISABLE_LIVE_BROADCAST', '0').lower() in ('1', 'true', 'yes', 'on'):
            return True
    except Exception as e:
        logger.debug(f"Failed to check DISABLE_LIVE_BROADCAST: {e}")
    return False


def _sanitize_json_value(value, context=None):
    """Sanitize a value for JSON serialization (handle inf, nan, and out-of-range floats).
    
    Instead of setting values to None, we round/clamp them to reasonable ranges:
    - inf → reasonable max/min based on context
    - nan → 0 or context-appropriate default
    - Very large values → JSON-safe maximum (1e15)
    - Very small values → JSON-safe minimum (-1e15)
    - NumPy types → converted to native Python types
    
    Args:
        value: Value to sanitize
        context: Optional context hint (e.g., 'learning_rate', 'n_estimators') for smart defaults
    """
    # ✅ CRITICAL FIX: Handle NumPy types FIRST before checking Python types
    # NumPy types (float64, int64, ndarray, etc.) are not JSON serializable
    try:
        import numpy as np
        if isinstance(value, np.floating):
            # Convert numpy float to Python float, then check for nan/inf
            value = float(value)
        elif isinstance(value, np.integer):
            # Convert numpy int to Python int
            return int(value)
        elif isinstance(value, np.bool_):
            # Convert numpy bool to Python bool
            return bool(value)
        elif isinstance(value, np.ndarray):
            # Convert numpy array to list, recursively sanitize
            return [_sanitize_json_value(v, context=context) for v in value.tolist()]
        elif hasattr(np, 'generic') and isinstance(value, np.generic):
            # Catch-all for other numpy scalar types
            return _sanitize_json_value(value.item(), context=context)
    except ImportError:
        pass  # numpy not installed, skip
    except Exception as e:
        logger.debug(f"Failed to use numpy for sanitization: {e}")
        # any other error, skip and try standard handling
    
    if isinstance(value, float):
        if math.isnan(value):
            # For NaN, use HPO mid-range or default values (from optuna_hpo_pilot.py)
            if context == 'learning_rate':
                return 0.05  # HPO mid-range: (0.01 + 0.25) / 2 ≈ 0.13, but use 0.05 as safe default
            elif context == 'n_estimators':
                return 500  # HPO mid-range: (150 + 900) / 2 ≈ 525, use 500
            elif context == 'max_depth':
                return 5  # HPO mid-range: (2 + 8) / 2 = 5
            elif context == 'confidence' or context == 'score':
                return 0.0  # Default confidence/score
            else:
                return 0.0  # General default
        
        if math.isinf(value):
            # For infinity, use HPO max/min values (from optuna_hpo_pilot.py)
            # These are the actual max values used in HPO optimization
            if value > 0:
                if context == 'learning_rate':
                    return 0.25  # HPO max: 0.25 (long horizons: 7d, 14d, 30d)
                elif context == 'n_estimators':
                    return 900  # HPO max: 900 (long horizons: 7d, 14d, 30d)
                elif context == 'max_depth':
                    return 8  # HPO max: 8 (long horizons: 7d, 14d, 30d)
                elif context == 'confidence' or context == 'score':
                    return 1.0  # Max confidence/score
                else:
                    return 1e15  # JSON-safe maximum
            else:  # negative infinity
                if context == 'learning_rate':
                    return 0.01  # HPO min: 0.01
                elif context == 'n_estimators':
                    return 100  # HPO min: 100-150 (use 100 as safe min)
                elif context == 'max_depth':
                    return 2  # HPO min: 2 (short horizons: 1d, 3d)
                elif context == 'confidence' or context == 'score':
                    return 0.0  # Min confidence/score
                else:
                    return -1e15  # JSON-safe minimum
        
        # Check for very large or very small values that might cause JSON issues
        # JSON max safe integer: 2^53 - 1 ≈ 9e15, but we use 1e15 for float safety
        if value > 1e15:
            # Clamp to HPO max values (from optuna_hpo_pilot.py)
            if context == 'learning_rate':
                return 0.25  # HPO max
            elif context == 'n_estimators':
                return 900  # HPO max
            elif context == 'max_depth':
                return 8  # HPO max
            elif context == 'confidence' or context == 'score':
                return 1.0
            else:
                return 1e15  # JSON-safe maximum
        elif value < -1e15:
            # Clamp to HPO min values
            if context == 'learning_rate':
                return 0.01  # HPO min
            elif context == 'n_estimators':
                return 100  # HPO min
            elif context == 'max_depth':
                return 2  # HPO min
            elif context == 'confidence' or context == 'score':
                return 0.0
            else:
                return -1e15  # JSON-safe minimum
        
        # Round very small values near zero to avoid precision issues
        if abs(value) < 1e-10:
            return 0.0
        
        return value
    elif isinstance(value, dict):
        # For dicts, try to infer context from key names
        return {k: _sanitize_json_value(v, context=_infer_context(k)) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v, context=context) for v in value]
    else:
        return value


def _find_problematic_values(obj, path="", max_depth=5):
    """Find problematic float values (inf, nan, or very large) in nested structure."""
    if max_depth <= 0:
        return []
    
    problematic = []
    try:
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj) or abs(obj) > 1e15:
                problematic.append(f"{path}: {obj}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                problematic.extend(_find_problematic_values(v, new_path, max_depth - 1))
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                new_path = f"{path}[{i}]"
                problematic.extend(_find_problematic_values(v, new_path, max_depth - 1))
    except Exception as e:
        logger.debug(f"Failed to find problematic values at {path}: {e}")
    return problematic


def _infer_context(key):
    """Infer context from key name for smart default values."""
    key_lower = str(key).lower()
    if 'learning_rate' in key_lower or 'lr' in key_lower or 'eta' in key_lower:
        return 'learning_rate'
    elif 'n_estimator' in key_lower or 'n_tree' in key_lower or 'num_tree' in key_lower:
        return 'n_estimators'
    elif 'max_depth' in key_lower or 'depth' in key_lower:
        return 'max_depth'
    elif 'confidence' in key_lower or 'conf' in key_lower:
        return 'confidence'
    elif 'score' in key_lower or 'r2' in key_lower or 'accuracy' in key_lower:
        return 'score'
    else:
        return None


def broadcast_user_signal(symbol: str, result: dict, flask_app=None) -> None:
    """Broadcast compact user signal via internal API.
    
    ✅ TEMPORARILY DISABLED for debugging WebSocket issues
    """
    logger.warning(f"🚫 broadcast_user_signal called for {symbol} - DISABLED for debugging")
    return  # Disabled for debugging
