import os
import sys
import logging
import math
import json

import requests


logger = logging.getLogger(__name__)


def _in_training_context() -> bool:
    try:
        argv = ' '.join(sys.argv) if hasattr(sys, 'argv') else ''
    except Exception:
        argv = ''
    if 'bulk_train_all.py' in argv:
        return True
    try:
        if os.getenv('DISABLE_LIVE_BROADCAST', '0').lower() in ('1', 'true', 'yes', 'on'):
            return True
    except Exception:
        pass
    return False


def _sanitize_json_value(value, context=None):
    """Sanitize a value for JSON serialization (handle inf, nan, and out-of-range floats).
    
    Instead of setting values to None, we round/clamp them to reasonable ranges:
    - inf → reasonable max/min based on context
    - nan → 0 or context-appropriate default
    - Very large values → JSON-safe maximum (1e15)
    - Very small values → JSON-safe minimum (-1e15)
    
    Args:
        value: Value to sanitize
        context: Optional context hint (e.g., 'learning_rate', 'n_estimators') for smart defaults
    """
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
    except Exception:
        pass
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

    Training contexts are gated off to avoid noisy logs and unintended calls.
    """
    try:
        if _in_training_context():
            logger.debug("Broadcast disabled in training context; skipping")
            return

        # Build compact payload
        visual_evidence = []
        try:
            vis = [p for p in (result.get('patterns') or []) if (p.get('source') == 'VISUAL_YOLO')]
            vis_sorted = sorted(vis, key=lambda p: float(p.get('confidence', 0.0)), reverse=True)
            for p in vis_sorted[:3]:
                visual_evidence.append({
                    'pattern': p.get('pattern'),
                    'confidence': float(p.get('confidence', 0.0))
                })
        except Exception:
            visual_evidence = []

        signal_data = {
            'symbol': symbol,
            'overall_signal': result.get('overall_signal', {}),
            'patterns': result.get('patterns', []),
            'visual': visual_evidence,
            'current_price': result.get('current_price', 0),
            'timestamp': result.get('timestamp')
        }
        payload = {
            'user_id': 1,
            'signal_data': signal_data
        }
        
        # ⚡ CRITICAL FIX: Sanitize payload for JSON serialization
        # Handle inf, nan, and out-of-range float values by rounding/clamping to reasonable ranges
        # This preserves best parameters instead of losing them
        try:
            payload = _sanitize_json_value(payload)
            # Test JSON serialization before sending
            json.dumps(payload)
        except (ValueError, TypeError, OverflowError) as json_err:
            # ⚡ DEBUG: Log the problematic values to find root cause
            logger.warning(f"⚠️ JSON serialization error for {symbol}: {json_err}")
            try:
                # Find problematic values
                problematic = _find_problematic_values(payload)
                if problematic:
                    logger.warning(f"⚠️ Problematic values for {symbol}: {problematic[:200]}")  # Limit log size
            except Exception:
                pass
            logger.warning(f"⚠️ Skipping broadcast for {symbol} due to JSON error")
            return

        # Prefer Flask config for token; fallback to environment
        token = None
        if flask_app is not None:
            try:
                token = flask_app.config.get('INTERNAL_API_TOKEN')
            except Exception:
                token = None
        if not token:
            token = os.getenv('INTERNAL_API_TOKEN')
        if not token:
            logger.warning("INTERNAL_API_TOKEN not configured - skipping live signal broadcast")
            return

        headers = {
            'Content-Type': 'application/json',
            'X-Internal-Token': token
        }
        resp = requests.post(
            'http://localhost:5000/api/internal/broadcast-user-signal',
            json=payload,
            headers=headers,
            timeout=5
        )
        if resp.status_code == 200:
            logger.info(f"🔔 Live signal sent for {symbol}")
        else:
            logger.warning(f"⚠️ Live signal failed for {symbol}: {resp.status_code}")

        # Additionally emit to stock room so all subscribers (watchlist) receive updates
        try:
            if flask_app is not None and hasattr(flask_app, 'socketio'):
                # ⚡ CRITICAL FIX: Sanitize result before socketio broadcast
                sanitized_result = _sanitize_json_value(result)
                try:
                    # Test JSON serialization
                    json.dumps(sanitized_result)
                except (ValueError, TypeError, OverflowError) as json_err:
                    logger.warning(f"⚠️ SocketIO JSON serialization error for {symbol}: {json_err}, skipping socketio broadcast")
                else:
                    flask_app.socketio.emit('pattern_analysis', {
                        'symbol': symbol,
                        'data': sanitized_result,
                        'timestamp': result.get('timestamp')
                    }, room=f'stock_{symbol}')
                    logger.debug(f"Stock room broadcast sent for {symbol}")
        except Exception as e:
            logger.warning(f"Stock room broadcast error for {symbol}: {e}")
    except Exception as e:
        # ⚡ CRITICAL FIX: More specific error message for JSON issues
        error_msg = str(e)
        if 'JSON' in error_msg or 'not JSON compliant' in error_msg or 'Out of range float' in error_msg:
            logger.warning(f"⚠️ Live signal broadcast error for {symbol}: Out of range float values are not JSON compliant")
        else:
            logger.warning(f"⚠️ Live signal broadcast error for {symbol}: {e}")
