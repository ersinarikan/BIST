from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def attach_broadcast_helper(app: Any, socketio: Any) -> None:
    """Attach a safe broadcast_log helper onto the Flask app instance.

    Emits once: prefers 'admin' room; falls back to broadcast on failure.
    """

    def broadcast_log(level: str, message: str, category: str = 'system') -> None:
        payload = {
            'level': level,
            'message': message,
            'category': category,
            'timestamp': datetime.now().isoformat(),
        }
        try:
            socketio.emit('log_update', payload, to='admin')
        except Exception as e:
            logger.debug(f"Failed to emit to admin room: {e}")
            try:
                socketio.emit('log_update', payload)
            except Exception as e2:
                logger.debug(f"Failed to emit broadcast: {e2}")

    try:
        setattr(app, 'broadcast_log', broadcast_log)
    except Exception as e:
        logger.debug(f"Failed to set broadcast_log on app: {e}")
        # As a last resort just ignore; caller can fall back

