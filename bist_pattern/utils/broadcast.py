from __future__ import annotations

from datetime import datetime
from typing import Any


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
        except Exception:
            try:
                socketio.emit('log_update', payload)
            except Exception:
                pass

    try:
        setattr(app, 'broadcast_log', broadcast_log)
    except Exception:
        # As a last resort just ignore; caller can fall back
        pass

