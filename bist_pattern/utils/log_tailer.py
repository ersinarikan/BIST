from __future__ import annotations
import os
import time
import threading
import logging
from datetime import datetime
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def start_gunicorn_log_tailer(app: Any, socketio: Any) -> None:
    """Best-effort tailer for gunicorn logs; emits to websocket.

    Controlled by ENV: ENABLE_GUNICORN_TAIL=true
    """
    # ⚡ DISABLED TEMPORARILY: To debug WebSocket stability
    return

    try:
        enabled = str(os.getenv('ENABLE_GUNICORN_TAIL', 'False')).lower() == 'true'
        if not enabled:
            return

        log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        files: Iterable[str] = (
            os.path.join(log_dir, 'gunicorn_error.log'),
            os.path.join(log_dir, 'gunicorn_access.log'),
        )

        def _tail(path: str, category: str) -> None:
            try:
                if not os.path.exists(path):
                    return
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(0, os.SEEK_END)
                    while True:
                        line = f.readline()
                        if not line:
                            time.sleep(1)
                            continue
                        # ✅ CRITICAL FIX: Sanitize log payload to prevent parse errors
                        try:
                            from bist_pattern.core.broadcaster import _sanitize_json_value
                            import json
                            payload = {
                                'level': 'INFO',
                                'message': line.strip()[:1000],  # Limit message length
                                'category': category[:50],  # Limit category length
                                'timestamp': datetime.now().isoformat(),
                            }
                            sanitized_payload = _sanitize_json_value(payload)
                            json.dumps(sanitized_payload)  # Test serialization
                            # ✅ FIX: Only send to admin room, use 'room' parameter not 'to'
                            socketio.emit('log_update', sanitized_payload, room='admin')
                        except Exception as e:
                            logger.debug(f"Failed to emit log update: {e}")
                            # Log tailer is best-effort
            except Exception as e:
                logger.debug(f"Failed to tail log file {path}: {e}")

        for fpath in files:
            t = threading.Thread(target=_tail, args=(fpath, 'gunicorn'), daemon=True)
            t.start()
    except Exception as e:
        logger.debug(f"Failed to start gunicorn log tailer: {e}")
