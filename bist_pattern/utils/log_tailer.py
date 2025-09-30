from __future__ import annotations
import os
import time
import threading
from datetime import datetime
from typing import Any, Iterable


def start_gunicorn_log_tailer(app: Any, socketio: Any) -> None:
    """Best-effort tailer for gunicorn logs; emits to websocket.

    Controlled by ENV: ENABLE_GUNICORN_TAIL=true
    """

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
                        payload = {
                            'level': 'INFO',
                            'message': line.strip(),
                            'category': category,
                            'timestamp': datetime.now().isoformat(),
                        }
                        try:
                            socketio.emit('log_update', payload, to='admin')
                            socketio.emit('log_update', payload)
                        except Exception:
                            pass
            except Exception:
                pass

        for fpath in files:
            t = threading.Thread(target=_tail, args=(fpath, 'gunicorn'), daemon=True)
            t.start()
    except Exception:
        pass


