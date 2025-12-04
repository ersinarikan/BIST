from flask_login import LoginManager
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_wtf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy

try:
    from models import db as models_db  # reuse the project's singleton
except Exception:
    models_db = None

# Reuse existing db instance if present to avoid multiple bindings
if models_db is not None:
    db = models_db
else:
    db = SQLAlchemy()

login_manager = LoginManager()
migrate = Migrate()
socketio = SocketIO()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")


def init_extensions(app):
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    # Resolve Socket.IO message queue for multi-worker support (Redis recommended)
    message_queue = (
        app.config.get("SOCKETIO_MESSAGE_QUEUE")
        or app.config.get("REDIS_URL")
        or None
    )
    cors_origins = app.config.get("CORS_ORIGINS") or "*"
    socketio.init_app(
        app,
        async_mode="gevent",
        message_queue=message_queue,
        cors_allowed_origins=cors_origins,
        logger=False,
        engineio_logger=False,
        ping_timeout=app.config.get("SOCKETIO_PING_TIMEOUT", 30),
        ping_interval=app.config.get("SOCKETIO_PING_INTERVAL", 20),
    )
    # Expose socketio and broadcaster on app for backward compatibility
    app.socketio = socketio

    # ✅ CRITICAL FIX: Disable broadcast_log - it causes parse errors and unnecessary WebSocket traffic
    def broadcast_log(level, message, category='system'):
        """Broadcast disabled - use API endpoints instead"""
        # Also log to stdout for journalctl visibility
        import logging
        logger = logging.getLogger(f'broadcast.{category}')
        log_msg = f"[{level}] {message}"
        
        if level.upper() == 'ERROR':
            logger.error(log_msg)
        elif level.upper() == 'WARNING':
            logger.warning(log_msg)
        elif level.upper() == 'SUCCESS':
            logger.info(f"✅ {message}")
        else:
            logger.info(log_msg)
        
        # Force flush for immediate journalctl visibility
        import sys
        sys.stdout.flush()

    app.broadcast_log = broadcast_log

    csrf.init_app(app)
    try:
        if hasattr(app, "view_functions") and "socketio" in app.view_functions:
            csrf.exempt(app.view_functions["socketio"])
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Failed to exempt socketio from CSRF: {e}")
    limiter.init_app(app)
