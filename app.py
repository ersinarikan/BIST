import os
from datetime import datetime
from flask import Flask, jsonify, request, redirect, url_for
from flask_login import LoginManager, current_user
from flask_mail import Mail
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit, join_room, leave_room
from config import config
from models import db, User
from bist_pattern.core.config_manager import ConfigManager
from bist_pattern.utils.error_handler import ErrorHandler
import logging
import time
import threading
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect

# Logger setup
# Include PID and module name in logs for clearer provenance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d %(name)s:%(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# System availability flags - module başında tanımla
try:
    import advanced_patterns  # noqa: F401
    ADVANCED_PATTERNS_AVAILABLE = True
except ImportError:
    ADVANCED_PATTERNS_AVAILABLE = False
    logger.warning("⚠️ Advanced patterns modülü yüklenemedi")

try:
    import visual_pattern_detector  # noqa: F401
    VISUAL_PATTERNS_AVAILABLE = True
except ImportError:
    VISUAL_PATTERNS_AVAILABLE = False
    logger.warning("⚠️ Visual patterns modülü yüklenemedi")

try:
    import ml_prediction_system  # noqa: F401
    ML_PREDICTION_AVAILABLE = True
except ImportError:
    ML_PREDICTION_AVAILABLE = False
    logger.warning("⚠️ ML Prediction modülü yüklenemedi")

try:
    from working_automation import get_working_automation_pipeline  # noqa: F401
    AUTOMATED_PIPELINE_AVAILABLE = True
except ImportError:
    AUTOMATED_PIPELINE_AVAILABLE = False
    logger.warning("⚠️ Automated Pipeline modülü yüklenemedi")

# Gevent availability check
try:
    import gevent  # noqa: F401
    GEVENT_AVAILABLE = True
except ImportError:
    GEVENT_AVAILABLE = False
    logger.warning("⚠️ Gevent modülü yüklenemedi")

# ==========================================
# EXTENSIONS INITIALIZATION
# ==========================================
# Use the shared SQLAlchemy instance from models
login_manager = LoginManager()
mail = Mail()
migrate = Migrate()
socketio = SocketIO()
csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")

# ==========================================
# FLASK APP FACTORY & CONFIG
# ==========================================


def create_app(config_name='default'):
    """Flask app factory"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Map INTERNAL_API_TOKEN from environment if present (keeps existing config fallback)
    try:
        # ✅ FIX: Use ConfigManager for consistent config access
        app.config['INTERNAL_API_TOKEN'] = ConfigManager.get('INTERNAL_API_TOKEN', app.config.get('INTERNAL_API_TOKEN'))
    except Exception as e:
        ErrorHandler.handle(e, 'app_init_internal_token', level='debug')
        pass

    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    migrate.init_app(app, db)
    # Configure optional cross-process message queue for Socket.IO (required for multi-worker)
    try:
        # ✅ FIX: Use ConfigManager for consistent config access
        mq_url = ConfigManager.get('SOCKETIO_MESSAGE_QUEUE', app.config.get('SOCKETIO_MESSAGE_QUEUE'))
    except Exception as e:
        ErrorHandler.handle(e, 'app_init_socketio_mq', level='debug')
        mq_url = None

    socketio.init_app(
        app,
        async_mode='gevent',
        cors_allowed_origins=app.config.get('CORS_ORIGINS', '*'),
        logger=False,
        engineio_logger=False,
        ping_timeout=90,
        ping_interval=45,
        message_queue=mq_url,
    )
    if not mq_url:
        logger.warning("Socket.IO message_queue not configured; cross-process emits may not reach clients. Set SOCKETIO_MESSAGE_QUEUE (e.g., redis://localhost:6379/0)")
    csrf.init_app(app)
    # Exempt Socket.IO transport (polling POSTs) from CSRF to prevent 400s on /socket.io/
    try:
        if hasattr(app, 'view_functions') and 'socketio' in app.view_functions:
            csrf.exempt(app.view_functions['socketio'])
    except Exception as _csrf_socketio_err:
        logger.info(f"CSRF exempt for socketio failed: {_csrf_socketio_err}")
    limiter.init_app(app)

    # Register patterns blueprint (exposes /api/visual-analysis)
    try:
        from blueprints.api_patterns import api_patterns as _api_patterns  # type: ignore
        app.register_blueprint(_api_patterns)
    except Exception as _bp_err:
        logger.warning(f"api_patterns blueprint registration skipped: {_bp_err}")

    # Template auto-reload to avoid stale cached templates in production
    try:
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.jinja_env.auto_reload = True
    except Exception:
        pass

    # Optional: tail gunicorn logs and broadcast to Live System Logs (read-only)
    def _start_log_tailer():
        try:
            # ✅ FIX: Use ConfigManager for consistent config access
            enabled = str(ConfigManager.get('ENABLE_GUNICORN_TAIL', 'False')).lower() == 'true'
            if not enabled:
                return
            # ✅ FIX: Use ConfigManager for consistent config access
            log_dir = ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            files = [
                os.path.join(log_dir, 'gunicorn_error.log'),
                os.path.join(log_dir, 'gunicorn_access.log')
            ]

            def _tail(path: str, category: str):
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
                            try:
                                app.broadcast_log('INFO', line.strip(), category)
                            except Exception:
                                pass
                except Exception:
                    pass

            for fpath in files:
                t = threading.Thread(target=_tail, args=(fpath, 'gunicorn'), daemon=True)
                t.start()
        except Exception:
            pass

    _start_log_tailer()

    # Debug: Log selected routes to verify blueprint registration
    try:
        routes_to_check = ('/api/pattern-summary', '/api/pattern-analysis', '/api/visual-analysis', '/api/internal/visual-analysis')
        for rule in app.url_map.iter_rules():
            text = str(rule)
            if any(seg in text for seg in routes_to_check):
                logger.info(f"ROUTE_REGISTERED {text}")
    except Exception as _route_log_err:
        logger.debug(f"route log skipped: {_route_log_err}")

    # Graceful shutdown for continuous loop
    # Signal handling moved to gevent-compatible version
    try:
        import signal
        
        def _graceful_stop(signum, frame):
            """Gevent-safe graceful shutdown handler"""
            try:
                # Use gevent-safe spawn to avoid BlockingIOError on signal wakeup fd
                if GEVENT_AVAILABLE:
                    from gevent import spawn as gevent_spawn

                    def _shutdown_task():
                        try:
                            from working_automation import get_working_automation_pipeline
                            pipeline = get_working_automation_pipeline()
                            if getattr(pipeline, 'is_running', False):
                                pipeline.stop_scheduler()
                                logger.info('Graceful shutdown: pipeline stopped')
                        except Exception as _e:
                            logger.warning(f'Graceful shutdown failed: {_e}')

                    gevent_spawn(_shutdown_task)
                else:
                    from working_automation import get_working_automation_pipeline
                    pipeline = get_working_automation_pipeline()
                    if getattr(pipeline, 'is_running', False):
                        pipeline.stop_scheduler()
                        logger.info('Graceful shutdown: pipeline stopped')
            except Exception as _e:
                logger.warning(f'Signal handler error: {_e}')
        
        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _graceful_stop)
    except Exception as e:
        logger.warning(f'Signal handler setup failed: {e}')

    # SocketIO was already initialized above with async_mode and CORS; avoid duplicate initialization
    
    # Auto-start automation pipeline on application startup
    def _auto_start_automation():
        """Automatically start automation pipeline when service starts"""
        try:
            if not AUTOMATED_PIPELINE_AVAILABLE:
                logger.info("⚠️ Automation pipeline not available - skipping auto-start")
                return
            
            # Check if auto-start is enabled (default: True)
            # ✅ FIX: Use ConfigManager for consistent config access
            auto_start = str(ConfigManager.get('AUTO_START_CYCLE', 'True')).lower() in ('true', '1', 'yes', 'on')
            if not auto_start:
                logger.info("⚠️ AUTO_START_CYCLE disabled - pipeline will not auto-start")
                return
            
            # Delay startup to allow app to fully initialize
            import threading

            def _delayed_start():
                try:
                    import time
                    time.sleep(3)  # Wait 3 seconds for app to be ready
                    
                    with app.app_context():
                        from working_automation import get_working_automation_pipeline
                        pipeline = get_working_automation_pipeline()
                        
                        if pipeline and not getattr(pipeline, 'is_running', False):
                            success = pipeline.start_scheduler()
                            if success:
                                logger.info("✅ Automation pipeline auto-started on service startup")
                            else:
                                logger.warning("⚠️ Automation pipeline auto-start failed")
                        else:
                            logger.info("ℹ️ Automation pipeline already running")
                except Exception as e:
                    logger.error(f"❌ Automation pipeline auto-start error: {e}")
            
            thread = threading.Thread(target=_delayed_start, daemon=True, name='AutoStartCycle')
            thread.start()
            logger.info("🔄 Automation pipeline auto-start scheduled (3s delay)")
            
        except Exception as e:
            logger.warning(f"Auto-start automation setup failed: {e}")
    
    _auto_start_automation()
    
    # Login Manager (use the globally initialized instance)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'  # Updated for blueprint routing

    # RBAC helpers
    def is_admin(user) -> bool:
        try:
            # Prefer explicit role flag, fallback to username/email for backward compatibility
            if user and getattr(user, 'role', None) == 'admin':
                return True
            admin_email = app.config.get('ADMIN_EMAIL')
            return bool(user and (getattr(user, 'username', None) == 'systemadmin' or (admin_email and getattr(user, 'email', None) == admin_email)))
        except Exception:
            return False

    from functools import wraps

    def admin_required(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                if current_user.is_authenticated and is_admin(current_user):
                    return fn(*args, **kwargs)
            except Exception:
                pass
            if request.path.startswith('/api/'):
                return jsonify({'status': 'unauthorized'}), 401
            return redirect(url_for('login'))
        return wrapper
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Mail/Migrate already initialized via init_app above
    
    # Optional OAuth setup
    oauth = None
    try:
        from authlib.integrations.flask_client import OAuth
        oauth = OAuth(app)
        if app.config.get('GOOGLE_CLIENT_ID') and app.config.get('GOOGLE_CLIENT_SECRET'):
            oauth.register(
                name='google',
                client_id=app.config.get('GOOGLE_CLIENT_ID'),
                client_secret=app.config.get('GOOGLE_CLIENT_SECRET'),
                server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
                client_kwargs={'scope': 'openid email profile'}
            )
        # Apple Sign-In (optional)
        if app.config.get('APPLE_CLIENT_ID'):
            oauth.register(
                name='apple',
                client_id=app.config.get('APPLE_CLIENT_ID'),
                client_secret=app.config.get('APPLE_CLIENT_SECRET'),
                server_metadata_url='https://appleid.apple.com/.well-known/openid-configuration',
                client_kwargs={'scope': 'name email'}
            )
    except Exception as _oauth_err:
        logger.info(f"OAuth not initialized: {_oauth_err}")

    # Security: CSRF, Rate limit, CORS (basic)
    # CSRF is already initialized via csrf.init_app(app) above

    # Limiter already attached via init_app; default limits can be configured via env if needed

    def internal_route(f):
        """Internal-only route: require INTERNAL_API_TOKEN or allow localhost if explicitly enabled."""
        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                configured_token = app.config.get('INTERNAL_API_TOKEN')
                header_token = request.headers.get('X-Internal-Token')
                # Allow localhost only if explicitly enabled via env or config
                # Default True to preserve local internal communications unless explicitly disabled
                # ✅ FIX: Use ConfigManager for consistent config access
                allow_localhost = str(ConfigManager.get('INTERNAL_ALLOW_LOCALHOST', str(app.config.get('INTERNAL_ALLOW_LOCALHOST', 'True')))).lower() == 'true'
                remote_ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
                is_local = remote_ip in ('127.0.0.1', '::1', 'localhost')

                # Prefer token auth
                if configured_token and header_token == configured_token:
                    return f(*args, **kwargs)
                # Fallback: localhost-only if allowed
                if allow_localhost and is_local:
                    return f(*args, **kwargs)
            except Exception:
                pass
            return jsonify({'status': 'forbidden'}), 403

        # Exempt from CSRF and rate limit after wrapping
        try:
            limiter.exempt(wrapper)
        except Exception:
            pass
        try:
            csrf.exempt(wrapper)
        except Exception:
            pass
        return wrapper

    try:
        from flask_cors import CORS
        cors_origins = app.config.get('CORS_ORIGINS') or []
        if cors_origins:
            CORS(app, origins=cors_origins, supports_credentials=True)
    except Exception as _cors_err:
        logger.warning(f"CORS init failed: {_cors_err}")

    # CSRF Configuration - exempt API endpoints
    app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # Disable CSRF globally, enable per-route as needed
    
    @app.before_request
    def _api_csrf_exempt():
        """API endpoints don't need CSRF (they use tokens/auth instead)."""
        pass  # CSRF disabled globally via config

    # ==========================================
    # BLUEPRINT REGISTRATION  
    # ==========================================
    # All HTTP routes moved to blueprints for better modularity
    try:
        from bist_pattern.blueprints.register_all import register_all_blueprints
        register_all_blueprints(app, csrf)
        logger.info("✅ All blueprints registered successfully")
    except Exception as e:
        logger.error(f"❌ Blueprint registration error: {e}")


    # ==========================================
    # REALTIME WEBSOCKET HANDLERS
    # ==========================================



    @socketio.on('connect')
    def handle_connect(auth):
        logger.info(f"🔗 Client connected: {request.sid}")
        emit('status', {
            'message': 'Connected to BIST AI System', 
            'timestamp': datetime.now().isoformat(),
            'connection_id': request.sid
        })
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"❌ Client disconnected: {request.sid}")
    
    @socketio.on('join_admin')
    def handle_join_admin():
        join_room('admin')
        logger.info(f"👤 Client joined admin room: {request.sid}")
        emit('room_joined', {'room': 'admin', 'message': 'Admin dashboard connected'})
    
    @socketio.on('join_user')
    def handle_join_user(data):
        user_id = data.get('user_id', 'anonymous')
        join_room(f'user_{user_id}')
        logger.info(f"👤 Client joined user room: {request.sid} -> user_{user_id}")
        emit('room_joined', {'room': f'user_{user_id}', 'message': 'User interface connected'})
    
    @socketio.on('subscribe_stock')
    def handle_subscribe_stock(data):
        symbol = data.get('symbol', '').upper()
        if symbol:
            join_room(f'stock_{symbol}')
            logger.info(f"📈 Client subscribed to {symbol}: {request.sid}")
            emit('subscription_confirmed', {'symbol': symbol, 'message': f'Subscribed to {symbol} updates'})
    
    @socketio.on('unsubscribe_stock')
    def handle_unsubscribe_stock(data):
        symbol = data.get('symbol', '').upper()
        if symbol:
            leave_room(f'stock_{symbol}')
            logger.info(f"📉 Client unsubscribed from {symbol}: {request.sid}")
            emit('subscription_removed', {'symbol': symbol, 'message': f'Unsubscribed from {symbol}'})
    
    @socketio.on('request_pattern_analysis')
    def handle_pattern_request(data):
        symbol = data.get('symbol', '').upper()
        if symbol:
            try:
                # Cache-only: do not compute here
                from bist_pattern.core.cache import cache_get as _cache_get  # type: ignore
                cache_key = f"pattern_analysis:{symbol}"
                result = None
                try:
                    result = _cache_get(cache_key)
                except Exception:
                    result = None
                if not result:
                    result = {'symbol': symbol, 'status': 'pending'}
                # Send to requesting client
                emit('pattern_analysis', {
                    'symbol': symbol,
                    'data': result,
                    'timestamp': datetime.now().isoformat()
                })
                # Also broadcast to stock room for other subscribers
                socketio.emit('pattern_analysis', {
                    'symbol': symbol,
                    'data': result,
                    'timestamp': datetime.now().isoformat()
                }, room=f'stock_{symbol}')  # type: ignore[call-arg]
                logger.info(f"📊 Pattern analysis (cache-only) sent for {symbol} to {request.sid} and stock room")
            except Exception as e:
                emit('error', {'message': f'Pattern analysis failed for {symbol}: {str(e)}'})
                logger.error(f"Pattern analysis error for {symbol}: {e}")
    
    # Real-time log broadcasting function
    def broadcast_log(level, message, category='system'):
        socketio.emit('log_update', {
            'level': level,
            'message': message,
            'category': category,
            'timestamp': datetime.now().isoformat()
        }, room='admin')  # type: ignore[call-arg]
    
    # Store socketio instance globally for background tasks
    app.socketio = socketio
    app.broadcast_log = broadcast_log
    
    # ==========================================
    # CALIBRATION DEFAULT STATE (BYPASS=TRUE)
    # ==========================================
    try:
        # If BYPASS_ISOTONIC_CALIBRATION is truthy (default '1'), persist a closed state to file
        # ✅ FIX: Use ConfigManager for consistent config access
        _bypass_env = str(ConfigManager.get('BYPASS_ISOTONIC_CALIBRATION', '1')).strip().lower() in ('1', 'true', 'yes', 'on')
        logger.info(f"🔧 Calibration startup: BYPASS_ISOTONIC_CALIBRATION={ConfigManager.get('BYPASS_ISOTONIC_CALIBRATION', '1')}, enabled={_bypass_env}")
        
        # ✅ NOTE: We don't force bypass based on skipped_horizons anymore
        # Reason: Online adjustment should be globally enabled (toggle open), but per-horizon
        # adjustment is already skipped in pattern_detector.py if horizon has insufficient data.
        # This allows adjustment to work for horizons WITH data (1d, 3d) even if other
        # horizons (7d, 14d, 30d) have insufficient data.
        # 
        # Per-horizon skipping is handled in pattern_detector.py._get_empirical_confidence_adjustment()
        # by checking skipped_horizons from param_store.json
        
        if _bypass_env:
            import json as _json
            from datetime import datetime as _dt
            # ✅ FIX: Use ConfigManager for consistent config access
            _log_dir = ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            os.makedirs(_log_dir, exist_ok=True)
            _state_path = os.path.join(_log_dir, 'calibration_state.json')
            # Merge with existing content if present, but force bypass=true
            cur = {}
            try:
                if os.path.exists(_state_path):
                    with open(_state_path, 'r') as rf:
                        cur = _json.load(rf) or {}
            except Exception:
                cur = {}
            cur['bypass'] = True
            # Keep previous penalty_factor if any, else provide conservative default
            if 'penalty_factor' not in cur or cur.get('penalty_factor') is None:
                cur['penalty_factor'] = 0.95
            cur['updated_at'] = _dt.now().isoformat()
            _tmp = _state_path + '.tmp'
            try:
                with open(_tmp, 'w') as wf:
                    wf.write(_json.dumps(cur, ensure_ascii=False))
                os.replace(_tmp, _state_path)
                logger.info(f"✅ Calibration bypass persisted to {_state_path}")
            except Exception as e:
                logger.warning(f"⚠️ Atomic write failed, trying fallback: {e}")
                # Non-atomic fallback
                with open(_state_path, 'w') as wf:
                    wf.write(_json.dumps(cur, ensure_ascii=False))
                logger.info(f"✅ Calibration bypass persisted (fallback) to {_state_path}")
    except Exception as e:
        logger.error(f"❌ Calibration startup error: {e}")
    
    return app

# Duplike flag tanımlamaları kaldırıldı - bunlar artık module başında tanımlı

# Global pattern detector instance
_pattern_detector = None


def get_pattern_detector():
    """Pattern detector singleton'ını döndür"""
    global _pattern_detector
    if _pattern_detector is None:
        from pattern_detector import HybridPatternDetector
        _pattern_detector = HybridPatternDetector()
    return _pattern_detector


def get_pipeline_with_context():
    """Pipeline'ı app context ile döndür"""
    if AUTOMATED_PIPELINE_AVAILABLE:
        return get_working_automation_pipeline()  # type: ignore
    return None

# Flask app instance
app = create_app(os.getenv('FLASK_ENV', 'default'))
# socketio zaten factory içinde init edildi; burada yeniden atamaya gerek yok

if __name__ == '__main__':
    # Environment variables'dan değerleri al
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # SocketIO ile çalıştır
    socketio.run(app, host=host, port=port, debug=debug)
