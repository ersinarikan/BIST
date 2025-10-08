"""
Configuration Management for BIST Pattern Detection
"""
import os
import logging

logger = logging.getLogger(__name__)

# Üretimde .env kullanılmıyor; yalnızca mevcut ortam değişkenlerine güveniyoruz.
# .env yükleme mantığı bilinçli olarak kaldırıldı (sadelik ve deterministik davranış için).

 
class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
    if not SECRET_KEY:
        import secrets
        SECRET_KEY = secrets.token_urlsafe(32)
        # Avoid printing secrets in production logs; provide minimal guidance instead
        print(
            "⚠️ FLASK_SECRET_KEY is not set. A temporary key was generated for this process. "
            "Set FLASK_SECRET_KEY in the environment for production."
        )
    
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    # Cookie Security
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = os.getenv('SESSION_COOKIE_HTTPONLY', 'True').lower() == 'true'
    SESSION_COOKIE_SAMESITE = os.getenv('SESSION_COOKIE_SAMESITE', 'Strict')
    REMEMBER_COOKIE_SECURE = os.getenv('REMEMBER_COOKIE_SECURE', 'True').lower() == 'true'
    REMEMBER_COOKIE_HTTPONLY = os.getenv('REMEMBER_COOKIE_HTTPONLY', 'True').lower() == 'true'
    PREFERRED_URL_SCHEME = os.getenv('PREFERRED_URL_SCHEME', 'https')
    
    # Database - PostgreSQL only (no SQLite fallback for production)
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is required!")
        raise ValueError("DATABASE_URL must be configured for PostgreSQL connection")
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 120,
        'pool_pre_ping': True
    }
    
    # Redis (for sessions and caching)
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    # Flask-SocketIO message queue (defaults to REDIS_URL)
    SOCKETIO_MESSAGE_QUEUE = os.getenv('SOCKETIO_MESSAGE_QUEUE', os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    # Socket.IO tunables
    SOCKETIO_PING_TIMEOUT = int(os.getenv('SOCKETIO_PING_TIMEOUT', '30'))
    SOCKETIO_PING_INTERVAL = int(os.getenv('SOCKETIO_PING_INTERVAL', '20'))
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600))
    
    # Google OAuth2
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
    GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
    
    # Apple Sign-In
    APPLE_CLIENT_ID = os.getenv('APPLE_CLIENT_ID')
    APPLE_TEAM_ID = os.getenv('APPLE_TEAM_ID')
    APPLE_KEY_ID = os.getenv('APPLE_KEY_ID')
    APPLE_PRIVATE_KEY_PATH = os.getenv('APPLE_PRIVATE_KEY_PATH')
    # Optional: direct client secret if generated externally
    APPLE_CLIENT_SECRET = os.getenv('APPLE_CLIENT_SECRET')
    
    # Email Configuration
    MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USE_SSL = os.getenv('MAIL_USE_SSL', 'False').lower() == 'true'
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv(
        'MAIL_DEFAULT_SENDER',
        'BIST Pattern Detection <noreply@bistpattern.com>',
    )
    
    # Email Verification
    EMAIL_VERIFICATION_SECRET = os.getenv('EMAIL_VERIFICATION_SECRET', SECRET_KEY)
    EMAIL_VERIFICATION_SALT = os.getenv('EMAIL_VERIFICATION_SALT', 'email-verification')
    EMAIL_VERIFICATION_EXPIRES = 3600  # 1 hour

    # Data Collector / Symbol Controls
    AUTO_CREATE_STOCKS = os.getenv('AUTO_CREATE_STOCKS', 'True').lower() == 'true'
    VALIDATE_SYMBOLS = os.getenv('VALIDATE_SYMBOLS', 'False').lower() == 'true'
    ALLOWED_SYMBOL_PATTERN = os.getenv('ALLOWED_SYMBOL_PATTERN', r'^[A-Z0-9]{3,6}$')

    # Yahoo Finance fetch settings
    YF_MAX_RETRIES = int(os.getenv('YF_MAX_RETRIES', '3'))
    YF_BACKOFF_BASE_SECONDS = float(os.getenv('YF_BACKOFF_BASE_SECONDS', '1.5'))
    YF_USER_AGENTS = [
        ua.strip() for ua in os.getenv('YF_USER_AGENTS', '').split('|') if ua.strip()
    ]
    
    # Enhanced Yahoo Finance with curl_cffi settings
    ENABLE_YAHOO_FALLBACK = os.getenv('ENABLE_YAHOO_FALLBACK', 'True').lower() == 'true'
    YF_ENHANCED_ENABLED = os.getenv('YF_ENHANCED_ENABLED', 'True').lower() == 'true'
    YF_MIN_DELAY = float(os.getenv('YF_MIN_DELAY', '2.0'))
    YF_MAX_DELAY = float(os.getenv('YF_MAX_DELAY', '6.0'))
    YF_BURST_DELAY = float(os.getenv('YF_BURST_DELAY', '10.0'))
    YF_BURST_THRESHOLD = int(os.getenv('YF_BURST_THRESHOLD', '5'))
    YF_SESSION_POOL_SIZE = int(os.getenv('YF_SESSION_POOL_SIZE', '5'))
    YF_FALLBACK_TIMEOUT = float(os.getenv('YF_FALLBACK_TIMEOUT', '60.0'))

    # Gunicorn Configuration
    GUNICORN_WORKERS = int(os.getenv('GUNICORN_WORKERS', '1'))
    GUNICORN_WORKER_CLASS = os.getenv('GUNICORN_WORKER_CLASS', 'geventwebsocket.gunicorn.workers.GeventWebSocketWorker')

    # News/RSS Configuration
    NEWS_SOURCES = [
        source.strip() for source in os.getenv('NEWS_SOURCES', '').split(',') if source.strip()
    ]
    NEWS_LOOKBACK_HOURS = int(os.getenv('NEWS_LOOKBACK_HOURS', '24'))
    NEWS_MAX_ITEMS = int(os.getenv('NEWS_MAX_ITEMS', '10'))

    # Pipeline Configuration - SYMBOL_FLOW only
    PIPELINE_MODE = os.getenv('PIPELINE_MODE', 'SYMBOL_FLOW')
    STATUS_GRACE_SECONDS = int(os.getenv('STATUS_GRACE_SECONDS', '5'))
    RUNNING_FLAG_KEY = os.getenv('RUNNING_FLAG_KEY', 'automation:running')
    RUNNING_FLAG_TTL = int(os.getenv('RUNNING_FLAG_TTL', '90'))
    RUNNING_FLAG_HEARTBEAT_SECONDS = int(os.getenv('RUNNING_FLAG_HEARTBEAT_SECONDS', '20'))
    AUTO_START_PIPELINE = os.getenv('AUTO_START_PIPELINE', 'True').lower() == 'true'

    # Machine Learning Cache Paths
    TRANSFORMERS_CACHE = os.getenv('TRANSFORMERS_CACHE', '/opt/bist-pattern/cache/huggingface')
    HF_HOME = os.getenv('HF_HOME', '/opt/bist-pattern/cache/huggingface')

    # Database connection alternatives (backwards compatibility)
    # Keep only env-driven values; no secrets in defaults
    DB_HOST = os.getenv('DB_HOST')
    try:
        DB_PORT = int(os.getenv('DB_PORT', '5432'))
    except Exception:
        DB_PORT = 5432
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    
    # Security: Read password from secure file if available
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    if not DB_PASSWORD:
        try:
            password_file = os.getenv('DB_PASSWORD_FILE', '/opt/bist-pattern/.secrets/db_password')
            if os.path.exists(password_file):
                with open(password_file, 'r') as f:
                    DB_PASSWORD = f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read database password from file: {e}")
            DB_PASSWORD = None

    # Internal API security - REQUIRE token in production
    INTERNAL_API_TOKEN = os.getenv('INTERNAL_API_TOKEN')
    if not INTERNAL_API_TOKEN:
        logger.warning("⚠️ INTERNAL_API_TOKEN not set - internal API will be disabled")
    
    # Localhost access control (default: disabled for security)
    INTERNAL_ALLOW_LOCALHOST = os.getenv('INTERNAL_ALLOW_LOCALHOST', 'False').lower() == 'true'
    
    BIST_API_URL = os.getenv('BIST_API_URL', 'http://localhost:5000')
    PUBLIC_BASE_URL = os.getenv('PUBLIC_BASE_URL', 'http://localhost:5000')
    
    # CORS - more restrictive default
    cors_env = os.getenv('CORS_ORIGINS', '')
    if cors_env:
        CORS_ORIGINS = [o.strip() for o in cors_env.split(',') if o.strip()]
    else:
        # Default to localhost only for development
        CORS_ORIGINS = ['http://localhost:5000', 'https://localhost:5000']

    # Scheduler control
    DISABLE_INTERNAL_SCHEDULER = os.getenv('DISABLE_INTERNAL_SCHEDULER', 'False').lower() == 'true'

    # Priority symbols for quick tests (comma-separated)
    PRIORITY_SYMBOLS = [
        s.strip().upper()
        for s in os.getenv('PRIORITY_SYMBOLS', '').split(',')
        if s.strip()
    ]

    # Collection behavior
    COLLECTION_SCOPE = os.getenv('COLLECTION_SCOPE', 'ALL')  # ALL | PRIORITY | DB_ACTIVE
    COLLECTOR_BATCH_SIZE = int(os.getenv('COLLECTOR_BATCH_SIZE', '50'))
    COLLECTION_PERIOD = os.getenv('COLLECTION_PERIOD', '3mo')
    PRIORITY_PERIOD = os.getenv('PRIORITY_PERIOD', '6mo')
    COLLECTOR_MAX_WORKERS = int(os.getenv('COLLECTOR_MAX_WORKERS', '5'))
    COLLECTOR_DELAY_RANGE = os.getenv('COLLECTOR_DELAY_RANGE', '1,3')
    BATCH_SLEEP_SECONDS = int(float(os.getenv('BATCH_SLEEP_SECONDS', '3')))
    BROADCAST_PROGRESS = os.getenv('BROADCAST_PROGRESS', 'True').lower() == 'true'
    MIN_HISTORY_DAYS = int(os.getenv('MIN_HISTORY_DAYS', '365'))

    # Model/Detection tuning (env via systemd override)
    ENABLE_ENHANCED_ML = os.getenv('ENABLE_ENHANCED_ML', 'True').lower() in ('1', 'true', 'yes')
    ENABLE_BASIC_ML = os.getenv('ENABLE_BASIC_ML', 'True').lower() in ('1', 'true', 'yes')
    ENABLE_FINGPT = os.getenv('ENABLE_FINGPT', 'True').lower() in ('1', 'true', 'yes')
    ENABLE_YOLO = os.getenv('ENABLE_YOLO', 'True').lower() in ('1', 'true', 'yes')
    YOLO_MIN_CONF = float(os.getenv('YOLO_MIN_CONF', '0.33'))
    CONSENSUS_DELTA = int(os.getenv('CONSENSUS_DELTA', '2'))
    VOL_HIGH_THRESHOLD = float(os.getenv('VOL_HIGH_THRESHOLD', '0.10'))
    VOL_LOW_THRESHOLD = float(os.getenv('VOL_LOW_THRESHOLD', '0.03'))
    DATA_CACHE_TTL = int(os.getenv('DATA_CACHE_TTL', '60'))

    # Pattern detector tuning
    PATTERN_CACHE_TTL = int(os.getenv('PATTERN_CACHE_TTL', '300'))  # seconds
    PATTERN_DATA_DAYS = int(os.getenv('PATTERN_DATA_DAYS', '365'))  # default lookback

    # Scheduler/loop timing
    FULL_CYCLE_SLEEP_SECONDS = int(float(os.getenv('FULL_CYCLE_SLEEP_SECONDS', '300')))
    SYMBOL_SLEEP_SECONDS = float(os.getenv('SYMBOL_SLEEP_SECONDS', '0.3'))

    # Development auth bypass (for UI integration before OAuth)
    DEV_AUTH_BYPASS = os.getenv('DEV_AUTH_BYPASS', 'False').lower() == 'true'

    # Admin account - must be provided via environment
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')
    ADMIN_DEFAULT_PASSWORD = os.getenv('ADMIN_DEFAULT_PASSWORD')

    # Paths for models and logs
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', '/opt/bist-pattern/yolo/patterns_all_v1.pt')
    ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', '/opt/bist-pattern/.cache/enhanced_ml_models')
    CATBOOST_TRAIN_DIR = os.getenv('CATBOOST_TRAIN_DIR', '/opt/bist-pattern/.cache/catboost')
    BIST_LOG_PATH = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        return None

 
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
 
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    @staticmethod
    def init_app(app):
        # Delegate to base init (staticmethod) for compatibility
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': ProductionConfig,
}
