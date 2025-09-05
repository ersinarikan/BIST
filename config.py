"""
Configuration Management for BIST Pattern Detection
"""
import os

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
        print("⚠️ FLASK_SECRET_KEY is not set. A temporary key was generated for this process. Set FLASK_SECRET_KEY in the environment for production.")
    
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    # Cookie Security
    SESSION_COOKIE_SECURE = os.getenv('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = os.getenv('SESSION_COOKIE_HTTPONLY', 'True').lower() == 'true'
    SESSION_COOKIE_SAMESITE = os.getenv('SESSION_COOKIE_SAMESITE', 'Strict')
    REMEMBER_COOKIE_SECURE = os.getenv('REMEMBER_COOKIE_SECURE', 'True').lower() == 'true'
    REMEMBER_COOKIE_HTTPONLY = os.getenv('REMEMBER_COOKIE_HTTPONLY', 'True').lower() == 'true'
    PREFERRED_URL_SCHEME = os.getenv('PREFERRED_URL_SCHEME', 'https')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/bist_pattern_db')
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 120,
        'pool_pre_ping': True
    }
    
    # Redis (for sessions and caching)
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
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
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER', 'BIST Pattern Detection <noreply@bistpattern.com>')
    
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
    YF_USER_AGENTS = [ua.strip() for ua in os.getenv('YF_USER_AGENTS', '').split('|') if ua.strip()]

    # Internal API security
    INTERNAL_API_TOKEN = os.getenv('INTERNAL_API_TOKEN', None)
    BIST_API_URL = os.getenv('BIST_API_URL', 'http://localhost:5000')
    # CORS
    CORS_ORIGINS = [o.strip() for o in os.getenv('CORS_ORIGINS', '').split(',') if o.strip()]

    # Scheduler control
    DISABLE_INTERNAL_SCHEDULER = os.getenv('DISABLE_INTERNAL_SCHEDULER', 'False').lower() == 'true'

    # Priority symbols for quick tests (comma-separated)
    PRIORITY_SYMBOLS = [s.strip().upper() for s in os.getenv('PRIORITY_SYMBOLS', '').split(',') if s.strip()]

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

    # Pattern detector tuning
    PATTERN_CACHE_TTL = int(os.getenv('PATTERN_CACHE_TTL', '300'))  # seconds
    PATTERN_DATA_DAYS = int(os.getenv('PATTERN_DATA_DAYS', '365'))  # default lookback

    # Development auth bypass (for UI integration before OAuth)
    DEV_AUTH_BYPASS = os.getenv('DEV_AUTH_BYPASS', 'False').lower() == 'true'

    # Admin account
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@bistpattern.com')
    ADMIN_DEFAULT_PASSWORD = os.getenv('ADMIN_DEFAULT_PASSWORD', '5ex5CHAN*')
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    @classmethod
    def init_app(cls, app):
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
    'default': ProductionConfig
}
