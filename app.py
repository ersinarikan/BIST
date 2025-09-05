import os
from datetime import datetime, timedelta
from decimal import Decimal
from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, url_for, make_response
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from flask_mail import Mail
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit, join_room, leave_room
from config import config
from models import db, User, Stock, StockPrice
import logging
import time
import threading
import time
import json
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
import threading

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System availability flags - module başında tanımla
try:
    from advanced_patterns import AdvancedPatternDetector
    ADVANCED_PATTERNS_AVAILABLE = True
except ImportError:
    ADVANCED_PATTERNS_AVAILABLE = False
    logger.warning("⚠️ Advanced patterns modülü yüklenemedi")

try:
    from visual_pattern_detector import get_visual_pattern_system
    VISUAL_PATTERNS_AVAILABLE = True
except ImportError:
    VISUAL_PATTERNS_AVAILABLE = False
    logger.warning("⚠️ Visual patterns modülü yüklenemedi")

try:
    from ml_prediction_system import get_ml_prediction_system
    ML_PREDICTION_AVAILABLE = True
except ImportError:
    ML_PREDICTION_AVAILABLE = False
    logger.warning("⚠️ ML Prediction modülü yüklenemedi")

try:
    from scheduler import get_automated_pipeline
    AUTOMATED_PIPELINE_AVAILABLE = True
except ImportError:
    AUTOMATED_PIPELINE_AVAILABLE = False
    logger.warning("⚠️ Automated Pipeline modülü yüklenemedi")

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

def internal_route(f):
    """Decorator to exempt internal routes from rate limiting."""
    limiter.exempt(f)
    return f

# ==========================================
# FLASK APP FACTORY & CONFIG
# ==========================================
def create_app(config_name='default'):
    """Flask app factory"""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(
        app,
        async_mode='gevent',
        cors_allowed_origins=app.config.get('CORS_ORIGINS', '*'),
        logger=False,
        engineio_logger=False,
        ping_timeout=30,
        ping_interval=20,
    )
    csrf.init_app(app)
    # Exempt Socket.IO transport (polling POSTs) from CSRF to prevent 400s on /socket.io/
    try:
        if hasattr(app, 'view_functions') and 'socketio' in app.view_functions:
            csrf.exempt(app.view_functions['socketio'])
    except Exception as _csrf_socketio_err:
        logger.info(f"CSRF exempt for socketio failed: {_csrf_socketio_err}")
    limiter.init_app(app)

    # Template auto-reload to avoid stale cached templates in production
    try:
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.jinja_env.auto_reload = True
    except Exception:
        pass

    # Optional: tail gunicorn logs and broadcast to Live System Logs (read-only)
    def _start_log_tailer():
        try:
            enabled = str(os.getenv('ENABLE_GUNICORN_TAIL', 'False')).lower() == 'true'
            if not enabled:
                return
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
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

    # Graceful shutdown for continuous loop
    try:
        import signal
        def _graceful_stop(signum, frame):
            try:
                from scheduler import get_automated_pipeline
                pipeline = get_automated_pipeline()
                if getattr(pipeline, 'is_running', False):
                    pipeline.stop_scheduler()
                    app.logger.info('Graceful shutdown: pipeline stopped')
            except Exception as _e:
                app.logger.warning(f'Graceful shutdown failed: {_e}')
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _graceful_stop)
    except Exception:
        pass

    # SocketIO was already initialized above with async_mode and CORS; avoid duplicate initialization
    
    # Login Manager (use the globally initialized instance)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

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
    try:
        from flask_wtf.csrf import generate_csrf
    except Exception as _csrf_err:
        logger.warning(f"CSRF import failed: {_csrf_err}")

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
                allow_localhost = str(os.getenv('INTERNAL_ALLOW_LOCALHOST', str(app.config.get('INTERNAL_ALLOW_LOCALHOST', 'True')))).lower() == 'true'
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

    # Routes
    @app.before_request
    def _maybe_exempt_api_from_csrf():
        try:
            # Exempt JSON APIs; forms (like /login) remain protected
            if request.path.startswith('/api/'):
                setattr(request, 'csrf_processing_exempt', True)
            # Also exempt Socket.IO polling endpoints to avoid 400 on POST /socket.io/
            if request.path.startswith('/socket.io'):
                setattr(request, 'csrf_processing_exempt', True)
        except Exception:
            pass
    @app.route('/')
    def index():
        try:
            if current_user.is_authenticated:
                return redirect(url_for('dashboard' if is_admin(current_user) else 'user_dashboard'))
        except Exception:
            pass
        return redirect(url_for('login'))

    @app.route('/api')
    def api_info():
        return jsonify({
            "message": "BIST Pattern Detection API",
            "status": "running",
            "version": "2.2.0",
            "database": "PostgreSQL",
            "features": ["Real-time Data", "Yahoo Finance", "Scheduler", "Dashboard", "Automation"]
        })

    @app.route('/dashboard')
    @login_required
    @admin_required
    def dashboard():
        """Real-time monitoring dashboard"""
        try:
            # Check if template exists
            template_path = os.path.join(app.template_folder, 'dashboard.html')
            if not os.path.exists(template_path):
                return jsonify({
                    'error': 'Dashboard template not found',
                    'message': 'Real-time dashboard is being deployed',
                    'status': 'template_missing'
                }), 404
            # No-cache response to always serve latest template and scripts
            resp = make_response(render_template('dashboard.html'))
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
            return resp
        except Exception as e:
            logger.error(f"Dashboard render error: {e}")
            return jsonify({
                'error': 'Dashboard render failed',
                'message': str(e),
                'status': 'render_error'
            }), 500

    # ================================
    # Auth routes (Email + Google OAuth)
    # ================================

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        try:
            if request.method == 'GET':
                return render_template('login.html', google_enabled=bool(oauth and getattr(oauth, 'google', None)), apple_enabled=bool(oauth and getattr(oauth, 'apple', None)), csrf_token=generate_csrf())

            # POST: email/password
            email = (request.form.get('email') or '').strip().lower()
            password = request.form.get('password') or ''
            if not email or not password:
                return render_template('login.html', error='E-posta ve şifre gerekli', google_enabled=bool(oauth and getattr(oauth, 'google', None)))
            user = User.query.filter_by(email=email).first()
            if user and user.check_password(password):
                # Update last login time and IP
                try:
                    user.last_login = datetime.now()
                    user.last_login_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                    db.session.commit()
                except Exception:
                    try:
                        db.session.rollback()
                    except Exception:
                        pass
                login_user(user)
                return redirect(url_for('dashboard' if is_admin(user) else 'user_dashboard'))
            return render_template('login.html', error='Geçersiz bilgiler', google_enabled=bool(oauth and getattr(oauth, 'google', None)), csrf_token=generate_csrf())
        except Exception as e:
            logger.error(f"Login error: {e}")
            try:
                return render_template('login.html', error='Sistem hatası', csrf_token=generate_csrf()), 500
            except Exception:
                return render_template('login.html', error='Sistem hatası'), 500

    @app.route('/logout')
    def logout():
        try:
            logout_user()
        except Exception:
            pass
        return redirect(url_for('login'))

    @app.route('/auth/google')
    def auth_google():
        if not oauth or not getattr(oauth, 'google', None):
            return redirect(url_for('login'))
        redirect_uri = url_for('auth_google_callback', _external=True)
        return oauth.google.authorize_redirect(redirect_uri)

    @app.route('/auth/google/callback')
    def auth_google_callback():
        try:
            if not oauth or not getattr(oauth, 'google', None):
                return redirect(url_for('login'))
            token = oauth.google.authorize_access_token()
            userinfo = token.get('userinfo') or {}
            if not userinfo:
                # Some providers return via userinfo endpoint
                resp = oauth.google.get('userinfo')
                userinfo = resp.json() if resp else {}

            email = (userinfo.get('email') or '').lower()
            if not email:
                return redirect(url_for('login'))
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(email=email, provider='google', provider_id=userinfo.get('sub'), first_name=userinfo.get('given_name'), last_name=userinfo.get('family_name'), avatar_url=userinfo.get('picture'), email_verified=True, is_active=True)
                db.session.add(user)
                db.session.commit()
            try:
                user.last_login = datetime.now()
                user.last_login_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                db.session.commit()
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass
            login_user(user)
            return redirect(url_for('user_dashboard'))
        except Exception as e:
            logger.error(f"Google OAuth error: {e}")
            return redirect(url_for('login'))

    @app.route('/auth/apple')
    def auth_apple():
        if not oauth or not getattr(oauth, 'apple', None):
            return redirect(url_for('login'))
        redirect_uri = url_for('auth_apple_callback', _external=True)
        return oauth.apple.authorize_redirect(redirect_uri)

    @app.route('/auth/apple/callback')
    def auth_apple_callback():
        try:
            if not oauth or not getattr(oauth, 'apple', None):
                return redirect(url_for('login'))
            token = oauth.apple.authorize_access_token()
            userinfo = token.get('userinfo') or {}
            email = (userinfo.get('email') or '').lower()
            if not email:
                # Apple çoğu zaman email'i ilk girişte verir; yoksa token id_token içinden parse edilebilir
                email = (token.get('id_token_claims') or {}).get('email', '').lower()
            if not email:
                return redirect(url_for('login'))
            user = User.query.filter_by(email=email).first()
            if not user:
                user = User(email=email, provider='apple', provider_id=(userinfo.get('sub') or (token.get('id_token_claims') or {}).get('sub')), first_name=userinfo.get('name'), email_verified=True, is_active=True)
                db.session.add(user)
                db.session.commit()
            try:
                user.last_login = datetime.now()
                user.last_login_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
                db.session.commit()
            except Exception:
                try:
                    db.session.rollback()
                except Exception:
                    pass
            login_user(user)
            return redirect(url_for('user_dashboard'))
        except Exception as e:
            logger.error(f"Apple OAuth error: {e}")
            return redirect(url_for('login'))
    
    @app.route('/user')
    @login_required
    def user_dashboard():
        """User interface for stock tracking and signals (no-cache)"""
        from flask import make_response
        resp = make_response(render_template('user_dashboard.html'))
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    
    @app.route('/health')
    def health():
        try:
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            
            # Database stats
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            
            return jsonify({
                "status": "healthy", 
                "database": "connected",
                "stocks": total_stocks,
                "price_records": total_prices,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"status": "unhealthy", "database": "disconnected", "error": str(e)}), 500
    

    
    @app.route('/stocks')
    def stocks_page():
        return render_template('stocks.html')
    
    @app.route('/analysis')
    def analysis_page():
        return render_template('analysis.html')
    
    @app.route('/api/stocks')
    def api_stocks():
        try:
            stocks = Stock.query.limit(50).all()  # Daha fazla hisse göster
            return jsonify([{
                "symbol": stock.symbol,
                "name": stock.name,
                "sector": stock.sector
            } for stock in stocks])
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ================================
    # Watchlist API (DB-backed)
    # ================================

    def _get_effective_user():
        """Resolve current user (Flask-Login or DEV_AUTH_BYPASS)"""
        try:
            from flask_login import current_user
            if current_user.is_authenticated:
                return current_user
        except Exception:
            pass
        # Development bypass: allow X-User-Id or default admin (1)
        try:
            if app.config.get('DEV_AUTH_BYPASS'):
                user_id_header = request.headers.get('X-User-Id')
                user_id = int(user_id_header) if user_id_header else 1
                return User.query.get(user_id)
        except Exception:
            return None
        return None

    @app.route('/api/watchlist', methods=['GET'])
    def get_watchlist():
        try:
            user = _get_effective_user()
            if not user:
                return jsonify({'status': 'unauthorized'}), 401
            from models import Watchlist
            items = Watchlist.query.filter_by(user_id=user.id).all()
            return jsonify({
                'status': 'success',
                'user_id': user.id,
                'watchlist': [item.to_dict() for item in items]
            })
        except Exception as e:
            logger.error(f"Watchlist get error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/watchlist', methods=['POST'])
    @csrf.exempt
    def add_watchlist():
        try:
            user = _get_effective_user()
            if not user:
                return jsonify({'status': 'unauthorized'}), 401
            data = request.get_json() or {}
            symbol = (data.get('symbol') or '').upper().strip()
            if not symbol:
                return jsonify({'status': 'error', 'error': 'symbol is required'}), 400

            # Ensure stock exists
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                if app.config.get('AUTO_CREATE_STOCKS', True):
                    stock = Stock(symbol=symbol, name=f"{symbol} Hisse Senedi", sector=data.get('sector') or 'Unknown')
                    db.session.add(stock)
                    db.session.flush()
                else:
                    return jsonify({'status': 'error', 'error': 'stock not found'}), 404

            from models import Watchlist
            item = Watchlist.query.filter_by(user_id=user.id, stock_id=stock.id).first()
            if not item:
                item = Watchlist(user_id=user.id, stock_id=stock.id)
                db.session.add(item)

            # Optional fields
            item.notes = data.get('notes')
            if 'alert_enabled' in data:
                item.alert_enabled = bool(data.get('alert_enabled'))
            if 'alert_threshold_buy' in data and data.get('alert_threshold_buy') is not None:
                try:
                    item.alert_threshold_buy = float(data.get('alert_threshold_buy'))
                except Exception:
                    pass
            if 'alert_threshold_sell' in data and data.get('alert_threshold_sell') is not None:
                try:
                    item.alert_threshold_sell = float(data.get('alert_threshold_sell'))
                except Exception:
                    pass

            db.session.commit()

            # Trigger initial analysis broadcast for this symbol so UI updates immediately
            try:
                result = get_pattern_detector().analyze_stock(symbol)
                if hasattr(app, 'socketio') and result:
                    socketio.emit('pattern_analysis', {
                        'symbol': symbol,
                        'data': result,
                        'timestamp': datetime.now().isoformat()
                    }, room=f'stock_{symbol}')
            except Exception:
                pass

            return jsonify({'status': 'success', 'item': item.to_dict()})
        except Exception as e:
            logger.error(f"Watchlist add error: {e}")
            try:
                db.session.rollback()
            except Exception:
                pass
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/watchlist/<symbol>', methods=['DELETE'])
    @csrf.exempt
    def delete_watchlist(symbol):
        try:
            user = _get_effective_user()
            if not user:
                return jsonify({'status': 'unauthorized'}), 401
            if not symbol:
                return jsonify({'status': 'error', 'error': 'symbol is required'}), 400
            symbol = symbol.upper().strip()
            from models import Watchlist
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                return jsonify({'status': 'error', 'error': 'stock not found'}), 404
            item = Watchlist.query.filter_by(user_id=user.id, stock_id=stock.id).first()
            if not item:
                return jsonify({'status': 'error', 'error': 'watchlist item not found'}), 404
            db.session.delete(item)
            db.session.commit()
            return jsonify({'status': 'success', 'message': f'{symbol} removed'})
        except Exception as e:
            logger.error(f"Watchlist delete error: {e}")
            try:
                db.session.rollback()
            except Exception:
                pass
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    @app.route('/api/stock-prices/<symbol>')
    def get_stock_prices(symbol):
        try:
            from sqlalchemy import desc
            stock = Stock.query.filter_by(symbol=symbol.upper()).first()
            if not stock:
                return jsonify({'error': 'Hisse bulunamadı'}), 404
            
            # Son 60 günlük veri
            prices = StockPrice.query.filter_by(stock_id=stock.id)\
                        .order_by(desc(StockPrice.date))\
                        .limit(60).all()
            
            if not prices:
                return jsonify({'error': 'Fiyat verisi bulunamadı'}), 404
            
            # JSON formatına çevir
            price_data = []
            for price in reversed(prices):  # Tarihe göre sırala
                price_data.append({
                    'date': price.date.strftime('%Y-%m-%d'),
                    'open': float(price.open_price),
                    'high': float(price.high_price),
                    'low': float(price.low_price),
                    'close': float(price.close_price),
                    'volume': int(price.volume)
                })
            
            return jsonify({
                'symbol': symbol.upper(),
                'name': stock.name,
                'sector': stock.sector,
                'data': price_data,
                'total_records': len(price_data)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/dashboard-stats')
    def dashboard_stats():
        try:
            from sqlalchemy import func, desc
            
            # Temel istatistikler
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            
            # En çok veri olan hisseler
            stock_with_most_data = db.session.query(
                Stock.symbol,
                func.count(StockPrice.id).label('price_count')
            ).join(StockPrice).group_by(Stock.symbol)\
            .order_by(desc('price_count')).limit(5).all()
            
            # Sektör dağılımı
            sector_stats = db.session.query(
                Stock.sector,
                func.count(Stock.id).label('stock_count')
            ).group_by(Stock.sector)\
            .order_by(desc('stock_count')).limit(10).all()
            
            top_stocks_data = [{'symbol': s, 'count': c} for s, c in stock_with_most_data]
            sector_data = [{'sector': s, 'count': c} for s, c in sector_stats]

            return jsonify({
                'total_stocks': total_stocks,
                'total_prices': total_prices,
                'top_stocks': top_stocks_data,
                'sectors': sector_data,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
            logger.error(f"Dashboard stats error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data-collection/status')
    def data_collection_status():
        try:
            # Basit durum bilgisi
            from sqlalchemy import func, desc
            
            # En son veri tarihi
            latest_date = db.session.query(func.max(StockPrice.date)).scalar()
            
            # Günlük veri sayısı
            latest_count = 0
            if latest_date:
                latest_count = StockPrice.query.filter_by(date=latest_date).count()
            
            return jsonify({
                'status': 'active',
                'latest_data_date': str(latest_date) if latest_date else None,
                'latest_day_records': latest_count,
                'total_records': StockPrice.query.count(),
                'message': 'Veri toplama sistemi aktif'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data-collection/manual', methods=['POST'])
    @csrf.exempt
    def manual_data_collection():
        try:
            import threading
            
            def collect_data():
                try:
                    from advanced_collector import AdvancedBISTCollector
                    collector = AdvancedBISTCollector()
                    result = collector.collect_priority_stocks()
                    logger.info(f'Manuel veri toplama tamamlandı: {result}')
                except Exception as e:
                    logger.error(f'Manuel veri toplama hatası: {e}')
            
            # Arkaplanda çalıştır
            thread = threading.Thread(target=collect_data, daemon=True)
            thread.start()
            
            return jsonify({
                'status': 'started', 
                'message': 'Manuel veri toplama arkaplanda başlatıldı'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/favicon.ico')
    def favicon():
        """Favicon serve et"""
        return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
    
    @app.route('/api/test-data')
    def test_data():
        return jsonify({'message': 'Bu bir test verisidir!'})
    
    @app.route('/api/pattern-analysis/<symbol>')
    def pattern_analysis(symbol):
        """Hisse için pattern analizi"""
        try:
            # Global singleton instance kullan - duplike instance oluşturma
            result = get_pattern_detector().analyze_stock(symbol.upper())
            
            # Simulation integration - aktif simulation varsa signal'i işle
            try:
                from simulation_engine import get_simulation_engine
                from models import SimulationSession
                
                # Aktif simulation session'ları bul
                active_sessions = SimulationSession.query.filter_by(status='active').all()
                
                if active_sessions and result.get('status') == 'success':
                    simulation_engine = get_simulation_engine()
                    
                    for session in active_sessions:
                        # Her aktif session için signal'i işle
                        trade = simulation_engine.process_signal(
                            session_id=session.id,
                            symbol=symbol.upper(),
                            signal_data=result
                        )
                        
                        if trade:
                            logger.info(f"🤖 Simulation trade executed: {trade.trade_type} {trade.quantity}x{symbol} @ {trade.price}")
                            
                            # WebSocket ile simulation update broadcast
                            if hasattr(app, 'socketio'):
                                app.socketio.emit('simulation_trade', {
                                    'session_id': session.id,
                                    'trade': trade.to_dict(),
                                    'timestamp': datetime.now().isoformat()
                                }, room='admin')
                        
            except Exception as sim_error:
                logger.warning(f"Simulation processing failed: {sim_error}")
                # Simulation hatası ana analizi etkilemesin
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Pattern analysis error for {symbol}: {e}")
            return jsonify({
                'symbol': symbol,
                'status': 'error',
                'error': str(e)
            }), 500
    
    @app.route('/api/pattern-summary')
    def pattern_summary():
        """Genel pattern özeti"""
        try:
            # Öncelikli hisseler
            priority_stocks = ['THYAO', 'AKBNK', 'GARAN', 'EREGL', 'ASELS', 'VAKBN', 'MGROS', 'FROTO']
            
            # Global singleton instance kullan - duplike instance oluşturma
            summary = get_pattern_detector().get_pattern_summary(priority_stocks)
            
            return jsonify(summary)
            
        except Exception as e:
            logger.error(f"Pattern summary error: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
    
    # Stock search API
    @app.route('/api/stocks/search')
    def search_stocks():
        """BIST hisse arama - Database'den Full-Text Search"""
        try:
            query = request.args.get('q', '').strip()
            limit = int(request.args.get('limit', 50))
            
            if not query:
                # Tüm hisseleri döndür (limit ile)
                stocks = Stock.query.limit(limit).all()
            else:
                # Full-text search - symbol, name, sector'da ara
                search_pattern = f"%{query.upper()}%"
                stocks = Stock.query.filter(
                    db.or_(
                        Stock.symbol.ilike(search_pattern),
                        Stock.name.ilike(search_pattern),
                        Stock.sector.ilike(search_pattern)
                    )
                ).limit(limit).all()
            
            # Response formatı
            result_stocks = []
            for stock in stocks:
                # Son fiyat bilgisi al
                latest_price = StockPrice.query.filter_by(stock_id=stock.id)\
                    .order_by(StockPrice.date.desc()).first()
                
                result_stocks.append({
                    'id': stock.id,
                    'symbol': stock.symbol,
                    'name': stock.name or stock.symbol,
                    'sector': stock.sector or 'N/A',
                    'price': float(latest_price.close_price) if latest_price else None,
                    'last_update': latest_price.date.isoformat() if latest_price else None
                })
            
            return jsonify({
                'status': 'success',
                'query': query,
                'stocks': result_stocks,
                'total': len(result_stocks),
                'message': f'{len(result_stocks)} hisse bulundu'
            })
                
        except Exception as e:
            logger.error(f"Stock search error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e),
                'stocks': []
            }), 500
    
    # Internal API for WebSocket broadcasting
    @app.route('/api/internal/broadcast-log', methods=['POST'])
    @csrf.exempt
    def internal_broadcast_log():
        """Internal endpoint for broadcasting logs from scheduler daemon"""
        try:
            data = request.get_json() or {}
            level = data.get('level', 'INFO')
            message = data.get('message', '')
            category = data.get('category', 'system')
            # Shared secret + localhost fallback (opsiyonel)
            token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            try:
                remote_ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
                is_local = remote_ip in ('127.0.0.1', '::1', 'localhost')
                allow_localhost = str(os.getenv('INTERNAL_ALLOW_LOCALHOST', str(app.config.get('INTERNAL_ALLOW_LOCALHOST', 'True')))).lower() == 'true'
            except Exception:
                is_local, allow_localhost = False, True
            if expected:
                if token != expected and not (allow_localhost and is_local):
                    return jsonify({'status': 'unauthorized'}), 401
            elif not is_local:
                # Token yoksa sadece localhost'a izin ver
                return jsonify({'status': 'unauthorized'}), 401
            
            # Broadcast log to connected clients
            app.broadcast_log(level, message, category)
            
            return jsonify({'status': 'success', 'message': 'Log broadcasted'})
        except Exception as e:
            logger.error(f"Internal broadcast error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/internal/broadcast-user-signal', methods=['POST'])
    @internal_route
    def internal_broadcast_user_signal():
        """Internal endpoint to broadcast a user-specific trading signal over WebSocket"""
        try:
            # Shared secret kontrolü (opsiyonel)
            token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                return jsonify({'status': 'unauthorized'}), 401

            data = request.get_json() or {}
            user_id = data.get('user_id')
            signal_data = data.get('signal_data')

            if not user_id or not signal_data:
                return jsonify({'status': 'error', 'error': 'user_id and signal_data are required'}), 400

            room = f'user_{user_id}'
            app.socketio.emit('user_signal', {
                'user_id': user_id,
                'signal': signal_data,
                'timestamp': datetime.now().isoformat()
            }, room=room)

            return jsonify({'status': 'success', 'message': f'signal broadcasted to {room}'})
        except Exception as e:
            logger.error(f"Internal user signal broadcast error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # New: Internal automation endpoints (token-protected, CSRF-exempt)
    @app.route('/api/internal/automation/<action>', methods=['POST'])
    @internal_route
    def internal_automation_control(action):
        try:
            # Token doğrulaması @internal_route tarafından yapıldı (token veya localhost)
            # Burada tekrar doğrulamaya gerek yok; çift kontrol 401 üretip kullanım zorluğu yaratıyordu.

            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({'status': 'unavailable', 'message': 'Automated Pipeline sistemi mevcut değil'}), 503

            pipeline = get_pipeline_with_context()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline not initialized'}), 500

            action = (action or '').lower()
            if action == 'start':
                if getattr(pipeline, 'is_running', False):
                    return jsonify({'status': 'already_running'})
                ok = pipeline.start_scheduler()
                return jsonify({'status': 'started' if ok else 'error'})
            elif action == 'stop':
                if not getattr(pipeline, 'is_running', False):
                    return jsonify({'status': 'already_stopped'})
                ok = pipeline.stop_scheduler()
                return jsonify({'status': 'stopped' if ok else 'error'})
            else:
                return jsonify({'status': 'error', 'message': 'invalid action'}), 400
        except Exception as e:
            logger.error(f"Internal automation error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/internal/automation/status')
    @internal_route
    def internal_automation_status():
        try:
            # Token doğrulaması @internal_route tarafından yapıldı

            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({'status': 'unavailable'}), 503
            pipeline = get_pipeline_with_context()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline not initialized'}), 500
            return jsonify({'status': 'success', 'scheduler_status': pipeline.get_scheduler_status()})
        except Exception as e:
            logger.error(f"Internal automation status error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/internal/automation/run-task/<task_name>', methods=['POST'])
    @internal_route
    def internal_run_automation_task(task_name):
        try:
            token_header = request.headers.get('Authorization', '') or ''
            token = None
            if token_header.lower().startswith('bearer '):
                token = token_header.split(' ', 1)[1].strip()
            if not token:
                token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                return jsonify({'status': 'unauthorized'}), 401

            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({'status': 'unavailable'}), 503
            pipeline = get_pipeline_with_context()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline not initialized'}), 500
            result = pipeline.run_manual_task(task_name)
            ok = bool(result) or isinstance(result, dict)
            return jsonify({'status': 'success' if ok else 'error', 'result': result})
        except Exception as e:
            logger.error(f"Internal automation run task error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @app.route('/api/internal/automation/full-cycle', methods=['POST'])
    @internal_route
    def internal_full_cycle():
        """Tek çağrıda tam döngü: veri toplama → AI analiz → toplu tahmin.
        ?async=1 verilirse arka planda çalışır ve 202 döner; ilerleme pipeline_status.json'a yazılır.
        """
        try:
            # Token doğrulaması (Authorization: Bearer veya X-Internal-Token)
            token_header = request.headers.get('Authorization', '') or ''
            token = None
            if token_header.lower().startswith('bearer '):
                token = token_header.split(' ', 1)[1].strip()
            if not token:
                token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                return jsonify({'status': 'unauthorized'}), 401

            # Yardımcı: pipeline durumu dosyasına yaz
            def _append_status(phase: str, state: str, details: dict | None = None):
                try:
                    import json, os
                    from datetime import datetime
                    log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                    os.makedirs(log_dir, exist_ok=True)
                    status_file = os.path.join(log_dir, 'pipeline_status.json')
                    payload = {'history': []}
                    try:
                        if os.path.exists(status_file):
                            with open(status_file, 'r') as rf:
                                payload = json.load(rf) or {'history': []}
                    except Exception:
                        payload = {'history': []}
                    entry = {
                        'phase': phase,
                        'state': state,
                        'timestamp': datetime.now().isoformat(),
                        'details': details or {}
                    }
                    payload.setdefault('history', []).append(entry)
                    payload['history'] = payload['history'][-200:]
                    with open(status_file, 'w') as wf:
                        json.dump(payload, wf)
                except Exception:
                    pass

            # Yayın (opsiyonel)
            def _broadcast(level, message, category='pipeline'):
                try:
                    if hasattr(app, 'broadcast_log'):
                        app.broadcast_log(level, message, category)
                except Exception:
                    pass

            def _run_cycle():
                # 1) Veri toplama
                _append_status('data_collection', 'start', {})
                _broadcast('INFO', '📊 Tam veri toplama başlıyor', 'collector')
                try:
                    from advanced_collector import AdvancedBISTCollector
                    collector = AdvancedBISTCollector()
                    col_res = collector.collect_all_stocks_parallel()
                    _append_status('data_collection', 'end', col_res or {})
                except Exception as e:
                    _append_status('data_collection', 'error', {'error': str(e)})

                # 2) AI analiz
                _append_status('ai_analysis', 'start', {})
                _broadcast('INFO', '🧠 AI analizi başlıyor', 'ai_analysis')
                analyzed = 0
                total = 0
                try:
                    from pattern_detector import HybridPatternDetector
                    det = HybridPatternDetector()
                    with app.app_context():
                        from models import Stock
                        symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
                    total = len(symbols)
                    for sym in symbols[:600]:
                        try:
                            det.analyze_stock(sym)
                            analyzed += 1
                        except Exception:
                            continue
                    _append_status('ai_analysis', 'end', {'analyzed': analyzed, 'total': total})
                except Exception as e:
                    _append_status('ai_analysis', 'error', {'error': str(e)})

                # 3) Toplu ML tahmin
                _append_status('bulk_predictions', 'start', {})
                _broadcast('INFO', '🤖 ML bulk predictions starting...', 'ml')
                try:
                    pipeline = get_pipeline_with_context()
                    bulk = pipeline.run_bulk_predictions_all() if pipeline else False
                    count = 0
                    try:
                        if isinstance(bulk, dict):
                            count = len(bulk.get('predictions') or {})
                    except Exception:
                        pass
                    _append_status('bulk_predictions', 'end', {'symbols': count})
                except Exception as e:
                    _append_status('bulk_predictions', 'error', {'error': str(e)})

                _broadcast('SUCCESS', '✅ Full cycle tamamlandı', 'pipeline')
                return {'analyzed': analyzed, 'total': total}

            # Mod: async veya sync
            do_async = (request.args.get('async') or '').lower() in ('1', 'true', 'yes')
            if do_async:
                import threading
                t = threading.Thread(target=_run_cycle, daemon=True)
                t.start()
                return jsonify({'status': 'accepted', 'mode': 'async'}), 202
            else:
                result = _run_cycle()
                return jsonify({'status': 'success', 'mode': 'sync', 'result': result})
        except Exception as e:
            logger.error(f"Internal full cycle error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # CLI Commands
    @app.cli.command()
    def init_db():
        """Initialize the database."""
        db.create_all()
        print('Database initialized!')
    
    @app.cli.command()
    def create_admin():
        """Create admin user."""
        from werkzeug.security import generate_password_hash
        
        admin = User(email=app.config.get('ADMIN_EMAIL', 'admin@bistpattern.com'))
        admin.username = 'systemadmin'
        admin.role = 'admin'
        # Default password from config (do NOT print it)
        admin.password_hash = generate_password_hash(app.config.get('ADMIN_DEFAULT_PASSWORD', '5ex5CHAN*'))
        admin.first_name = 'System'
        admin.last_name = 'Administrator'
        admin.is_active = True
        admin.email_verified = True
        
        try:
            db.session.add(admin)
            db.session.commit()
            print('Admin user created!')
        except Exception as e:
            print(f'Error creating admin: {e}')
    
    @app.route('/api/visual-analysis/<symbol>')
    def visual_pattern_analysis(symbol):
        """Visual pattern analysis endpoint"""
        try:
            from visual_pattern_detector import get_visual_pattern_system
            
            # Visual pattern system'i al
            visual_system = get_visual_pattern_system()
            
            # Sistem bilgilerini kontrol et
            system_info = visual_system.get_system_info()
            if not system_info['yolo_available']:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'YOLOv8 sistemi mevcut değil. "pip install ultralytics" ile yükleyin.',
                    'system_info': system_info
                })
            
            # Hisse verisini al
            stock_data = get_pattern_detector().get_stock_data(symbol)
            if stock_data is None or len(stock_data) < 20:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı'
                })
            
            # Visual analiz yap
            result = visual_system.analyze_stock_visual(symbol, stock_data)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Visual analysis error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/system-info')
    def system_info():
        """Sistem bilgilerini döndür"""
        try:
            from visual_pattern_detector import get_visual_pattern_system
            visual_system = get_visual_pattern_system()
            
            # FinGPT analyzer (optional)
            try:
                from fingpt_analyzer import get_fingpt_analyzer
                fingpt_analyzer = get_fingpt_analyzer()
                fingpt_available = True
            except ImportError:
                fingpt_analyzer = None
                fingpt_available = False
            
            # Database stats
            db_stats = { 'stocks': 0, 'price_records': 0 }
            try:
                db_stats['stocks'] = Stock.query.count()
                db_stats['price_records'] = StockPrice.query.count()
            except Exception as e:
                logger.warning(f"Database stats query failed: {e}")

            info = {
                'hybrid_detector': {
                    'status': 'active',
                    'cache_size': len(get_pattern_detector().cache) 
                },
                'visual_patterns': visual_system.get_system_info(),
                'sentiment_analysis': {
                    'model_loaded': hasattr(fingpt_analyzer, 'model'),
                    'status': 'active' if hasattr(fingpt_analyzer, 'model') else 'inactive'
                },
                'advanced_patterns': ADVANCED_PATTERNS_AVAILABLE,
                'ml_predictions': {
                    'available': ML_PREDICTION_AVAILABLE,
                    'status': 'active' if ML_PREDICTION_AVAILABLE else 'inactive'
                },
                'automated_pipeline': {
                    'available': AUTOMATED_PIPELINE_AVAILABLE,
                    'status': 'active' if AUTOMATED_PIPELINE_AVAILABLE else 'inactive'
                },
                'database': db_stats
            }
            
            return jsonify(info)
            
        except Exception as e:
            logger.error(f"System info error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/ml-prediction/<symbol>')
    def ml_prediction_analysis(symbol):
        """ML tabanlı fiyat tahmini"""
        try:
            if not ML_PREDICTION_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'ML Prediction sistemi mevcut değil'
                })
            
            # Hisse verisini al
            stock_data = get_pattern_detector().get_stock_data(symbol, days=365)  # 1 yıllık veri
            if stock_data is None or len(stock_data) < 100:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı (minimum 100 gün gerekli)'
                })
            
            # ML prediction system'i al
            ml_system = get_pattern_detector().ml_predictor
            
            # Sentiment analizi ekle
            sentiment_score = None
            try:
                from fingpt_analyzer import get_fingpt_analyzer
                fingpt = get_fingpt_analyzer()
                if fingpt.model_loaded:
                    # Basit sentiment analizi için dummy text
                    test_text = f"{symbol} stock analysis"
                    sentiment_result = fingpt.analyze_sentiment(test_text)
                    if sentiment_result['status'] == 'success':
                        # positive: 1, negative: 0, neutral: 0.5
                        if sentiment_result['sentiment'] == 'positive':
                            sentiment_score = sentiment_result['scores']['positive']
                        elif sentiment_result['sentiment'] == 'negative':
                            sentiment_score = 1 - sentiment_result['scores']['negative']
                        else:
                            sentiment_score = 0.5
            except Exception as e:
                logger.warning(f"Sentiment analizi eklenemedi: {e}")
            
            # Tahmin yap
            predictions = ml_system.predict_prices(symbol, stock_data, sentiment_score)
            
            if predictions:
                result = {
                    'symbol': symbol,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'current_price': float(stock_data['close'].iloc[-1]),
                    'predictions': predictions,
                    'sentiment_integrated': sentiment_score is not None,
                    'data_points': len(stock_data)
                }
            else:
                # Model eğitimi gerekebilir
                logger.info(f"{symbol} için model eğitimi başlatılıyor...")
                training_result = ml_system.train_models(symbol, stock_data)
                
                if training_result:
                    # Eğitimden sonra tekrar tahmin yap
                    predictions = ml_system.predict_prices(symbol, stock_data, sentiment_score)
                    result = {
                        'symbol': symbol,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'current_price': float(stock_data['close'].iloc[-1]),
                        'predictions': predictions or {},
                        'sentiment_integrated': sentiment_score is not None,
                        'data_points': len(stock_data),
                        'model_trained': True
                    }
                else:
                    result = {
                        'symbol': symbol,
                        'status': 'error',
                        'message': 'Model eğitimi başarısız',
                        'timestamp': datetime.now().isoformat()
                    }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/train-ml-model/<symbol>')
    def train_ml_model(symbol):
        """Belirli bir hisse için ML modelini eğit"""
        try:
            if not ML_PREDICTION_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'ML Prediction sistemi mevcut değil'
                })
            
            # Hisse verisini al (2 yıllık veri)
            stock_data = get_pattern_detector().get_stock_data(symbol, days=730)
            if stock_data is None or len(stock_data) < 200:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı (minimum 200 gün gerekli)'
                })
            
            # ML prediction system'i al
            ml_system = get_pattern_detector().ml_predictor
            
            # Model eğitimi
            training_result = ml_system.train_models(symbol, stock_data)
            
            if training_result:
                result = {
                    'symbol': symbol,
                    'status': 'success',
                    'message': 'Model eğitimi tamamlandı',
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(stock_data),
                    'models_trained': list(training_result.keys())
                }
            else:
                result = {
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Model eğitimi başarısız',
                    'timestamp': datetime.now().isoformat()
                }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/alerts/configs', methods=['GET', 'POST'])
    @csrf.exempt
    def alert_configs():
        """Alert konfigürasyonları yönetimi"""
        try:
            from alert_system import get_alert_system, AlertConfig
            
            alert_system = get_alert_system()
            
            if request.method == 'GET':
                # Mevcut konfigürasyonları döndür
                configs = alert_system.get_alert_configs()
                return jsonify({
                    'status': 'success',
                    'configs': configs,
                    'count': len(configs)
                })
            
            elif request.method == 'POST':
                # Yeni konfigürasyon ekle
                data = request.get_json()
                
                if not data or 'symbol' not in data:
                    return jsonify({'error': 'Symbol gerekli'}), 400
                
                config = AlertConfig(
                    symbol=data['symbol'].upper(),
                    min_signal_strength=data.get('min_signal_strength', 70.0),
                    signal_types=data.get('signal_types', ['BULLISH', 'BEARISH']),
                    email_enabled=data.get('email_enabled', False),
                    webhook_enabled=data.get('webhook_enabled', False),
                    email_addresses=data.get('email_addresses', []),
                    webhook_url=data.get('webhook_url', ''),
                    check_interval_minutes=data.get('check_interval_minutes', 15),
                    active=data.get('active', True)
                )
                
                config_id = alert_system.add_alert_config(config)
                
                if config_id:
                    return jsonify({
                        'status': 'success',
                        'config_id': config_id,
                        'message': f'{config.symbol} için alert konfigürasyonu eklendi'
                    })
                else:
                    return jsonify({'error': 'Alert config eklenemedi'}), 500
            
        except Exception as e:
            logger.error(f"Alert configs error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/alerts/configs/<config_id>', methods=['DELETE'])
    @csrf.exempt
    def delete_alert_config(config_id):
        """Alert konfigürasyonu sil"""
        try:
            from alert_system import get_alert_system
            
            alert_system = get_alert_system()
            
            if alert_system.remove_alert_config(config_id):
                return jsonify({
                    'status': 'success',
                    'message': 'Alert konfigürasyonu silindi'
                })
            else:
                return jsonify({'error': 'Konfigürasyon bulunamadı'}), 404
            
        except Exception as e:
            logger.error(f"Delete alert config error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/alerts/history')
    def alert_history():
        """Alert geçmişi"""
        try:
            from alert_system import get_alert_system
            
            alert_system = get_alert_system()
            limit = request.args.get('limit', 50, type=int)
            
            history = alert_system.get_alert_history(limit)
            
            return jsonify({
                'status': 'success',
                'alerts': history,
                'count': len(history)
            })
            
        except Exception as e:
            logger.error(f"Alert history error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/alerts/start', methods=['POST'])
    def start_alert_monitoring():
        """Alert monitoring başlat"""
        try:
            from alert_system import get_alert_system
            
            alert_system = get_alert_system()
            alert_system.start_monitoring()
            
            return jsonify({
                'status': 'success',
                'message': 'Alert monitoring başlatıldı'
            })
            
        except Exception as e:
            logger.error(f"Start monitoring error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/alerts/stop', methods=['POST'])
    def stop_alert_monitoring():
        """Alert monitoring durdur"""
        try:
            from alert_system import get_alert_system
            
            alert_system = get_alert_system()
            alert_system.stop_monitoring()
            
            return jsonify({
                'status': 'success',
                'message': 'Alert monitoring durduruldu'
            })
            
        except Exception as e:
            logger.error(f"Stop monitoring error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/alerts/test/<symbol>')
    def test_alert(symbol):
        """Test alert gönder"""
        try:
            from alert_system import get_alert_system
            
            alert_system = get_alert_system()
            
            if alert_system.test_alert(symbol.upper()):
                return jsonify({
                    'status': 'success',
                    'message': f'{symbol} için test alert gönderildi'
                })
            else:
                return jsonify({'error': 'Test alert gönderilemedi'}), 500
            
        except Exception as e:
            logger.error(f"Test alert error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data-collection/start', methods=['POST'])
    @csrf.exempt
    def start_data_collection():
        """BIST veri toplama işlemini başlat"""
        try:
            from data_collector import get_data_collector
            
            data = request.get_json() or {}
            period = data.get('period', '2y')
            max_workers = data.get('max_workers', 3)
            
            collector = get_data_collector()
            
            # Background task olarak başlat
            import threading
            
            def collect_data():
                try:
                    result = collector.collect_all_data(max_workers=max_workers, period=period)
                    logger.info(f"Data collection completed: {result}")
                except Exception as e:
                    logger.error(f"Data collection error: {e}")
            
            thread = threading.Thread(target=collect_data, daemon=True)
            thread.start()
            
            return jsonify({
                'status': 'started',
                'message': 'BIST veri toplama başlatıldı',
                'period': period,
                'max_workers': max_workers
            })
            
        except Exception as e:
            logger.error(f"Data collection start error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data-collection/stats')
    def data_collection_stats():
        """Veri toplama istatistikleri"""
        try:
            from data_collector import get_data_collector
            
            collector = get_data_collector()
            stats = collector.get_collection_stats()
            
            return jsonify({
                'status': 'success',
                'stats': stats
            })
            
        except Exception as e:
            logger.error(f"Data stats error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/data-collection/update/<symbol>')
    def update_stock_data(symbol):
        """Tek bir hisse için veri güncelle"""
        try:
            from data_collector import get_data_collector
            
            days = request.args.get('days', 30, type=int)
            
            collector = get_data_collector()
            success = collector.update_single_stock(symbol.upper(), days)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': f'{symbol} verisi güncellendi',
                    'days': days
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} verisi güncellenemedi'
                })
            
        except Exception as e:
            logger.error(f"Data update error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhanced-ml/train/<symbol>')
    def train_enhanced_ml(symbol):
        """Enhanced ML model eğitimi"""
        try:
            from enhanced_ml_system import get_enhanced_ml_system
            
            # Hisse verisini al
            stock_data = get_pattern_detector().get_stock_data(symbol, days=730)
            if stock_data is None or len(stock_data) < 200:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı (minimum 200 gün gerekli)'
                })
            
            enhanced_ml = get_enhanced_ml_system()
            
            # Model eğitimi
            training_result = enhanced_ml.train_enhanced_models(symbol, stock_data)
            
            if training_result:
                result = {
                    'symbol': symbol,
                    'status': 'success',
                    'message': 'Enhanced ML model eğitimi tamamlandı',
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(stock_data),
                    'models_trained': list(training_result.keys()),
                    'model_types': ['XGBoost', 'LightGBM', 'CatBoost']
                }
            else:
                result = {
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Enhanced ML model eğitimi başarısız',
                    'timestamp': datetime.now().isoformat()
                }
            
            return jsonify(result)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Enhanced ML training error: {e}")
            logger.error(f"Full traceback: {error_details}")
            return jsonify({
                'error': str(e),
                'traceback': error_details,
                'status': 'error'
            }), 500
    
    @app.route('/api/enhanced-ml/predict/<symbol>')
    def enhanced_ml_prediction(symbol):
        """Enhanced ML tahmin"""
        try:
            from enhanced_ml_system import get_enhanced_ml_system
            
            # Hisse verisini al
            stock_data = get_pattern_detector().get_stock_data(symbol, days=365)
            if stock_data is None or len(stock_data) < 100:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı'
                })
            
            enhanced_ml = get_enhanced_ml_system()
            
            # Tahmin yap
            predictions = enhanced_ml.predict_enhanced(symbol, stock_data)
            
            if predictions:
                result = {
                    'symbol': symbol,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'current_price': float(stock_data['close'].iloc[-1]),
                    'predictions': predictions,
                    'data_points': len(stock_data),
                    'enhanced_models': True
                }
            else:
                result = {
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Enhanced ML tahmin yapılamadı - model eğitimi gerekli',
                    'timestamp': datetime.now().isoformat()
                }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Enhanced ML prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhanced-ml/features/<symbol>')
    def enhanced_ml_features(symbol):
        """Feature importance analizi"""
        try:
            from enhanced_ml_system import get_enhanced_ml_system
            
            enhanced_ml = get_enhanced_ml_system()
            model_type = request.args.get('model', 'xgboost')
            top_n = request.args.get('top', 20, type=int)
            
            top_features = enhanced_ml.get_top_features(symbol, model_type, top_n)
            
            if top_features:
                result = {
                    'symbol': symbol,
                    'status': 'success',
                    'model_type': model_type,
                    'top_features': top_features,
                    'feature_count': top_n
                }
            else:
                result = {
                    'symbol': symbol,
                    'status': 'error',
                    'message': f'{symbol} için feature importance bulunamadı'
                }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhanced-ml/info')
    def enhanced_ml_info():
        """Enhanced ML sistem bilgileri"""
        try:
            from enhanced_ml_system import get_enhanced_ml_system
            
            enhanced_ml = get_enhanced_ml_system()
            info = enhanced_ml.get_system_info()
            
            return jsonify({
                'status': 'success',
                'system_info': info
            })
            
        except Exception as e:
            logger.error(f"Enhanced ML info error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/simple-ml/train/<symbol>')
    def train_simple_ml(symbol):
        """Simple Enhanced ML model eğitimi"""
        try:
            from simple_enhanced_ml import get_simple_enhanced_ml
            
            # Data al
            stock_data = get_pattern_detector().get_stock_data(symbol, days=365)
            if stock_data is None or len(stock_data) < 100:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı'
                })
            
            simple_ml = get_simple_enhanced_ml()
            
            # Train
            training_result = simple_ml.train_simple_models(symbol, stock_data)
            
            if training_result:
                return jsonify({
                    'symbol': symbol,
                    'status': 'success',
                    'message': 'Simple Enhanced ML eğitimi tamamlandı',
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(stock_data),
                    'models_trained': list(training_result.keys())
                })
            else:
                return jsonify({
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Simple Enhanced ML eğitimi başarısız',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Simple ML training error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/simple-ml/predict/<symbol>')
    def simple_ml_prediction(symbol):
        """Simple Enhanced ML tahmin"""
        try:
            from simple_enhanced_ml import get_simple_enhanced_ml
            
            # Data al
            stock_data = get_pattern_detector().get_stock_data(symbol, days=200)
            if stock_data is None or len(stock_data) < 50:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı'
                })
            
            simple_ml = get_simple_enhanced_ml()
            
            # Predict
            predictions = simple_ml.predict_simple(symbol, stock_data)
            
            if predictions:
                return jsonify({
                    'symbol': symbol,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'current_price': float(stock_data['close'].iloc[-1]),
                    'predictions': predictions,
                    'model_type': 'simple_enhanced_ml'
                })
            else:
                return jsonify({
                    'symbol': symbol,
                    'status': 'error',
                    'message': 'Simple ML tahmin yapılamadı - model eğitimi gerekli',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Simple ML prediction error: {e}")
            return jsonify({'error': str(e)}), 500


    @app.route('/api/user/predictions/<symbol>')
    def user_predictions(symbol):
        """Kullanıcı paneli için 1/3/7/14/30g tahminleri döndür (enhanced -> simple -> basic fallback)."""
        try:
            horizons = [1, 3, 7, 14, 30]
            symbol = symbol.upper()

            # 1) Veriyi hazırla (1 yıl)
            stock_data = get_pattern_detector().get_stock_data(symbol, days=365)
            if stock_data is None or len(stock_data) < 50:
                return jsonify({
                    'status': 'error',
                    'message': f'{symbol} için yeterli veri bulunamadı',
                    'symbol': symbol
                }), 404

            def _normalize_predictions(raw, current):
                """Farklı dönen şemaları  {'1d':price, '3d':price,...} şekline normalize et"""
                out = {}
                if not raw:
                    return out
                # 1) Dict içindeki doğrudan horizon anahtarları
                for key, val in (raw.items() if isinstance(raw, dict) else []):
                    k = str(key).lower()
                    if k in ('1d','d1','one_day','day1','1day'):
                        out['1d'] = float(val.get('price') if isinstance(val, dict) else val)
                    elif k in ('3d','d3','three_day','day3','3day'):
                        out['3d'] = float(val.get('price') if isinstance(val, dict) else val)
                    elif k in ('7d','d7','seven_day','day7','7day'):
                        out['7d'] = float(val.get('price') if isinstance(val, dict) else val)
                    elif k in ('14d','d14','fourteen_day','day14','14day'):
                        out['14d'] = float(val.get('price') if isinstance(val, dict) else val)
                    elif k in ('30d','d30','thirty_day','day30','30day'):
                        out['30d'] = float(val.get('price') if isinstance(val, dict) else val)
                # 2) Liste/dict altında generic alanlar (horizon, days, target/prediction/value)
                def _pick_num(x):
                    if isinstance(x, (int, float)):
                        return float(x)
                    if isinstance(x, dict):
                        for cand in ('price','prediction','value','target','y'):
                            if cand in x and isinstance(x[cand], (int,float)):
                                return float(x[cand])
                    return None
                if isinstance(raw, list):
                    for item in raw:
                        days = None
                        if isinstance(item, dict):
                            for cand in ('horizon','days','d','day'):
                                if cand in item:
                                    try:
                                        days = int(str(item[cand]).replace('d',''))
                                    except Exception:
                                        pass
                            val = _pick_num(item)
                            if days in (1,3,7,14,30) and isinstance(val,(int,float)):
                                out[f'{days}d'] = float(val)
                # 3) Eğer hiçbir şey bulunamadıysa boş dön
                return out

            result_payload = {
                'status': 'success',
                'symbol': symbol,
                'current_price': float(stock_data['close'].iloc[-1]),
                'predictions': {},
                'horizons': horizons
            }

            # 2) Enhanced ML varsa kullan (opsiyonel; env flag)
            used_model = None
            if str(os.getenv('ENABLE_ENHANCED_ML', 'False')).lower() == 'true':
                try:  # Enhanced
                    from enhanced_ml_system import get_enhanced_ml_system
                    enhanced_ml = get_enhanced_ml_system()
                    enhanced_preds = enhanced_ml.predict_enhanced(symbol, stock_data)
                    if enhanced_preds:
                        result_payload['predictions'] = _normalize_predictions(enhanced_preds, result_payload['current_price']) or enhanced_preds
                        used_model = 'enhanced_ml'
                except Exception:
                    pass

            # 3) Fallback: simple enhanced ml (opsiyonel; env flag)
            if not result_payload['predictions'] and str(os.getenv('ENABLE_SIMPLE_ML', 'False')).lower() == 'true':
                try:  # Simple
                    from simple_enhanced_ml import get_simple_enhanced_ml
                    simple_ml = get_simple_enhanced_ml()
                    simple_preds = simple_ml.predict_simple(symbol, stock_data)
                    if simple_preds:
                        result_payload['predictions'] = _normalize_predictions(simple_preds, result_payload['current_price']) or simple_preds
                        used_model = 'simple_enhanced_ml'
                except Exception:
                    pass

            # 4) Fallback: basic ml predictor (opsiyonel; env flag)
            if not result_payload['predictions'] and ML_PREDICTION_AVAILABLE and str(os.getenv('ENABLE_BASIC_ML', 'False')).lower() == 'true':
                try:  # Basic
                    ml_system = get_pattern_detector().ml_predictor
                    basic_preds = ml_system.predict_prices(symbol, stock_data, None) or {}
                    result_payload['predictions'] = _normalize_predictions(basic_preds, result_payload['current_price']) or basic_preds
                    used_model = 'basic_ml'
                except Exception:
                    pass

            # 5) Son hal - minimum şema uyumu
            # Her durumda normalize edilmiş anahtarları doldurmaya çalış
            normalized = _normalize_predictions(result_payload['predictions'], result_payload['current_price'])
            if normalized:
                result_payload['predictions'] = normalized
            # Basit fallback: model yoksa son 5 gün ortalama günlük getiriyi üstel projeksiyonla uygula
            if not result_payload['predictions']:
                try:
                    close = stock_data['close']
                    avg_ret = float(close.pct_change().dropna().tail(5).mean()) if len(close) > 6 else 0.0
                    base = result_payload['current_price']
                    naive = {
                        '1d': float(base * ((1 + avg_ret) ** 1)),
                        '3d': float(base * ((1 + avg_ret) ** 3)),
                        '7d': float(base * ((1 + avg_ret) ** 7)),
                        '14d': float(base * ((1 + avg_ret) ** 14)),
                        '30d': float(base * ((1 + avg_ret) ** 30)),
                    }
                    result_payload['predictions'] = naive
                    used_model = used_model or 'naive'
                except Exception:
                    pass
            result_payload['model'] = used_model or 'none'
            return jsonify(result_payload)
        except Exception as e:
            logger.error(f"User predictions error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/watchlist/predictions')
    @login_required
    def watchlist_predictions():
        """Kullanıcının watchlist'indeki tüm hisseler için 1/3/7/14/30 günlük tahminleri döndürür.
        Öncelik sırası: Enhanced ML (varsa) → Basic ML (bulk dosyadan) → boş.
        Kaynak: /opt/bist-pattern/logs/ml_bulk_predictions.json
        """
        try:
            # Watchlist sembollerini al
            from models import Watchlist, Stock, StockPrice
            user_id = current_user.id
            items = Watchlist.query.filter_by(user_id=user_id).all()
            symbols = [item.stock.symbol for item in items if getattr(item, 'stock', None)]

            # Bulk predictions dosyasını yükle
            predictions_map = {}
            try:
                import json as _json
                log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                fpath = os.path.join(log_dir, 'ml_bulk_predictions.json')
                if os.path.exists(fpath):
                    with open(fpath, 'r') as rf:
                        data = _json.load(rf) or {}
                        predictions_map = (data.get('predictions') or {}) if isinstance(data, dict) else {}
            except Exception:
                predictions_map = {}

            def _normalize(raw):
                """{'1d':{'price':..}} veya {'predictions': {'1d':{'ensemble_prediction':..}}} gibi
                farklı şemaları {'1d':float,'3d':float,'7d':float,'14d':float,'30d':float} formatına çevirir."""
                out = {}
                if not raw:
                    return out
                try:
                    if isinstance(raw, dict) and 'predictions' in raw and isinstance(raw['predictions'], dict):
                        raw = raw['predictions']
                    for k, v in (raw.items() if isinstance(raw, dict) else []):
                        kk = str(k).lower()
                        if kk in ('1d','3d','7d','14d','30d'):
                            if isinstance(v, dict):
                                # enhanced
                                if 'ensemble_prediction' in v and isinstance(v['ensemble_prediction'], (int,float)):
                                    out[kk] = float(v['ensemble_prediction'])
                                # basic
                                elif 'price' in v and isinstance(v['price'], (int,float)):
                                    out[kk] = float(v['price'])
                            elif isinstance(v, (int,float)):
                                out[kk] = float(v)
                except Exception:
                    return {}
                return out

            # DB'den son fiyatları al (tek sefer okuma için mapping)
            last_close_by_symbol = {}
            try:
                sym_set = set(symbols)
                if sym_set:
                    stocks = Stock.query.filter(Stock.symbol.in_(sym_set)).all()
                    id_to_sym = {s.id: s.symbol for s in stocks}
                    # En son tarihli fiyat kayıtlarını çek
                    latest = StockPrice.query.order_by(StockPrice.stock_id, StockPrice.date.desc()).all()
                    for rec in latest:
                        sym = id_to_sym.get(rec.stock_id)
                        if sym and sym not in last_close_by_symbol:
                            last_close_by_symbol[sym] = float(rec.close_price)
                        # tüm semboller için ilk karşılaşılan en son kayıt yeterli
                        if len(last_close_by_symbol) == len(sym_set):
                            break
            except Exception:
                pass

            response_items = []
            for sym in symbols:
                pred_entry = predictions_map.get(sym) or {}
                # Tercih: enhanced → basic
                model_used = None
                normalized = {}
                if isinstance(pred_entry, dict) and pred_entry.get('enhanced'):
                    normalized = _normalize(pred_entry.get('enhanced'))
                    model_used = 'enhanced'
                if not normalized and isinstance(pred_entry, dict) and pred_entry.get('basic'):
                    normalized = _normalize(pred_entry.get('basic'))
                    model_used = model_used or 'basic'

                response_items.append({
                    'symbol': sym,
                    'current_price': last_close_by_symbol.get(sym),
                    'predictions': normalized,
                    'model': model_used or 'none'
                })

            return jsonify({
                'status': 'success',
                'count': len(response_items),
                'items': response_items
            })
        except Exception as e:
            logger.error(f"Watchlist predictions error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # ================================
    # AUTOMATED PIPELINE ENDPOINTS
    # ================================

    @app.route('/api/automation/start', methods=['POST'])
    @csrf.exempt
    @login_required
    @admin_required
    def start_automation():
        """Automated Pipeline'ı başlat"""
        try:
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Automated Pipeline sistemi mevcut değil'
                }), 503
            
            pipeline = get_pipeline_with_context()
            
            if pipeline.is_running:
                return jsonify({
                    'status': 'already_running',
                    'message': 'Automated Pipeline zaten çalışıyor'
                })
            
            # Continuous mode default
            os.environ['PIPELINE_MODE'] = os.getenv('PIPELINE_MODE', 'CONTINUOUS_FULL')
            success = pipeline.start_scheduler()
            
            if success:
                return jsonify({
                    'status': 'started',
                    'message': 'Automated Pipeline başarıyla başlatıldı',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Automated Pipeline başlatılamadı'
                }), 500
                
        except Exception as e:
            logger.error(f"Automation start error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Automation başlatma hatası: {str(e)}'
            }), 500

    @app.route('/api/automation/stop', methods=['POST'])
    @csrf.exempt
    @login_required
    @admin_required
    def stop_automation():
        """Automated Pipeline'ı durdur"""
        try:
            # Helper: Always clear pipeline history file on stop requests
            def _clear_pipeline_history_file():
                try:
                    import json, os
                    log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                    os.makedirs(log_dir, exist_ok=True)
                    status_file = os.path.join(log_dir, 'pipeline_status.json')
                    with open(status_file, 'w') as f:
                        json.dump({'history': []}, f)
                    return True
                except Exception:
                    return False
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Automated Pipeline sistemi mevcut değil'
                }), 503
            
            pipeline = get_pipeline_with_context()
            
            if not pipeline.is_running:
                _clear_pipeline_history_file()
                return jsonify({
                    'status': 'already_stopped',
                    'message': 'Automated Pipeline zaten durmuş'
                })
            
            success = pipeline.stop_scheduler()
            
            if success:
                _clear_pipeline_history_file()
                return jsonify({
                    'status': 'stopped',
                    'message': 'Automated Pipeline başarıyla durduruldu',
                    'timestamp': datetime.now().isoformat()
                })
            else:
                _clear_pipeline_history_file()
                return jsonify({
                    'status': 'error',
                    'message': 'Automated Pipeline durdurulamadı'
                }), 500
                
        except Exception as e:
            logger.error(f"Automation stop error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Automation durdurma hatası: {str(e)}'
            }), 500

    @app.route('/api/automation/status')
    @login_required
    @admin_required
    def automation_status():
        """Automated Pipeline durumu - Continuous-only basitleştirilmiş"""
        try:
            # Optional cache and source controls to stabilize UI
            cache_ttl_seconds = int(os.getenv('STATUS_CACHE_TTL', '15'))
            source = (request.args.get('source') or 'auto').lower()  # auto|internal|external
            use_cache = (request.args.get('cache') or '1') in ('1', 'true', 'yes')
            # Simple in-process cache on app object
            if not hasattr(app, '_status_cache'):
                app._status_cache = {'ts': 0.0, 'payload': None}
            if use_cache and app._status_cache.get('payload') is not None:
                if (time.time() - float(app._status_cache.get('ts') or 0)) < cache_ttl_seconds:
                    return jsonify(app._status_cache['payload'])
            
            # Internal scheduler status (continuous mode)
            internal_status = {'is_running': False, 'scheduled_jobs': 0, 'next_runs': [], 'thread_alive': False}
            if AUTOMATED_PIPELINE_AVAILABLE:
                try:
                    pipeline = get_pipeline_with_context()
                    internal_status = pipeline.get_scheduler_status()
                except:
                    pass

            # Grace window: Eğer kısa süre önce RUNNING ise ve şu an STOPPED görünüyorsa, kısa bir süre RUNNING göster
            try:
                grace_seconds = int(os.getenv('STATUS_GRACE_SECONDS', '5'))
                if internal_status and isinstance(internal_status.get('is_running'), bool):
                    if internal_status.get('is_running'):
                        setattr(app, '_last_running_ts', time.time())
                    else:
                        last_ts = getattr(app, '_last_running_ts', 0)
                        if last_ts and (time.time() - float(last_ts)) < grace_seconds:
                            internal_status['is_running'] = True
            except Exception:
                pass
            
            payload = {
                'status': 'success',
                'available': True,
                'scheduler_status': internal_status,
                'external_scheduler': {'is_running': False, 'message': 'removed'},
                'timestamp': datetime.now().isoformat(),
                'mode': 'CONTINUOUS_FULL'
            }
            # Save cache
            if use_cache:
                app._status_cache = {'ts': time.time(), 'payload': payload}
            return jsonify(payload)
            
        except Exception as e:
            logger.error(f"Automation status error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Automation status hatası: {str(e)}'
            }), 500

    def _check_external_scheduler():
        """External scheduler daemon durumunu kontrol et (PID path fallback'lı)."""
        try:
            # PID dosya yolu: env ile override edilebilir
            explicit_pid = os.getenv('BIST_PID_FILE')
            if explicit_pid:
                candidates = [explicit_pid]
            else:
                base_dir = os.getenv('BIST_PID_PATH', '/opt/bist-pattern')
                candidates = [
                    os.path.join(base_dir, 'scheduler_daemon.pid'),
                    os.path.join(os.getcwd(), 'scheduler_daemon.pid')
                ]
            pid_file = None
            for cand in candidates:
                if os.path.exists(cand):
                    pid_file = cand
                    break
            if not pid_file:
                return {
                    'is_running': False,
                    'message': 'PID file not found in expected paths',
                    'scheduled_jobs': 0,
                    'next_runs': [],
                    'thread_alive': False
                }

            # PID file'ı oku
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Process'in çalışıp çalışmadığını kontrol et
            try:
                os.kill(pid, 0)  # Signal 0 - existence check
                return {
                    'is_running': True,
                    'message': f'External scheduler running with PID {pid} ({pid_file})',
                    'pid': pid,
                    'scheduled_jobs': 1,  # External scheduler en az bir job'a sahip
                    'next_runs': [{'job': 'external_scheduler', 'next_run': 'continuous'}],
                    'thread_alive': True
                }
            except OSError:
                # Process yok, PID file eski
                return {
                    'is_running': False,
                    'message': f'Process with PID {pid} not found ({pid_file})',
                    'scheduled_jobs': 0,
                    'next_runs': [],
                    'thread_alive': False
                }

        except Exception as e:
            return {
                'is_running': False,
                'message': f'Error checking external scheduler: {str(e)}',
                'scheduled_jobs': 0,
                'next_runs': [],
                'thread_alive': False
            }

    @app.route('/api/automation/health')
    @login_required
    @admin_required
    def automation_health():
        """Sistem sağlık kontrolü - Artık daemon tarafından yazılan dosyayı okur"""
        try:
            import json
            health_status_path = os.path.join(os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs'), 'health_status.json')
            
            if not os.path.exists(health_status_path):
                # Dosya henüz oluşturulmamış olabilir
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Health status not available yet. Scheduler may be starting.'
                }), 503

            with open(health_status_path, 'r') as f:
                health_data = json.load(f)

            # WebSocket ve API durumunu ekle (bunlar app context'inde daha iyi bilinir)
            health_data.setdefault('systems', {})
            health_data['systems']['flask_api'] = {'status': 'healthy', 'details': 'API is responsive'}
            health_data['systems']['websocket'] = {'status': 'connected', 'details': 'Socket.IO is active'}
            # Database status
            try:
                from sqlalchemy import text
                db.session.execute(text('SELECT 1'))
                health_data['systems']['database'] = {'status': 'connected'}
            except Exception as _db_err:
                health_data['systems']['database'] = {'status': 'error', 'details': str(_db_err)}
            # Automation engine status
            try:
                pipeline = get_pipeline_with_context()
                is_running = bool(pipeline and getattr(pipeline, 'is_running', False))
                health_data['systems']['automation_engine'] = {'status': 'running' if is_running else 'stopped'}
            except Exception as _auto_err:
                health_data['systems']['automation_engine'] = {'status': 'error', 'details': str(_auto_err)}

            return jsonify({
                'status': 'success',
                'health_check': health_data,
            })

        except Exception as e:
            logger.error(f"Health check read error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Could not read health status file: {str(e)}'
            }), 500

    @app.route('/api/automation/run-task/<task_name>', methods=['POST'])
    @csrf.exempt
    @login_required
    @admin_required
    def run_automation_task(task_name):
        try:
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Automated Pipeline sistemi mevcut değil'
                }), 503
            
            valid_tasks = [
                'data_collection', 'model_retraining', 'health_check', 
                'status_report', 'weekly_collection'
            ]
            
            if task_name not in valid_tasks:
                return jsonify({
                    'status': 'error',
                    'message': f'Geçersiz görev: {task_name}',
                    'valid_tasks': valid_tasks
                }), 400
            
            pipeline = get_pipeline_with_context()
            result = pipeline.run_manual_task(task_name)
            
            if result:
                return jsonify({
                    'status': 'success',
                    'message': f'{task_name} görevi başarıyla çalıştırıldı',
                    'task': task_name,
                    'result': result if isinstance(result, dict) else True,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'{task_name} görevi çalıştırılamadı',
                    'task': task_name
                }), 500
                
        except Exception as e:
            logger.error(f"Manual task error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Manuel görev hatası: {str(e)}'
            }), 500

    @app.route('/api/automation/report')
    @login_required
    @admin_required
    def automation_report():
        """Günlük sistem raporu"""
        try:
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Automated Pipeline sistemi mevcut değil'
                }), 503
            
            pipeline = get_pipeline_with_context()
            report = pipeline.daily_status_report()
            
            return jsonify({
                'status': 'success',
                'report': report,
                'timestamp': datetime.now().isoformat(),
                'last_run_stats': pipeline.last_run_stats
            })
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Rapor oluşturma hatası: {str(e)}'
            }), 500

    # ==========================================
    # PAPER TRADING SIMULATION API ENDPOINTS
    # ==========================================

    @app.route('/api/simulation/start', methods=['POST'])
    @csrf.exempt
    def start_simulation():
        """Yeni paper trading simulation başlat"""
        try:
            from simulation_engine import get_simulation_engine
            
            data = request.get_json() or {}
            user_id = data.get('user_id', 1)  # Default admin user
            initial_balance = float(data.get('initial_balance', 100.0))
            duration_hours = int(data.get('duration_hours', 48))
            session_name = data.get('session_name', 'AI Performance Test')
            trade_amount = float(data.get('trade_amount', 10000.0))  # Dashboard'dan gelecek
            
            simulation_engine = get_simulation_engine()
            # Trade amount'u engine'e set et
            simulation_engine.fixed_trade_amount = Decimal(str(trade_amount))
            
            session = simulation_engine.create_session(
                user_id=user_id,
                initial_balance=initial_balance,
                duration_hours=duration_hours,
                session_name=session_name
            )
            
            logger.info(f"✅ New simulation started: {session.id}")
            
            return jsonify({
                'status': 'success',
                'message': 'Simulation başlatıldı',
                'session': session.to_dict()
            })
            
        except Exception as e:
            logger.error(f"❌ Simulation start error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Simulation başlatma hatası: {str(e)}'
            }), 500

    @app.route('/api/simulation/<int:session_id>/status')
    def simulation_status(session_id):
        """Simulation durumunu getir"""
        try:
            from simulation_engine import get_simulation_engine
            from models import SimulationSession
            
            session = SimulationSession.query.get(session_id)
            if not session:
                return jsonify({
                    'status': 'error',
                    'message': 'Simulation session bulunamadı'
                }), 404
            
            simulation_engine = get_simulation_engine()
            performance = simulation_engine.get_session_performance(session_id)
            
            return jsonify({
                'status': 'success',
                'performance': performance
            })
            
        except Exception as e:
            logger.error(f"❌ Simulation status error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Simulation status hatası: {str(e)}'
            }), 500

    @app.route('/api/simulation/<int:session_id>/stop', methods=['POST'])
    @csrf.exempt
    def stop_simulation(session_id):
        """Simulation'ı durdur"""
        try:
            from models import SimulationSession
            
            session = SimulationSession.query.get(session_id)
            if not session:
                return jsonify({
                    'status': 'error',
                    'message': 'Simulation session bulunamadı'
                }), 404
            
            session.status = 'completed'
            session.end_time = datetime.now()
            db.session.commit()
            
            logger.info(f"✅ Simulation stopped: {session_id}")
            
            return jsonify({
                'status': 'success',
                'message': 'Simulation durduruldu',
                'session': session.to_dict()
            })
            
        except Exception as e:
            logger.error(f"❌ Simulation stop error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Simulation durdurma hatası: {str(e)}'
            }), 500

    @app.route('/api/simulation/list')
    def list_simulations():
        """Tüm simulation session'ları listele"""
        try:
            from models import SimulationSession
            from sqlalchemy import desc
            
            sessions = SimulationSession.query.order_by(desc(SimulationSession.created_at)).limit(20).all()
            
            return jsonify({
                'status': 'success',
                'sessions': [session.to_dict() for session in sessions]
            })
            
        except Exception as e:
            logger.error(f"❌ Simulation list error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Simulation listesi hatası: {str(e)}'
            }), 500

    @app.route('/api/simulation/process-signal', methods=['POST'])
    @internal_route
    def process_simulation_signal():
        """
        Process an incoming trading signal for active simulations.
        """
        try:
            from simulation_engine import get_simulation_engine
            
            data = request.get_json()
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'JSON data gerekli'
                }), 400
            
            session_id = data.get('session_id')
            symbol = data.get('symbol')
            signal_data = data.get('signal_data')
            
            if not all([session_id, symbol, signal_data]):
                return jsonify({
                    'status': 'error',
                    'message': 'session_id, symbol ve signal_data gerekli'
                }), 400
            
            simulation_engine = get_simulation_engine()
            trade = simulation_engine.process_signal(session_id, symbol, signal_data)
            
            if trade:
                return jsonify({
                    'status': 'success',
                    'message': 'Signal işlendi, trade execute edildi',
                    'trade': trade.to_dict()
                })
            else:
                return jsonify({
                    'status': 'success',
                    'message': 'Signal işlendi, trade execute edilmedi',
                    'trade': None
                })
            
        except Exception as e:
            logger.error(f"❌ Signal processing error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Signal processing hatası: {str(e)}'
            }), 500

    def _read_pipeline_history(max_items: int = 6):
        """Read pipeline status history from JSON file (scheduler writes)."""
        try:
            import json
            log_path = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            status_file = os.path.join(log_path, 'pipeline_status.json')
            if not os.path.exists(status_file):
                return []
            with open(status_file, 'r') as f:
                data = json.load(f) or {}
            history = data.get('history', [])
            return history[-max_items:]
        except Exception:
            return []

    @app.route('/api/automation/pipeline-history')
    @login_required
    @admin_required
    def api_pipeline_history():
        try:
            hist = _read_pipeline_history(50) or []  # preserve append order (newest already last)
            resp = jsonify({'status': 'success', 'history': hist})
            # Prevent browser/proxy caching
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return resp
        except Exception as e:
            logger.error(f"Pipeline history error: {e}")
            return jsonify({'status': 'error', 'error': str(e)})

    @app.route('/api/recent-tasks')
    def recent_tasks():
        """Recent Tasks endpoint for dashboard"""
        try:
            from sqlalchemy import func, desc
            from models import SimulationSession, SimulationTrade
            
            # Eğer otomasyon çalışmıyorsa, listeyi boş döndür (kullanıcı isteği)
            try:
                pipeline = get_pipeline_with_context()
                if not (pipeline and getattr(pipeline, 'is_running', False)):
                    current_time = datetime.now()
                    # Gerçek sistem istatistikleri yine sağlansın
                    total_stocks = Stock.query.count()
                    total_prices = StockPrice.query.count()
                    latest_date = db.session.query(func.max(StockPrice.date)).scalar()
                    return jsonify({
                        'status': 'success',
                        'tasks': [],
                        'count': 0,
                        'last_update': current_time.isoformat(),
                        'system_stats': {
                            'stocks': total_stocks,
                            'prices': total_prices,
                            'latest_date': str(latest_date) if latest_date else None
                        }
                    })
            except Exception:
                pass

            # Eğer otomasyon ÇALIŞIYOR ise: tek-kayıtlı "anlık durum" görünümü
            try:
                is_running = bool(pipeline and getattr(pipeline, 'is_running', False))
            except Exception:
                is_running = False

            current_time = datetime.now()

            if is_running:
                # Pipeline history'den en son 'start' kaydını bul (faz belirleme için güvenli yöntem)
                try:
                    history = _read_pipeline_history(20) or []
                except Exception:
                    history = []

                last_start = None
                for h in reversed(history):
                    if h.get('state') == 'start' and h.get('phase') in ('data_collection', 'ai_analysis', 'incremental_cycle'):
                        last_start = h
                        break

                tasks = []
                if last_start:
                    phase = last_start.get('phase')
                    ts = last_start.get('timestamp') or current_time.isoformat()
                    if phase == 'data_collection':
                        label, icon, desc = 'Veri Toplama', '📊', 'Tüm hisseler indiriliyor'
                    elif phase == 'ai_analysis':
                        label, icon, desc = 'AI Pattern Analizi', '🧠', 'Aktif hisseler analiz ediliyor'
                    else:  # incremental_cycle
                        label, icon, desc = 'Incremental Döngü', '🔄', 'Sembol bazlı: veri→analiz→tahmin'
                    tasks.append({
                        'id': 1,
                        'task': label,
                        'description': desc,
                        'status': 'running',
                        'timestamp': ts.split('T')[-1][:8],
                        'icon': icon,
                        'type': f'pipeline_{phase}'
                    })
                else:
                    # Ara durumda: yalnızca scheduler çalışıyor bilgisini göster
                    tasks.append({
                        'id': 1,
                        'task': 'Scheduler',
                        'description': 'Sürekli tam döngü: Toplama→Analiz',
                        'status': 'running',
                        'timestamp': current_time.strftime('%H:%M:%S'),
                        'icon': '⏰',
                        'type': 'scheduler_status'
                    })

                # Scheduler kartını ikinci olarak ekle (varsa yinelenmesin)
                if not any(t.get('type') == 'scheduler_status' for t in tasks):
                    tasks.append({
                        'id': len(tasks) + 1,
                        'task': 'Scheduler',
                        'description': 'Sürekli tam döngü: Toplama→Analiz',
                        'status': 'running',
                        'timestamp': current_time.strftime('%H:%M:%S'),
                        'icon': '⏰',
                        'type': 'scheduler_status'
                    })

                # Bu modda sadece tek/iki kart döndür
                return jsonify({
                    'status': 'success',
                    'tasks': tasks,
                    'count': len(tasks),
                    'last_update': current_time.isoformat(),
                })

            # Otomasyon çalışmıyorsa veya history alınamadıysa; kapsamlı (eski) görünüm
            current_time = datetime.now()
            
            # Get real database stats
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            
            # Get latest data info
            latest_date = db.session.query(func.max(StockPrice.date)).scalar()
            latest_count = 0
            if latest_date:
                latest_count = StockPrice.query.filter_by(date=latest_date).count()
            
            # Get real recent tasks from simulation trades and system activities
            tasks = []
            task_id = 1
            
            # 1. AI Analysis Results (son 1 saat içindeki sinyaller)
            try:
                # Son 1 saatteki toplam sinyalleri say
                one_hour_ago = current_time - timedelta(hours=1)
                
                # Toplam aktif simulation trades
                total_buy_signals = SimulationTrade.query.filter(
                    SimulationTrade.trade_type == 'BUY',
                    SimulationTrade.execution_time >= one_hour_ago
                ).count()
                
                total_sell_signals = SimulationTrade.query.filter(
                    SimulationTrade.trade_type == 'SELL',
                    SimulationTrade.execution_time >= one_hour_ago
                ).count()
                
                if total_buy_signals > 0 or total_sell_signals > 0:
                    tasks.append({
                        'id': task_id,
                        'task': 'AI Sinyal Analizi',
                        'description': f'Son 1 saat: {total_buy_signals} ALIM, {total_sell_signals} SATIM sinyali',
                        'status': 'completed',
                        'timestamp': current_time.strftime('%H:%M:%S'),
                        'icon': '🎯',
                        'type': 'signal_analysis'
                    })
                    task_id += 1
                
                # AI Analizi durumu (pipeline history'den gerçek durum)
                try:
                    history = _read_pipeline_history(20)
                except Exception:
                    history = []

                last_ai = None
                for h in reversed(history):
                    if h.get('phase') == 'ai_analysis':
                        last_ai = h
                        break

                ai_status = 'running'
                ai_desc = f"{Stock.query.filter_by(is_active=True).count()} hisse için 5-katmanlı analiz"
                ai_time = current_time.strftime('%H:%M:%S')
                if last_ai:
                    state = last_ai.get('state')
                    details = last_ai.get('details', {}) or {}
                    ts = last_ai.get('timestamp')
                    if ts:
                        ai_time = ts.split('T')[-1][:8]
                    if state == 'end':
                        ai_status = 'completed'
                        ai_desc = f"{details.get('analyzed', 0)}/{details.get('total', 0)} hisse, {details.get('signals', 0)} sinyal"
                    elif state == 'error':
                        ai_status = 'failed'
                        ai_desc = 'AI analizi hata verdi'
                    else:
                        ai_status = 'running'
                        ai_desc = f"{details.get('analyzed', 0)}+ hisse analiz ediliyor"

                tasks.append({
                    'id': task_id,
                    'task': 'AI Pattern Analizi',
                    'description': ai_desc,
                    'status': ai_status,
                    'timestamp': ai_time,
                    'icon': '🧠',
                    'type': 'ai_analysis'
                })
                task_id += 1
                
            except Exception as e:
                logger.warning(f"AI analiz istatistik hatası: {e}")
            
            # 2. Recent simulation trades
            recent_trades = SimulationTrade.query.join(Stock)\
                .order_by(desc(SimulationTrade.execution_time)).limit(3).all()
            
            for trade in recent_trades:
                tasks.append({
                    'id': task_id,
                    'task': f'{trade.trade_type} Signal',
                    'description': f'{trade.stock.symbol}: {trade.quantity} adet @ {trade.price}₺',
                    'status': 'completed',
                    'timestamp': trade.execution_time.strftime('%H:%M:%S'),
                    'icon': '🟢' if trade.trade_type == 'BUY' else '🔴',
                    'type': 'simulation_trade'
                })
                task_id += 1
            
            # 2. Data collection status (gerçek)
            if latest_count > 0:
                tasks.append({
                    'id': task_id,
                    'task': 'Veri Toplama',
                    'description': f'{latest_count} hisse güncellendi',
                    'status': 'completed',
                    'timestamp': current_time.strftime('%H:%M:%S'),
                    'icon': '📊',
                    'type': 'data_collection'
                })
                task_id += 1
            
            # 3. Active simulation status
            active_sessions = SimulationSession.query.filter_by(status='active').count()
            if active_sessions > 0:
                tasks.append({
                    'id': task_id,
                    'task': 'AI Simulation',
                    'description': f'{active_sessions} aktif simulation çalışıyor',
                    'status': 'running',
                    'timestamp': current_time.strftime('%H:%M:%S'),
                    'icon': '🤖',
                    'type': 'simulation_status'
                })
                task_id += 1
            
            # 4. Scheduler status (dynamic description and actual state)
            pipeline_mode = os.getenv('PIPELINE_MODE', 'SCHEDULED').upper()
            if pipeline_mode == 'CONTINUOUS_FULL':
                sched_desc = 'Sürekli tam döngü: Toplama→Analiz'
            else:
                sched_desc = 'Zamanlanmış pipeline'

            # Determine actual running state from internal pipeline
            sched_status = 'stopped'
            try:
                pipeline = get_pipeline_with_context()
                if pipeline and getattr(pipeline, 'is_running', False):
                    sched_status = 'running'
            except Exception:
                pass

            tasks.append({
                'id': task_id,
                'task': 'Scheduler',
                'description': sched_desc,
                'status': sched_status,
                'timestamp': current_time.strftime('%H:%M:%S'),
                'icon': '⏰',
                'type': 'scheduler_status'
            })
            task_id += 1

            # 5. Pipeline history (data collection / analysis phases)
            try:
                for h in reversed(_read_pipeline_history(6)):
                    phase = h.get('phase', '')
                    state = h.get('state', '')
                    ts = h.get('timestamp', '')
                    icon = '📊' if phase == 'data_collection' else '🧠'
                    label = 'Veri Toplama' if phase == 'data_collection' else 'AI Analizi'
                    status_map = {'start': 'running', 'end': 'completed', 'error': 'failed'}
                    tasks.append({
                        'id': task_id,
                        'task': label,
                        'description': f"{label} {state}",
                        'status': status_map.get(state, 'running'),
                        'timestamp': ts.split('T')[-1][:8] if ts else current_time.strftime('%H:%M:%S'),
                        'icon': icon,
                        'type': f'pipeline_{phase}'
                    })
                    task_id += 1
            except Exception:
                pass
            
            # Update data collection task with real numbers
            for task in tasks:
                if task['type'] == 'data_collection' and total_stocks > 0:
                    task['description'] = f'{total_stocks} hisse, {total_prices:,} fiyat kaydı aktif'
                    if latest_date:
                        task['description'] += f' (Son: {latest_date})'
            
            return jsonify({
                'status': 'success',
                'tasks': tasks,
                'count': len(tasks),
                'last_update': current_time.isoformat(),
                'system_stats': {
                    'stocks': total_stocks,
                    'prices': total_prices,
                    'latest_date': str(latest_date) if latest_date else None
                }
            })
            
        except Exception as e:
            logger.error(f"Recent tasks error: {e}")
            return jsonify({
                'status': 'error', 
                'message': f'Recent tasks hatası: {str(e)}',
                'tasks': []
            }), 500

    # WebSocket Event Handlers
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
                result = get_pattern_detector().analyze_stock(symbol)
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
                }, room=f'stock_{symbol}')
                
                logger.info(f"📊 Pattern analysis sent for {symbol} to {request.sid} and stock room")
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
        }, room='admin')
    
    # Store socketio instance globally for background tasks
    app.socketio = socketio
    app.broadcast_log = broadcast_log
    
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
        return get_automated_pipeline()
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
