import os
from datetime import datetime, timedelta
from decimal import Decimal
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_login import LoginManager
from flask_mail import Mail
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit, join_room, leave_room
from config import config
from models import db, User, Stock, StockPrice
import logging
import threading
import time

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

def create_app(config_name=None):
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'production')
    
    # Ensure template directory exists
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir, exist_ok=True)
    
    app = Flask(__name__, template_folder=template_dir)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Login Manager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Mail
    mail = Mail(app)
    
    # Migration
    migrate = Migrate(app, db)
    
    # Routes
    @app.route('/')
    def index():
        return jsonify({
            "message": "BIST Pattern Detection API",
            "status": "running",
            "version": "2.2.0",
            "database": "PostgreSQL", 
            "features": ["Real-time Data", "Yahoo Finance", "Scheduler", "Dashboard", "Automation"]
        })

    @app.route('/dashboard')
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
            
            return render_template('dashboard.html')
        except Exception as e:
            logger.error(f"Dashboard render error: {e}")
            return jsonify({
                'error': 'Dashboard render failed',
                'message': str(e),
                'status': 'render_error'
            }), 500
    
    @app.route('/user')
    def user_dashboard():
        """User interface for stock tracking and signals"""
        return render_template('user_dashboard.html')
    
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
            return jsonify({"status": "unhealthy", "error": str(e)}), 500
    

    
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
            
            return jsonify({
                'total_stocks': total_stocks,
                'total_prices': total_prices,
                'top_stocks': [{'symbol': s[0], 'count': s[1]} for s in stock_with_most_data],
                'sectors': [{'sector': s[0], 'count': s[1]} for s in sector_stats],
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
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
        return jsonify({
            'status': 'success',
            'message': 'Test endpoint çalışıyor',
            'timestamp': datetime.now().isoformat(),
            'version': '2.1.0'
        })
    
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
    def internal_broadcast_log():
        """Internal endpoint for broadcasting logs from scheduler daemon"""
        try:
            data = request.get_json()
            level = data.get('level', 'INFO')
            message = data.get('message', '')
            category = data.get('category', 'system')
            
            # Broadcast log to connected clients
            app.broadcast_log(level, message, category)
            
            return jsonify({'status': 'success', 'message': 'Log broadcasted'})
        except Exception as e:
            logger.error(f"Internal broadcast error: {e}")
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
        
        admin = User(email='admin@bistpattern.com')
        admin.username = 'admin'
        admin.password_hash = generate_password_hash('admin123')
        admin.first_name = 'System'
        admin.last_name = 'Administrator'
        admin.is_active = True
        admin.is_email_verified = True
        
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
                'database': {
                    'stocks': Stock.query.count(),
                    'price_records': StockPrice.query.count()
                }
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

    # ================================
    # AUTOMATED PIPELINE ENDPOINTS
    # ================================

    @app.route('/api/automation/start', methods=['POST'])
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
    def stop_automation():
        """Automated Pipeline'ı durdur"""
        try:
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Automated Pipeline sistemi mevcut değil'
                }), 503
            
            pipeline = get_pipeline_with_context()
            
            if not pipeline.is_running:
                return jsonify({
                    'status': 'already_stopped',
                    'message': 'Automated Pipeline zaten durmuş'
                })
            
            success = pipeline.stop_scheduler()
            
            if success:
                return jsonify({
                    'status': 'stopped',
                    'message': 'Automated Pipeline başarıyla durduruldu',
                    'timestamp': datetime.now().isoformat()
                })
            else:
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
    def automation_status():
        """Automated Pipeline durumu - External Scheduler Daemon kontrol et"""
        try:
            # External scheduler daemon status check
            external_scheduler_status = _check_external_scheduler()
            
            # Backward compatibility için internal scheduler da kontrol et
            internal_status = {'is_running': False, 'scheduled_jobs': 0, 'next_runs': [], 'thread_alive': False}
            if AUTOMATED_PIPELINE_AVAILABLE:
                try:
                    pipeline = get_pipeline_with_context()
                    internal_status = pipeline.get_scheduler_status()
                except:
                    pass
            
            # External scheduler varsa onu prioritize et
            final_status = external_scheduler_status if external_scheduler_status['is_running'] else internal_status
            
            return jsonify({
                'status': 'success',
                'available': True,
                'scheduler_status': final_status,
                'external_scheduler': external_scheduler_status,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Automation status error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Automation status hatası: {str(e)}'
            }), 500

    def _check_external_scheduler():
        """External scheduler daemon durumunu kontrol et"""
        try:
            pid_file = '/opt/bist-pattern/scheduler_daemon.pid'
            
            if not os.path.exists(pid_file):
                return {
                    'is_running': False,
                    'message': 'PID file not found',
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
                    'message': f'External scheduler running with PID {pid}',
                    'pid': pid,
                    'scheduled_jobs': 1,  # External scheduler always has jobs
                    'next_runs': [{'job': 'external_scheduler', 'next_run': 'continuous'}],
                    'thread_alive': True
                }
            except OSError:
                # Process yok, PID file eski
                return {
                    'is_running': False,
                    'message': f'Process with PID {pid} not found',
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
    def automation_health():
        """Sistem sağlık kontrolü"""
        try:
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Automated Pipeline sistemi mevcut değil'
                }), 503
            
            # Health check migrated to daemon - return basic status
            health_status = {
                'overall_status': 'migrated',
                'message': 'Health check is now handled by scheduler_daemon.py',
                'systems': {
                    'automation': {'status': 'active', 'details': 'Running via daemon'},
                    'flask_api': {'status': 'healthy', 'details': 'API endpoints working'}
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                'status': 'success',
                'health_check': health_status,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Health check hatası: {str(e)}'
            }), 500

    @app.route('/api/automation/run-task/<task_name>', methods=['POST'])
    def run_manual_task(task_name):
        """Manuel görev çalıştır"""
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
    def process_simulation_signal():
        """Pattern signal'i simulation engine'e gönder"""
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

    @app.route('/api/recent-tasks')
    def recent_tasks():
        """Recent Tasks endpoint for dashboard"""
        try:
            from sqlalchemy import func, desc
            from models import SimulationSession, SimulationTrade
            
            # Simulated task history - in production, this would come from a task history table
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
                
                # Toplam analiz edilen hisse sayısı (yaklaşık)
                total_stocks = Stock.query.filter_by(is_active=True).count()
                tasks.append({
                    'id': task_id,
                    'task': 'AI Pattern Analizi',
                    'description': f'{total_stocks} hisse için 5-katmanlı analiz aktif',
                    'status': 'running',
                    'timestamp': current_time.strftime('%H:%M:%S'),
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
            
            # 4. Scheduler status
            tasks.append({
                'id': task_id,
                'task': 'Scheduler',
                'description': 'Her 15 dakikada veri çekiyor',
                'status': 'running',
                'timestamp': current_time.strftime('%H:%M:%S'),
                'icon': '⏰',
                'type': 'scheduler_status'
            })
            
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
app = create_app()
socketio = app.socketio

if __name__ == '__main__':
    # Environment variables'dan değerleri al
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # SocketIO ile çalıştır
    socketio.run(app, host=host, port=port, debug=debug)
