# BIST Pattern Detection System

> **Enterprise-grade AI-powered stock pattern detection and prediction system for Borsa Istanbul (BIST)**

## 📊 Project Overview

BIST-Pattern is a comprehensive financial analysis platform that combines advanced AI/ML techniques with traditional technical analysis to detect stock patterns, predict price movements, and provide automated trading signals for the Turkish stock market.

### Key Features

- 🤖 **AI Pattern Detection**: YOLOv8-based visual pattern recognition
- 📈 **ML Price Prediction**: Ensemble models (XGBoost, LightGBM, CatBoost)
- 📰 **Sentiment Analysis**: FinGPT-powered news sentiment analysis
- 🔄 **Automated Analysis**: Continuous monitoring and pattern detection
- 🎯 **Multi-Source Validation**: Cross-validation across BASIC/ADVANCED/YOLO/ML systems
- 📊 **Real-time Dashboard**: Live WebSocket updates and interactive visualizations
- 🔐 **Enterprise Security**: JWT, CSRF protection, rate limiting, audit logging

---

## 🏗️ Architecture

### Technology Stack

**Backend:**
- **Framework**: Flask 2.x + Flask-SocketIO
- **Database**: PostgreSQL 14+
- **Cache/Queue**: Redis 7+
- **Server**: Gunicorn + Gevent workers
- **ORM**: SQLAlchemy 2.x
- **Migration**: Flask-Migrate (Alembic)

**AI/ML:**
- **Visual Detection**: YOLOv8 (Ultralytics)
- **ML Models**: XGBoost, LightGBM, CatBoost
- **Sentiment Analysis**: FinGPT (Hugging Face Transformers)
- **Technical Analysis**: TA-Lib
- **Data Processing**: Pandas, NumPy

**Frontend:**
- **UI Framework**: Bootstrap 5.3
- **Charts**: Chart.js 4.x
- **Real-time**: Socket.IO Client
- **JavaScript**: Vanilla ES6+

### Project Statistics

```
Total Code Lines: ~15,000+
Python Files: 67
Templates: 6 (2,451 lines HTML/JS)
Blueprints: 12 modules
Core Modules: 10 modules
HTTP Routes: 127 endpoints
WebSocket Events: 7
Database Models: 7
```

---

## 📁 Project Structure

```
/opt/bist-pattern/
├── app.py                          # Main Flask application (3,104 lines)
├── config.py                       # Configuration management
├── models.py                       # SQLAlchemy database models
├── pattern_detector.py             # Core pattern detection (1,581 lines)
├── gunicorn.conf.py               # Production server config
│
├── bist_pattern/                   # Core package
│   ├── blueprints/                # Modular routes
│   │   ├── admin_dashboard.py     # Admin panel routes
│   │   ├── api_automation.py      # Automation control API
│   │   ├── api_health.py          # Health check endpoints
│   │   ├── api_internal.py        # Internal API (WebSocket broadcast, etc.)
│   │   ├── api_metrics.py         # System metrics API
│   │   ├── api_public.py          # Public API endpoints
│   │   ├── api_recent.py          # Recent tasks API
│   │   ├── api_simulation.py      # Trading simulation API
│   │   ├── api_watchlist.py       # Watchlist management
│   │   ├── auth.py                # Authentication routes
│   │   ├── web.py                 # Web page routes
│   │   └── register_all.py        # Blueprint registration
│   │
│   └── core/                      # Core business logic
│       ├── auth_manager.py        # Authentication & authorization
│       ├── basic_pattern_detector.py  # Basic TA patterns
│       ├── cache.py               # Caching utilities
│       ├── csrf_security.py       # CSRF protection
│       ├── decorators.py          # Custom decorators
│       ├── ml_coordinator.py      # ML prediction coordination
│       ├── news_sentiment_system.py   # News sentiment analysis
│       ├── pattern_coordinator.py # Pattern detection coordination
│       ├── pattern_validator.py   # Multi-stage pattern validation
│       └── unified_collector.py   # Data collection system
│
├── templates/                     # Jinja2 templates
│   ├── base.html                  # Base template with common layout
│   ├── dashboard.html             # Admin dashboard (1,659 lines)
│   ├── user_dashboard.html        # User interface (1,686 lines)
│   ├── login.html                 # Login page
│   ├── stocks.html                # Stock listing
│   └── analysis.html              # Stock analysis page
│
├── static/                        # Static assets
│   ├── favicon.ico
│   ├── favicon.svg
│   └── pattern_translations.js    # Pattern name translations (TR/EN)
│
├── yolo/                          # YOLO model files
│   └── patterns_all_v7_rectblend.pt  # Trained pattern detection model
│
├── simple_ml_models/              # Simple ML prediction models
├── enhanced_ml_models/            # Advanced ensemble models
│
├── logs/                          # Application logs
├── cache/                         # Hugging Face model cache
├── datasets/                      # YOLO training datasets
│
└── scripts/                       # Utility scripts
    ├── bulk_train_all.py          # Bulk ML model training
    ├── backtest_selection_policy.py   # Backtest configuration
    └── post_train_enhanced_check.py   # Model validation
```

---

## 🔄 Data Flow & Pipelines

### 1. Data Collection Pipeline

```
External Sources → Collector → Database → Cache
     │
     ├─→ Yahoo Finance (yfinance)
     ├─→ Enhanced Yahoo Finance Wrapper (retry, session pooling)
     ├─→ RSS News Feeds (7 sources)
     └─→ Market Data APIs

Flow:
1. unified_collector.py fetches price data
2. Data validated and sanitized
3. Stored in StockPrice table
4. Cached for quick access
```

**Key Module**: `bist_pattern/core/unified_collector.py`
- Handles data collection from multiple sources
- Implements retry logic and error handling
- Manages session pooling for performance
- Validates and sanitizes OHLCV data

### 2. Pattern Detection Pipeline

```
Stock Data → Multi-Stage Detection → Validation → Results
                    │
                    ├─→ BASIC: TA indicators (RSI, MACD, BB)
                    ├─→ ADVANCED: Complex formations (H&S, Double Top/Bottom)
                    ├─→ YOLO: Visual pattern recognition
                    └─→ VALIDATION: Cross-source confirmation

Flow:
1. pattern_detector.py coordinates detection
2. Each detector analyzes independently:
   - Basic: Moving averages, oscillators
   - Advanced: Head & Shoulders, Double Tops, etc.
   - YOLO: Visual chart analysis
3. pattern_validator.py validates findings
4. Results merged and returned
```

**Key Components:**

**A. Basic Pattern Detection**
- File: `bist_pattern/core/basic_pattern_detector.py`
- Detects: MA crossovers, RSI signals, MACD divergences
- Fast, lightweight analysis

**B. Advanced Pattern Detection**
- File: `advanced_patterns.py`
- Detects: HEAD_AND_SHOULDERS, INVERSE_HEAD_AND_SHOULDERS, DOUBLE_TOP, DOUBLE_BOTTOM
- Uses price action analysis

**C. Visual Pattern Detection**
- File: `visual_pattern_detector.py`, `visual_pattern_async.py`
- YOLOv8 model for chart image analysis
- Async processing for performance

**D. Pattern Validation**
- File: `bist_pattern/core/pattern_validator.py`
- Three-stage validation:
  1. **BASIC**: Baseline detection (weight: 0.3)
  2. **ADVANCED**: Confirms structure (weight: 0.3)
  3. **YOLO**: Visual evidence (weight: 0.4)
- Standalone patterns (ADVANCED/YOLO) accepted if confidence ≥ 0.55
- Multi-source confirmation boosts confidence

### 3. ML Prediction Pipeline

```
Historical Data → Feature Engineering → Ensemble Models → Predictions
                          │
                          ├─→ Technical indicators (50+ features)
                          ├─→ Price patterns
                          ├─→ Volume analysis
                          └─→ Sentiment scores

Models:
├─→ XGBoost (Tree-based)
├─→ LightGBM (Gradient boosting)
└─→ CatBoost (Categorical features)

Output:
├─→ 1-day, 3-day, 7-day, 14-day, 30-day predictions
└─→ Confidence scores and delta percentages
```

**Key Modules:**

**A. ML Coordinator**
- File: `bist_pattern/core/ml_coordinator.py`
- Coordinates Basic + Enhanced ML systems
- Manages model ensemble
- Calibrates predictions

**B. Enhanced ML System**
- File: `enhanced_ml_system.py`
- Advanced feature engineering
- Multi-horizon predictions
- Auto-training and model refresh

**C. Simple ML System**
- File: `ml_prediction_system.py`
- Lightweight predictions
- Quick response times

### 4. Sentiment Analysis Pipeline

```
News Sources → Collection → Analysis → Scoring
      │
      ├─→ RSS Feeds (async)
      ├─→ Financial news sites
      └─→ Stock-specific news

Processing:
1. news_sentiment_async.py collects news
2. fingpt_analyzer.py analyzes sentiment
3. Scores aggregated per stock
4. Integrated into pattern analysis
```

**Key Files:**
- `fingpt_analyzer.py`: FinGPT-based sentiment analysis
- `news_sentiment_async.py`: Async news collection
- `rss_news_async.py`: RSS feed parser

### 5. Automation Pipeline

```
Scheduler → Stock Selection → Analysis → Storage → Broadcasting
    │
    ├─→ Continuous mode (APScheduler)
    ├─→ Priority stocks first
    └─→ Configurable intervals

Flow:
1. working_automation.py manages scheduler
2. Stocks analyzed in batches
3. Results stored in database
4. WebSocket broadcasts to connected clients
5. Metrics tracked and logged
```

**Key File**: `working_automation.py`
- Implements continuous analysis scheduling
- Manages thread pools
- Broadcasts updates via WebSocket
- Tracks automation metrics

---

## 🔌 API Endpoints

### Authentication & User Management

```http
POST   /login                    # User login
POST   /logout                   # User logout
GET    /auth/google             # Google OAuth
GET    /auth/google/callback    # OAuth callback
GET    /auth/apple              # Apple OAuth
GET    /auth/apple/callback     # OAuth callback
```

### Stock Data & Analysis

```http
GET    /api/stocks                          # List all stocks
GET    /api/stock-prices/<symbol>           # Price history
GET    /api/pattern-analysis/<symbol>       # Pattern analysis for stock
GET    /api/pattern-summary                 # Pattern summary (priority stocks)
GET    /api/visual-analysis/<symbol>        # YOLO visual analysis
GET    /api/user/predictions/<symbol>       # ML predictions
```

### Watchlist Management

```http
GET    /api/watchlist                       # Get user's watchlist
POST   /api/watchlist                       # Add to watchlist
DELETE /api/watchlist/<symbol>              # Remove from watchlist
```

### Dashboard & Statistics

```http
GET    /api/dashboard-stats                 # Dashboard statistics
GET    /api/system-info                     # System information
GET    /api/recent-tasks                    # Recent analysis tasks
```

### Automation Control

```http
GET    /api/automation/status               # Automation status
POST   /api/automation/start                # Start automation
POST   /api/automation/stop                 # Stop automation
GET    /api/automation/health               # Health check
GET    /api/automation/pipeline-history     # Analysis history
```

### Internal APIs

```http
POST   /api/internal/broadcast-log          # Broadcast log message (internal)
GET    /api/internal/automation/status      # Internal automation status
```

### Simulation

```http
GET    /api/simulation/sessions             # List simulation sessions
POST   /api/simulation/sessions             # Create simulation
GET    /api/simulation/sessions/<id>        # Get session details
POST   /api/simulation/sessions/<id>/start  # Start simulation
```

### Admin Panel

```http
GET    /admin                               # Admin dashboard
GET    /admin/users                         # User management
GET    /admin/metrics                       # System metrics
GET    /admin/logs                          # System logs
```

---

## 🔌 WebSocket Events

### Client → Server

```javascript
// Connection management
socket.on('connect')
socket.on('disconnect')

// Room management
socket.emit('join_admin')               // Join admin room
socket.emit('join_user')                // Join user room
socket.emit('subscribe_stock', {symbol})   // Subscribe to stock updates
socket.emit('unsubscribe_stock', {symbol}) // Unsubscribe

// Request analysis
socket.emit('request_pattern_analysis', {symbol})
```

### Server → Client

```javascript
// Log broadcasts
socket.on('log_broadcast', (data) => {
    // { level, message, category, timestamp }
})

// Analysis updates
socket.on('analysis_complete', (data) => {
    // { symbol, patterns, predictions, timestamp }
})

// Live signals
socket.on('live_signal', (data) => {
    // { symbol, signal, confidence, patterns }
})

// Automation status
socket.on('automation_status_update', (data) => {
    // { is_running, scheduled_jobs, next_runs }
})

// Simulation updates
socket.on('simulation_trade', (data) => {
    // { session_id, trade, timestamp }
})
```

---

## 💾 Database Models

### Core Models

**1. User**
```python
- id: Integer (PK)
- username: String(80), unique
- email: String(120), unique
- password_hash: String(255)
- role: String(20)  # 'admin' or 'user'
- created_at: DateTime
- last_login: DateTime
- last_login_ip: String(45)
```

**2. Stock**
```python
- id: Integer (PK)
- symbol: String(10), unique
- name: String(200)
- sector: String(100)
- created_at: DateTime
- updated_at: DateTime
```

**3. StockPrice**
```python
- id: Integer (PK)
- stock_id: Integer (FK → Stock)
- date: Date
- open_price: Numeric(10, 2)
- high_price: Numeric(10, 2)
- low_price: Numeric(10, 2)
- close_price: Numeric(10, 2)
- volume: BigInteger
- created_at: DateTime
```

**4. Watchlist**
```python
- id: Integer (PK)
- user_id: Integer (FK → User)
- stock_id: Integer (FK → Stock)
- added_at: DateTime
- notes: Text
```

**5. SimulationSession**
```python
- id: Integer (PK)
- user_id: Integer (FK → User)
- name: String(100)
- initial_capital: Numeric(15, 2)
- current_capital: Numeric(15, 2)
- status: String(20)  # 'active', 'paused', 'completed'
- start_date: DateTime
- end_date: DateTime
```

**6. SimulationTrade**
```python
- id: Integer (PK)
- session_id: Integer (FK → SimulationSession)
- stock_id: Integer (FK → Stock)
- trade_type: String(10)  # 'BUY' or 'SELL'
- quantity: Integer
- price: Numeric(10, 2)
- timestamp: DateTime
```

**7. AuditLog**
```python
- id: Integer (PK)
- user_id: Integer (FK → User, nullable)
- action: String(100)
- details: Text
- ip_address: String(45)
- timestamp: DateTime
```

---

## ⚙️ Configuration

### Environment Variables

All configuration is managed through environment variables defined in:
`/etc/systemd/system/bist-pattern.service.d/99-consolidated.conf`

**Core System:**
```bash
PYTHONPATH=/opt/bist-pattern
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

**Database:**
```bash
DATABASE_URL=postgresql://user:pass@localhost:5432/bist_pattern_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bist_pattern_db
DB_USER=bist_user
```

**Redis:**
```bash
REDIS_URL=redis://127.0.0.1:6379/0
SOCKETIO_MESSAGE_QUEUE=redis://127.0.0.1:6379/0
```

**Security:**
```bash
FLASK_SECRET_KEY=<generated>
JWT_SECRET_KEY=<generated>
INTERNAL_API_TOKEN=<generated>
SESSION_COOKIE_SECURE=True
CSRF_ENABLED=True
```

**Pattern Detection:**
```bash
ENABLE_YOLO=True
YOLO_MODEL_PATH=/opt/bist-pattern/yolo/patterns_all_v7_rectblend.pt
YOLO_MIN_CONF=0.45

ENABLE_PATTERN_VALIDATION=True
PATTERN_MIN_VALIDATION_CONF=0.5
PATTERN_STANDALONE_MIN_CONF=0.55
PATTERN_BASIC_WEIGHT=0.3
PATTERN_ADVANCED_WEIGHT=0.3
PATTERN_YOLO_WEIGHT=0.4
PATTERN_MATCH_THRESHOLD=0.7
```

**ML Configuration:**
```bash
ML_MIN_DATA_DAYS=200
ML_MAX_MODEL_AGE_DAYS=7
ML_TRAINING_COOLDOWN_HOURS=6
```

**Automation:**
```bash
AUTOMATION_INTERVAL_MINUTES=15
AUTOMATION_WORKERS=2
PATTERN_COORDINATOR_WORKERS=1
```

---

## 🚀 Deployment

### System Requirements

- **OS**: Ubuntu 22.04 LTS or newer
- **Python**: 3.10+
- **PostgreSQL**: 14+
- **Redis**: 7+
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 50GB minimum
- **CPU**: 4 cores minimum

### Installation

1. **Clone repository:**
```bash
cd /opt
git clone <repository-url> bist-pattern
cd bist-pattern
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup database:**
```bash
# Create PostgreSQL database and user
sudo -u postgres psql
CREATE DATABASE bist_pattern_db;
CREATE USER bist_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE bist_pattern_db TO bist_user;
\q

# Run migrations
flask db upgrade
```

5. **Configure systemd:**
```bash
sudo cp /etc/systemd/system/bist-pattern.service.d/99-consolidated.conf.example \
        /etc/systemd/system/bist-pattern.service.d/99-consolidated.conf
sudo nano /etc/systemd/system/bist-pattern.service.d/99-consolidated.conf
# Edit configuration as needed

sudo systemctl daemon-reload
sudo systemctl enable bist-pattern
sudo systemctl start bist-pattern
```

6. **Verify installation:**
```bash
sudo systemctl status bist-pattern
curl http://localhost:5000/health
```

### Production Checklist

- [ ] Generate secure secrets (FLASK_SECRET_KEY, JWT_SECRET_KEY)
- [ ] Configure PostgreSQL with proper credentials
- [ ] Setup Redis with authentication
- [ ] Configure HTTPS/SSL (nginx reverse proxy)
- [ ] Setup firewall rules
- [ ] Configure log rotation
- [ ] Setup automated backups
- [ ] Test disaster recovery
- [ ] Monitor system resources
- [ ] Setup alerting (email, Slack, etc.)

---

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Coverage report
pytest --cov=bist_pattern tests/
```

### Manual Testing
```bash
# Test pattern detection
curl http://localhost:5000/api/pattern-analysis/THYAO | jq

# Test ML predictions
curl http://localhost:5000/api/user/predictions/AKBNK | jq

# Test automation status
curl -H "Authorization: Bearer <token>" \
     http://localhost:5000/api/automation/status | jq
```

---

## 📊 Performance

### Optimization Strategies

1. **Caching:**
   - Pattern analysis results cached (TTL: 300s)
   - Stock price data cached (TTL: 60s)
   - Redis-backed cache for scalability

2. **Async Processing:**
   - YOLO analysis runs in background threads
   - News collection uses async HTTP
   - WebSocket broadcasts non-blocking

3. **Database:**
   - Indexed foreign keys
   - Connection pooling
   - Query optimization

4. **Rate Limiting:**
   - API endpoints rate-limited
   - Per-user quotas
   - Burst protection

### Monitoring

```bash
# System metrics
curl http://localhost:5000/api/metrics

# Recent tasks
curl http://localhost:5000/api/recent-tasks

# Automation health
curl http://localhost:5000/api/automation/health
```

---

## 🔒 Security

### Implemented Measures

1. **Authentication:**
   - JWT tokens for API access
   - Session-based auth for web interface
   - OAuth support (Google, Apple)
   - Password hashing (bcrypt)

2. **Authorization:**
   - Role-based access control (RBAC)
   - Admin/User roles
   - Route-level decorators

3. **Protection:**
   - CSRF protection (Flask-WTF)
   - Rate limiting (Flask-Limiter)
   - SQL injection prevention (SQLAlchemy ORM)
   - XSS protection (Jinja2 auto-escaping)

4. **Audit:**
   - All admin actions logged
   - IP tracking
   - Login history

---

## 📝 License

[Your License Here]

---

## 👥 Contributors

[Your Team/Contributors]

---

## 📞 Support

For issues and questions:
- **Email**: [your-email]
- **GitHub Issues**: [repo-url/issues]

---

**Last Updated**: September 30, 2025
**Version**: 1.0.0 (Pre-Refactor)
