# 🎯 BIST AI Pattern Detection & Trading System

**Advanced AI-powered stock analysis and paper trading system for BIST (Borsa Istanbul) stocks.**

## 🚀 Features

### 🧠 5-Layer AI Analysis System
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Advanced Patterns**: Head & Shoulders, Double Top/Bottom, Triangles, Cup & Handle
- **Visual AI (YOLOv8)**: Computer vision pattern detection on charts
- **ML Predictions**: XGBoost, LightGBM, LSTM, CatBoost ensemble
- **Hybrid Decision Engine**: Weighted confidence scoring

### 📊 Real-Time Features
- **606 BIST Stocks**: Comprehensive coverage of all BIST stocks
- **30-minute Analysis**: Full AI analysis every 30 minutes
- **15-minute Data Collection**: Real-time price data updates
- **WebSocket Integration**: Live dashboard updates
- **User Watchlists**: Personalized stock tracking

### 💰 Paper Trading Simulation
- **Unlimited Budget**: Virtual trading with parametric amounts
- **Real AI Signals**: Trades based on actual AI analysis (>60% confidence)
- **Performance Tracking**: Detailed P&L, win rates, trade history
- **Multi-timeframe**: 12h, 24h, 48h, 72h, 1 week simulations

### 🎨 Modern Dashboard
- **Admin Panel**: System monitoring, automation control
- **User Dashboard**: Personal watchlists, real-time signals
- **Confidence Tooltips**: Signal strength explanations
- **Recent Tasks**: Live system activity feed

## 🛠 Tech Stack

- **Backend**: Python Flask, PostgreSQL, SQLAlchemy
- **AI/ML**: scikit-learn, XGBoost, LightGBM, TensorFlow, YOLOv8
- **Real-time**: WebSocket (Flask-SocketIO)
- **Data Source**: Yahoo Finance API
- **Frontend**: Bootstrap 5, JavaScript
- **Infrastructure**: Nginx, Gunicorn, Systemd
- **Scheduler**: APScheduler for automated tasks

## 📋 Installation

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv postgresql nginx
```

### Setup
```bash
# Clone repository
git clone https://github.com/ersinarikan/bist.git
cd bist

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Database setup
sudo -u postgres createdb bist_pattern
sudo -u postgres createuser bist_user

# Environment configuration
cp .env.example .env
# Edit .env with your database credentials

# Initialize database
flask db upgrade

# Run stock data collection
python3 advanced_collector.py

# Start services
sudo systemctl start bist-pattern
sudo systemctl start bist-scheduler
```

## 🎮 Usage

### Admin Dashboard
Access: `http://localhost:5000/dashboard`
- Monitor system health
- Control automation
- View AI simulation results
- Check recent activities

### User Dashboard  
Access: `http://localhost:5000/user`
- Add stocks to watchlist
- Receive personalized signals
- View confidence explanations
- Real-time updates

### API Endpoints
```python
# Get pattern analysis
GET /api/pattern-analysis/{symbol}

# Start simulation
POST /api/simulation/start

# Search stocks
GET /api/stocks/search?q={query}

# Recent tasks
GET /api/recent-tasks
```

## 🧪 AI Model Details

### Confidence Scoring
- **85-100%**: All AI models agree - Very strong signal
- **70-84%**: Strong signal - Some mixed data
- **55-69%**: Weak signal - Be careful
- **<55%**: Very weak signal - No action recommended

### Signal Generation
1. **Data Collection**: 15-minute intervals for all 606 stocks
2. **Technical Analysis**: TA-Lib indicators calculation
3. **Pattern Detection**: Advanced pattern recognition
4. **Visual Analysis**: YOLOv8 chart pattern detection
5. **ML Prediction**: 4-model ensemble prediction
6. **Final Decision**: Weighted confidence scoring

## 📈 Performance

- **Analysis Speed**: 606 stocks in ~30 minutes
- **Signal Accuracy**: 60%+ confidence threshold
- **Real-time Latency**: <2 seconds WebSocket updates
- **Data Freshness**: 15-minute price updates
- **Uptime**: 99.9% with systemd monitoring

## 🔧 Configuration

### Environment Variables
```bash
FLASK_ENV=production
DATABASE_URL=postgresql://user:pass@localhost/bist_pattern
SECRET_KEY=your-secret-key
REDIS_URL=redis://localhost:6379
```

### Scheduler Configuration
- **Data Collection**: Every 15 minutes
- **AI Analysis**: Every 30 minutes  
- **Model Training**: Daily at 20:00
- **Health Checks**: Every 15 minutes

## 📊 Monitoring

### Systemd Services
```bash
# Check service status
sudo systemctl status bist-pattern
sudo systemctl status bist-scheduler

# View logs
journalctl -u bist-pattern -f
journalctl -u bist-scheduler -f
```

### Performance Metrics
- System health indicators
- Real-time trade execution stats
- AI analysis success rates
- User engagement metrics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Yahoo Finance for stock data API
- BIST for market structure
- YOLOv8 for computer vision capabilities
- Open source ML libraries

## 📞 Support

For support, please open an issue on GitHub or contact the development team.

---

**⚡ Built with AI-First approach for next-generation trading analysis ⚡**
