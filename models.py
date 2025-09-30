"""
Database Models for BIST Pattern Detection
PostgreSQL + SQLAlchemy Implementation
"""
# flake8: noqa
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Enhanced User Model with OAuth and Email Verification"""
    __tablename__ = 'users'
    __table_args__ = (
        db.Index('idx_user_last_login', 'last_login'),
    )
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)  # Null for OAuth users
    
    # User Profile
    first_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=True)
    avatar_url = db.Column(db.String(500), nullable=True)
    
    # Authentication Provider
    provider = db.Column(db.String(20), default='email')  # email, google, apple
    provider_id = db.Column(db.String(255), nullable=True)  # OAuth provider ID
    
    # Email Verification
    email_verified = db.Column(db.Boolean, default=False)
    email_verification_token = db.Column(db.String(255), nullable=True)
    email_verification_sent_at = db.Column(db.DateTime, nullable=True)
    
    # Account Status
    is_active = db.Column(db.Boolean, default=True)
    is_premium = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    # RBAC / Security
    role = db.Column(db.String(20), nullable=False, default='user', index=True)
    last_login_ip = db.Column(db.String(45), nullable=True)
    
    # Settings
    timezone = db.Column(db.String(50), default='Europe/Istanbul')
    language = db.Column(db.String(5), default='tr')
    email_notifications = db.Column(db.Boolean, default=True)
    push_notifications = db.Column(db.Boolean, default=True)
    
    # Relationships
    watchlist = db.relationship('Watchlist', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def __init__(self, email, **kwargs):
        self.email = email.lower()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password"""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)
    
    def generate_verification_token(self):
        """Generate email verification token"""
        self.email_verification_token = str(uuid.uuid4())
        self.email_verification_sent_at = datetime.utcnow()
        return self.email_verification_token
    
    @property
    def full_name(self):
        """Get full name"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username or self.email.split('@')[0]
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'full_name': self.full_name,
            'avatar_url': self.avatar_url,
            'provider': self.provider,
            'email_verified': self.email_verified,
            'is_premium': self.is_premium,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.email}>'

class Stock(db.Model):
    """Stock Information"""
    __tablename__ = 'stocks'
    
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(255), nullable=False)
    sector = db.Column(db.String(100), nullable=True)
    market_cap = db.Column(db.BigInteger, nullable=True)
    
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    prices = db.relationship('StockPrice', backref='stock', lazy='dynamic', cascade='all, delete-orphan')
    watchlist_items = db.relationship('Watchlist', backref='stock', lazy='dynamic')
    
    def __repr__(self):
        return f'<Stock {self.symbol}: {self.name}>'

class StockPrice(db.Model):
    """Stock Price Data"""
    __tablename__ = 'stock_prices'
    
    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    
    date = db.Column(db.Date, nullable=False, index=True)
    open_price = db.Column(db.Numeric(10, 4), nullable=False)
    high_price = db.Column(db.Numeric(10, 4), nullable=False)
    low_price = db.Column(db.Numeric(10, 4), nullable=False)
    close_price = db.Column(db.Numeric(10, 4), nullable=False)
    volume = db.Column(db.BigInteger, nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('stock_id', 'date', name='unique_stock_date'),
        db.Index('idx_stock_date', 'stock_id', 'date')
    )
    
    def __repr__(self):
        return f'<StockPrice {self.stock.symbol} {self.date}: {self.close_price}>'

class Watchlist(db.Model):
    """User Watchlist"""
    __tablename__ = 'watchlist'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    
    # Watchlist Settings
    notes = db.Column(db.Text, nullable=True)
    alert_enabled = db.Column(db.Boolean, default=True)
    alert_threshold_buy = db.Column(db.Numeric(10, 4), nullable=True)
    alert_threshold_sell = db.Column(db.Numeric(10, 4), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('user_id', 'stock_id', name='unique_user_stock'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': getattr(self.stock, 'symbol', None),
            'name': getattr(self.stock, 'name', None),
            'notes': self.notes,
            'alert_enabled': self.alert_enabled,
            'alert_threshold_buy': float(self.alert_threshold_buy) if self.alert_threshold_buy else None,
            'alert_threshold_sell': float(self.alert_threshold_sell) if self.alert_threshold_sell else None,
            'created_at': (self.created_at.isoformat() if self.created_at else None)
        }
    
    def __repr__(self):
        return f'<Watchlist {self.user.email}: {self.stock.symbol}>'


# ==========================================
# PAPER TRADING / SIMULATION MODELS
# ==========================================

class SimulationSession(db.Model):
    """Paper Trading Simulation Session"""
    __tablename__ = 'simulation_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Session Config
    session_name = db.Column(db.String(100), nullable=False, default='AI Performance Test')
    initial_balance = db.Column(db.Numeric(15, 2), nullable=False, default=100.00)
    duration_hours = db.Column(db.Integer, nullable=False, default=48)
    
    # Session Status
    status = db.Column(db.String(20), nullable=False, default='active')  # active, completed, paused
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    
    # Performance Metrics
    current_balance = db.Column(db.Numeric(15, 2), nullable=False, default=100.00)
    total_trades = db.Column(db.Integer, nullable=False, default=0)
    winning_trades = db.Column(db.Integer, nullable=False, default=0)
    losing_trades = db.Column(db.Integer, nullable=False, default=0)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trades = db.relationship('SimulationTrade', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    snapshots = db.relationship('PortfolioSnapshot', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    
    @property
    def profit_loss(self):
        """Calculate profit/loss"""
        return float(self.current_balance) - float(self.initial_balance)
    
    @property
    def profit_loss_percentage(self):
        """Calculate profit/loss percentage"""
        if float(self.initial_balance) == 0:
            return 0
        return (self.profit_loss / float(self.initial_balance)) * 100
    
    @property
    def win_rate(self):
        """Calculate win rate percentage"""
        if self.total_trades == 0:
            return 0
        return (self.winning_trades / self.total_trades) * 100
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_name': self.session_name,
            'initial_balance': float(self.initial_balance),
            'current_balance': float(self.current_balance),
            'duration_hours': self.duration_hours,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'profit_loss': self.profit_loss,
            'profit_loss_percentage': round(self.profit_loss_percentage, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<SimulationSession {self.id}: {self.session_name} - {self.status}>'


class SimulationTrade(db.Model):
    """Individual Paper Trading Transaction"""
    __tablename__ = 'simulation_trades'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('simulation_sessions.id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    
    # Trade Details
    trade_type = db.Column(db.String(10), nullable=False)  # BUY, SELL
    quantity = db.Column(db.Numeric(10, 4), nullable=False)  # Fractional shares için
    price = db.Column(db.Numeric(10, 4), nullable=False)
    total_amount = db.Column(db.Numeric(15, 2), nullable=False)
    
    # Signal Information
    signal_source = db.Column(db.String(50), nullable=True)  # MACD, RSI, PATTERN, etc.
    signal_confidence = db.Column(db.Numeric(5, 2), nullable=True)
    pattern_detected = db.Column(db.String(50), nullable=True)  # DOUBLE_TOP, etc.
    
    # Trade Status
    status = db.Column(db.String(20), nullable=False, default='executed')  # executed, pending, cancelled
    execution_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Performance (for closed positions)
    profit_loss = db.Column(db.Numeric(15, 2), nullable=True)
    profit_loss_percentage = db.Column(db.Numeric(5, 2), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = db.relationship('Stock', backref='simulation_trades')
    
    def calculate_profit_loss(self, sell_price):
        """Calculate profit/loss for a position"""
        if self.trade_type == 'BUY':
            return (float(sell_price) - float(self.price)) * self.quantity
        else:
            return (float(self.price) - float(sell_price)) * self.quantity
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.stock.symbol,
            'trade_type': self.trade_type,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'total_amount': float(self.total_amount),
            'signal_source': self.signal_source,
            'signal_confidence': float(self.signal_confidence) if self.signal_confidence else None,
            'pattern_detected': self.pattern_detected,
            'status': self.status,
            'execution_time': self.execution_time.isoformat(),
            'profit_loss': float(self.profit_loss) if self.profit_loss else None,
            'profit_loss_percentage': float(self.profit_loss_percentage) if self.profit_loss_percentage else None,
            'created_at': self.created_at.isoformat()
        }
    
    def __repr__(self):
        return f'<SimulationTrade {self.trade_type} {self.quantity}x{self.stock.symbol} @ {self.price}>'


class PortfolioSnapshot(db.Model):
    """Portfolio Balance Snapshots for Performance Tracking"""
    __tablename__ = 'portfolio_snapshots'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('simulation_sessions.id'), nullable=False)
    
    # Portfolio State
    cash_balance = db.Column(db.Numeric(15, 2), nullable=False)
    total_portfolio_value = db.Column(db.Numeric(15, 2), nullable=False)
    total_stocks_value = db.Column(db.Numeric(15, 2), nullable=False, default=0)
    
    # Performance Metrics
    total_profit_loss = db.Column(db.Numeric(15, 2), nullable=False, default=0)
    total_profit_loss_percentage = db.Column(db.Numeric(5, 2), nullable=False, default=0)
    
    # Active Positions Count
    active_positions = db.Column(db.Integer, nullable=False, default=0)
    
    # Timestamp
    snapshot_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'cash_balance': float(self.cash_balance),
            'total_portfolio_value': float(self.total_portfolio_value),
            'total_stocks_value': float(self.total_stocks_value),
            'total_profit_loss': float(self.total_profit_loss),
            'total_profit_loss_percentage': float(self.total_profit_loss_percentage),
            'active_positions': self.active_positions,
            'snapshot_time': self.snapshot_time.isoformat()
        }
    
    def __repr__(self):
        return f'<PortfolioSnapshot {self.session_id}: {self.total_portfolio_value} TL>'
