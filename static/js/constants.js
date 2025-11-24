/**
 * User Dashboard Constants
 * Tüm magic number ve string sabitler burada
 */

// Signal confidence thresholds
export const SIGNAL_CONFIDENCE = {
  VERY_HIGH: 85,
  HIGH: 70,
  MEDIUM: 60,
  LOW: 55,
  SAFE_HOLD: 65
};

// Movement thresholds (percentage)
export const MOVEMENT_THRESHOLDS = {
  SMALL: 0.5,
  MEDIUM: 1.5,
  CALIBRATED: {
    '1d': 0.008,  // 0.8%
    '3d': 0.021,  // 2.1%
    '7d': 0.030,  // 3.0%
    '14d': 0.030, // 3.0%
    '30d': 0.025  // 2.5%
  }
};

// Horizon definitions
export const HORIZONS = {
  '1d': { label: '1 Gün', days: 1 },
  '3d': { label: '3 Gün', days: 3 },
  '7d': { label: '7 Gün', days: 7 },
  '14d': { label: '14 Gün', days: 14 },
  '30d': { label: '30 Gün', days: 30 }
};

// Model types and labels
export const MODELS = {
  BASIC: 'basic',
  ENHANCED: 'enhanced'
};

export const MODEL_LABELS = {
  [MODELS.BASIC]: 'Temel',
  [MODELS.ENHANCED]: 'Gelişmiş'
};

// Pattern sources
export const PATTERN_SOURCES = {
  ML_PREDICTOR: 'ML_PREDICTOR',
  ENHANCED_ML: 'ENHANCED_ML',
  VISUAL_YOLO: 'VISUAL_YOLO',
  ADVANCED_TA: 'ADVANCED_TA',
  FINGPT: 'FINGPT',
  BASIC: 'BASIC'
};

export const SOURCE_LABELS = {
  [PATTERN_SOURCES.ML_PREDICTOR]: 'Temel Analiz',
  [PATTERN_SOURCES.ENHANCED_ML]: 'Gelişmiş Analiz',
  [PATTERN_SOURCES.VISUAL_YOLO]: 'Görsel',
  [PATTERN_SOURCES.ADVANCED_TA]: 'Teknik Analiz',
  [PATTERN_SOURCES.FINGPT]: 'Sezgisel',
  [PATTERN_SOURCES.BASIC]: 'Temel'
};

// Badge colors for pattern sources
export const BADGE_COLORS = {
  [PATTERN_SOURCES.VISUAL_YOLO]: 'primary',
  [PATTERN_SOURCES.ENHANCED_ML]: 'warning',
  [PATTERN_SOURCES.ML_PREDICTOR]: 'info',
  [PATTERN_SOURCES.FINGPT]: 'success',
  [PATTERN_SOURCES.ADVANCED_TA]: 'danger',
  default: 'secondary'
};

// Signal types
export const SIGNALS = {
  BULLISH: 'BULLISH',
  BEARISH: 'BEARISH',
  HOLD: 'HOLD',
  NEUTRAL: 'NEUTRAL'
};

// Signal labels (Turkish)
export const SIGNAL_LABELS = {
  [SIGNALS.BULLISH]: {
    high: 'Yüksek alım sinyali',
    medium: 'Alım sinyali',
    low: 'Bekleme'
  },
  [SIGNALS.BEARISH]: {
    high: 'Yüksek satış sinyali',
    medium: 'Satış sinyali',
    low: 'Bekleme'
  },
  [SIGNALS.HOLD]: 'Bekleme',
  [SIGNALS.NEUTRAL]: 'Nötr'
};

// CSS classes for signals
export const SIGNAL_CLASSES = {
  [SIGNALS.BULLISH]: 'signal-buy',
  [SIGNALS.BEARISH]: 'signal-sell',
  [SIGNALS.HOLD]: 'signal-hold text-muted',
  [SIGNALS.NEUTRAL]: 'signal-hold text-muted'
};

// Icon classes for signals
export const SIGNAL_ICONS = {
  [SIGNALS.BULLISH]: 'fas fa-arrow-up',
  [SIGNALS.BEARISH]: 'fas fa-arrow-down',
  [SIGNALS.HOLD]: 'fas fa-minus',
  [SIGNALS.NEUTRAL]: 'fas fa-minus'
};

// API endpoints (without /api prefix - added by APIClient baseURL)
export const API_ENDPOINTS = {
  WATCHLIST: '/watchlist',
  WATCHLIST_PREDICTIONS: '/watchlist/predictions',
  BATCH_PREDICTIONS: '/batch/predictions',
  BATCH_PATTERN_ANALYSIS: '/batch/pattern-analysis',
  PATTERN_ANALYSIS: '/pattern-analysis',
  STOCK_SEARCH: '/stocks/search',
  STOCK_PRICES: '/stock-prices',
  USER_PREDICTIONS: '/user/predictions'
};

// WebSocket events
export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  CONNECT_ERROR: 'connect_error',
  JOIN_USER: 'join_user',
  JOIN_ADMIN: 'join_admin',
  SUBSCRIBE_STOCK: 'subscribe_stock',
  UNSUBSCRIBE_STOCK: 'unsubscribe_stock',
  REQUEST_PATTERN_ANALYSIS: 'request_pattern_analysis',
  PATTERN_ANALYSIS: 'pattern_analysis',
  USER_SIGNAL: 'user_signal',
  ROOM_JOINED: 'room_joined',
  SUBSCRIPTION_CONFIRMED: 'subscription_confirmed',
  ERROR: 'error'
};

// WebSocket configuration
export const WS_CONFIG = {
  PATH: '/socket.io',
  TRANSPORTS: ['websocket', 'polling'],
  RECONNECTION_DELAY: 3000,  // ✅ FIX: Increased from 1500ms to 3000ms to reduce server load
  RECONNECTION_ATTEMPTS: 8,
  TIMEOUT: 10000  // ✅ FIX: Reduced from 20000ms to 10000ms for faster failure detection
};

// Cache and throttling
export const CACHE = {
  PREDICTION_FETCH_THROTTLE: 5, // seconds
  NOTIFICATION_THROTTLE: 8000, // milliseconds
  MAX_LIVE_SIGNALS: 5
};

// UI limits
export const UI_LIMITS = {
  MAX_PATTERN_BADGES: 6,
  MAX_SEARCH_RESULTS: 50,
  SEARCH_MIN_LENGTH: 2,
  DEBOUNCE_DELAY: 300 // milliseconds
};

// Chart colors
export const CHART_COLORS = {
  PRIMARY: '#0d6efd',
  PRIMARY_ALPHA: 'rgba(13,110,253,0.08)',
  DANGER: '#dc3545',
  DANGER_ALPHA: 'rgba(220,53,69,0.08)'
};

// Notification types
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  INFO: 'info',
  WARNING: 'warning'
};

// LocalStorage keys
export const STORAGE_KEYS = {
  WATCHED_STOCKS: 'watchedStocks',
  DEBUG_UI: 'DEBUG_UI'
};

// Alert types
export const ALERT_TYPES = {
  ALL: 'all',
  BUY: 'buy',
  SELL: 'sell'
};

// Default values
export const DEFAULTS = {
  HORIZON: '7d',
  SORT_ORDER: 'desc',
  DIRECTION_FILTER: 'all',
  MIN_ABS_CHANGE: 0
};

