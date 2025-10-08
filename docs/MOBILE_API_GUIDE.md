# Mobile Application API Guide
## BIST Pattern User Dashboard - Mobile API Reference

Bu dokümantasyon, mobil uygulama geliştirmek için gerekli tüm API endpoint'lerini içerir.

---

## 🔐 Authentication

### Login
```http
POST /login
Content-Type: application/x-www-form-urlencoded

email=user@example.com&password=secret
```

**Response:**
- Success: Redirect to `/user` with session cookie
- Error: 401/403 with error message

### Logout
```http
GET /logout
```

**Response:** Redirect to `/login`

### Session Check
```http
GET /api/
```

**Response:**
```json
{
  "message": "BIST Pattern Detection API",
  "status": "running",
  "version": "2.2.0",
  "database": "PostgreSQL",
  "features": ["Real-time Data", "Yahoo Finance", "Scheduler", "Dashboard", "Automation"]
}
```

---

## 📊 Watchlist Management

### Get User Watchlist
```http
GET /api/watchlist
Cookie: session=<session_token>
```

**Response:**
```json
{
  "status": "success",
  "user_id": 4,
  "watchlist": [
    {
      "id": 1,
      "symbol": "AEFES",
      "name": "Anadolu Efes",
      "notes": null,
      "alert_enabled": true,
      "alert_threshold_buy": null,
      "alert_threshold_sell": null,
      "created_at": "2025-10-08T10:00:00"
    }
  ]
}
```

### Add to Watchlist
```http
POST /api/watchlist
Content-Type: application/json
Cookie: session=<session_token>

{
  "symbol": "THYAO",
  "alert_enabled": true,
  "notes": "İzleniyor",
  "alert_threshold_buy": 120.0,
  "alert_threshold_sell": 80.0
}
```

**Response:**
```json
{
  "status": "success",
  "item": {
    "id": 2,
    "symbol": "THYAO",
    "name": "Türk Hava Yolları",
    ...
  }
}
```

### Remove from Watchlist
```http
DELETE /api/watchlist/THYAO
Cookie: session=<session_token>
```

**Response:**
```json
{
  "status": "success",
  "message": "THYAO removed"
}
```

---

## 🔮 Predictions & Analysis

### Get Watchlist Predictions (Batch)
**En hızlı yöntem - tüm watchlist için tek istekle**

```http
POST /api/batch/predictions
Content-Type: application/json
Cookie: session=<session_token>

{
  "symbols": ["AEFES", "ARCLK", "ASELS"]
}
```

**Response:**
```json
{
  "status": "success",
  "count": 3,
  "timestamp": "2025-10-08T18:00:00",
  "source_timestamp": "2025-10-08T17:30:00",
  "results": {
    "AEFES": {
      "status": "success",
      "predictions": {
        "1d": 14.03,
        "3d": 14.04,
        "7d": 14.06,
        "14d": 14.12,
        "30d": 14.22
      },
      "confidences": {
        "1d": 0.68,
        "3d": 0.67,
        "7d": 0.62,
        "14d": 0.54,
        "30d": 0.34
      },
      "current_price": 14.02,
      "model": "enhanced",
      "source_timestamp": "2025-10-08T17:30:00",
      "analysis_timestamp": "2025-10-08T17:53:55"
    }
  }
}
```

### Get Single Symbol Predictions
```http
GET /api/user/predictions/THYAO
Cookie: session=<session_token>
```

**Response:**
```json
{
  "status": "success",
  "symbol": "THYAO",
  "predictions": {
    "1d": 120.5,
    "3d": 122.0,
    "7d": 125.0,
    "14d": 128.0,
    "30d": 135.0
  },
  "current_price": 120.0
}
```

---

## 🎯 Pattern Analysis

### Get Batch Pattern Analysis
**En hızlı yöntem - cache-only, tüm semboller için**

```http
POST /api/batch/pattern-analysis
Content-Type: application/json

{
  "symbols": ["AEFES", "ARCLK", "ASELS"]
}
```

**Response:**
```json
{
  "status": "success",
  "count": 3,
  "timestamp": "2025-10-08T18:00:00",
  "results": {
    "AEFES": {
      "symbol": "AEFES",
      "status": "success",
      "timestamp": "2025-10-08T17:53:55",
      "current_price": 14.02,
      "from_cache": true,
      "stale": false,
      "stale_seconds": 120.5,
      "indicators": {
        "sma_20": 14.5,
        "rsi": 45.2,
        "macd": -0.05,
        ...
      },
      "patterns": [
        {
          "pattern": "HAMMER",
          "signal": "BULLISH",
          "confidence": 0.75,
          "source": "ADVANCED_TA",
          "range": {
            "start_index": 50,
            "end_index": 59
          }
        }
      ],
      "overall_signal": {
        "signal": "BULLISH",
        "confidence": 0.69,
        "strength": 69,
        "reasoning": "12 sinyal analiz edildi"
      },
      "ml_unified": {
        "1d": {
          "basic": {
            "price": 14.02,
            "confidence": null,
            "delta_pct": 0.0,
            "evidence": {...}
          },
          "enhanced": {
            "price": 14.03,
            "confidence": 0.68,
            "delta_pct": 0.001,
            "evidence": {
              "pattern_score": 0.0,
              "sentiment_score": 0.0,
              "w_pat": 0.12,
              "w_sent": 0.10
            }
          },
          "best": "enhanced"
        },
        "3d": {...},
        "7d": {...},
        "14d": {...},
        "30d": {...}
      }
    }
  }
}
```

### Get Single Symbol Pattern Analysis
```http
GET /api/pattern-analysis/THYAO?fast=1
```

**Parameters:**
- `fast=1`: Cache-only, no fresh computation
- `v=<timestamp>`: Cache buster

**Response:** Same structure as batch result for single symbol

---

## 📈 Stock Data

### Search Stocks
```http
GET /api/stocks/search?q=thyao&limit=50
```

**Response:**
```json
{
  "status": "success",
  "query": "thyao",
  "total": 1,
  "stocks": [
    {
      "id": 1,
      "symbol": "THYAO",
      "name": "Türk Hava Yolları",
      "sector": "Ulaştırma",
      "price": 120.5,
      "last_update": "2025-10-08"
    }
  ]
}
```

### Get Stock Price History
```http
GET /api/stock-prices/THYAO?days=60
```

**Response:**
```json
{
  "status": "success",
  "symbol": "THYAO",
  "data": [
    {
      "date": "2025-10-08",
      "open": 119.5,
      "high": 121.0,
      "low": 119.0,
      "close": 120.5,
      "volume": 1250000
    },
    ...
  ]
}
```

### Get All Stocks
```http
GET /api/stocks
```

**Response:**
```json
{
  "status": "success",
  "stocks": [
    {
      "id": 1,
      "symbol": "THYAO",
      "name": "Türk Hava Yolları"
    },
    ...
  ]
}
```

---

## 🔔 Real-time Updates (WebSocket)

### Connect to WebSocket
```javascript
const socket = io('https://your-domain.com', {
  path: '/socket.io',
  transports: ['websocket', 'polling'],
  withCredentials: true
});
```

### Events to Listen

#### Connection
```javascript
socket.on('connect', () => {
  console.log('Connected:', socket.id);
});
```

#### Join User Room
```javascript
socket.emit('join_user', { user_id: 4 });

socket.on('room_joined', (data) => {
  // { room: 'user_4', message: 'User interface connected' }
});
```

#### Subscribe to Stock Updates
```javascript
socket.emit('subscribe_stock', { symbol: 'THYAO' });

socket.on('subscription_confirmed', (data) => {
  // { symbol: 'THYAO', message: 'Subscribed...' }
});
```

#### Receive Pattern Analysis Updates
```javascript
socket.on('pattern_analysis', (data) => {
  // {
  //   symbol: 'THYAO',
  //   data: { ...analysis... },
  //   timestamp: '2025-10-08T18:00:00'
  // }
});
```

#### Receive Live Signals
```javascript
socket.on('user_signal', (data) => {
  // {
  //   signal: {
  //     symbol: 'THYAO',
  //     overall_signal: {
  //       signal: 'BULLISH',
  //       confidence: 0.75
  //     }
  //   },
  //   timestamp: '2025-10-08T18:00:00'
  // }
});
```

---

## 📱 Mobile App Recommended Flow

### 1. App Launch
```
1. Check session validity: GET /api/
2. If not logged in: Show login screen
3. If logged in: Load dashboard
```

### 2. Dashboard Load
```
Sequential:
1. GET /api/watchlist → Load user's symbols
2. POST /api/batch/predictions → Get all predictions in one call
3. POST /api/batch/pattern-analysis → Get all analyses in one call

Parallel:
- Connect WebSocket
- Join user room
- Subscribe to all watchlist symbols
```

### 3. Live Updates
```
WebSocket:
- pattern_analysis: Update card data
- user_signal: Show notification
```

### 4. Add Stock
```
1. GET /api/stocks/search?q=thyao → Search
2. POST /api/watchlist → Add to watchlist
3. socket.emit('subscribe_stock', {symbol: 'THYAO'}) → Subscribe
4. Refresh batch data
```

### 5. Detail View
```
1. GET /api/pattern-analysis/THYAO?fast=1 → Get full analysis
2. GET /api/stock-prices/THYAO?days=60 → Get price history
3. Render charts, patterns, ML summary
```

---

## 🔑 Authentication for Mobile

### Option 1: Session-based (Recommended for PWA)
```http
POST /login
→ Returns session cookie
→ Include cookie in subsequent requests
```

### Option 2: OAuth (Google/Apple Sign-in)
```http
GET /auth/google
→ Redirect flow
→ Returns session
```

### Option 3: API Token (For native apps)
**Not currently implemented - would need to add:**
```http
POST /api/auth/token
{
  "email": "user@example.com",
  "password": "secret"
}
→ Returns: { "token": "jwt_token_here" }
```

---

## 📊 API Endpoints Summary

### Core Endpoints (Must Have)
```
✅ GET  /api/watchlist              - Kullanıcının hisseleri
✅ POST /api/watchlist              - Hisse ekle
✅ DELETE /api/watchlist/{symbol}   - Hisse çıkar
✅ POST /api/batch/predictions      - Toplu tahminler (HIZLI!)
✅ POST /api/batch/pattern-analysis - Toplu analiz (HIZLI!)
✅ GET  /api/stocks/search          - Hisse ara
```

### Detail Endpoints (Nice to Have)
```
✅ GET /api/user/predictions/{symbol}  - Tek sembol tahmin
✅ GET /api/pattern-analysis/{symbol}  - Tek sembol analiz
✅ GET /api/stock-prices/{symbol}      - Fiyat geçmişi
```

### WebSocket Events
```
✅ connect / disconnect
✅ join_user
✅ subscribe_stock / unsubscribe_stock
✅ pattern_analysis (update)
✅ user_signal (notification)
```

---

## 🚀 Performance Tips

### 1. Use Batch APIs
```javascript
// ❌ YAVAS: 10 hisse için 10 istek
for (symbol of watchlist) {
  await fetch(`/api/pattern-analysis/${symbol}`);
}

// ✅ HIZLI: 10 hisse için 1 istek
await fetch('/api/batch/pattern-analysis', {
  method: 'POST',
  body: JSON.stringify({ symbols: watchlist })
});
```

### 2. Cache Strategy
```javascript
// Local cache with TTL
const cache = {
  predictions: {
    data: {...},
    timestamp: Date.now(),
    ttl: 30000 // 30 seconds
  }
};

// Throttle API calls
if (Date.now() - cache.predictions.timestamp < cache.predictions.ttl) {
  return cache.predictions.data; // Use cached
}
```

### 3. WebSocket for Live Updates
```javascript
// Don't poll APIs every second!
// Use WebSocket for real-time updates
socket.on('pattern_analysis', (data) => {
  updateUI(data.symbol, data.data);
});
```

---

## 📱 Sample Mobile Screen → API Mapping

### Home Screen (Watchlist)
```
APIs:
1. GET /api/watchlist → Symbol list
2. POST /api/batch/predictions → All predictions
3. POST /api/batch/pattern-analysis → All signals

Display:
- Symbol name
- Current price
- Signal (Buy/Sell/Hold)
- Confidence %
- Prediction badges (Gelişmiş 7D, Temel 7D)
```

### Stock Detail Screen
```
APIs:
1. GET /api/pattern-analysis/{symbol} → Full analysis
2. GET /api/stock-prices/{symbol}?days=60 → Chart data

Display:
- Price chart with pattern overlays
- Technical indicators (RSI, MACD, Bollinger)
- Detected patterns list
- ML predictions (1D/3D/7D/14D/30D)
- ML summary with evidence
- Volume tier
```

### Search Screen
```
API:
1. GET /api/stocks/search?q={query}

Display:
- Search results
- Symbol, name, sector
- Latest price
```

### Notifications Screen
```
WebSocket:
- user_signal events

Display:
- Live signals for watchlist stocks
- Timestamp
- Confidence level
```

---

## 🔧 Example Mobile Implementation

### React Native Example
```javascript
import { io } from 'socket.io-client';

class BISTAPIClient {
  constructor(baseURL, sessionCookie) {
    this.baseURL = baseURL;
    this.sessionCookie = sessionCookie;
    this.socket = null;
  }

  async getWatchlist() {
    const response = await fetch(`${this.baseURL}/api/watchlist`, {
      credentials: 'include',
      headers: {
        'Cookie': this.sessionCookie
      }
    });
    return await response.json();
  }

  async getBatchPredictions(symbols) {
    const response = await fetch(`${this.baseURL}/api/batch/predictions`, {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'Cookie': this.sessionCookie
      },
      body: JSON.stringify({ symbols })
    });
    return await response.json();
  }

  connectWebSocket(userId) {
    this.socket = io(this.baseURL, {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
      withCredentials: true
    });

    this.socket.on('connect', () => {
      this.socket.emit('join_user', { user_id: userId });
    });

    return this.socket;
  }

  subscribeToStocks(symbols) {
    symbols.forEach(symbol => {
      this.socket.emit('subscribe_stock', { symbol });
    });
  }
}

// Usage
const api = new BISTAPIClient('https://your-domain.com', sessionCookie);
const watchlist = await api.getWatchlist();
const predictions = await api.getBatchPredictions(watchlist.map(w => w.symbol));
```

---

## 🔐 Security Notes

### Session Management
- Sessions are stored server-side (Flask-Login)
- Cookie-based authentication
- HTTPS required for production
- Implement token refresh for long-lived sessions

### Rate Limiting
- API has rate limiting enabled
- Batch endpoints preferred (single request instead of multiple)
- WebSocket preferred for live updates (not polling)

### CORS
- Configure CORS_ORIGINS for mobile app domains
- Set in environment: `CORS_ORIGINS=https://mobile.yourdomain.com`

---

## 📊 Data Refresh Strategy

### On App Launch
```
1. GET /api/watchlist (cache: none)
2. POST /api/batch/predictions (cache: 30s)
3. POST /api/batch/pattern-analysis (cache: 60s)
4. Connect WebSocket
```

### On Pull-to-Refresh
```
1. Force fetch batch predictions (bypass cache)
2. Force fetch batch pattern analysis (bypass cache)
3. Update UI
```

### Background Updates
```
1. WebSocket events (real-time)
2. Periodic batch fetch every 5 minutes (when app active)
3. No polling when app in background
```

---

## 🎨 UI Components → API Mapping

### StockCard Component
```javascript
{
  symbol: "AEFES",
  name: "Anadolu Efes",
  currentPrice: 14.02,  // from batch/predictions
  signal: {             // from batch/pattern-analysis
    label: "Bekleme",
    confidence: 69,
    type: "BULLISH"
  },
  predictions: {        // from batch/predictions
    "1d": 14.03,
    "7d": 14.06,
    ...
  },
  badges: [             // from batch/pattern-analysis
    { label: "Gelişmiş 7D", color: "warning" },
    { label: "Düşen Yıldız", color: "danger" }
  ]
}
```

### ChartView Component
```javascript
{
  priceHistory: [],     // from /api/stock-prices
  patterns: [],         // from /api/pattern-analysis
  indicators: {},       // from /api/pattern-analysis
  mlSummary: {}         // from /api/pattern-analysis.ml_unified
}
```

---

## ⚡ Optimization Checklist

- [x] Use batch APIs for multiple symbols
- [x] Implement local caching with TTL
- [x] Use WebSocket for live updates (not polling)
- [x] Lazy load detail data (only when user opens detail)
- [x] Compress images/assets
- [x] Implement offline mode with cached data
- [x] Use pagination for long lists
- [x] Debounce search input
- [x] Show loading states
- [x] Handle errors gracefully

---

## 📝 Additional API Endpoints (Optional)

### User Profile
```http
GET /api/user/profile
→ { email, username, premium status, settings }
```

### System Health
```http
GET /api/health
→ { status, uptime, features_enabled }
```

### Cache Report
```http
GET /api/watchlist/cache-report
→ Which symbols have fresh data
```

---

## 🚦 HTTP Status Codes

```
200 - Success
401 - Unauthorized (not logged in)
403 - Forbidden (no permission)
404 - Not Found (symbol doesn't exist)
500 - Server Error
```

---

## 💡 Recommended Mobile Stack

### React Native
- react-native-chart-kit (for charts)
- socket.io-client (for WebSocket)
- AsyncStorage (for caching)
- React Navigation (for screens)

### Flutter
- fl_chart (for charts)
- socket_io_client (for WebSocket)
- shared_preferences (for caching)
- provider (for state management)

### PWA (Web-based)
- Chart.js (same as web dashboard)
- Socket.IO (same as web dashboard)
- Service Worker (for offline)
- IndexedDB (for caching)

---

**Need more details? Let me know which platform you're using!**

