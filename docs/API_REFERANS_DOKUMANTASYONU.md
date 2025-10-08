# BIST Pattern API Referans Dokümantasyonu

## 📚 Tüm API Endpoint'leri - Detaylı Request/Response Rehberi

Bu dokümantasyon, BIST Pattern sisteminin tüm API endpoint'lerini, ne tür veriler istediğini, ne tür cevaplar verdiğini detaylı olarak açıklar.

---

## 📋 İçindekiler

1. [Authentication API](#authentication-api)
2. [Watchlist API](#watchlist-api)
3. [Predictions API](#predictions-api)
4. [Pattern Analysis API](#pattern-analysis-api)
5. [Stock Data API](#stock-data-api)
6. [Internal API](#internal-api)
7. [Health & Status API](#health--status-api)
8. [WebSocket Events](#websocket-events)

---

## 🔐 Authentication API

### POST /login

**Ne Sorar:** Kullanıcı email ve şifre

**Request:**
```http
POST /login HTTP/1.1
Content-Type: application/x-www-form-urlencoded

email=kullanici@example.com&password=sifreburaya
```

**Ne Cevap Verir:**
- **Başarılı:** HTTP 302 redirect `/user` veya `/dashboard` + session cookie
- **Başarısız:** HTTP 200 + login.html (hata mesajıyla)

**Session Cookie Örneği:**
```
Set-Cookie: session=eyJ1c2VyX2lkIjo0fQ...; Path=/; HttpOnly
```

**Kullanım:**
```bash
curl -X POST https://your-domain.com/login \
  -d "email=user@example.com&password=secret" \
  -c cookies.txt  # Cookie'yi sakla
```

---

### GET /logout

**Ne Sorar:** Hiçbir şey (sadece session cookie)

**Request:**
```http
GET /logout HTTP/1.1
Cookie: session=...
```

**Ne Cevap Verir:** HTTP 302 redirect `/login`

---

### GET /auth/google

**Ne Sorar:** Hiçbir şey

**Ne Cevap Verir:** Google OAuth redirect URL

**Kullanım:** Browser'da açılır, kullanıcı Google ile giriş yapar

---

## 📊 Watchlist API

### GET /api/watchlist

**Ne Sorar:** Sadece authentication (session cookie)

**Request:**
```http
GET /api/watchlist HTTP/1.1
Cookie: session=...
```

**Ne Cevap Verir:**

**Başarılı Response:**
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
    },
    {
      "id": 2,
      "symbol": "ARCLK",
      "name": "Arçelik",
      "notes": "İzleniyor",
      "alert_enabled": true,
      "alert_threshold_buy": 120.0,
      "alert_threshold_sell": 100.0,
      "created_at": "2025-10-08T11:30:00"
    }
  ]
}
```

**Hatalı Response (Unauthorized):**
```json
{
  "status": "unauthorized"
}
```
HTTP Status: 401

**Field Açıklamaları:**
- `user_id`: Kullanıcının ID'si (integer)
- `watchlist`: Liste (array) - kullanıcının takip ettiği hisseler
- `id`: Watchlist item ID (integer)
- `symbol`: Hisse kodu (string, uppercase)
- `name`: Hisse adı (string, nullable)
- `notes`: Kullanıcı notları (string, nullable)
- `alert_enabled`: Alarm aktif mi? (boolean)
- `alert_threshold_buy`: Alım alarm eşiği (float, nullable)
- `alert_threshold_sell`: Satış alarm eşiği (float, nullable)
- `created_at`: Eklenme tarihi (ISO 8601 string)

---

### POST /api/watchlist

**Ne Sorar:** Hisse kodu ve alarm ayarları

**Request:**
```http
POST /api/watchlist HTTP/1.1
Content-Type: application/json
Cookie: session=...

{
  "symbol": "THYAO",
  "alert_enabled": true,
  "notes": "Türk Hava Yolları takipte",
  "alert_threshold_buy": 125.0,
  "alert_threshold_sell": 100.0
}
```

**Request Field'ları:**
- `symbol`: **Zorunlu** - Hisse kodu (string, büyük harf)
- `alert_enabled`: Opsiyonel - Alarm aktif mi? (boolean, default: true)
- `notes`: Opsiyonel - Kullanıcı notu (string, max 500 karakter)
- `alert_threshold_buy`: Opsiyonel - Alım eşiği (float)
- `alert_threshold_sell`: Opsiyonel - Satış eşiği (float)

**Ne Cevap Verir:**

**Başarılı:**
```json
{
  "status": "success",
  "item": {
    "id": 7,
    "symbol": "THYAO",
    "name": "Türk Hava Yolları",
    "notes": "Türk Hava Yolları takipte",
    "alert_enabled": true,
    "alert_threshold_buy": 125.0,
    "alert_threshold_sell": 100.0,
    "created_at": "2025-10-08T18:30:00"
  }
}
```

**Hatalı (Symbol yok):**
```json
{
  "status": "error",
  "error": "stock not found"
}
```
HTTP Status: 404

**Hatalı (Zaten var):**
```json
{
  "status": "error",
  "error": "already in watchlist"
}
```
HTTP Status: 400

---

### DELETE /api/watchlist/{symbol}

**Ne Sorar:** URL'de hisse kodu

**Request:**
```http
DELETE /api/watchlist/THYAO HTTP/1.1
Cookie: session=...
```

**Ne Cevap Verir:**

**Başarılı:**
```json
{
  "status": "success",
  "message": "THYAO removed"
}
```

**Hatalı (Bulunamadı):**
```json
{
  "status": "error",
  "error": "watchlist item not found"
}
```
HTTP Status: 404

---

## 🔮 Predictions API

### POST /api/batch/predictions

**⚡ ÖNERİLEN YÖNTEM - Çok hızlı!**

**Ne Sorar:** Sembol listesi (array)

**Request:**
```http
POST /api/batch/predictions HTTP/1.1
Content-Type: application/json

{
  "symbols": ["AEFES", "ARCLK", "ASELS", "THYAO"]
}
```

**Request Limitleri:**
- Minimum: 1 sembol
- Maksimum: 50 sembol
- Semboller büyük harf olmalı

**Ne Cevap Verir:**

**Başarılı Response:**
```json
{
  "status": "success",
  "count": 4,
  "timestamp": "2025-10-08T18:30:00.123456",
  "source_timestamp": "2025-10-08T18:25:00.000000",
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
      "source_timestamp": "2025-10-08T18:25:00",
      "analysis_timestamp": "2025-10-08T17:53:55"
    },
    "ARCLK": {
      "status": "success",
      "predictions": {
        "1d": 117.60,
        "3d": 117.80,
        "7d": 118.20,
        "14d": 118.80,
        "30d": 119.50
      },
      "confidences": {
        "1d": 0.72,
        "3d": 0.70,
        "7d": 0.68,
        "14d": 0.62,
        "30d": 0.55
      },
      "current_price": 117.20,
      "source_timestamp": "2025-10-08T18:25:00",
      "analysis_timestamp": "2025-10-08T18:10:32"
    },
    "ASELS": {
      "status": "pending"
    }
  }
}
```

**Field Açıklamaları:**
- `status`: "success" veya "error" (string)
- `count`: Döndürülen sonuç sayısı (integer)
- `timestamp`: Yanıt oluşturulma zamanı (ISO 8601 string)
- `source_timestamp`: Kaynak verinin üretilme zamanı (ISO 8601 string)
- `results`: Object - her sembol için sonuçlar
  - `{SYMBOL}.status`: "success", "pending" veya "error"
  - `{SYMBOL}.predictions`: Object - horizon → fiyat mapping
    - `1d`: 1 günlük tahmin (float)
    - `3d`: 3 günlük tahmin (float)
    - `7d`: 7 günlük tahmin (float)
    - `14d`: 14 günlük tahmin (float)
    - `30d`: 30 günlük tahmin (float)
  - `{SYMBOL}.confidences`: Object - horizon → güven mapping
    - Her horizon için 0.0-1.0 arası güven skoru (float)
  - `{SYMBOL}.current_price`: Güncel fiyat (float)
  - `{SYMBOL}.source_timestamp`: Tahmin kaynağının zamanı
  - `{SYMBOL}.analysis_timestamp`: Analizin zamanı

**Hatalı Response:**
```json
{
  "status": "error",
  "message": "Provide 1-50 symbols"
}
```
HTTP Status: 400

**Veri Kaynağı:**
- `/opt/bist-pattern/logs/ml_bulk_predictions.json`
- Cache-only (fresh hesaplama YOK)
- Automation cycle tarafından üretilir

---

### GET /api/user/predictions/{symbol}

**Ne Sorar:** URL'de sembol, session cookie

**Request:**
```http
GET /api/user/predictions/THYAO HTTP/1.1
Cookie: session=...
```

**Ne Cevap Verir:**

**Başarılı:**
```json
{
  "status": "success",
  "symbol": "THYAO",
  "predictions": {
    "1d": 120.50,
    "3d": 122.00,
    "7d": 125.00,
    "14d": 128.00,
    "30d": 135.00
  },
  "current_price": 120.00
}
```

**Hatalı (Veri yok):**
```json
{
  "status": "error",
  "message": "THYAO için yeterli veri bulunamadı",
  "symbol": "THYAO"
}
```
HTTP Status: 404

**Not:** Bu endpoint **hesaplama yapar** (batch'den farklı). Daha yavaş ama daha güncel olabilir.

---

### GET /api/watchlist/predictions

**Ne Sorar:** Session cookie (kullanıcının watchlist'inden semboller)

**Request:**
```http
GET /api/watchlist/predictions HTTP/1.1
Cookie: session=...
```

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "count": 3,
  "items": [
    {
      "symbol": "AEFES",
      "current_price": 14.02,
      "predictions": {
        "1d": 14.03,
        "3d": 14.04,
        "7d": 14.06,
        "14d": 14.12,
        "30d": 14.22
      },
      "model": "enhanced",
      "data_days": 365,
      "last_date": "2025-10-08",
      "name": "Anadolu Efes",
      "last_signal": {
        "signal": "BULLISH",
        "confidence": 0.69,
        "timestamp": "2025-10-08T17:53:55"
      }
    }
  ]
}
```

**Field Açıklamaları:**
- `model`: Hangi model kullanıldı? "basic", "enhanced" veya null
- `data_days`: Kaç günlük veri var? (integer)
- `last_date`: Son veri tarihi (ISO date string)
- `last_signal`: En son sinyal snapshot (object)

---

## 🎯 Pattern Analysis API

### POST /api/batch/pattern-analysis

**⚡ ÖNERİLEN YÖNTEM - Çok hızlı!**

**Ne Sorar:** Sembol listesi

**Request:**
```http
POST /api/batch/pattern-analysis HTTP/1.1
Content-Type: application/json

{
  "symbols": ["AEFES", "ARCLK", "THYAO"]
}
```

**Request Limitleri:**
- Minimum: 1 sembol
- Maksimum: 50 sembol

**Ne Cevap Verir:**

**Başarılı Response:**
```json
{
  "status": "success",
  "results": {
    "AEFES": {
      "symbol": "AEFES",
      "status": "success",
      "timestamp": "2025-10-08T17:53:55.037474",
      "current_price": 14.02,
      "from_cache": true,
      "stale": false,
      "stale_seconds": 180.5,
      "data_points": 365,
      "indicators": {
        "sma_20": 14.50,
        "sma_50": 14.80,
        "ema_12": 14.35,
        "ema_26": 14.55,
        "rsi": 45.2,
        "macd": -0.05,
        "macd_signal": -0.03,
        "macd_histogram": -0.02,
        "bb_upper": 15.20,
        "bb_lower": 13.80,
        "bb_position": 0.42,
        "resistance": 15.50,
        "support": 13.50
      },
      "patterns": [
        {
          "pattern": "HAMMER",
          "signal": "BULLISH",
          "confidence": 0.75,
          "source": "ADVANCED_TA",
          "detection_method": "talib",
          "strength": 75,
          "range": {
            "start_index": 350,
            "end_index": 364
          },
          "validation_stages": ["ADVANCED"],
          "validation_score": 0.3
        },
        {
          "pattern": "ML_PREDICTOR_7D",
          "signal": "BULLISH",
          "confidence": 0.62,
          "strength": 62,
          "source": "ML_PREDICTOR",
          "delta_pct": 0.0285
        },
        {
          "pattern": "ENHANCED_ML_7D",
          "signal": "BULLISH",
          "confidence": 0.40,
          "strength": 40,
          "source": "ENHANCED_ML",
          "delta_pct": 0.0028
        }
      ],
      "overall_signal": {
        "signal": "BULLISH",
        "confidence": 0.6889919081883552,
        "strength": 68,
        "reasoning": "12 sinyal analiz edildi",
        "signals": [
          {
            "signal": "BULLISH",
            "confidence": 0.595,
            "source": "RSI Oversold"
          },
          {
            "signal": "BEARISH",
            "confidence": 0.51,
            "source": "MACD Negative"
          }
        ]
      },
      "ml_unified": {
        "1d": {
          "basic": {
            "price": 14.024966666666666,
            "confidence": null,
            "delta_pct": 0.00035547566332228,
            "reliability": null,
            "evidence": {
              "pattern_score": 0.0,
              "sentiment_score": 0.0,
              "w_pat": 0.12,
              "w_sent": 0.1,
              "contrib_conf": 0.0,
              "source": "fallback"
            }
          },
          "enhanced": {
            "price": 14.031448071264423,
            "confidence": 0.6768718789592032,
            "delta_pct": 0.0008199339031014502,
            "reliability": null,
            "evidence": {
              "pattern_score": 0.0,
              "sentiment_score": 0.0,
              "w_pat": 0.12,
              "w_sent": 0.1,
              "contrib_conf": 0.0,
              "source": "fallback"
            }
          },
          "best": "enhanced"
        },
        "7d": {
          "basic": {
            "price": 14.322666666666665,
            "confidence": 0.25,
            "delta_pct": 0.02161903759398496,
            "reliability": 0.6,
            "evidence": {
              "pattern_score": -0.24169732541115233,
              "sentiment_score": 0.0,
              "contrib_conf": -0.0009384155273437506,
              "w_pat": 0.06,
              "w_sent": 0.05,
              "contrib_delta": 0.0005397260273972602
            }
          },
          "enhanced": {
            "price": 14.05959702605991,
            "confidence": 0.4010150462690355,
            "delta_pct": 0.0028420629799928575,
            "reliability": 0.6236020740038384,
            "evidence": {
              "pattern_score": -0.24169732541115233,
              "sentiment_score": 0.0,
              "contrib_conf": -0.014505116270923616,
              "w_pat": 0.06,
              "w_sent": 0.05,
              "contrib_delta": 0.0005397260273972602
            }
          },
          "best": "enhanced"
        }
      }
    },
    "ARCLK": {
      "status": "pending"
    }
  },
  "count": 2,
  "timestamp": "2025-10-08T18:30:00"
}
```

**Field Açıklamaları (Pattern Analysis):**

**indicators:** Teknik göstergeler
- `sma_20`: 20 günlük basit hareketli ortalama (float)
- `sma_50`: 50 günlük basit hareketli ortalama (float)
- `ema_12`: 12 günlük üstel hareketli ortalama (float)
- `ema_26`: 26 günlük üstel hareketli ortalama (float)
- `rsi`: Relative Strength Index, 0-100 arası (float)
- `macd`: MACD değeri (float)
- `macd_signal`: MACD sinyal çizgisi (float)
- `macd_histogram`: MACD histogram (float)
- `bb_upper`: Bollinger Üst Band (float)
- `bb_lower`: Bollinger Alt Band (float)
- `bb_position`: Bollinger pozisyonu, 0-1 arası (float)
- `resistance`: Direnç seviyesi (float)
- `support`: Destek seviyesi (float)

**patterns:** Tespit edilen formasyonlar (array)
- `pattern`: Formasyon adı (string) - "HAMMER", "DOUBLE_TOP", vs.
- `signal`: Sinyal yönü - "BULLISH", "BEARISH", "NEUTRAL" (string)
- `confidence`: Güven skoru, 0-1 arası (float)
- `source`: Kaynak - "ADVANCED_TA", "VISUAL_YOLO", "ML_PREDICTOR", "ENHANCED_ML", "FINGPT" (string)
- `detection_method`: Tespit yöntemi - "talib", "visual", "ml" (string)
- `strength`: Güç, 0-100 arası (integer)
- `range`: Formasyon aralığı (object, nullable)
  - `start_index`: Başlangıç bar index'i (integer)
  - `end_index`: Bitiş bar index'i (integer)
- `validation_stages`: Doğrulama aşamaları (array of strings)
- `validation_score`: Doğrulama skoru, 0-1 arası (float)
- `delta_pct`: ML pattern'larda tahmin edilen değişim yüzdesi (float, decimal)

**overall_signal:** Genel sinyal özeti
- `signal`: "BULLISH", "BEARISH", "NEUTRAL" (string)
- `confidence`: Güven skoru, 0-1 arası (float)
- `strength`: Güç, 0-100 arası (integer)
- `reasoning`: Açıklama metni (string)
- `signals`: Tüm sinyallerin detayı (array)

**ml_unified:** ML birleşik tahminler (object)
- Her horizon (1d, 3d, 7d, 14d, 30d) için:
  - `basic`: Temel ML modeli (object, nullable)
    - `price`: Tahmin edilen fiyat (float)
    - `confidence`: Güven skoru, 0-1 arası (float, nullable)
    - `delta_pct`: Değişim yüzdesi, decimal (float)
    - `reliability`: Model güvenilirliği, 0-1 arası (float, nullable)
    - `evidence`: Kanıt detayları (object)
      - `pattern_score`: Pattern skoru, -1 ile 1 arası (float)
      - `sentiment_score`: Sentiment skoru, -1 ile 1 arası (float)
      - `contrib_conf`: Confidence'a yapılan katkı (float)
      - `w_pat`: Pattern ağırlığı (float)
      - `w_sent`: Sentiment ağırlığı (float)
      - `contrib_booster`: Booster katkısı (float, sadece 1D için)
      - `booster_prob`: Booster probability (float, 0-1, sadece 1D için)
      - `contrib_delta`: Delta tilt katkısı (float)
      - `source`: "main_loop" veya "fallback" (string)
  - `enhanced`: Gelişmiş ML modeli (object, nullable) - aynı yapı
  - `best`: En iyi model - "basic" veya "enhanced" (string)

**Veri Kaynağı:**
- Cache: `/opt/bist-pattern/logs/pattern_cache/{SYMBOL}.json`
- Cache TTL: 300 saniye (5 dakika)
- `stale`: Cache geçerlilik durumu (boolean)
- `stale_seconds`: Cache yaşı saniye cinsinden (float)

---

### GET /api/pattern-analysis/{symbol}

**Ne Sorar:** URL'de sembol, query parametreleri (opsiyonel)

**Request:**
```http
GET /api/pattern-analysis/THYAO?fast=1&v=1728405600 HTTP/1.1
```

**Query Parametreleri:**
- `fast`: "1" ise sadece memory/Redis cache, dosya cache kabul edilir (opsiyonel)
- `v`: Cache buster - timestamp değeri (opsiyonel)

**Ne Cevap Verir:**

Batch API ile aynı yapıda ama tek sembol için.

**Pending Response (Cache yok):**
```json
{
  "symbol": "THYAO",
  "status": "pending"
}
```

**Not:** Bu endpoint **hesaplama YAPMAZ**. Sadece cache'den okur. Fresh analiz için automation cycle'ın çalışması gerekir.

---

## 📈 Stock Data API

### GET /api/stocks/search

**Ne Sorar:** Arama terimi (query)

**Request:**
```http
GET /api/stocks/search?q=türk&limit=20 HTTP/1.1
```

**Query Parametreleri:**
- `q`: Arama terimi - sembol, isim veya sektörde arar (string, **zorunlu**)
- `limit`: Maksimum sonuç sayısı (integer, default: 50, max: 50)

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "query": "türk",
  "total": 5,
  "stocks": [
    {
      "id": 1,
      "symbol": "THYAO",
      "name": "Türk Hava Yolları",
      "sector": "Ulaştırma",
      "price": 120.50,
      "last_update": "2025-10-08"
    },
    {
      "id": 2,
      "symbol": "TKFEN",
      "name": "Tekfen Holding",
      "sector": "Holding",
      "price": 45.20,
      "last_update": "2025-10-08"
    }
  ]
}
```

**Arama Mekanizması:**
- Sembol'de arar (ILIKE %türk%)
- İsim'de arar (ILIKE %türk%)
- Sektör'de arar (ILIKE %türk%)
- Büyük/küçük harf duyarsız
- SQL LIKE ile çalışır

**Boş Sonuç:**
```json
{
  "status": "success",
  "query": "asdfasdf",
  "total": 0,
  "stocks": []
}
```

---

### GET /api/stocks

**Ne Sorar:** Hiçbir şey

**Request:**
```http
GET /api/stocks HTTP/1.1
```

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "stocks": [
    {"id": 1, "symbol": "THYAO", "name": "Türk Hava Yolları"},
    {"id": 2, "symbol": "AKBNK", "name": "Akbank"},
    ...
    // Maksimum 1000 hisse
  ]
}
```

**Limit:** İlk 1000 hisse (alfabetik sıralı)

---

### GET /api/stock-prices/{symbol}

**Ne Sorar:** URL'de sembol, query parametresi (gün sayısı)

**Request:**
```http
GET /api/stock-prices/THYAO?days=60 HTTP/1.1
```

**Query Parametreleri:**
- `days`: Kaç günlük veri? (integer, default: 60, max: 365)

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "symbol": "THYAO",
  "days": 60,
  "data": [
    {
      "date": "2025-08-10",
      "open": 115.50,
      "high": 118.00,
      "low": 115.00,
      "close": 117.50,
      "volume": 1250000
    },
    {
      "date": "2025-08-11",
      "open": 117.50,
      "high": 120.00,
      "low": 116.80,
      "close": 119.20,
      "volume": 1580000
    },
    ...
    {
      "date": "2025-10-08",
      "open": 119.50,
      "high": 121.00,
      "low": 119.00,
      "close": 120.50,
      "volume": 1420000
    }
  ]
}
```

**Field Açıklamaları:**
- `date`: İşlem günü (ISO date string, YYYY-MM-DD)
- `open`: Açılış fiyatı (float)
- `high`: En yüksek fiyat (float)
- `low`: En düşük fiyat (float)
- `close`: Kapanış fiyatı (float)
- `volume`: İşlem hacmi (integer)

**Sıralama:** Eskiden yeniye (tarih artan)

**Veri Kaynağı:** PostgreSQL `stock_prices` tablosu

---

## 🔧 Internal API (Admin/Automation)

### GET /api/internal/automation/status

**Ne Sorar:** Internal token

**Request:**
```http
GET /api/internal/automation/status HTTP/1.1
X-Internal-Token: IBx_gsmQUL9oxymAgr67PxES7ACfKlk1Ex5F9jCCOFw
```

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "is_running": true,
  "current_cycle": 42,
  "last_run": "2025-10-08T18:25:00",
  "next_run": "2025-10-08T18:30:00",
  "interval_minutes": 5,
  "symbols_processed": 608,
  "errors": 0
}
```

---

### POST /api/internal/automation/start

**Ne Sorar:** Internal token

**Request:**
```http
POST /api/internal/automation/start HTTP/1.1
X-Internal-Token: IBx_gsmQUL9oxymAgr67PxES7ACfKlk1Ex5F9jCCOFw
Content-Type: application/json

{}
```

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "message": "Automation started",
  "is_running": true
}
```

**Hatalı (Zaten çalışıyor):**
```json
{
  "status": "error",
  "message": "Automation already running"
}
```

---

### POST /api/internal/automation/stop

**Ne Sorar:** Internal token

**Request:**
```http
POST /api/internal/automation/stop HTTP/1.1
X-Internal-Token: IBx_gsmQUL9oxymAgr67PxES7ACfKlk1Ex5F9jCCOFw
```

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "message": "Automation stopped",
  "is_running": false
}
```

---

### GET /api/internal/automation/volume/tiers

**Ne Sorar:** Symbol ve internal token

**Request:**
```http
GET /api/internal/automation/volume/tiers?symbol=THYAO HTTP/1.1
X-Internal-Token: IBx_gsmQUL9oxymAgr67PxES7ACfKlk1Ex5F9jCCOFw
```

**Ne Cevap Verir:**

```json
{
  "status": "success",
  "symbol": "THYAO",
  "tier": "high",
  "avg_volume": 1450000,
  "volume_30d": 1450000,
  "tier_thresholds": {
    "very_high": 5000000,
    "high": 1000000,
    "medium": 500000,
    "low": 100000,
    "very_low": 0
  }
}
```

**Tier Açıklamaları:**
- `very_high`: Çok Yüksek (> 5M)
- `high`: Yüksek (1M - 5M)
- `medium`: Orta (500K - 1M)
- `low`: Düşük (100K - 500K)
- `very_low`: Çok Düşük (< 100K)

---

## 💚 Health & Status API

### GET /api/

**Ne Sorar:** Hiçbir şey

**Request:**
```http
GET /api/ HTTP/1.1
```

**Ne Cevap Verir:**

```json
{
  "message": "BIST Pattern Detection API",
  "status": "running",
  "version": "2.2.0",
  "database": "PostgreSQL",
  "features": [
    "Real-time Data",
    "Yahoo Finance",
    "Scheduler",
    "Dashboard",
    "Automation"
  ]
}
```

**Kullanım:** API'nin çalışıp çalışmadığını kontrol etmek için

---

### GET /api/health

**Ne Sorar:** Hiçbir şey

**Request:**
```http
GET /api/health HTTP/1.1
```

**Ne Cevap Verir:**

```json
{
  "status": "healthy",
  "timestamp": "2025-10-08T18:30:00",
  "database": "connected",
  "cache": "active",
  "automation": "running"
}
```

---

## 🔔 WebSocket Events

### Bağlantı Kurma

**Client → Server:**
```javascript
const socket = io('https://your-domain.com', {
  path: '/socket.io',
  transports: ['websocket', 'polling']
});
```

**Server → Client (connect event):**
```json
{
  "message": "Connected to BIST AI System",
  "timestamp": "2025-10-08T18:30:00",
  "connection_id": "abc123xyz"
}
```

---

### join_user

**Client → Server:**
```javascript
socket.emit('join_user', {
  user_id: 4
});
```

**Server → Client (room_joined event):**
```json
{
  "room": "user_4",
  "message": "User interface connected"
}
```

---

### subscribe_stock

**Client → Server:**
```javascript
socket.emit('subscribe_stock', {
  symbol: 'THYAO'
});
```

**Server → Client (subscription_confirmed event):**
```json
{
  "symbol": "THYAO",
  "message": "Subscribed to THYAO updates"
}
```

---

### pattern_analysis (Server → Client)

**Ne Zaman Gönderilir:**
- Automation cycle bir sembolü analiz ettiğinde
- Real-time güncelleme olarak

**Event Data:**
```json
{
  "symbol": "THYAO",
  "data": {
    "symbol": "THYAO",
    "status": "success",
    "timestamp": "2025-10-08T18:30:00",
    "current_price": 120.50,
    "indicators": { ... },
    "patterns": [ ... ],
    "overall_signal": { ... },
    "ml_unified": { ... }
  },
  "timestamp": "2025-10-08T18:30:00"
}
```

**Nasıl Dinlenir:**
```javascript
socket.on('pattern_analysis', (data) => {
  console.log('Analiz güncellendi:', data.symbol);
  updateUI(data.symbol, data.data);
});
```

---

### user_signal (Server → Client)

**Ne Zaman Gönderilir:**
- Güçlü bir sinyal tespit edildiğinde (confidence > 0.70)
- Kullanıcının watchlist'indeki bir hisse için

**Event Data:**
```json
{
  "signal": {
    "symbol": "THYAO",
    "overall_signal": {
      "signal": "BULLISH",
      "confidence": 0.85,
      "strength": 85,
      "reasoning": "15 sinyal analiz edildi"
    },
    "patterns": [ ... ],
    "visual": [
      {
        "pattern": "DOUBLE_BOTTOM",
        "confidence": 0.78
      }
    ],
    "current_price": 120.50,
    "timestamp": "2025-10-08T18:30:00"
  },
  "timestamp": "2025-10-08T18:30:00"
}
```

**Nasıl Dinlenir:**
```javascript
socket.on('user_signal', (data) => {
  const signal = data.signal;
  showNotification(
    `${signal.symbol}: ${signal.overall_signal.signal}`,
    `Güven: %${Math.round(signal.overall_signal.confidence * 100)}`
  );
});
```

---

## 📊 API Response Standartları

### Başarılı Response
```json
{
  "status": "success",
  ... // diğer field'lar
}
```

### Hatalı Response
```json
{
  "status": "error",
  "error": "Hata mesajı burada",
  "message": "Detaylı açıklama"  // opsiyonel
}
```

### Pending Response (Veri henüz yok)
```json
{
  "status": "pending",
  "symbol": "THYAO",
  "message": "Analiz henüz yapılmadı"  // opsiyonel
}
```

---

## 🔑 Authentication Mekanizması

### Session-based (Mevcut)

**Login Flow:**
```
1. POST /login → email + password
2. Server checks credentials
3. If valid: creates session, sets cookie
4. Cookie otomatik tüm isteklerde gönderilir
5. Server her istekte session'ı doğrular
```

**Session Cookie:**
```
session=eyJ1c2VyX2lkIjo0LCJfZnJlc2giOmZhbHNlLCJfaWQiOiIxMjM0NTYifQ.ZyNxHw.abcdef...
```

**Session Doğrulama:**
- Her API isteğinde cookie otomatik gönderilir
- Server Flask-Login ile doğrular
- Geçersizse: 401 Unauthorized

---

## 🚀 API Kullanım Senaryoları

### Senaryo 1: Uygulama İlk Açılış

**Adımlar:**
```
1. GET /api/ 
   → API çalışıyor mu kontrol et
   
2. GET /api/watchlist
   → Kullanıcının hisselerini al
   Response: ["AEFES", "ARCLK", "ASELS", "THYAO", "BIMAS", "BRSAN"]
   
3. POST /api/batch/predictions
   Body: {"symbols": ["AEFES", "ARCLK", ...]}
   → Tüm tahminleri tek istekte al
   Response: 6 sembol için predictions
   
4. POST /api/batch/pattern-analysis
   Body: {"symbols": ["AEFES", "ARCLK", ...]}
   → Tüm analizleri tek istekte al
   Response: 6 sembol için analyses
   
5. WebSocket connect
   → Real-time updates için bağlan
   
6. socket.emit('join_user', {user_id: 4})
   → Kullanıcı odasına katıl
   
7. socket.emit('subscribe_stock', {symbol: 'AEFES'})
   socket.emit('subscribe_stock', {symbol: 'ARCLK'})
   ...
   → Her hisseye subscribe ol
```

**Toplam Süre:** ~1 saniye
- 3 HTTP request (paralel yapılabilir)
- 1 WebSocket bağlantısı
- 6 subscribe event'i

---

### Senaryo 2: Pull-to-Refresh

**Adımlar:**
```
1. POST /api/batch/predictions
   Body: {"symbols": ["AEFES", "ARCLK", ...]}
   → Tüm tahminleri yenile
   
2. POST /api/batch/pattern-analysis
   Body: {"symbols": ["AEFES", "ARCLK", ...]}
   → Tüm analizleri yenile
```

**Toplam Süre:** ~500ms

---

### Senaryo 3: Hisse Ekleme

**Adımlar:**
```
1. GET /api/stocks/search?q=thyao
   → Hisse ara
   Response: [{"symbol": "THYAO", "name": "Türk Hava Yolları", ...}]
   
2. POST /api/watchlist
   Body: {"symbol": "THYAO", "alert_enabled": true}
   → Watchlist'e ekle
   Response: {"status": "success", "item": {...}}
   
3. socket.emit('subscribe_stock', {symbol: 'THYAO'})
   → WebSocket subscribe
   
4. POST /api/batch/predictions
   Body: {"symbols": ["THYAO"]}
   → Yeni hisse için tahmin al
   
5. POST /api/batch/pattern-analysis
   Body: {"symbols": ["THYAO"]}
   → Yeni hisse için analiz al
```

**Toplam Süre:** ~800ms

---

### Senaryo 4: Detay Sayfası Açma

**Adımlar:**
```
1. GET /api/pattern-analysis/THYAO?fast=1
   → Analiz detaylarını al (cache-only, hızlı)
   
2. GET /api/stock-prices/THYAO?days=60
   → 60 günlük fiyat geçmişi al (grafik için)
```

**Paralel yapılabilir - Toplam Süre:** ~250ms

---

## 📐 Veri Tipleri ve Formatlar

### Tarih/Zaman Formatları

**ISO 8601 String:**
```
"2025-10-08T18:30:00.123456"
"2025-10-08T17:53:55"
```

**ISO Date String:**
```
"2025-10-08"
```

**Türkçe Locale:**
```javascript
// Frontend'de çevir:
new Date("2025-10-08T18:30:00").toLocaleString('tr-TR')
// → "08.10.2025 18:30:00"
```

---

### Para Formatı

**Backend'den Gelen:**
```json
{
  "price": 117.50,        // float, 2 decimal
  "current_price": 14.02  // float, 2 decimal
}
```

**Frontend'de Göster:**
```javascript
// Türkçe format:
new Intl.NumberFormat('tr-TR', {
  style: 'currency',
  currency: 'TRY'
}).format(117.50)
// → "₺117,50"
```

---

### Yüzde Formatı

**Backend'den Gelen:**
```json
{
  "delta_pct": 0.0285,      // decimal (2.85%)
  "confidence": 0.68,       // decimal (68%)
  "change_pct": -0.012      // decimal (-1.2%)
}
```

**Frontend'de Göster:**
```javascript
// Yüzde'ye çevir:
const pct = delta_pct * 100;  // 2.85
const formatted = `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
// → "+2.9%"
```

---

## ⚠️ Hata Durumları ve HTTP Status Kodları

### 200 OK
```json
{
  "status": "success",
  ...
}
```
İstek başarılı.

---

### 400 Bad Request
```json
{
  "status": "error",
  "error": "symbol is required"
}
```
Geçersiz parametre veya eksik field.

**Örnekler:**
- Symbol eksik
- Sembol sayısı limiti aşıldı (>50)
- Geçersiz format

---

### 401 Unauthorized
```json
{
  "status": "unauthorized"
}
```
Kullanıcı giriş yapmamış veya session süresi dolmuş.

**Çözüm:** Login ekranına yönlendir.

---

### 403 Forbidden
```json
{
  "status": "forbidden"
}
```
Internal API için token geçersiz veya eksik.

---

### 404 Not Found
```json
{
  "status": "error",
  "error": "stock not found"
}
```
İstenen kaynak bulunamadı.

**Örnekler:**
- Sembol database'de yok
- Watchlist item yok
- Veri henüz yok (pending dönmeli)

---

### 500 Internal Server Error
```json
{
  "status": "error",
  "error": "Database connection failed"
}
```
Server hatası.

**Çözüm:** Kullanıcıya hata göster, retry mekanizması.

---

## 🎯 API Best Practices

### 1. Batch API Kullan (Performans)

**❌ KÖTÜ (N+1 Problemi):**
```javascript
// 10 hisse için 10 istek!
for (const symbol of symbols) {
  await fetch(`/api/user/predictions/${symbol}`);
}
// Toplam: ~2 saniye
```

**✅ İYİ (Batch):**
```javascript
// 10 hisse için 1 istek!
await fetch('/api/batch/predictions', {
  method: 'POST',
  body: JSON.stringify({ symbols })
});
// Toplam: ~200ms
```

---

### 2. Cache Kullan (Network Trafiği Azalt)

```javascript
const cache = {
  predictions: null,
  timestamp: 0,
  ttl: 30000  // 30 saniye
};

async function getPredictions(symbols) {
  const now = Date.now();
  
  // Cache geçerliyse kullan
  if (cache.predictions && (now - cache.timestamp) < cache.ttl) {
    return cache.predictions;
  }
  
  // Yoksa API'den çek
  const data = await api.getBatchPredictions(symbols);
  cache.predictions = data;
  cache.timestamp = now;
  
  return data;
}
```

---

### 3. WebSocket Kullan (Polling Yapma)

**❌ KÖTÜ (Polling):**
```javascript
// Her 5 saniyede bir API çağrısı - Server'ı yorar!
setInterval(async () => {
  await fetch('/api/pattern-analysis/THYAO');
}, 5000);
```

**✅ İYİ (WebSocket):**
```javascript
// Sadece değişiklik olduğunda güncelleme gelir
socket.on('pattern_analysis', (data) => {
  if (data.symbol === 'THYAO') {
    updateUI(data.data);
  }
});
```

---

### 4. Error Handling

```javascript
async function getWatchlist() {
  try {
    const response = await fetch('/api/watchlist');
    const data = await response.json();
    
    if (response.status === 401) {
      // Session süresi dolmuş
      redirectToLogin();
      return;
    }
    
    if (data.status !== 'success') {
      throw new Error(data.error || 'Bilinmeyen hata');
    }
    
    return data.watchlist;
  } catch (error) {
    console.error('API Error:', error);
    showErrorToast(error.message);
    return [];
  }
}
```

---

### 5. Throttle ve Debounce

**Arama için Debounce:**
```javascript
let searchTimeout;

function onSearchInput(query) {
  clearTimeout(searchTimeout);
  
  searchTimeout = setTimeout(async () => {
    const results = await fetch(`/api/stocks/search?q=${query}`);
    showResults(results);
  }, 300);  // 300ms bekle, sonra ara
}
```

**Refresh için Throttle:**
```javascript
let lastRefreshTime = 0;

async function refreshData() {
  const now = Date.now();
  
  // 5 saniyede bir'den fazla refresh yapma
  if (now - lastRefreshTime < 5000) {
    console.log('Çok sık refresh, atlanıyor');
    return;
  }
  
  lastRefreshTime = now;
  await loadBatchData();
}
```

---

## 📊 Veri Akışı Diyagramı

### Watchlist Ekranı Data Flow

```
┌─────────────┐
│   Mobil     │
│ Uygulama    │
└──────┬──────┘
       │
       │ 1. GET /api/watchlist
       ├──────────────────────────►┌─────────────┐
       │                           │   Backend   │
       │◄──────────────────────────┤  (Flask)    │
       │  [AEFES, ARCLK, ASELS]    └──────┬──────┘
       │                                  │
       │ 2. POST /api/batch/predictions   │
       │    {symbols: [AEFES, ARCLK...]}  │
       ├──────────────────────────────────►
       │                                  │
       │                           ┌──────▼──────┐
       │                           │   Cache     │
       │                           │ (JSON File) │
       │                           └──────┬──────┘
       │◄──────────────────────────────────
       │  {AEFES: {1d:14.03,...}, ...}
       │
       │ 3. POST /api/batch/pattern-analysis
       ├──────────────────────────────────►
       │                                  │
       │◄──────────────────────────────────
       │  {AEFES: {patterns:[...], ...}}
       │
       │ 4. WebSocket connect
       ├══════════════════════════════════►
       │                                  │
       │  socket.on('pattern_analysis')   │
       │◄══════════════════════════════════
       │  Real-time updates
       │
       ▼
   ┌────────┐
   │   UI   │
   │ Render │
   └────────┘
```

---

## 🔄 Automation Cycle ile Veri Üretimi

### Cycle Nasıl Çalışır?

```
┌────────────────────────────────────────┐
│  Automation Cycle (Her 5 dakikada)    │
└────────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │ 1. Tüm hisseleri al    │ ← PostgreSQL
    │    (608 sembol)        │
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ 2. Her sembol için:    │
    │   - Fiyat verisi çek   │ ← Yahoo Finance
    │   - Pattern analizi    │ ← pattern_detector.py
    │   - ML tahminleri      │ ← enhanced_ml_system.py
    │   - Visual analiz      │ ← YOLO
    │   - Sentiment analiz   │ ← FinGPT
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ 3. Cache'e yaz:        │
    │   pattern_cache/       │ ← {SYMBOL}.json
    │   ml_bulk_predictions  │ ← Toplu tahminler
    │   signals_last         │ ← Son sinyaller
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │ 4. WebSocket broadcast │
    │   pattern_analysis     │ → Tüm bağlı client'lar
    │   user_signal          │ → İlgili kullanıcılar
    └────────────────────────┘
```

**Önemli:**
- Cycle ~5-10 dakika sürer (608 sembol için)
- API'ler cache'den okur (çok hızlı!)
- Fresh hesaplama YOK (cycle yapar)

---

## 📝 API Request Örnekleri (cURL)

### Watchlist Al
```bash
curl -X GET 'https://your-domain.com/api/watchlist' \
  -H 'Cookie: session=...' \
  -H 'Content-Type: application/json'
```

### Hisse Ekle
```bash
curl -X POST 'https://your-domain.com/api/watchlist' \
  -H 'Cookie: session=...' \
  -H 'Content-Type: application/json' \
  -d '{
    "symbol": "THYAO",
    "alert_enabled": true,
    "notes": "Test notu"
  }'
```

### Batch Predictions
```bash
curl -X POST 'https://your-domain.com/api/batch/predictions' \
  -H 'Content-Type: application/json' \
  -d '{
    "symbols": ["AEFES", "ARCLK", "THYAO"]
  }'
```

### Pattern Analysis
```bash
curl -X GET 'https://your-domain.com/api/pattern-analysis/THYAO?fast=1'
```

### Hisse Ara
```bash
curl -X GET 'https://your-domain.com/api/stocks/search?q=türk&limit=20'
```

### Fiyat Geçmişi
```bash
curl -X GET 'https://your-domain.com/api/stock-prices/THYAO?days=60'
```

---

## 🧪 Postman Collection

### Environment Variables
```json
{
  "base_url": "https://your-domain.com",
  "session_cookie": "session=eyJ1c2VyX2lkIjo0...",
  "internal_token": "IBx_gsmQUL9oxymAgr67PxES7ACfKlk1Ex5F9jCCOFw"
}
```

### Collection Örneği
```json
{
  "info": {
    "name": "BIST Pattern API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Auth",
      "item": [
        {
          "name": "Login",
          "request": {
            "method": "POST",
            "header": [],
            "body": {
              "mode": "urlencoded",
              "urlencoded": [
                {"key": "email", "value": "user@example.com"},
                {"key": "password", "value": "secret"}
              ]
            },
            "url": "{{base_url}}/login"
          }
        }
      ]
    },
    {
      "name": "Watchlist",
      "item": [
        {
          "name": "Get Watchlist",
          "request": {
            "method": "GET",
            "header": [
              {"key": "Cookie", "value": "{{session_cookie}}"}
            ],
            "url": "{{base_url}}/api/watchlist"
          }
        }
      ]
    }
  ]
}
```

---

## 🔍 Debugging ve Test

### Console'da Test (Browser)

**Watchlist Al:**
```javascript
fetch('/api/watchlist', {
  credentials: 'include'
})
.then(r => r.json())
.then(d => console.log(d));
```

**Batch Predictions:**
```javascript
fetch('/api/batch/predictions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({symbols: ['AEFES', 'THYAO']})
})
.then(r => r.json())
.then(d => console.log(d));
```

**WebSocket Test:**
```javascript
const socket = io();
socket.on('connect', () => console.log('Bağlandı:', socket.id));
socket.emit('join_user', {user_id: 4});
socket.on('pattern_analysis', d => console.log('Güncelleme:', d));
```

---

## 📊 Response Boyutları (Yaklaşık)

| Endpoint | Sembol Sayısı | Ortalama Boyut | Süre |
|----------|---------------|----------------|------|
| GET /api/watchlist | - | 2 KB | 50ms |
| POST /api/batch/predictions | 10 | 8 KB | 200ms |
| POST /api/batch/pattern-analysis | 10 | 80 KB | 300ms |
| GET /api/pattern-analysis/{symbol} | 1 | 8 KB | 150ms |
| GET /api/stock-prices/{symbol} | 1 (60 gün) | 5 KB | 100ms |
| GET /api/stocks/search | - (50 sonuç) | 15 KB | 80ms |

**Not:** Boyutlar ve süreler cache durumuna göre değişir.

---

## 🔐 Security Notları

### CSRF Protection
- API endpoint'leri CSRF'den muaf
- Form-based login CSRF korumalı değil (proxy uyumluluğu için)

### Rate Limiting
- API'lerde rate limiting var
- Varsayılan: 100 request/dakika per IP
- Batch API'ler önerilir (tek request)

### HTTPS Zorunlu (Production)
- Tüm API'ler HTTPS üzerinden çalışmalı
- Session cookie HttpOnly
- Credentials: include gerekli

---

## 📱 Mobil Uygulama İçin Öneriler

### İlk Yükleme Stratejisi
```dart
1. showLoadingScreen();
2. final watchlist = await api.getWatchlist();        // 50ms
3. final symbols = watchlist.map((w) => w.symbol);
4. await Future.wait([                                // Paralel!
     api.getBatchPredictions(symbols),               // 200ms
     api.getBatchPatternAnalysis(symbols),           // 300ms
   ]);
5. ws.connect();                                      // 150ms
6. hideLoadingScreen();
// Toplam: ~700ms (çok hızlı!)
```

### Refresh Stratejisi
```dart
Future<void> onRefresh() async {
  final symbols = watchlist.map((w) => w.symbol).toList();
  
  await Future.wait([
    api.getBatchPredictions(symbols),
    api.getBatchPatternAnalysis(symbols),
  ]);
  
  setState(() {});
}
```

### Detay Sayfası Stratejisi
```dart
Future<void> openDetail(String symbol) async {
  // Cache'den hızlı göster
  showCachedData(symbol);
  
  // Paralel olarak güncel veri yükle
  final results = await Future.wait([
    api.getPatternAnalysis(symbol, fast: true),
    api.getStockPrices(symbol, days: 60),
  ]);
  
  updateUI(results[0], results[1]);
}
```

---

## 🎨 UI için Yardımcı Fonksiyonlar

### Signal Label (Türkçe)
```dart
String getSignalLabel(String signal, double confidence) {
  if (signal == 'BULLISH') {
    if (confidence >= 0.85) return 'Yüksek Alım Sinyali';
    if (confidence >= 0.70) return 'Alım Sinyali';
    if (confidence >= 0.55) return 'Zayıf Alım';
    return 'Bekleme';
  } else if (signal == 'BEARISH') {
    if (confidence >= 0.85) return 'Yüksek Satış Sinyali';
    if (confidence >= 0.70) return 'Satış Sinyali';
    if (confidence >= 0.55) return 'Zayıf Satış';
    return 'Bekleme';
  }
  return 'Nötr';
}
```

### Model Label (Türkçe)
```dart
String getModelLabel(String? model) {
  if (model == 'enhanced') return 'Gelişmiş';
  if (model == 'basic') return 'Temel';
  return 'Bilinmiyor';
}
```

### Source Label (Türkçe)
```dart
String getSourceLabel(String source) {
  const labels = {
    'ML_PREDICTOR': 'Temel Analiz',
    'ENHANCED_ML': 'Gelişmiş Analiz',
    'VISUAL_YOLO': 'Görsel',
    'ADVANCED_TA': 'Teknik Analiz',
    'FINGPT': 'Sezgisel',
  };
  return labels[source] ?? source;
}
```

---

## ⚡ Performance Metrikleri

### Gerçek Ölçümler (Production)

**İlk Yükleme (6 hisse):**
```
GET /api/watchlist                     → 52ms
POST /api/batch/predictions            → 187ms
POST /api/batch/pattern-analysis       → 312ms
WebSocket connect + join + subscribe   → 145ms
────────────────────────────────────────────────
TOPLAM: ~696ms
```

**Refresh (6 hisse):**
```
POST /api/batch/predictions            → 195ms
POST /api/batch/pattern-analysis       → 305ms
────────────────────────────────────────────────
TOPLAM: ~500ms
```

**Detay Açma:**
```
GET /api/pattern-analysis/THYAO        → 142ms
GET /api/stock-prices/THYAO            → 98ms
────────────────────────────────────────────────
TOPLAM: ~240ms
```

---

## 📋 Endpoint Özet Tablosu

| Endpoint | Method | Auth | Ne Sorar | Ne Verir | Süre |
|----------|--------|------|----------|----------|------|
| /login | POST | ❌ | email, password | Session cookie | 100ms |
| /logout | GET | ✅ | - | Redirect | 50ms |
| /api/watchlist | GET | ✅ | - | Watchlist array | 50ms |
| /api/watchlist | POST | ✅ | symbol, alerts | Added item | 80ms |
| /api/watchlist/{symbol} | DELETE | ✅ | symbol | Success msg | 60ms |
| /api/batch/predictions | POST | ❌ | symbols array | Predictions map | 200ms |
| /api/batch/pattern-analysis | POST | ❌ | symbols array | Analyses map | 300ms |
| /api/user/predictions/{symbol} | GET | ✅ | symbol | Single prediction | 150ms |
| /api/pattern-analysis/{symbol} | GET | ❌ | symbol, fast? | Single analysis | 150ms |
| /api/stocks/search | GET | ❌ | q, limit? | Stock array | 80ms |
| /api/stocks | GET | ❌ | - | Stock array (1000) | 120ms |
| /api/stock-prices/{symbol} | GET | ❌ | symbol, days? | Price history | 100ms |
| /api/ | GET | ❌ | - | API info | 10ms |
| /api/health | GET | ❌ | - | Health status | 20ms |

**Auth Sütunu:**
- ✅: Session cookie gerekli
- ❌: Public endpoint (authentication gerekmez)

---

## 🔔 WebSocket Event Tablosu

| Event | Direction | Ne Sorar | Ne Verir |
|-------|-----------|----------|----------|
| connect | S→C | - | connection_id, timestamp |
| disconnect | S→C | - | - |
| join_user | C→S | user_id | room_joined event |
| join_admin | C→S | - | room_joined event |
| subscribe_stock | C→S | symbol | subscription_confirmed |
| unsubscribe_stock | C→S | symbol | - |
| request_pattern_analysis | C→S | symbol | pattern_analysis event |
| pattern_analysis | S→C | - | symbol, data, timestamp |
| user_signal | S→C | - | signal data |
| room_joined | S→C | - | room, message |
| subscription_confirmed | S→C | - | symbol, message |
| error | S→C | - | message |

**Direction:**
- C→S: Client → Server
- S→C: Server → Client

---

## 💾 Cache Mekanizması

### Memory/Redis Cache
```
TTL: 300 saniye (5 dakika)
Key Format: pattern_analysis:{SYMBOL}
Kullanım: İlk öncelik
```

### File Cache
```
Path: /opt/bist-pattern/logs/pattern_cache/{SYMBOL}.json
TTL: 300 saniye (5 dakika)
Kullanım: Memory cache miss'te
```

### Bulk Predictions File
```
Path: /opt/bist-pattern/logs/ml_bulk_predictions.json
Update: Automation cycle her çalıştığında
Format: {"predictions": {"{SYMBOL}": {...}}}
```

### Cache Yaşlandırma
```
stale: false  → Fresh (< 300 saniye)
stale: true   → Eski (> 300 saniye ama hala kullanılabilir)
```

**API Davranışı:**
- Cache varsa (fresh veya stale): Hemen döndürür
- Cache yoksa: `status: "pending"` döndürür
- Fresh hesaplama yapılmaz (cycle bekle)

---

## 🛡️ Error Recovery Stratejileri

### Network Timeout
```dart
try {
  final response = await api.getBatchPredictions(symbols)
    .timeout(Duration(seconds: 30));
} on TimeoutException {
  // Cache'den göster veya retry
  return getCachedPredictions();
} catch (e) {
  // Hata göster
  showErrorDialog('Bağlantı hatası: $e');
}
```

### 401 Unauthorized
```dart
if (response.statusCode == 401) {
  // Session süresi dolmuş
  await logout();
  Navigator.pushReplacement(
    context,
    MaterialPageRoute(builder: (_) => LoginScreen()),
  );
}
```

### WebSocket Disconnect
```dart
socket.on('disconnect', () {
  setState(() => isConnected = false);
  
  // 3 saniye sonra otomatik reconnect
  Future.delayed(Duration(seconds: 3), () {
    if (!isConnected) {
      socket.connect();
    }
  });
});
```

### Partial Data (Pending)
```dart
final result = predictions['THYAO'];
if (result['status'] == 'pending') {
  // Veri henüz hazır değil
  showPlaceholder('Analiz bekleniyor...');
  
  // WebSocket'ten güncelleme gelince göster
  socket.on('pattern_analysis', (data) {
    if (data['symbol'] == 'THYAO') {
      updateUI(data['data']);
    }
  });
}
```

---

## 📊 Veri Tutarlılığı

### ml_unified vs predictions vs enhanced_predictions

**3 farklı format var:**

**1. enhanced_predictions (Ham ML):**
```json
{
  "7d": {
    "ensemble_prediction": 14.06,
    "confidence": 0.62,
    "models": {
      "xgboost": {...},
      "lightgbm": {...},
      "catboost": {...}
    }
  }
}
```

**2. predictions (Basit format - batch API):**
```json
{
  "7d": 14.06
}
```

**3. ml_unified (Birleşik format - en detaylı):**
```json
{
  "7d": {
    "basic": {
      "price": 14.32,
      "confidence": 0.25,
      "delta_pct": 0.0216,
      "evidence": {...}
    },
    "enhanced": {
      "price": 14.06,
      "confidence": 0.62,
      "delta_pct": 0.0028,
      "evidence": {...}
    },
    "best": "enhanced"
  }
}
```

**Mobil Uygulama Önerisi:**
- Kart ekranında: `predictions` kullan (basit, hızlı)
- Detay ekranında: `ml_unified` kullan (detaylı, evidence var)

---

## 🎯 API Seçim Rehberi

### Ne Zaman Hangi API?

**Watchlist Ekranı:**
```
✅ POST /api/batch/predictions         → Tüm tahminler
✅ POST /api/batch/pattern-analysis    → Tüm sinyaller
```

**Detay Ekranı:**
```
✅ GET /api/pattern-analysis/{symbol}  → Tam analiz
✅ GET /api/stock-prices/{symbol}      → Grafik verisi
```

**Arama:**
```
✅ GET /api/stocks/search              → Hisse ara
```

**Hisse Ekleme/Çıkarma:**
```
✅ POST /api/watchlist                 → Ekle
✅ DELETE /api/watchlist/{symbol}      → Çıkar
```

**Real-time Updates:**
```
✅ WebSocket pattern_analysis          → Otomatik güncelleme
✅ WebSocket user_signal               → Push notification
```

---

## 📖 API Versiyonlama

**Mevcut Versiyon:** 2.2.0

**Breaking Changes:**
- Versiyon değişikliklerinde backward compatibility korunur
- Yeni field'lar eklenir (eski field'lar kaldırılmaz)
- Deprecated field'lar için 6 ay grace period

**Version Header (gelecekte):**
```http
X-API-Version: 2.2.0
```

---

## 📞 Destek ve İletişim

**API Sorunları:**
- Log kontrolü: `/opt/bist-pattern/logs/gunicorn_error.log`
- Status: `GET /api/health`
- Automation: `GET /api/internal/automation/status`

**Documentation Updates:**
- Bu dosya: `/opt/bist-pattern/docs/API_REFERANS_DOKUMANTASYONU.md`
- Flutter rehberi: `/opt/bist-pattern/docs/FLUTTER_MOBIL_UYGULAMA_REHBERI.md`

---

**Son Güncelleme:** 08 Ekim 2025
**API Versiyon:** 2.2.0
**Dokümantasyon Versiyon:** 1.0

