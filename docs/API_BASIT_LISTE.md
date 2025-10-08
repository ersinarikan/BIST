# BIST Pattern API - Basit Liste

## 📌 Önemli Notlar

**Base URL:** `https://your-domain.com/api`

**Authentication:** Session cookie (login sonrası otomatik)

**Internal Token:** `.env` dosyasında `INTERNAL_API_TOKEN` ile ayarla (örneklerde gösterilen token'ı değiştir!)

---

## 🟢 GET Endpoint'leri

### GET /api/watchlist
**Ne sorar:** Session cookie  
**Cevap:** Kullanıcının takip ettiği hisseler listesi
```json
{"status":"success", "watchlist":[{"symbol":"AEFES","name":"Anadolu Efes"}]}
```

---

### GET /api/stocks/search?q={query}
**Ne sorar:** Arama kelimesi (q parametresi)  
**Cevap:** Bulunan hisseler listesi
```json
{"status":"success", "stocks":[{"symbol":"THYAO","name":"Türk Hava Yolları","sector":"Ulaştırma"}]}
```

---

### GET /api/stocks
**Ne sorar:** Hiçbir şey  
**Cevap:** Tüm hisseler (max 1000)
```json
{"status":"success", "stocks":[{"id":1,"symbol":"THYAO","name":"Türk Hava Yolları"}]}
```

---

### GET /api/stock-prices/{symbol}?days={gün}
**Ne sorar:** Sembol (URL) ve gün sayısı (query)  
**Cevap:** Fiyat geçmişi (grafik için)
```json
{"status":"success", "data":[{"date":"2025-10-08","close":120.5,"volume":1250000}]}
```

---

### GET /api/pattern-analysis/{symbol}?fast=1
**Ne sorar:** Sembol (URL), fast=1 (cache-only)  
**Cevap:** Pattern analizi (cache'den, hesaplama YOK)
```json
{"symbol":"THYAO", "current_price":120.5, "patterns":[...], "ml_unified":{...}}
```

---

### GET /api/user/predictions/{symbol}
**Ne sorar:** Sembol (URL), session cookie  
**Cevap:** Tek sembol için tahminler
```json
{"status":"success", "predictions":{"1d":120.5,"3d":122.0,"7d":125.0}}
```

---

### GET /api/
**Ne sorar:** Hiçbir şey  
**Cevap:** API bilgisi (çalışıyor mu kontrolü)
```json
{"status":"running", "version":"2.2.0"}
```

---

### GET /api/health
**Ne sorar:** Hiçbir şey  
**Cevap:** Sistem sağlığı
```json
{"status":"healthy", "database":"connected", "automation":"running"}
```

---

### GET /api/internal/automation/status
**Ne sorar:** X-Internal-Token header  
**Cevap:** Automation durumu (admin)
```json
{"is_running":true, "current_cycle":42, "symbols_processed":608}
```

---

### GET /api/internal/automation/volume/tiers?symbol={symbol}
**Ne sorar:** Symbol (query), X-Internal-Token header  
**Cevap:** Hacim seviyesi
```json
{"symbol":"THYAO", "tier":"high", "avg_volume":1450000}
```

---

## 🔵 POST Endpoint'leri

### POST /api/batch/predictions
**Ne gönderir:** Sembol listesi
```json
{"symbols":["AEFES","ARCLK","THYAO"]}
```
**Cevap:** Tüm semboller için tahminler (tek istekte!)
```json
{"status":"success", "results":{"AEFES":{"predictions":{"1d":14.03,"7d":14.06}}}}
```

---

### POST /api/batch/pattern-analysis
**Ne gönderir:** Sembol listesi
```json
{"symbols":["AEFES","ARCLK","THYAO"]}
```
**Cevap:** Tüm semboller için analizler (tek istekte!)
```json
{"status":"success", "results":{"AEFES":{"patterns":[...],"overall_signal":{...}}}}
```

---

### POST /api/watchlist
**Ne gönderir:** Hisse bilgileri
```json
{"symbol":"THYAO", "alert_enabled":true, "notes":"Test"}
```
**Cevap:** Eklenen item
```json
{"status":"success", "item":{"id":7,"symbol":"THYAO"}}
```

---

### POST /login
**Ne gönderir:** Email ve şifre (form-urlencoded)
```
email=user@example.com&password=secret
```
**Cevap:** Redirect + session cookie
```
302 Redirect → /user
Set-Cookie: session=...
```

---

### POST /api/internal/automation/start
**Ne gönderir:** X-Internal-Token header, boş body
```json
{}
```
**Cevap:** Başlatma durumu
```json
{"status":"success", "message":"Automation started", "is_running":true}
```

---

### POST /api/internal/automation/stop
**Ne gönderir:** X-Internal-Token header, boş body
```json
{}
```
**Cevap:** Durdurma durumu
```json
{"status":"success", "message":"Automation stopped", "is_running":false}
```

---

## 🔴 DELETE Endpoint'leri

### DELETE /api/watchlist/{symbol}
**Ne sorar:** Symbol (URL), session cookie  
**Cevap:** Silme onayı
```json
{"status":"success", "message":"THYAO removed"}
```

---

## 🔔 WebSocket Events

### Client → Server (emit)

**join_user**
```javascript
socket.emit('join_user', {user_id: 4})
```

**subscribe_stock**
```javascript
socket.emit('subscribe_stock', {symbol: 'THYAO'})
```

**unsubscribe_stock**
```javascript
socket.emit('unsubscribe_stock', {symbol: 'THYAO'})
```

---

### Server → Client (on)

**pattern_analysis** - Analiz güncellendi
```json
{"symbol":"THYAO", "data":{...}, "timestamp":"2025-10-08T18:30:00"}
```

**user_signal** - Yeni sinyal
```json
{"signal":{"symbol":"THYAO","overall_signal":{...}}}
```

**room_joined** - Odaya katıldı
```json
{"room":"user_4", "message":"User interface connected"}
```

---

## 🔑 Internal API Token Kullanımı

### Header Formatı:
```http
X-Internal-Token: YOUR_SECURE_TOKEN_HERE
```

### Token Ayarlama (.env):
```bash
# .env dosyasına ekle:
INTERNAL_API_TOKEN=yeni_guvenli_token_buraya_1a2b3c4d5e6f
```

### Token Oluşturma:
```bash
# Python ile güvenli token:
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Veya OpenSSL ile:
openssl rand -base64 32
```

**⚠️ ÖNEMLİ:** Default token'ı mutlaka değiştir! Production'da kullanma!

---

## ⚡ Hızlı Referans

**Watchlist için:** `GET /api/watchlist`

**Tahminler için (batch):** `POST /api/batch/predictions`

**Analizler için (batch):** `POST /api/batch/pattern-analysis`

**Hisse ara:** `GET /api/stocks/search?q=...`

**Hisse ekle:** `POST /api/watchlist`

**Hisse çıkar:** `DELETE /api/watchlist/{symbol}`

**Grafik için fiyat:** `GET /api/stock-prices/{symbol}?days=60`

**WebSocket bağlan:** `io('https://domain.com', {path:'/socket.io'})`

---

## 📊 Response Kodları

- **200**: Başarılı
- **302**: Redirect (login sonrası)
- **400**: Geçersiz istek
- **401**: Giriş gerekli
- **403**: Yetkisiz (internal API için)
- **404**: Bulunamadı
- **500**: Server hatası

---

**Basit ve öz! Daha fazla detay için diğer dokümanlara bak.**

