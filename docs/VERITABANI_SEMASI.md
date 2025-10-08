# BIST Pattern Veritabanı Şeması

## 📊 PostgreSQL Database Schema

Bu dokümantasyon, BIST Pattern sisteminin tüm veritabanı tablolarını, sütunlarını ve ilişkilerini detaylı olarak açıklar.

---

## 📋 İçindekiler

1. [Tablo Özeti](#tablo-özeti)
2. [User Yönetimi](#user-yönetimi)
3. [Hisse Senedi Verileri](#hisse-senedi-verileri)
4. [Kullanıcı Watchlist](#kullanıcı-watchlist)
5. [Paper Trading](#paper-trading)
6. [ML Prediction Logging](#ml-prediction-logging)
7. [İlişki Diyagramı](#ilişki-diyagramı)
8. [Index'ler](#indexler)

---

## 📊 Tablo Özeti

| Tablo | Amaç | Satır Sayısı (Tahmini) |
|-------|------|------------------------|
| **users** | Kullanıcı hesapları | ~100 |
| **stocks** | Hisse senedi bilgileri | ~600 |
| **stock_prices** | Günlük OHLCV verileri | ~220,000 |
| **watchlist** | Kullanıcı takip listeleri | ~1,000 |
| **simulation_sessions** | Paper trading oturumları | ~50 |
| **simulation_trades** | Paper trading işlemleri | ~500 |
| **portfolio_snapshots** | Portföy anlık görüntüleri | ~1,000 |
| **predictions_log** | ML tahmin kayıtları | ~50,000 |
| **outcomes_log** | Tahmin sonuçları | ~30,000 |
| **metrics_daily** | Günlük metrikler | ~10,000 |

---

## 👤 User Yönetimi

### Table: **users**

**Açıklama:** Kullanıcı hesapları, authentication ve profil bilgileri

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **email** | VARCHAR(255) | NO | - | Email (unique, index) |
| **username** | VARCHAR(80) | YES | NULL | Kullanıcı adı (unique) |
| **password_hash** | VARCHAR(255) | YES | NULL | Şifre hash (OAuth'ta null) |
| **first_name** | VARCHAR(100) | YES | NULL | Ad |
| **last_name** | VARCHAR(100) | YES | NULL | Soyad |
| **avatar_url** | VARCHAR(500) | YES | NULL | Profil resmi URL |
| **provider** | VARCHAR(20) | NO | 'email' | Auth provider (email/google/apple) |
| **provider_id** | VARCHAR(255) | YES | NULL | OAuth provider ID |
| **email_verified** | BOOLEAN | NO | FALSE | Email doğrulandı mı? |
| **email_verification_token** | VARCHAR(255) | YES | NULL | Doğrulama token'ı |
| **email_verification_sent_at** | DATETIME | YES | NULL | Token gönderim zamanı |
| **is_active** | BOOLEAN | NO | TRUE | Hesap aktif mi? |
| **is_premium** | BOOLEAN | NO | FALSE | Premium üyelik |
| **created_at** | DATETIME | NO | NOW() | Oluşturulma tarihi |
| **last_login** | DATETIME | YES | NULL | Son giriş zamanı |
| **role** | VARCHAR(20) | NO | 'user' | Rol (user/admin) |
| **last_login_ip** | VARCHAR(45) | YES | NULL | Son giriş IP |
| **timezone** | VARCHAR(50) | NO | 'Europe/Istanbul' | Saat dilimi |
| **language** | VARCHAR(5) | NO | 'tr' | Dil (tr/en) |
| **email_notifications** | BOOLEAN | NO | TRUE | Email bildirimleri |
| **push_notifications** | BOOLEAN | NO | TRUE | Push bildirimleri |

**Index'ler:**
- PRIMARY KEY: `id`
- UNIQUE: `email`, `username`
- INDEX: `email`, `role`, `last_login`

**İlişkiler:**
- `watchlist` → Kullanıcının takip listesi (1:N)
- `simulation_sessions` → Kullanıcının paper trading oturumları (1:N)

**Örnek Data:**
```sql
INSERT INTO users (email, username, password_hash, role, is_active)
VALUES ('admin@bist.com', 'admin', 'pbkdf2:sha256:...', 'admin', TRUE);
```

---

## 📈 Hisse Senedi Verileri

### Table: **stocks**

**Açıklama:** Hisse senedi ana bilgileri (sembol, ad, sektör)

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **symbol** | VARCHAR(20) | NO | - | Hisse kodu (unique, index) |
| **name** | VARCHAR(255) | NO | - | Hisse adı |
| **sector** | VARCHAR(100) | YES | NULL | Sektör |
| **market_cap** | BIGINT | YES | NULL | Piyasa değeri |
| **is_active** | BOOLEAN | NO | TRUE | Aktif mi? |
| **created_at** | DATETIME | NO | NOW() | Oluşturulma tarihi |
| **updated_at** | DATETIME | NO | NOW() | Güncellenme tarihi (auto) |

**Index'ler:**
- PRIMARY KEY: `id`
- UNIQUE: `symbol`
- INDEX: `symbol`

**İlişkiler:**
- `stock_prices` → Fiyat verileri (1:N)
- `watchlist` → Watchlist itemları (1:N)
- `simulation_trades` → Paper trading işlemleri (1:N)

**Örnek Data:**
```sql
INSERT INTO stocks (symbol, name, sector, market_cap)
VALUES ('THYAO', 'Türk Hava Yolları', 'Ulaştırma', 25000000000);
```

---

### Table: **stock_prices**

**Açıklama:** Günlük OHLCV (Open, High, Low, Close, Volume) fiyat verileri

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **stock_id** | INTEGER | NO | - | Foreign key → stocks.id |
| **date** | DATE | NO | - | İşlem günü |
| **open_price** | NUMERIC(10,4) | NO | - | Açılış fiyatı |
| **high_price** | NUMERIC(10,4) | NO | - | En yüksek fiyat |
| **low_price** | NUMERIC(10,4) | NO | - | En düşük fiyat |
| **close_price** | NUMERIC(10,4) | NO | - | Kapanış fiyatı |
| **volume** | BIGINT | NO | - | İşlem hacmi |
| **created_at** | DATETIME | NO | NOW() | Kayıt zamanı |

**Index'ler:**
- PRIMARY KEY: `id`
- UNIQUE: `(stock_id, date)` - Her sembol için her gün sadece 1 kayıt
- INDEX: `stock_id`, `date`
- COMPOSITE INDEX: `(stock_id, date)`

**Foreign Keys:**
- `stock_id` → `stocks.id` (CASCADE DELETE)

**Örnek Data:**
```sql
INSERT INTO stock_prices (stock_id, date, open_price, high_price, low_price, close_price, volume)
VALUES (1, '2025-10-08', 119.50, 121.00, 119.00, 120.50, 1420000);
```

**Veri Boyutu:**
- ~365 gün/sembol × 600 sembol = ~220,000 satır
- Günlük büyüme: ~600 satır
- Yıllık büyüme: ~220,000 satır

---

## ⭐ Kullanıcı Watchlist

### Table: **watchlist**

**Açıklama:** Kullanıcıların takip ettiği hisseler ve alarm ayarları

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **user_id** | INTEGER | NO | - | Foreign key → users.id |
| **stock_id** | INTEGER | NO | - | Foreign key → stocks.id |
| **notes** | TEXT | YES | NULL | Kullanıcı notları |
| **alert_enabled** | BOOLEAN | NO | TRUE | Alarm aktif mi? |
| **alert_threshold_buy** | NUMERIC(10,4) | YES | NULL | Alım alarm fiyatı |
| **alert_threshold_sell** | NUMERIC(10,4) | YES | NULL | Satış alarm fiyatı |
| **created_at** | DATETIME | NO | NOW() | Oluşturulma tarihi |
| **updated_at** | DATETIME | NO | NOW() | Güncellenme tarihi (auto) |

**Index'ler:**
- PRIMARY KEY: `id`
- UNIQUE: `(user_id, stock_id)` - Bir kullanıcı aynı hisseyi 1 kez ekleyebilir
- INDEX: `user_id`, `stock_id`

**Foreign Keys:**
- `user_id` → `users.id` (CASCADE DELETE)
- `stock_id` → `stocks.id` (CASCADE DELETE)

**Örnek Data:**
```sql
INSERT INTO watchlist (user_id, stock_id, notes, alert_threshold_buy)
VALUES (4, 1, 'İzleniyor', 125.00);
```

---

## 🎮 Paper Trading

### Table: **simulation_sessions**

**Açıklama:** Paper trading oturumları (sanal portföy testi)

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **user_id** | INTEGER | NO | - | Foreign key → users.id |
| **session_name** | VARCHAR(100) | NO | 'AI Performance Test' | Oturum adı |
| **initial_balance** | NUMERIC(15,2) | NO | 100.00 | Başlangıç bakiyesi (TL) |
| **duration_hours** | INTEGER | NO | 48 | Süre (saat) |
| **status** | VARCHAR(20) | NO | 'active' | Durum (active/completed/paused) |
| **start_time** | DATETIME | NO | NOW() | Başlangıç zamanı |
| **end_time** | DATETIME | YES | NULL | Bitiş zamanı |
| **current_balance** | NUMERIC(15,2) | NO | 100.00 | Güncel bakiye |
| **total_trades** | INTEGER | NO | 0 | Toplam işlem sayısı |
| **winning_trades** | INTEGER | NO | 0 | Kazanan işlem sayısı |
| **losing_trades** | INTEGER | NO | 0 | Kaybeden işlem sayısı |
| **created_at** | DATETIME | NO | NOW() | Oluşturulma |
| **updated_at** | DATETIME | NO | NOW() | Güncellenme (auto) |

**Properties (Computed):**
- `profit_loss`: current_balance - initial_balance
- `profit_loss_percentage`: (profit_loss / initial_balance) × 100
- `win_rate`: (winning_trades / total_trades) × 100

**Foreign Keys:**
- `user_id` → `users.id`

**İlişkiler:**
- `simulation_trades` → İşlemler (1:N)
- `portfolio_snapshots` → Anlık görüntüler (1:N)

---

### Table: **simulation_trades**

**Açıklama:** Paper trading işlemleri (alım/satım)

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **session_id** | INTEGER | NO | - | Foreign key → simulation_sessions.id |
| **stock_id** | INTEGER | NO | - | Foreign key → stocks.id |
| **trade_type** | VARCHAR(10) | NO | - | İşlem tipi (BUY/SELL) |
| **quantity** | NUMERIC(10,4) | NO | - | Miktar (fractional shares) |
| **price** | NUMERIC(10,4) | NO | - | İşlem fiyatı |
| **total_amount** | NUMERIC(15,2) | NO | - | Toplam tutar |
| **signal_source** | VARCHAR(50) | YES | NULL | Sinyal kaynağı (MACD/RSI/PATTERN) |
| **signal_confidence** | NUMERIC(5,2) | YES | NULL | Sinyal güveni |
| **pattern_detected** | VARCHAR(50) | YES | NULL | Tespit edilen pattern |
| **status** | VARCHAR(20) | NO | 'executed' | Durum (executed/pending/cancelled) |
| **execution_time** | DATETIME | NO | NOW() | Gerçekleşme zamanı |
| **profit_loss** | NUMERIC(15,2) | YES | NULL | Kar/Zarar (pozisyon kapatıldığında) |
| **profit_loss_percentage** | NUMERIC(5,2) | YES | NULL | Kar/Zarar % |
| **created_at** | DATETIME | NO | NOW() | Kayıt zamanı |

**Foreign Keys:**
- `session_id` → `simulation_sessions.id`
- `stock_id` → `stocks.id`

---

### Table: **portfolio_snapshots**

**Açıklama:** Portföy performans takibi için anlık görüntüler

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **session_id** | INTEGER | NO | - | Foreign key → simulation_sessions.id |
| **cash_balance** | NUMERIC(15,2) | NO | - | Nakit bakiye |
| **total_portfolio_value** | NUMERIC(15,2) | NO | - | Toplam portföy değeri |
| **total_stocks_value** | NUMERIC(15,2) | NO | 0 | Hisse değeri toplamı |
| **total_profit_loss** | NUMERIC(15,2) | NO | 0 | Toplam kar/zarar |
| **total_profit_loss_percentage** | NUMERIC(5,2) | NO | 0 | Toplam kar/zarar % |
| **active_positions** | INTEGER | NO | 0 | Açık pozisyon sayısı |
| **snapshot_time** | DATETIME | NO | NOW() | Snapshot zamanı |

**Foreign Keys:**
- `session_id` → `simulation_sessions.id`

---

## 🤖 ML Prediction Logging

### Table: **predictions_log**

**Açıklama:** ML tahminlerinin gerçek zamanlı kaydı (feedback loop için)

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **stock_id** | INTEGER | YES | NULL | Foreign key → stocks.id |
| **symbol** | VARCHAR(20) | NO | - | Hisse kodu |
| **horizon** | VARCHAR(10) | NO | - | Tahmin ufku (1d/3d/7d/14d/30d) |
| **ts_pred** | DATETIME | NO | NOW() | Tahmin zamanı |
| **price_now** | NUMERIC(14,4) | YES | NULL | Tahmin anındaki fiyat |
| **pred_price** | NUMERIC(14,4) | YES | NULL | Tahmin edilen fiyat |
| **delta_pred** | NUMERIC(8,4) | YES | NULL | Tahmin edilen değişim % |
| **model** | VARCHAR(12) | YES | NULL | Kullanılan model (basic/enhanced) |
| **unified_best** | VARCHAR(12) | YES | NULL | En iyi seçilen model |
| **confidence** | NUMERIC(4,2) | YES | NULL | Güven skoru (0-1) |
| **param_version** | VARCHAR(64) | YES | NULL | Parametre versiyonu |
| **pat_score** | NUMERIC(6,3) | YES | NULL | Pattern evidence skoru |
| **sent_score** | NUMERIC(6,3) | YES | NULL | Sentiment evidence skoru |
| **visual_bullish** | BOOLEAN | YES | NULL | Visual YOLO bullish var mı? |
| **visual_bearish** | BOOLEAN | YES | NULL | Visual YOLO bearish var mı? |
| **created_at** | DATETIME | NO | NOW() | Kayıt zamanı |

**Index'ler:**
- PRIMARY KEY: `id`
- INDEX: `symbol`, `ts_pred`, `horizon`, `stock_id`
- COMPOSITE: `(symbol, ts_pred)`, `(stock_id, ts_pred)`, `(horizon, ts_pred)`

**Foreign Keys:**
- `stock_id` → `stocks.id`

**Veri Akışı:**
```
pattern_detector.py analyze_stock()
  ↓
ML tahmin üretilir (1d,3d,7d,14d,30d)
  ↓
predictions_log'a kaydedilir
  ↓
Sonra outcomes_log ile eşleştirilerek değerlendirilir
```

**Örnek Data:**
```sql
INSERT INTO predictions_log (symbol, horizon, ts_pred, price_now, pred_price, delta_pred, model, confidence)
VALUES ('THYAO', '7d', NOW(), 120.50, 125.00, 0.0373, 'enhanced', 0.68);
```

---

### Table: **outcomes_log**

**Açıklama:** Tahminlerin gerçekleşen sonuçları (evaluation)

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **prediction_id** | INTEGER | NO | - | Foreign key → predictions_log.id |
| **ts_eval** | DATETIME | NO | NOW() | Değerlendirme zamanı |
| **price_eval** | NUMERIC(14,4) | YES | NULL | Değerlendirme anındaki fiyat |
| **delta_real** | NUMERIC(8,4) | YES | NULL | Gerçekleşen değişim % |
| **dir_hit** | BOOLEAN | YES | NULL | Yön doğru mu? (up/down) |
| **abs_err** | NUMERIC(8,4) | YES | NULL | Mutlak hata |
| **mape** | NUMERIC(8,4) | YES | NULL | Mean Absolute Percentage Error |
| **pnl** | NUMERIC(12,2) | YES | NULL | Profit/Loss (sanal) |
| **regime_vol20** | NUMERIC(8,4) | YES | NULL | 20 günlük volatilite |
| **regime_vol60** | NUMERIC(8,4) | YES | NULL | 60 günlük volatilite |
| **created_at** | DATETIME | NO | NOW() | Kayıt zamanı |

**Index'ler:**
- PRIMARY KEY: `id`
- INDEX: `prediction_id`, `ts_eval`

**Foreign Keys:**
- `prediction_id` → `predictions_log.id`

**İlişki:**
```
predictions_log (tahmin)
    ↓ (1:1)
outcomes_log (sonuç)
```

**Örnek Data:**
```sql
-- 7 gün sonra değerlendirme
INSERT INTO outcomes_log (prediction_id, ts_eval, price_eval, delta_real, dir_hit, abs_err, mape)
VALUES (123, NOW(), 124.50, 0.0332, TRUE, 0.0041, 0.33);
```

---

### Table: **metrics_daily**

**Açıklama:** Günlük toplu metrikler (sembol × horizon bazında)

**Sütunlar:**

| Sütun | Tip | Null? | Varsayılan | Açıklama |
|-------|-----|-------|------------|----------|
| **id** | INTEGER | NO | AUTO | Primary key |
| **date** | DATE | NO | - | Metrik günü |
| **symbol** | VARCHAR(20) | NO | - | Hisse kodu |
| **horizon** | VARCHAR(10) | NO | - | Tahmin ufku (1d/3d/7d/14d/30d) |
| **acc** | NUMERIC(6,4) | YES | NULL | Accuracy (yön doğruluğu) |
| **precision** | NUMERIC(6,4) | YES | NULL | Precision |
| **recall** | NUMERIC(6,4) | YES | NULL | Recall |
| **mae** | NUMERIC(8,4) | YES | NULL | Mean Absolute Error |
| **mape** | NUMERIC(8,4) | YES | NULL | Mean Absolute Percentage Error |
| **brier** | NUMERIC(8,4) | YES | NULL | Brier Score |
| **pnl** | NUMERIC(14,2) | YES | NULL | Profit/Loss (sanal) |
| **sharpe** | NUMERIC(6,3) | YES | NULL | Sharpe Ratio |
| **max_dd** | NUMERIC(6,3) | YES | NULL | Maximum Drawdown |
| **created_at** | DATETIME | NO | NOW() | Kayıt zamanı |

**Index'ler:**
- PRIMARY KEY: `id`
- UNIQUE: `(date, symbol, horizon)` - Her gün × sembol × horizon için 1 kayıt
- INDEX: `date`, `symbol`, `horizon`

**Örnek Data:**
```sql
INSERT INTO metrics_daily (date, symbol, horizon, acc, precision, mape)
VALUES ('2025-10-08', 'THYAO', '7d', 0.6800, 0.7200, 2.45);
```

**Aggregation:**
- Günlük olarak predictions_log ve outcomes_log'dan hesaplanır
- Calibration ve parameter optimization için kullanılır

---

## 🔗 İlişki Diyagramı (ERD)

```
┌─────────────┐
│    users    │
│  (id: PK)   │
└──────┬──────┘
       │
       │ 1:N
       ▼
┌─────────────┐         ┌─────────────┐
│  watchlist  │    N:1  │   stocks    │
│  (id: PK)   ├────────→│  (id: PK)   │
│user_id: FK  │         │symbol: UQ   │
│stock_id: FK │         └──────┬──────┘
└─────────────┘                │
                               │ 1:N
                               ▼
                        ┌─────────────┐
                        │stock_prices │
                        │  (id: PK)   │
                        │stock_id: FK │
                        │date: UQ     │
                        └─────────────┘

┌─────────────┐
│    users    │
└──────┬──────┘
       │ 1:N
       ▼
┌──────────────────┐
│simulation_sessions│
│    (id: PK)      │
│  user_id: FK     │
└────────┬─────────┘
         │ 1:N
         ├────────────────────┐
         │                    │
         ▼                    ▼
┌─────────────────┐   ┌────────────────────┐
│simulation_trades│   │portfolio_snapshots │
│    (id: PK)     │   │    (id: PK)        │
│ session_id: FK  │   │  session_id: FK    │
│  stock_id: FK   │   └────────────────────┘
└─────────────────┘

┌─────────────┐
│   stocks    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│predictions_log  │
│   (id: PK)      │
│ stock_id: FK    │
│ symbol, horizon │
└────────┬────────┘
         │ 1:1
         ▼
┌─────────────────┐
│ outcomes_log    │
│   (id: PK)      │
│prediction_id: FK│
└─────────────────┘

┌─────────────────┐
│ metrics_daily   │
│   (id: PK)      │
│ date, symbol,   │
│ horizon (UQ)    │
└─────────────────┘
```

---

## 🔍 Index Stratejisi

### Sık Kullanılan Sorgular ve Index'leri

**1. Kullanıcının watchlist'i:**
```sql
SELECT * FROM watchlist WHERE user_id = 4;
```
Index: `user_id` ✅

**2. Sembol için fiyat geçmişi:**
```sql
SELECT * FROM stock_prices 
WHERE stock_id = 1 
ORDER BY date DESC 
LIMIT 60;
```
Index: `(stock_id, date)` ✅

**3. Sembol için tahminler:**
```sql
SELECT * FROM predictions_log 
WHERE symbol = 'THYAO' 
AND horizon = '7d'
ORDER BY ts_pred DESC;
```
Index: `(symbol, ts_pred)` ✅

**4. Günlük metrikler:**
```sql
SELECT * FROM metrics_daily
WHERE date = '2025-10-08'
AND symbol = 'THYAO';
```
Index: `(date, symbol, horizon)` UNIQUE ✅

---

## 📊 Veri Boyutları ve Büyüme

### Mevcut Boyutlar (Tahmini)

| Tablo | Satır Sayısı | Boyut | Günlük Büyüme |
|-------|--------------|-------|---------------|
| users | ~100 | 50 KB | +1-5 |
| stocks | ~600 | 100 KB | +0-2 |
| stock_prices | ~220,000 | 50 MB | +600 |
| watchlist | ~1,000 | 200 KB | +5-20 |
| simulation_sessions | ~50 | 20 KB | +1-5 |
| simulation_trades | ~500 | 100 KB | +10-50 |
| portfolio_snapshots | ~1,000 | 150 KB | +20-100 |
| predictions_log | ~50,000 | 15 MB | +3,000 |
| outcomes_log | ~30,000 | 10 MB | +500 |
| metrics_daily | ~10,000 | 5 MB | +600 |
| **TOPLAM** | **~312,000** | **~80 MB** | **~5,000/gün** |

### Retention Policy

**stock_prices:**
- Son 2 yıl saklanır
- Eski veriler arşivlenir

**predictions_log / outcomes_log:**
- Son 6 ay saklanır
- Eski veriler metrics_daily'ye aggregate edilir

**metrics_daily:**
- Süresiz saklanır (küçük boyut)

---

## 🛠️ Database Migrations

### Migration Dizini
```
migrations/versions/
├── 20250821_add_user_role_last_login_ip.py
└── ... (diğer migration'lar)
```

### Yeni Migration Oluşturma
```bash
flask db migrate -m "migration açıklaması"
flask db upgrade
```

### Migration Geri Alma
```bash
flask db downgrade
```

---

## 🔧 Maintenance İşlemleri

### Vacuum (PostgreSQL)
```sql
-- Boş alanları temizle, index'leri optimize et
VACUUM ANALYZE stock_prices;
VACUUM ANALYZE predictions_log;
```

### Eski Veri Temizleme
```sql
-- 2 yıldan eski fiyat verilerini sil
DELETE FROM stock_prices 
WHERE date < CURRENT_DATE - INTERVAL '2 years';

-- 6 aydan eski tahmin loglarını sil
DELETE FROM predictions_log 
WHERE ts_pred < CURRENT_TIMESTAMP - INTERVAL '6 months';
```

### Index Rebuild
```sql
REINDEX TABLE stock_prices;
REINDEX TABLE predictions_log;
```

---

## 📋 Tablo Detayları (SQL CREATE)

### users
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    username VARCHAR(80) UNIQUE,
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    avatar_url VARCHAR(500),
    provider VARCHAR(20) DEFAULT 'email' NOT NULL,
    provider_id VARCHAR(255),
    email_verified BOOLEAN DEFAULT FALSE NOT NULL,
    email_verification_token VARCHAR(255),
    email_verification_sent_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    is_premium BOOLEAN DEFAULT FALSE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    last_login TIMESTAMP,
    role VARCHAR(20) DEFAULT 'user' NOT NULL,
    last_login_ip VARCHAR(45),
    timezone VARCHAR(50) DEFAULT 'Europe/Istanbul' NOT NULL,
    language VARCHAR(5) DEFAULT 'tr' NOT NULL,
    email_notifications BOOLEAN DEFAULT TRUE NOT NULL,
    push_notifications BOOLEAN DEFAULT TRUE NOT NULL
);

CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_user_role ON users(role);
CREATE INDEX idx_user_last_login ON users(last_login);
```

---

### stocks
```sql
CREATE TABLE stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT TRUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_stock_symbol ON stocks(symbol);
```

---

### stock_prices
```sql
CREATE TABLE stock_prices (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open_price NUMERIC(10,4) NOT NULL,
    high_price NUMERIC(10,4) NOT NULL,
    low_price NUMERIC(10,4) NOT NULL,
    close_price NUMERIC(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT unique_stock_date UNIQUE (stock_id, date)
);

CREATE INDEX idx_stock_date ON stock_prices(stock_id, date);
CREATE INDEX idx_date ON stock_prices(date);
```

---

### watchlist
```sql
CREATE TABLE watchlist (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    notes TEXT,
    alert_enabled BOOLEAN DEFAULT TRUE NOT NULL,
    alert_threshold_buy NUMERIC(10,4),
    alert_threshold_sell NUMERIC(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT unique_user_stock UNIQUE (user_id, stock_id)
);

CREATE INDEX idx_watchlist_user ON watchlist(user_id);
CREATE INDEX idx_watchlist_stock ON watchlist(stock_id);
```

---

### predictions_log
```sql
CREATE TABLE predictions_log (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    symbol VARCHAR(20) NOT NULL,
    horizon VARCHAR(10) NOT NULL,
    ts_pred TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    price_now NUMERIC(14,4),
    pred_price NUMERIC(14,4),
    delta_pred NUMERIC(8,4),
    model VARCHAR(12),
    unified_best VARCHAR(12),
    confidence NUMERIC(4,2),
    param_version VARCHAR(64),
    pat_score NUMERIC(6,3),
    sent_score NUMERIC(6,3),
    visual_bullish BOOLEAN,
    visual_bearish BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_pred_symbol_time ON predictions_log(symbol, ts_pred);
CREATE INDEX idx_pred_stock_time ON predictions_log(stock_id, ts_pred);
CREATE INDEX idx_pred_horizon_time ON predictions_log(horizon, ts_pred);
```

---

### outcomes_log
```sql
CREATE TABLE outcomes_log (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions_log(id),
    ts_eval TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    price_eval NUMERIC(14,4),
    delta_real NUMERIC(8,4),
    dir_hit BOOLEAN,
    abs_err NUMERIC(8,4),
    mape NUMERIC(8,4),
    pnl NUMERIC(12,2),
    regime_vol20 NUMERIC(8,4),
    regime_vol60 NUMERIC(8,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE INDEX idx_outcome_prediction ON outcomes_log(prediction_id);
CREATE INDEX idx_outcome_eval_time ON outcomes_log(ts_eval);
```

---

### metrics_daily
```sql
CREATE TABLE metrics_daily (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    horizon VARCHAR(10) NOT NULL,
    acc NUMERIC(6,4),
    precision NUMERIC(6,4),
    recall NUMERIC(6,4),
    mae NUMERIC(8,4),
    mape NUMERIC(8,4),
    brier NUMERIC(8,4),
    pnl NUMERIC(14,2),
    sharpe NUMERIC(6,3),
    max_dd NUMERIC(6,3),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    CONSTRAINT unique_metrics_day_sym_hor UNIQUE (date, symbol, horizon)
);

CREATE INDEX idx_metrics_date ON metrics_daily(date);
CREATE INDEX idx_metrics_symbol ON metrics_daily(symbol);
CREATE INDEX idx_metrics_horizon ON metrics_daily(horizon);
```

---

## 🔑 Constraint'ler Özeti

| Tablo | Constraint | Tip | Açıklama |
|-------|------------|-----|----------|
| users | email | UNIQUE | Email benzersiz olmalı |
| users | username | UNIQUE | Username benzersiz olmalı |
| stocks | symbol | UNIQUE | Sembol benzersiz olmalı |
| stock_prices | (stock_id, date) | UNIQUE | Her sembol için her gün 1 kayıt |
| watchlist | (user_id, stock_id) | UNIQUE | Kullanıcı aynı hisseyi 1 kez ekler |
| metrics_daily | (date, symbol, horizon) | UNIQUE | Her gün × sembol × horizon için 1 kayıt |

---

## 📈 Örnek Sorgular

### Kullanıcının watchlist'indeki hisseler için güncel fiyatlar
```sql
SELECT 
    w.id,
    s.symbol,
    s.name,
    sp.close_price,
    sp.date
FROM watchlist w
JOIN stocks s ON w.stock_id = s.id
LEFT JOIN LATERAL (
    SELECT close_price, date
    FROM stock_prices
    WHERE stock_id = s.id
    ORDER BY date DESC
    LIMIT 1
) sp ON TRUE
WHERE w.user_id = 4
ORDER BY s.symbol;
```

---

### Son 7 gündeki tahmin performansı
```sql
SELECT 
    pl.symbol,
    pl.horizon,
    AVG(CASE WHEN ol.dir_hit THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(ol.abs_err) as mae,
    AVG(ol.mape) as mape,
    COUNT(*) as count
FROM predictions_log pl
JOIN outcomes_log ol ON pl.id = ol.prediction_id
WHERE pl.ts_pred >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY pl.symbol, pl.horizon
ORDER BY accuracy DESC;
```

---

### En çok takip edilen 10 hisse
```sql
SELECT 
    s.symbol,
    s.name,
    COUNT(w.id) as follower_count
FROM stocks s
LEFT JOIN watchlist w ON s.id = w.stock_id
GROUP BY s.id, s.symbol, s.name
ORDER BY follower_count DESC
LIMIT 10;
```

---

## 🔐 Security ve Permissions

### User Roles
- **user**: Normal kullanıcı (watchlist, predictions görüntüleme)
- **admin**: Yönetici (automation kontrolü, tüm veriler)

### Row Level Security (RLS)
```sql
-- Watchlist: Kullanıcı sadece kendi kayıtlarını görebilir
CREATE POLICY watchlist_user_policy ON watchlist
FOR ALL
TO public
USING (user_id = current_setting('app.user_id')::INTEGER);
```

---

## 📊 Backup Stratejisi

### Günlük Backup
```bash
# PostgreSQL dump
pg_dump -U postgres bist_pattern > backup_$(date +%Y%m%d).sql

# Sadece schema
pg_dump -U postgres --schema-only bist_pattern > schema.sql

# Sadece data
pg_dump -U postgres --data-only bist_pattern > data.sql
```

### Restore
```bash
psql -U postgres bist_pattern < backup_20251008.sql
```

---

## 🔄 Migration History

### Mevcut Migration'lar
```bash
flask db current    # Mevcut versiyon
flask db history    # Tüm geçmiş
flask db upgrade    # En son versiyona yükselt
flask db downgrade  # Bir önceki versiyona geri dön
```

---

## 📝 Notlar

### NUMERIC vs FLOAT
- **NUMERIC(10,4)**: Kesin ondalık (fiyatlar için)
- **FLOAT**: Yaklaşık ondalık (hesaplamalar için)

### CASCADE DELETE
- Watchlist: User silinince watchlist'i de silinir
- Stock Prices: Stock silinince fiyatları da silinir

### Auto Update
- `updated_at`: Her UPDATE'te otomatik güncellenir (onupdate trigger)

---

**Son Güncelleme:** 08 Ekim 2025  
**DB Engine:** PostgreSQL 14+  
**ORM:** SQLAlchemy 2.x  
**Migration Tool:** Flask-Migrate (Alembic)

