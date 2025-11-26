# Admin Dashboard Alım-Satım Simülasyonu Analizi
## Borsacı Gözüyle ve Kod Gözüyle Detaylı İnceleme

---

## 📊 GENEL BAKIŞ

**Simülasyon Türü:** Forward Simulation (Gerçek Zamanlı İşlem Simülasyonu)
**Dosya:** `bist_pattern/simulation/forward_engine.py`
**API Endpoint:** `/api/internal/simulation/forward-start`

---

## 🎯 BORSACI GÖZÜYLE ANALİZ

### 1. STRATEJİ ÖZETİ

**Yaklaşım:** Momentum/Confidence-Based Portfolio Yönetimi
- En yüksek confidence'a sahip sinyalleri seçer
- Confidence'a göre ağırlıklı pozisyon alır
- Stop-loss ve confidence düşüşü ile çıkış yapar
- Pozisyon boşalınca yeni sinyallerle doldurur

**Avantajlar:**
- ✅ Basit ve anlaşılır mantık
- ✅ Risk yönetimi var (stop-loss)
- ✅ Dinamik portföy rotasyonu

**Dezavantajlar:**
- ⚠️ Confidence'a aşırı bağımlılık
- ⚠️ Trend takibi yok (sadece sinyal bazlı)
- ⚠️ Volatilite kontrolü yok
- ⚠️ Sektör çeşitlendirmesi yok

---

### 2. ALIM KARARLARI (Entry Logic)

#### 2.1. İlk Alımlar (start_simulation)

**Mantık:**
```python
# En yüksek confidence'lı topN sinyal seçilir
best_signals = _get_best_signals(horizon, topN)

# Confidence'a göre ağırlık hesaplanır
weight = signal['confidence'] / total_confidence
allocation = initial_capital * weight
shares = int(allocation / price)
```

**Borsacı Değerlendirmesi:**

✅ **DOĞRU:**
- Confidence'a göre ağırlıklandırma mantıklı
- TopN ile portföy çeşitlendirmesi var
- Commission hesabı doğru

⚠️ **SORUNLAR:**

1. **Confidence Normalizasyonu Eksik:**
   - Eğer tüm sinyaller düşük confidence'lıysa (örn: 0.1, 0.15, 0.2), yine de %100 sermaye kullanılır
   - Düşük confidence'lı sinyallerde daha az pozisyon alınmalı

2. **Minimum Confidence Threshold Yok:**
   - Örn: confidence < 0.3 ise hiç alım yapma
   - Şu an en düşük confidence'lı sinyal bile alınabilir

3. **Fiyat Validasyonu Yetersiz:**
   ```python
   if not price or price <= 0:
       continue  # Sadece 0 veya negatif kontrolü
   ```
   - Çok düşük fiyatlar (penny stocks) kontrol edilmiyor
   - Likidite kontrolü yok

4. **Sektör Çeşitlendirmesi Yok:**
   - Tüm topN sinyal aynı sektörden olabilir
   - Sektör riski konsantrasyonu oluşabilir

---

#### 2.2. Rotasyon Alımları (check_and_trade)

**Mantık:**
```python
# Boş pozisyon slotları için yeni sinyaller
if len(positions) < params['topN']:
    available_slots = params['topN'] - len(positions)
    new_candidates = _get_best_signals(horizon, params['topN'], exclude_symbols=held_symbols)
```

**Borsacı Değerlendirmesi:**

✅ **DOĞRU:**
- Mevcut pozisyonları exclude etme mantıklı
- Boş slotları doldurma stratejisi doğru

⚠️ **SORUNLAR:**

1. **Cash Kullanımı:**
   ```python
   allocation = cash * weight  # Sadece kalan cash kullanılıyor
   ```
   - Eğer cash azaldıysa, yeni pozisyonlar çok küçük olabilir
   - Portföy dengesizliği oluşabilir

2. **Timing Problemi:**
   - Her cycle'da (10-15 kez/gün) kontrol ediliyor
   - Aynı gün içinde çok sık alım-satım yapılabilir
   - Transaction cost artabilir

---

### 3. SATIŞ KARARLARI (Exit Logic)

#### 3.1. Stop-Loss

**Mantık:**
```python
pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
if pnl_pct <= -params['stop_loss_pct']:
    should_sell = True
    sell_reason = 'stop_loss'
```

**Borsacı Değerlendirmesi:**

✅ **DOĞRU:**
- Stop-loss mekanizması var
- Entry price'a göre hesaplama doğru

⚠️ **SORUNLAR:**

1. **Trailing Stop-Loss Yok:**
   - Sadece entry price'a göre sabit stop-loss
   - Eğer pozisyon kârdayken geri düşerse, kârı koruyamaz
   - Örnek: Entry=100, Stop=97, Fiyat=110'a çıktı, sonra 98'e düştü → Stop-loss tetiklenmez ama kâr kaybolur

2. **Volatilite-Aware Stop-Loss Yok:**
   - Tüm semboller için aynı stop-loss yüzdesi
   - Volatil semboller için stop-loss çok sıkı olabilir
   - Düşük volatil semboller için stop-loss çok gevşek olabilir

3. **Gap Down Kontrolü Yok:**
   - Eğer fiyat gap down ile stop-loss'un altına düşerse, stop-loss fiyatında satış yapılır
   - Gerçek piyasada gap down durumunda daha düşük fiyattan satış yapılabilir

---

#### 3.2. Sell Signal (Sinyal Bazlı Çıkış)

**Mantık:**
```python
delta = float(recent.delta_pred or 0.0)
action = 'buy' if delta > 0 else 'sell' if delta < 0 else 'hold'

if action == 'sell':
    should_sell = True
    sell_reason = 'sell_signal'
```

**Borsacı Değerlendirmesi:**

✅ **DOĞRU:**
- Model sinyali değiştiğinde çıkış yapma mantıklı
- Trend değişimini yakalama potansiyeli var

⚠️ **SORUNLAR:**

1. **Sell Signal Threshold Yok:**
   - `delta < 0` ise hemen satış
   - Çok küçük negatif delta'lar için de satış yapılabilir (noise)
   - Minimum threshold olmalı (örn: delta < -0.01)

2. **Confidence Kontrolü Yok:**
   - Sell signal'ın confidence'ı düşükse, güvenilir olmayabilir
   - Sell signal confidence > 0.5 gibi bir threshold olmalı

3. **Kârda Satış Kontrolü Yok:**
   - Eğer pozisyon kârdayken sell signal gelirse, hemen satış
   - Kâr koruma mekanizması yok (örn: kâr > %5 ise sell signal'ı görmezden gel)

---

#### 3.3. Confidence Drop (Güven Düşüşü)

**Mantık:**
```python
current_conf = float(recent.confidence or 0.0)
if current_conf < pos['entry_confidence'] * (1 - params['relative_drop_threshold']):
    should_sell = True
    sell_reason = 'confidence_drop'
```

**Örnek:** Entry confidence=0.8, threshold=0.20
- Exit condition: current_conf < 0.8 * (1 - 0.20) = 0.8 * 0.8 = 0.64
- Yani confidence 0.8'den 0.64'e düşerse satış

**Borsacı Değerlendirmesi:**

✅ **DOĞRU:**
- Model güveni düştüğünde çıkış mantıklı
- Relative drop threshold ile esnek kontrol

⚠️ **SORUNLAR:**

1. **Absolute Threshold Yok:**
   - Eğer entry confidence çok düşükse (örn: 0.3), %20 düşüş = 0.24
   - 0.24 confidence ile pozisyon tutmak mantıksız
   - Minimum confidence threshold olmalı (örn: current_conf < 0.4 ise sat)

2. **Kârda Confidence Drop:**
   - Eğer pozisyon kârdayken confidence düşerse, hemen satış
   - Kâr koruma mekanizması yok

3. **Time-Based Confidence Decay Yok:**
   - Eski sinyallerin confidence'ı zamanla düşebilir
   - Pozisyon yaşına göre confidence decay olmalı

---

### 4. POZİSYON YÖNETİMİ

#### 4.1. Position Sizing

**Mantık:**
```python
weight = signal['confidence'] / total_confidence
allocation = initial_capital * weight  # veya cash * weight
shares = int(allocation / price)
```

**Borsacı Değerlendirmesi:**

✅ **DOĞRU:**
- Confidence'a göre ağırlıklandırma mantıklı
- Integer share hesaplama doğru

⚠️ **SORUNLAR:**

1. **Minimum Position Size Yok:**
   - Çok küçük pozisyonlar oluşabilir (örn: 1 share)
   - Minimum position size olmalı (örn: min 1000 TL)

2. **Maximum Position Size Yok:**
   - Tek bir pozisyona çok fazla sermaye ayrılabilir
   - Maximum position size olmalı (örn: max %30 sermaye)

3. **Cash Reserve Yok:**
   - Tüm cash kullanılıyor
   - Acil durumlar için cash reserve olmalı (örn: %10)

---

#### 4.2. Portfolio Rebalancing

**Mantık:**
- Her cycle'da (10-15 kez/gün) pozisyonlar kontrol edilir
- Boş slotlar doldurulur
- Stop-loss/sell signal/confidence drop ile çıkış yapılır

**Borsacı Değerlendirmesi:**

⚠️ **SORUNLAR:**

1. **Rebalancing Frequency Çok Yüksek:**
   - Her cycle'da kontrol = 10-15 kez/gün
   - Çok sık rebalancing transaction cost'u artırır
   - Günlük 1-2 kez yeterli olabilir

2. **Partial Exit Yok:**
   - Ya tam pozisyon tutulur ya da tamamen satılır
   - Kısmi çıkış (örn: %50) yok

3. **Position Aging Yok:**
   - Pozisyon yaşına göre farklı strateji yok
   - Eski pozisyonlar için farklı stop-loss olabilir

---

### 5. RİSK YÖNETİMİ

**Mevcut Risk Kontrolleri:**
- ✅ Stop-loss
- ✅ Confidence drop
- ✅ Sell signal

**Eksik Risk Kontrolleri:**
- ❌ Maximum drawdown kontrolü
- ❌ Portfolio-level stop-loss
- ❌ Volatilite kontrolü
- ❌ Sektör konsantrasyon limiti
- ❌ Correlation kontrolü
- ❌ Leverage kontrolü (şu an yok ama gelecekte eklenebilir)

---

## 💻 KOD GÖZÜYLE ANALİZ

### 1. VERİ KAYNAKLARI

#### 1.1. Signal Kaynağı (_get_best_signals)

**Kod:**
```python
cutoff = datetime.utcnow() - timedelta(hours=2)
recent_preds = PredictionsLog.query.filter(
    PredictionsLog.horizon.in_(eligible_horizons),
    PredictionsLog.ts_pred >= cutoff
).all()
```

**Kod Değerlendirmesi:**

✅ **DOĞRU:**
- Son 2 saatteki sinyalleri kullanma mantıklı
- Horizon filtering doğru

⚠️ **SORUNLAR:**

1. **Time Window Sabit:**
   - 2 saat window her zaman uygun olmayabilir
   - Horizon'a göre dinamik olmalı (örn: 1d için 1 saat, 30d için 24 saat)

2. **Duplicate Signal Handling:**
   ```python
   if pred.symbol not in symbol_best or conf > symbol_best[pred.symbol]['confidence']:
       symbol_best[pred.symbol] = {...}
   ```
   - Aynı sembol için en yüksek confidence alınıyor
   - Ama aynı sembol farklı horizon'larda birden fazla sinyal olabilir
   - Bu durumda hangi horizon'un sinyali alınmalı? (şu an en yüksek confidence)

3. **Signal Freshness Kontrolü Yok:**
   - 2 saat içindeki tüm sinyaller eşit ağırlıkta
   - Daha yeni sinyaller daha fazla ağırlık almalı

---

#### 1.2. Fiyat Kaynağı (_get_current_price)

**Kod:**
```python
sp = StockPrice.query.filter_by(stock_id=stock.id).order_by(StockPrice.date.desc()).first()
if sp and sp.close_price and sp.close_price > 0:
    return float(sp.close_price)
```

**Kod Değerlendirmesi:**

⚠️ **SORUNLAR:**

1. **Close Price Kullanımı:**
   - Sadece close price kullanılıyor
   - Gerçek piyasada alım-satım için bid/ask spread olmalı
   - Simülasyonda spread yok, bu gerçekçi değil

2. **Price Staleness Kontrolü Yok:**
   - Eğer son fiyat 1 gün önceyse, güncel değil
   - Price freshness kontrolü olmalı (örn: son 1 saat içinde)

3. **Market Hours Kontrolü Yok:**
   - Borsa kapalıyken eski fiyat kullanılıyor
   - Borsa açık/kapalı kontrolü olmalı

4. **Gap Handling Yok:**
   - Eğer fiyat gap up/down ile açıldıysa, close price yanıltıcı olabilir
   - Gap kontrolü olmalı

---

### 2. HESAPLAMA HATALARI

#### 2.1. Commission Hesaplama

**Kod:**
```python
cost = shares * price
comm = cost * params['commission']
total_cost = cost + comm
```

**Kod Değerlendirmesi:**

✅ **DOĞRU:**
- Commission hesaplama doğru
- Alım ve satımda commission uygulanıyor

⚠️ **SORUNLAR:**

1. **Commission Minimum Yok:**
   - Çok küçük işlemlerde commission çok düşük olabilir
   - Gerçek piyasada minimum commission olabilir (örn: min 5 TL)

2. **Commission Asymmetric:**
   - Alım ve satım commission'ı aynı
   - Gerçek piyasada farklı olabilir

---

#### 2.2. Equity Hesaplama

**Kod:**
```python
position_value = sum(
    p['shares'] * (_get_current_price(p['symbol']) or p['entry_price'])
    for p in positions
)
current_equity = cash + position_value
```

**Kod Değerlendirmesi:**

✅ **DOĞRU:**
- Equity hesaplama mantıklı
- Fallback olarak entry_price kullanılıyor

⚠️ **SORUNLAR:**

1. **Slippage Yok:**
   - Gerçek piyasada büyük işlemlerde slippage olur
   - Simülasyonda slippage yok

2. **Market Impact Yok:**
   - Büyük işlemler piyasayı etkileyebilir
   - Simülasyonda market impact yok

---

### 3. STATE YÖNETİMİ

#### 3.1. State File

**Kod:**
```python
STATE_FILE = 'logs/simulation_state.json'

def _read_state() -> Optional[Dict]:
    with open(STATE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def _write_state(state: Dict) -> None:
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
```

**Kod Değerlendirmesi:**

⚠️ **SORUNLAR:**

1. **File Locking Yok:**
   - Eğer birden fazla process aynı anda state'i okur/yazarsa, race condition olabilir
   - File locking (fcntl) kullanılmalı

2. **Atomic Write Yok:**
   - State yazma sırasında crash olursa, state corrupt olabilir
   - Atomic write (temp file + rename) kullanılmalı

3. **Concurrent Access:**
   - `check_and_trade()` automation cycle'da çağrılıyor
   - Aynı anda admin dashboard'dan status okunabilir
   - Race condition riski var

---

#### 3.2. State Persistence

**Kod:**
```python
state = {
    'active': True,
    'start_time': datetime.utcnow().isoformat(),
    'portfolio': {
        'cash': cash,
        'equity': initial_equity,
        'positions': positions
    },
    'trades': trades,
    'daily_snapshots': [...]
}
```

**Kod Değerlendirmesi:**

✅ **DOĞRU:**
- State structure iyi tasarlanmış
- Daily snapshots var

⚠️ **SORUNLAR:**

1. **State File Size:**
   - Tüm trades state'te tutuluyor
   - Uzun simülasyonlarda state file çok büyük olabilir
   - Trades'i ayrı bir dosyaya taşımalı veya limit koymalı

2. **State Backup Yok:**
   - State file corrupt olursa, simülasyon kaybolur
   - Periodic backup olmalı

---

### 4. HATA YÖNETİMİ

#### 4.1. Exception Handling

**Kod:**
```python
try:
    price = _get_current_price(symbol)
    if not price or price <= 0:
        logger.warning(f"No price for {symbol}, skipping")
        continue
except Exception as e:
    logger.warning(f"Error getting price for {symbol}: {e}")
    return None
```

**Kod Değerlendirmesi:**

✅ **DOĞRU:**
- Exception handling var
- Logging yapılıyor

⚠️ **SORUNLAR:**

1. **Silent Failures:**
   - Bazı hatalar sadece log'lanıyor, simülasyon devam ediyor
   - Kritik hatalarda simülasyon durdurulmalı mı?

2. **Partial Failure Handling:**
   - Bir pozisyon için fiyat alınamazsa, sadece o pozisyon skip ediliyor
   - Tüm pozisyonlar için fiyat alınamazsa ne olur?

---

### 5. PERFORMANS

#### 5.1. Database Queries

**Kod:**
```python
recent_preds = PredictionsLog.query.filter(
    PredictionsLog.horizon.in_(eligible_horizons),
    PredictionsLog.ts_pred >= cutoff
).all()

sp = StockPrice.query.filter_by(stock_id=stock.id).order_by(StockPrice.date.desc()).first()
```

**Kod Değerlendirmesi:**

⚠️ **SORUNLAR:**

1. **N+1 Query Problem:**
   - Her pozisyon için ayrı ayrı fiyat sorgusu yapılıyor
   - Batch query yapılmalı

2. **Index Kontrolü Yok:**
   - `PredictionsLog.ts_pred` ve `StockPrice.date` index'li mi?
   - Index kontrolü yapılmalı

3. **Query Optimization:**
   - Her cycle'da aynı sorgular tekrar yapılıyor
   - Cache mekanizması olabilir

---

## 🔴 KRİTİK MANTIK HATALARI

### 1. **Confidence Normalizasyonu Eksik**

**Problem:**
```python
total_confidence = sum(s['confidence'] for s in best_signals)
weight = signal['confidence'] / total_confidence
```

Eğer tüm sinyaller düşük confidence'lıysa (örn: [0.1, 0.15, 0.2]), yine de %100 sermaye kullanılır.

**Çözüm:**
```python
# Minimum confidence threshold
min_confidence = 0.3
filtered_signals = [s for s in best_signals if s['confidence'] >= min_confidence]

# Veya confidence'a göre sermaye kullanımı
total_confidence = sum(s['confidence'] for s in best_signals)
if total_confidence < min_total_confidence:
    # Daha az sermaye kullan
    capital_usage = min(1.0, total_confidence / min_total_confidence)
```

---

### 2. **Sell Signal Threshold Yok**

**Problem:**
```python
delta = float(recent.delta_pred or 0.0)
if delta < 0:
    should_sell = True
```

Çok küçük negatif delta'lar için de satış yapılabilir (noise).

**Çözüm:**
```python
SELL_SIGNAL_THRESHOLD = -0.01  # %1'den fazla düşüş
if delta < SELL_SIGNAL_THRESHOLD:
    should_sell = True
```

---

### 3. **Trailing Stop-Loss Yok**

**Problem:**
Sadece entry price'a göre sabit stop-loss var. Eğer pozisyon kârdayken geri düşerse, kârı koruyamaz.

**Çözüm:**
```python
# Trailing stop-loss
if current_price > pos['entry_price']:
    # Kârdayken, en yüksek fiyatı takip et
    if 'highest_price' not in pos:
        pos['highest_price'] = current_price
    else:
        pos['highest_price'] = max(pos['highest_price'], current_price)
    
    # Trailing stop: highest_price'ın %X altına düşerse sat
    trailing_stop_price = pos['highest_price'] * (1 - params['trailing_stop_pct'])
    if current_price <= trailing_stop_price:
        should_sell = True
        sell_reason = 'trailing_stop'
```

---

### 4. **File Locking Yok**

**Problem:**
State file'a concurrent access olabilir, race condition riski var.

**Çözüm:**
```python
import fcntl

def _write_state(state: Dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(state, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

---

### 5. **Price Staleness Kontrolü Yok**

**Problem:**
Eski fiyatlar kullanılabilir, gerçekçi değil.

**Çözüm:**
```python
def _get_current_price(symbol: str) -> Optional[float]:
    sp = StockPrice.query.filter_by(stock_id=stock.id).order_by(StockPrice.date.desc()).first()
    if sp and sp.close_price and sp.close_price > 0:
        # Price freshness kontrolü
        price_age = (datetime.utcnow() - sp.date).total_seconds() / 3600  # saat cinsinden
        if price_age > 4:  # 4 saatten eski fiyat kullanma
            logger.warning(f"Price too stale for {symbol}: {price_age:.1f} hours old")
            return None
        return float(sp.close_price)
    return None
```

---

## ✅ ÖNERİLER

### 1. **Kısa Vadeli İyileştirmeler**

1. **Minimum Confidence Threshold Ekle**
2. **Sell Signal Threshold Ekle**
3. **Price Staleness Kontrolü Ekle**
4. **File Locking Ekle**
5. **Minimum Position Size Ekle**

### 2. **Orta Vadeli İyileştirmeler**

1. **Trailing Stop-Loss Ekle**
2. **Volatilite-Aware Stop-Loss**
3. **Sektör Çeşitlendirmesi**
4. **Rebalancing Frequency Optimizasyonu**
5. **Partial Exit Mekanizması**

### 3. **Uzun Vadeli İyileştirmeler**

1. **Portfolio-Level Risk Kontrolleri**
2. **Correlation-Based Position Sizing**
3. **Market Regime Detection**
4. **Dynamic Confidence Weighting**
5. **Backtesting Framework**

---

## 📝 SONUÇ

**Genel Değerlendirme:**
- ✅ Temel mantık doğru ve çalışıyor
- ⚠️ Birçok iyileştirme fırsatı var
- 🔴 Birkaç kritik mantık hatası var (confidence normalization, sell signal threshold, trailing stop)

**Öncelik Sırası:**
1. **KRİTİK:** File locking, price staleness, sell signal threshold
2. **YÜKSEK:** Minimum confidence threshold, trailing stop-loss
3. **ORTA:** Sektör çeşitlendirmesi, volatilite-aware stop-loss
4. **DÜŞÜK:** Portfolio-level risk, correlation-based sizing

**Borsacı Gözüyle:** Basit ama etkili bir strateji. Risk yönetimi iyileştirilebilir.

**Kod Gözüyle:** Temiz kod, ama concurrent access ve error handling iyileştirilebilir.

