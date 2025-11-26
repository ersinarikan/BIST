# Simülasyon Mantık Analizi
## Model Performansını Doğrulama Açısından İnceleme

---

## 🎯 AMAÇ

**Simülasyonun amacı:** Geliştirdiğimiz uygulamanın (modelin) önerileri doğrultusunda hareket edildiğinde ne kadar kazançlı çıkılacağını test etmek - yani **modelin gerçekten işe yarayıp yaramadığını doğrulamak**.

---

## 🔍 MEVCUT MANTIK İNCELEMESİ

### 1. MODEL ÖNERİLERİNİ KULLANMA

#### ✅ DOĞRU YAPILANLAR:

**1.1. Alım Kararları (Entry):**
```python
# Model önerisi: delta_pred > 0 → buy signal
delta = float(pred.delta_pred or 0.0)
action = 'buy' if delta > 0 else 'sell' if delta < 0 else 'hold'

# Confidence'a göre ağırlıklandırma
weight = signal['confidence'] / total_confidence
allocation = initial_capital * weight
```

**Değerlendirme:**
- ✅ Modelin buy sinyallerini kullanıyor
- ✅ Confidence'a göre pozisyon büyüklüğü belirleniyor
- ✅ Modelin önerdiği sembolleri seçiyor

**1.2. Satış Kararları (Exit) - Model Sinyali:**
```python
# Model önerisi: delta_pred < 0 → sell signal
if action == 'sell':
    should_sell = True
    sell_reason = 'sell_signal'
```

**Değerlendirme:**
- ✅ Modelin sell sinyallerini kullanıyor
- ✅ Model "sat" dediğinde satış yapılıyor

---

### 2. ⚠️ SORUNLU MANTIKLAR

#### 2.1. STOP-LOSS MEKANİZMASI

**Mevcut Mantık:**
```python
# Stop-loss check
pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
if pnl_pct <= -params['stop_loss_pct']:
    should_sell = True
    sell_reason = 'stop_loss'
```

**Problem:**

**Model ne diyor?**
- Model: "14 gün sonra bu hisse %5 artacak" (horizon=14d, delta_pred=0.05)
- Modelin önerisi: 14 gün tut, sonra sat

**Simülasyon ne yapıyor?**
- 3. günde fiyat %3 düştü → Stop-loss tetiklendi → Satış yapıldı
- Modelin önerdiği süre (14 gün) dolmadan çıkış yapıldı

**Sonuç:**
- ❌ Modelin önerisi doğru olsa bile, simülasyon zarar gösterir
- ❌ Model performansı yanlış ölçülür
- ❌ Model "14 gün tut" dedi ama simülasyon 3. günde sattı

**Örnek Senaryo:**
```
Gün 1: Model "14d sonra %5 artış" → Alım yapıldı (100 TL)
Gün 3: Fiyat 97 TL'ye düştü (%3 düşüş) → Stop-loss → Satış (97 TL)
Gün 14: Fiyat 105 TL'ye çıktı (%5 artış) → Model haklıydı ama simülasyon zarar gösterdi
```

**Bu mantık model performansını ölçmüyor, kendi risk yönetimi performansını ölçüyor!**

---

#### 2.2. CONFIDENCE DROP MEKANİZMASI

**Mevcut Mantık:**
```python
# Check for relative confidence drop
current_conf = float(recent.confidence or 0.0)
if current_conf < pos['entry_confidence'] * (1 - params['relative_drop_threshold']):
    should_sell = True
    sell_reason = 'confidence_drop'
```

**Problem:**

**Model ne diyor?**
- Model: "14 gün sonra %5 artış, confidence=0.8"
- Modelin önerisi: 14 gün tut, confidence=0.8

**Simülasyon ne yapıyor?**
- 5. günde yeni bir prediction geldi, confidence=0.6 (%20 düşüş)
- Confidence drop → Satış yapıldı
- Modelin önerdiği süre (14 gün) dolmadan çıkış yapıldı

**Sonuç:**
- ❌ Modelin önerisi doğru olsa bile, simülasyon zarar gösterir
- ❌ Model performansı yanlış ölçülür
- ❌ Model "14 gün tut, confidence=0.8" dedi ama simülasyon 5. günde sattı

**Örnek Senaryo:**
```
Gün 1: Model "14d sonra %5 artış, conf=0.8" → Alım (100 TL)
Gün 5: Yeni prediction "conf=0.6" → Confidence drop → Satış (98 TL)
Gün 14: Fiyat 105 TL → Model haklıydı ama simülasyon zarar gösterdi
```

**Bu mantık model performansını ölçmüyor, confidence tracking performansını ölçüyor!**

---

#### 2.3. HORIZON KULLANIMI

**Mevcut Mantık:**
```python
# Model horizon'ı sadece sinyal seçiminde kullanılıyor
eligible_horizons = _get_eligible_horizons(max_days)  # [1d, 3d, 7d, 14d]
# Ama pozisyon tutma süresi horizon'a göre değil, stop-loss/confidence drop'a göre
```

**Problem:**

**Model ne diyor?**
- Model: "14 gün sonra %5 artış" (horizon=14d)
- Modelin önerisi: **14 gün tut**, sonra sat

**Simülasyon ne yapıyor?**
- Horizon sadece sinyal seçiminde kullanılıyor
- Pozisyon tutma süresi horizon'a göre değil
- Stop-loss veya confidence drop ile erken çıkış yapılıyor

**Sonuç:**
- ❌ Modelin önerdiği süre (horizon) göz ardı ediliyor
- ❌ Model "14 gün tut" dedi ama simülasyon 3-5 günde çıkış yapıyor
- ❌ Model performansı yanlış ölçülür

---

### 3. 🔴 KRİTİK MANTIK HATASI

**Ana Sorun:** Simülasyon, modelin önerdiği stratejiyi değil, kendi risk yönetimi stratejisini test ediyor.

**Modelin Önerdiği Strateji:**
1. "X gün sonra Y% artış/azalış olacak" (horizon + delta_pred)
2. "Bu önerinin güvenilirliği Z%" (confidence)
3. **Öneri:** X gün tut, sonra sat

**Simülasyonun Yaptığı:**
1. Modelin buy sinyallerini kullanıyor ✅
2. Modelin sell sinyallerini kullanıyor ✅
3. **AMA:** Stop-loss ve confidence drop ile erken çıkış yapıyor ❌
4. **AMA:** Modelin önerdiği süre (horizon) göz ardı ediliyor ❌

**Sonuç:**
- Simülasyon sonuçları model performansını değil, **risk yönetimi performansını** ölçüyor
- Model doğru olsa bile, simülasyon zarar gösterebilir
- Model performansı yanlış ölçülür

---

## 💡 DOĞRU MANTIK NASIL OLMALI?

### Senaryo 1: Model Performansını Ölçmek İçin

**Mantık:**
1. Model "X gün sonra Y% artış" dedi → Alım yap
2. **X gün bekle** (modelin önerdiği süre)
3. X gün sonra sat (modelin önerdiği zaman)
4. Sonucu ölç: Model haklı mıydı?

**Kod:**
```python
# Model önerisi: horizon=14d, delta_pred=0.05
# Pozisyon tutma süresi: 14 gün
entry_time = datetime.utcnow()
target_exit_time = entry_time + timedelta(days=14)

# 14 gün sonra otomatik satış
if datetime.utcnow() >= target_exit_time:
    should_sell = True
    sell_reason = 'horizon_reached'  # Modelin önerdiği süre doldu
```

**Bu mantık model performansını ölçer!**

---

### Senaryo 2: Risk Yönetimi ile Model Performansını Birlikte Ölçmek İçin

**Mantık:**
1. Model "X gün sonra Y% artış" dedi → Alım yap
2. **AMA:** Stop-loss veya confidence drop varsa erken çıkış yap
3. **AMA:** Modelin önerdiği süre (X gün) dolmadan çıkış yapılırsa, bu **risk yönetimi kararı**, model performansı değil
4. Sonuçları ayır:
   - Model performansı: Horizon dolduğunda ne oldu?
   - Risk yönetimi performansı: Erken çıkışlar ne kadar etkili?

**Kod:**
```python
# Model önerisi: horizon=14d
target_exit_time = entry_time + timedelta(days=14)

# Risk yönetimi kontrolleri
if stop_loss_triggered:
    should_sell = True
    sell_reason = 'stop_loss'  # Risk yönetimi kararı
    # Model performansını ölçme: Bu trade'i model performansına dahil etme
    # Çünkü model "14 gün tut" dedi, ama 3. günde çıkış yapıldı

elif confidence_drop:
    should_sell = True
    sell_reason = 'confidence_drop'  # Risk yönetimi kararı
    # Model performansını ölçme: Bu trade'i model performansına dahil etme

elif datetime.utcnow() >= target_exit_time:
    should_sell = True
    sell_reason = 'horizon_reached'  # Model performansı
    # Bu trade'i model performansına dahil et
```

**Bu mantık hem model performansını hem risk yönetimi performansını ölçer!**

---

## 📊 MEVCUT SİMÜLASYON NE ÖLÇÜYOR?

### Ölçülen Metrikler:

1. **Toplam P&L:** Risk yönetimi + Model performansı karışık
2. **Hit Rate:** Risk yönetimi + Model performansı karışık
3. **Return %:** Risk yönetimi + Model performansı karışık

### Ölçülmeyen Metrikler:

1. **Model Performansı (Saf):** Modelin önerdiği süre dolduğunda ne oldu?
2. **Risk Yönetimi Performansı (Saf):** Erken çıkışlar ne kadar etkili?
3. **Model vs Risk Yönetimi:** Hangisi daha etkili?

---

## 🎯 ÖNERİLER

### 1. Model Performansını Ölçmek İçin (Saf Test)

**Değişiklik:**
- Stop-loss ve confidence drop'u **devre dışı bırak** (veya opsiyonel yap)
- Modelin önerdiği süre (horizon) dolana kadar pozisyon tut
- Horizon dolduğunda otomatik satış

**Kod:**
```python
# Model önerisi: horizon=14d
target_exit_time = entry_time + timedelta(days=horizon_days)

# Sadece horizon dolduğunda satış
if datetime.utcnow() >= target_exit_time:
    should_sell = True
    sell_reason = 'horizon_reached'
```

**Bu mantık model performansını saf olarak ölçer!**

---

### 2. Model + Risk Yönetimi Performansını Birlikte Ölçmek İçin

**Değişiklik:**
- Stop-loss ve confidence drop'u **ayrı bir metrik olarak** ölç
- Model performansını **ayrı bir metrik olarak** ölç
- Her iki metrik de ayrı ayrı raporlanmalı

**Kod:**
```python
# Trade sonuçlarını kategorize et
if sell_reason == 'stop_loss' or sell_reason == 'confidence_drop':
    # Risk yönetimi kararı
    trade_category = 'risk_management'
elif sell_reason == 'horizon_reached':
    # Model performansı
    trade_category = 'model_performance'
elif sell_reason == 'sell_signal':
    # Model sinyali (bu da model performansı)
    trade_category = 'model_performance'

# Ayrı metrikler
model_performance_pnl = sum(t['profit'] for t in trades if t['category'] == 'model_performance')
risk_management_pnl = sum(t['profit'] for t in trades if t['category'] == 'risk_management')
```

**Bu mantık hem model performansını hem risk yönetimi performansını ölçer!**

---

### 3. Hibrit Yaklaşım (Önerilen)

**Mantık:**
1. Model performansını ölçmek için: Horizon dolana kadar tut (stop-loss/confidence drop yok)
2. Risk yönetimi performansını ölçmek için: Stop-loss ve confidence drop aktif
3. **AMA:** Her iki metrik de ayrı ayrı raporlanmalı

**Kod:**
```python
# İki mod: "model_test" veya "hybrid"
if simulation_mode == 'model_test':
    # Sadece model performansını ölç
    # Stop-loss ve confidence drop devre dışı
    if datetime.utcnow() >= target_exit_time:
        should_sell = True
        sell_reason = 'horizon_reached'

elif simulation_mode == 'hybrid':
    # Hem model hem risk yönetimi
    if stop_loss_triggered:
        should_sell = True
        sell_reason = 'stop_loss'
        trade_category = 'risk_management'
    elif confidence_drop:
        should_sell = True
        sell_reason = 'confidence_drop'
        trade_category = 'risk_management'
    elif datetime.utcnow() >= target_exit_time:
        should_sell = True
        sell_reason = 'horizon_reached'
        trade_category = 'model_performance'
```

**Bu mantık en esnek ve doğru ölçüm sağlar!**

---

## 📝 SONUÇ

### Mevcut Durum:

❌ **Simülasyon model performansını ölçmüyor**
- Stop-loss ve confidence drop ile erken çıkışlar model performansını bozuyor
- Modelin önerdiği süre (horizon) göz ardı ediliyor
- Sonuçlar risk yönetimi + model performansı karışık

### İdeal Durum:

✅ **Simülasyon model performansını ölçmeli**
- Modelin önerdiği süre (horizon) dolana kadar pozisyon tutulmalı
- Stop-loss ve confidence drop opsiyonel olmalı (veya ayrı metrik olarak ölçülmeli)
- Model performansı ve risk yönetimi performansı ayrı ayrı raporlanmalı

### Öncelik:

1. **KRİTİK:** Horizon-based exit mekanizması ekle
2. **YÜKSEK:** Stop-loss ve confidence drop'u opsiyonel yap
3. **ORTA:** Model performansı ve risk yönetimi performansını ayrı metrikler olarak ölç

**Bu değişikliklerle simülasyon, modelin gerçekten işe yarayıp yaramadığını doğru ölçecek!**

