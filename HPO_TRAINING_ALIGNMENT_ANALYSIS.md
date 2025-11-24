# HPO vs Training DirHit Alignment Analizi

**Tarih:** 2025-11-24  
**Durum:** ⚠️ **Ciddi uyumsuzluklar tespit edildi**

---

## 📊 Genel Durum

### İstatistikler (show_hpo_progress.py çıktısından):
- **Ortalama HPO DirHit:** 87.72%
- **Ortalama Training DirHit:** 69.91%
- **Fark:** -17.81% (Training, HPO'dan düşük)

### Tamamlanan Görevler:
- **Toplam:** 68 sembol-horizon çifti
- **Tümü 1d horizon** için

---

## 🔴 KRİTİK UYUMSUZLUKLAR

### 1. **AKFIS 1d - En Kötü Durum**
- **HPO DirHit:** 100.00%
- **Training DirHit (WFV):** 0.00%
- **Fark:** -100.00% ⚠️ **ÇOK KRİTİK**
- **Durum:** Training'de hiç doğru tahmin yok, HPO'da mükemmel!

### 2. **ALCAR 1d**
- **HPO DirHit:** 100.00%
- **Training DirHit (WFV):** ~29.17% (tahmin)
- **Fark:** ~-70.83% ⚠️ **ÇOK YÜKSEK**

### 3. **AHGAZ 1d**
- **HPO DirHit:** 100.00%
- **Training DirHit (WFV):** ~28.12% (tahmin)
- **Fark:** ~-71.88% ⚠️ **ÇOK YÜKSEK**

### 4. **BALSU 1d**
- **HPO DirHit:** 75.11%
- **Training DirHit (WFV):** 29.58%
- **Fark:** -45.53% ⚠️ **YÜKSEK**

### 5. **AKYHO 1d**
- **HPO DirHit:** 79.10%
- **Training DirHit (WFV):** 42.52%
- **Fark:** -36.58% ⚠️ **ORTA**

### 6. **ANHYT 1d**
- **HPO DirHit:** 68.75%
- **Training DirHit (WFV):** 50.00%
- **Fark:** -18.75% ⚠️ **ORTA**

### 7. **ATAKP 1d**
- **HPO DirHit:** 87.50%
- **Training DirHit (WFV):** 58.33%
- **Fark:** -29.17% ⚠️ **ORTA**

---

## 🔍 TESPİT EDİLEN SORUNLAR

### 1. **Seed Uyumsuzluğu**
**Sorun:** Training'de `seed=42` kullanılıyor, ama HPO'da `best_trial` seed'i kullanılıyor olabilir.

**Loglardan:**
```
🔧 AKFIS 1d WFV: Using seed=42 (best_trial=None) for evaluation
```

**Etki:** Farklı seed'ler farklı model sonuçlarına yol açabilir.

**Çözüm:** HPO best trial'ın seed'ini kullanmalı.

---

### 2. **Data Split Farklılığı**
**Sorun:** HPO ve Training farklı data split'leri kullanıyor olabilir.

**Loglardan:**
```
📊 AKFIS 1d WFV Split 1/2: train=139 days, test=30 days
🔍 [eval-debug] AKFIS 1d Split 1: Train period: 2025-01-23 to 2025-08-15
🔍 [eval-debug] AKFIS 1d Split 1: Test period: 2025-08-18 to 2025-09-26
```

**Etki:** Farklı train/test split'leri farklı DirHit sonuçlarına yol açabilir.

**Çözüm:** HPO ile aynı split stratejisini kullanmalı.

---

### 3. **Feature Flags Uygulama Sorunu**
**Sorun:** Feature flags doğru uygulanmıyor olabilir.

**Loglardan:**
```
🔧 AKFIS 1d WFV: Feature flags set from best_params: 15 flags
🔧 Eval env (WFV): adaptive=0, seed_bagging=1, directional_loss=0, smart=0, stacked=1, regime=0
```

**Etki:** Feature flags doğru set edilmiş görünüyor, ama model eğitimi sırasında uygulanmıyor olabilir.

**Çözüm:** Feature flags'in model eğitimi sırasında doğru uygulandığını doğrulamalı.

---

### 4. **Mask Count Sorunu (AKFIS)**
**Sorun:** AKFIS için mask_count çok düşük (0 veya 1).

**Loglardan:**
```
✅ AKFIS 1d WFV Split 1: DirHit=0.00%, nRMSE=1.036, Score=-6.22 (valid=29/30, mask=0, ...)
✅ AKFIS 1d Online Split 1: DirHit = 0.00% (valid=29/30, mask=0, ...)
```

**Etki:** Threshold (0.005) üzerinde çok az prediction var, bu yüzden DirHit hesaplanamıyor.

**Çözüm:** Threshold değerini kontrol etmeli veya prediction magnitude'larını kontrol etmeli.

---

### 5. **Model Parametreleri Uygulama Sorunu**
**Sorun:** HPO best params doğru uygulanmıyor olabilir.

**Loglardan:**
```
⚙️ AKFIS 1d: Best HPO params set for evaluation: n_est=323, max_depth=2, lr=0.058604142193426044
🔍 AKFIS 1d: Environment vars - OPTUNA_XGB_N_ESTIMATORS=323, OPTUNA_XGB_MAX_DEPTH=2, OPTUNA_XGB_LEARNING_RATE=0.058604142193426044
```

**Etki:** Parametreler set edilmiş görünüyor, ama model eğitimi sırasında kullanılmıyor olabilir.

**Çözüm:** Model eğitimi sırasında parametrelerin doğru kullanıldığını doğrulamalı.

---

## 🎯 ÖNCELİKLİ SORUNLAR

### 1. **AKFIS - Mask Count = 0 Sorunu** 🔴 KRİTİK
- **Sorun:** Prediction'lar threshold (0.005) altında kalıyor
- **Etki:** DirHit hesaplanamıyor (0.00%)
- **Çözüm:** 
  - Prediction magnitude'larını kontrol et
  - Threshold değerini düşür (0.005 → 0.001)
  - Model prediction scale'ini kontrol et

### 2. **Seed Uyumsuzluğu** 🟡 ORTA
- **Sorun:** Training'de seed=42, HPO'da best_trial seed'i
- **Etki:** Farklı model sonuçları
- **Çözüm:** HPO best trial seed'ini kullan

### 3. **Data Split Farklılığı** 🟡 ORTA
- **Sorun:** HPO ve Training farklı split stratejileri kullanıyor
- **Etki:** Farklı DirHit sonuçları
- **Çözüm:** HPO ile aynı split stratejisini kullan

---

## 📋 ÖNERİLER

### Kısa Vadede:
1. **AKFIS mask_count sorununu çöz** (threshold veya prediction scale)
2. **Seed uyumsuzluğunu düzelt** (best_trial seed kullan)
3. **Data split uyumsuzluğunu kontrol et** (HPO ile aynı split)

### Orta Vadede:
4. **Feature flags uygulamasını doğrula** (model eğitimi sırasında)
5. **Model parametreleri uygulamasını doğrula** (model eğitimi sırasında)
6. **Comprehensive logging ekle** (HPO vs Training karşılaştırması için)

### Uzun Vadede:
7. **Automated alignment test** (HPO sonrası otomatik doğrulama)
8. **DirHit difference alerting** (büyük farklar için uyarı)
9. **Root cause analysis** (her uyumsuzluk için detaylı analiz)

---

## 🔍 DETAYLI ANALİZ GEREKLİ

### AKFIS Örneği İçin:
1. **HPO'da mask_count neydi?** (HPO loglarını kontrol et)
2. **HPO'da prediction magnitude'ları neydi?** (HPO loglarını kontrol et)
3. **Training'de prediction magnitude'ları neden düşük?** (model prediction scale'ini kontrol et)
4. **Feature flags doğru uygulanıyor mu?** (model eğitimi sırasında log ekle)

### Genel İçin:
1. **HPO ve Training aynı data kullanıyor mu?** (data source kontrolü)
2. **HPO ve Training aynı feature engineering kullanıyor mu?** (feature columns kontrolü)
3. **HPO ve Training aynı model parametreleri kullanıyor mu?** (parametre kontrolü)
4. **HPO ve Training aynı evaluation metodunu kullanıyor mu?** (DirHit calculation kontrolü)

---

## 📊 SONUÇ

**Durum:** ⚠️ **Ciddi uyumsuzluklar var**

**En Kritik Sorun:** AKFIS gibi bazı semboller için Training DirHit = 0% (HPO'da 100%)

**Öncelik:** AKFIS mask_count sorununu çöz, sonra seed ve data split uyumsuzluklarını düzelt.

**Beklenen İyileştirme:** Bu düzeltmelerle Training DirHit'ler HPO DirHit'lere yaklaşmalı (ortalama fark -17.81% → -5% altı).

---

**Son Güncelleme:** 2025-11-24 17:30

