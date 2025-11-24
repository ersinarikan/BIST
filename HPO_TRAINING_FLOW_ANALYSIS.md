# HPO → Training Akış Analizi ve Kritik Bulgular

**Tarih:** 2025-11-24  
**Durum:** 🔴 **Kritik farklılıklar tespit edildi**

---

## 📋 Akış Özeti

### HPO Akışı (`optuna_hpo_with_feature_flags.py`):
1. **Data Fetching:** `fetch_prices(engine, symbol)` - **Cache BYPASS, direkt DB'den**
2. **Split Generation:** `generate_walkforward_splits(total_days, horizon, n_splits=4)`
3. **Her Split İçin:**
   - Model eğitimi: `ml.train_enhanced_models(sym, train_df)`
   - `raw_r2` değerleri hesaplanıyor (XGB, LGB, CatBoost için)
   - Walk-forward prediction: `ml.predict_enhanced(sym, cur)`
   - `predict_enhanced` içinde:
     - Model'lerden prediction alınıyor
     - `historical_r2` = model'lerin `raw_r2` değerleri
     - `smart_ensemble(historical_r2=historical_r2, ...)` → `ensemble_prediction`
   - `pred_return = pred_price / last_close - 1.0`
   - DirHit hesaplama: `dirhit(y_true, preds, thr=0.005)`
4. **Ortalama:** Tüm split'lerin DirHit'leri ortalanıyor

### Training Akışı (`continuous_hpo_training_pipeline.py`):
1. **Data Fetching:** `det.get_stock_data(symbol, days=0)` - **Cache kullanıyor olabilir**
2. **Split Generation:** `generate_walkforward_splits(total_days, horizon, n_splits=4)` - **Aynı fonksiyon**
3. **Her Split İçin:**
   - Model eğitimi: `ml_eval.train_enhanced_models(symbol, train_df_split)`
   - `raw_r2` değerleri hesaplanıyor (XGB, LGB, CatBoost için)
   - Walk-forward prediction: `ml_eval.predict_enhanced(symbol, cur)`
   - `predict_enhanced` içinde:
     - Model'lerden prediction alınıyor
     - `historical_r2` = model'lerin `raw_r2` değerleri
     - `smart_ensemble(historical_r2=historical_r2, ...)` → `ensemble_prediction`
   - `pred_return = pred_price / last_close - 1.0`
   - DirHit hesaplama: `_dirhit(y_true_split, preds, thr=0.005)`
4. **Ortalama:** Tüm split'lerin DirHit'leri ortalanıyor

---

## 🔴 KRİTİK BULGULAR

### 1. **Data Source Farklılığı** 🔴 **ÇOK KRİTİK**

**HPO:**
```python
df = fetch_prices(engine, sym)  # Cache BYPASS, direkt DB'den
# ⚡ CRITICAL FIX: Skip cache for HPO to ensure fresh data from DB
```

**Training:**
```python
df = det.get_stock_data(symbol, days=0)  # Cache KULLANIYOR!
# pattern_detector.py:387-390
cached = self._df_cache.get(symbol)
if cached and (now_ts - float(cached.get('ts', 0))) < float(self.data_cache_ttl):
    return df_cached  # Cache'den dönüyor!
```

**Etki:**
- HPO cache'i bypass ediyor, training cache kullanıyor
- Cache stale olabilir veya farklı veri içerebilir
- Bu farklı `total_days` değerlerine yol açabilir
- Farklı `total_days` → farklı split'ler → farklı train/test data
- **SONUÇ:** Farklı veri → farklı model eğitimi → farklı `raw_r2` → farklı ensemble ağırlıkları → farklı predictions → farklı DirHit

**Çözüm:**
- Training'de de cache'i bypass etmeli (HPO ile tutarlılık için)
- Veya `get_stock_data`'ya cache bypass parametresi ekle

---

### 2. **historical_r2 Farklılığı** 🔴 **ÇOK KRİTİK**

**Sorun:**
- `historical_r2` değerleri model eğitimi sırasında hesaplanıyor (`raw_r2`)
- Bu değerler `smart_ensemble`'ın ağırlıklarını etkiliyor
- Farklı train data → farklı `raw_r2` → farklı ensemble ağırlıkları → farklı `ensemble_prediction`

**Kod:**
```python
# enhanced_ml_system.py:4893-4908
historical_r2 = []
for info in model_predictions.values():
    r2_val = info.get('raw_r2')  # Model eğitimi sırasında hesaplanan R²
    if r2_val is not None:
        historical_r2.append(float(r2_val))
    else:
        # Fallback: confidence'den reverse-engineer
        conf = float(info.get('confidence', 0.5))
        approx_r2 = max(-0.5, min(0.8, (conf - 0.3) / 0.65 * 0.8))
        historical_r2.append(approx_r2)

# smart_ensemble bu historical_r2'yi kullanarak ağırlıkları hesaplıyor
ensemble_pred, final_weights = smart_ensemble(
    predictions=np.array(predictions_list),
    historical_r2=historical_r2,  # ⚠️ Bu değerler farklı olabilir!
    ...
)
```

**Etki:**
- HPO'da train_df ile eğitilen model'in `raw_r2` değerleri
- Training'de train_df_split ile eğitilen model'in `raw_r2` değerleri
- Eğer split'ler farklıysa veya data farklıysa, `raw_r2` farklı olur
- Bu da `smart_ensemble`'ın farklı ağırlıklar kullanmasına yol açar
- Sonuç: Farklı `ensemble_prediction` → farklı `pred_return` → farklı DirHit

**Örnek Senaryo (AHGAZ):**
1. HPO Split 1: train_df (448 gün) → XGB raw_r2=0.15, LGB raw_r2=0.12, Cat raw_r2=0.10
2. Training Split 1: train_df_split (448 gün ama farklı data?) → XGB raw_r2=0.08, LGB raw_r2=0.05, Cat raw_r2=0.03
3. HPO'da smart_ensemble ağırlıkları: [0.4, 0.35, 0.25] (XGB daha yüksek)
4. Training'de smart_ensemble ağırlıkları: [0.33, 0.33, 0.34] (daha eşit)
5. Farklı ağırlıklar → farklı ensemble_prediction → farklı DirHit

---

### 3. **Seed Uyumsuzluğu** ✅ **DÜZELTİLDİ**

**Önceki Sorun:**
- HPO: `ml.base_seeds = [42 + trial.number]` (trial 1262 → seed 1304)
- Training: `ml_eval.base_seeds = [42 + eval_seed]` (eval_seed=42 fallback)

**Düzeltme:**
- Training'de `best_trial_number` kullanılıyor
- `eval_seed = best_trial_number if best_trial_number is not None else 42`
- `ml_eval.base_seeds = [42 + eval_seed]`

**Durum:** ✅ Düzeltildi, ama doğrulanmalı

---

### 4. **Split Tutarlılığı** ⚠️ **KONTROL EDİLMELİ**

**Sorun:**
- HPO ve training aynı `generate_walkforward_splits` fonksiyonunu kullanıyor
- Ama `total_days` farklı olabilir (data source farklılığı nedeniyle)
- Farklı `total_days` → farklı split'ler → farklı train/test data

**Kontrol:**
- HPO'da: `total_days = len(df)` (fetch_prices'den gelen)
- Training'de: `total_days = len(df)` (get_stock_data'den gelen)
- Bu değerler aynı mı?

---

### 5. **Feature Flags Uygulama** ✅ **DOĞRU GÖRÜNÜYOR**

**Kontrol:**
- Best params'tan feature flags set ediliyor
- Smart ensemble params set ediliyor
- Environment variables doğru set ediliyor

**Durum:** ✅ Doğru görünüyor, ama doğrulanmalı

---

## 🎯 ÖNCELİKLİ SORUNLAR

### 1. **Data Source Tutarlılığı** 🔴 **KRİTİK**
- **Sorun:** HPO cache bypass, training cache kullanıyor olabilir
- **Etki:** Farklı veri → farklı split'ler → farklı sonuçlar
- **Çözüm:** Training'de de cache'i bypass et veya HPO'da da cache kullan

### 2. **historical_r2 Farklılığı** 🔴 **ÇOK KRİTİK**
- **Sorun:** `raw_r2` değerleri farklı train data ile farklı hesaplanıyor
- **Etki:** Farklı ensemble ağırlıkları → farklı predictions → farklı DirHit
- **Çözüm:** 
  - HPO'da hesaplanan `raw_r2` değerlerini best_params'a kaydet
  - Training'de bu kaydedilen `raw_r2` değerlerini kullan (model eğitimi sırasında değil, prediction sırasında)

### 3. **Split Tutarlılığı Doğrulama** 🟡 **ORTA**
- **Sorun:** Split'ler aynı mı?
- **Etki:** Farklı split'ler → farklı train/test data → farklı sonuçlar
- **Çözüm:** HPO ve training'de aynı split'leri kullandığını doğrula (log ekle)

---

## 🔍 DETAYLI İNCELEME GEREKLİ

### AHGAZ Örneği İçin:
1. **HPO'da kullanılan data:**
   - `fetch_prices` → kaç gün? Hangi tarih aralığı?
   - Split'ler: train_end_idx, test_end_idx değerleri?

2. **Training'de kullanılan data:**
   - `get_stock_data` → kaç gün? Hangi tarih aralığı?
   - Split'ler: train_end_idx, test_end_idx değerleri?

3. **HPO'da hesaplanan raw_r2 değerleri:**
   - Trial 1262, Split 1: XGB raw_r2=?, LGB raw_r2=?, Cat raw_r2=?
   - Bu değerler best_params'a kaydediliyor mu?

4. **Training'de hesaplanan raw_r2 değerleri:**
   - Split 1: XGB raw_r2=?, LGB raw_r2=?, Cat raw_r2=?
   - Bu değerler HPO ile aynı mı?

5. **Ensemble ağırlıkları:**
   - HPO'da: final_weights = ?
   - Training'de: final_weights = ?
   - Bu ağırlıklar aynı mı?

6. **Prediction değerleri:**
   - HPO'da: ensemble_prediction = ?
   - Training'de: ensemble_prediction = ?
   - Bu değerler aynı mı?

---

## 💡 ÖNERİLER

### Kısa Vadede:
1. **Data source tutarlılığı:** Training'de de cache'i bypass et
2. **historical_r2 kaydetme:** HPO'da hesaplanan `raw_r2` değerlerini best_params'a kaydet
3. **historical_r2 kullanma:** Training'de kaydedilen `raw_r2` değerlerini kullan

### Orta Vadede:
4. **Split doğrulama:** HPO ve training'de aynı split'leri kullandığını log ile doğrula
5. **Ensemble ağırlıkları loglama:** HPO ve training'de ensemble ağırlıklarını logla ve karşılaştır
6. **Prediction karşılaştırma:** HPO ve training'de aynı t için prediction değerlerini karşılaştır

### Uzun Vadede:
7. **Deterministic ensemble:** HPO'da hesaplanan ensemble ağırlıklarını best_params'a kaydet ve training'de kullan
8. **Comprehensive logging:** Tüm kritik değerleri (raw_r2, ensemble weights, predictions) logla
9. **Automated validation:** HPO sonrası otomatik olarak training ile karşılaştır ve farkları raporla

---

## 📊 SONUÇ

**Durum:** 🔴 **Kritik farklılıklar var**

**En Kritik Sorun:** `historical_r2` değerleri model eğitimi sırasında hesaplanıyor ve bu değerler `smart_ensemble`'ın ağırlıklarını etkiliyor. Eğer HPO ve training'de farklı train data kullanılıyorsa, farklı `raw_r2` değerleri hesaplanır, bu da farklı ensemble ağırlıklarına ve dolayısıyla farklı predictions'a yol açar.

**Öncelik:** 
1. Data source tutarlılığını sağla
2. HPO'da hesaplanan `raw_r2` değerlerini best_params'a kaydet
3. Training'de kaydedilen `raw_r2` değerlerini kullan (model eğitimi sırasında değil, prediction sırasında)

**Beklenen İyileştirme:** Bu düzeltmelerle Training DirHit'ler HPO DirHit'lere çok daha yakın olmalı.

---

**Son Güncelleme:** 2025-11-24 20:30

