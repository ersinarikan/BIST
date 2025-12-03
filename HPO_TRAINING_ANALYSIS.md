# HPO vs Training DirHit Farkları - Detaylı Analiz Raporu

## 📊 Özet

HPO sonuçları ile Training sonuçları arasında önemli farklar görülüyor. Bu rapor, bu farkların kök nedenlerini analiz ediyor.

## 🔍 Tespit Edilen Kritik Farklar

### 1. **Veri Kaynağı Farkı** ✅ ÇÖZÜLMÜŞ
- **HPO**: `fetch_prices()` kullanıyor (DB'den direkt, cache bypass)
- **Training**: `fetch_prices()` kullanıyor (aynı kaynak) ✅
- **Durum**: Kod incelemesinde training'in de `fetch_prices()` kullandığı görüldü (satır 3197)

### 2. **Adaptive Learning Durumu** ✅ ÇÖZÜLMÜŞ
- **HPO**: `ML_USE_ADAPTIVE_LEARNING = '0'` (kapalı)
- **Training**: `ML_USE_ADAPTIVE_LEARNING = '0'` (kapalı) ✅
- **Durum**: Training'de de adaptive learning kapalı (satır 3138)

### 3. **Seed Kullanımı** ⚠️ POTANSİYEL SORUN
- **HPO**: `ml.base_seeds = [42 + trial.number]` (her trial için farklı seed)
- **Training Evaluation**: `ml_eval.base_seeds = [42 + best_trial_number]` (best trial'ın seed'i)
- **Durum**: Seed doğru ayarlanıyor gibi görünüyor, ancak **model instance'ları farklı** olabilir

### 4. **Model Instance Yönetimi** ⚠️ KRİTİK SORUN
- **HPO**: Her trial için **YENİ** `EnhancedMLSystem()` instance'ı oluşturuluyor
- **Training Evaluation**: **YENİ** `EnhancedMLSystem()` instance'ı oluşturuluyor ✅
- **Ancak**: Training'de önce tüm df ile model eğitiliyor, sonra evaluation için train_df ile yeniden eğitiliyor
- **Sorun**: İlk eğitim (tüm df ile) evaluation'ı etkileyebilir mi?

### 5. **Split Stratejisi** ✅ AYNI
- **HPO**: `generate_walkforward_splits(total_days, horizon, n_splits=4)` - 4 split
- **Training Evaluation**: `generate_walkforward_splits(total_days, horizon, n_splits=4)` - 4 split ✅
- **Durum**: Aynı split fonksiyonu kullanılıyor

### 6. **Evaluation Spec Kullanımı** ⚠️ KONTROL EDİLMELİ
- **Training**: `evaluation_spec` varsa split'leri override ediyor
- **Sorun**: Eğer `evaluation_spec` yoksa veya farklıysa, split'ler farklı olabilir

### 7. **Low Support Gating** ✅ DÜZELTİLDİ
- **HPO**: `HPO_MIN_MASK_COUNT` ve `HPO_MIN_MASK_PCT` kontrolü yapılıyor (default: 0, 0)
- **Training**: `HPO_MIN_MASK_COUNT` ve `HPO_MIN_MASK_PCT` kontrolü yapılıyor (default: 0, 0) ✅
- **Durum**: Artık aynı default değerler kullanılıyor (düzeltme yapıldı)

### 8. **DirHit Hesaplama Mantığı** ✅ AYNI
- **HPO**: `dirhit(y_true, y_pred, thr=0.005)` - threshold mask kullanıyor
- **Training**: `_dirhit(y_true, y_pred, thr=0.005)` - aynı mantık ✅

## 🎯 Kök Neden Analizi

### Senaryo 1: Low Support Gating Farkı (EN MUHTEMEL)
**Problem**: Training'de low support kontrolü daha sıkı (min_mask_count=10, min_mask_pct=5.0), HPO'da ise default (0, 0).

**Etki**: 
- HPO'da tüm split'ler değerlendiriliyor
- Training'de bazı split'ler exclude ediliyor (mask_count < 10 veya mask_pct < 5.0)
- Bu, training DirHit'inin daha düşük çıkmasına neden olabilir

**Örnek**: 
- ADEL_1d: HPO DirHit=85.42% Training DirHit=42.21%
- Eğer HPO'da 4 split varsa ve training'de 2 split exclude edilirse, fark büyük olabilir

### Senaryo 2: Evaluation Spec Eksikliği
**Problem**: Training'de `evaluation_spec` yoksa veya farklıysa, split'ler farklı olabilir.

**Etki**: 
- HPO'da kullanılan split'ler ile training'de kullanılan split'ler farklı olabilir
- Bu, farklı test setleri üzerinde değerlendirme yapılmasına neden olur

### Senaryo 3: Model State Contamination
**Problem**: Training'de önce tüm df ile model eğitiliyor, sonra evaluation için train_df ile yeniden eğitiliyor.

**Etki**: 
- İlk eğitim model state'ini etkileyebilir (singleton cache, global state, vb.)
- Yeni instance oluşturuluyor ama bazı global state'ler temizlenmemiş olabilir

### Senaryo 4: Seed Bagging Farkı
**Problem**: HPO'da seed bagging açık/kapalı olabilir, training'de farklı olabilir.

**Etki**: 
- Seed bagging açık/kapalı durumu farklıysa, model eğitimi farklı olabilir

## 🔧 Önerilen Düzeltmeler

### 1. Low Support Gating Tutarlılığı ✅ DÜZELTİLDİ
```python
# continuous_hpo_training_pipeline.py satır 2438-2444
# DÜZELTME YAPILDI:
_min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))  # Default: 0 (HPO ile aynı)
_min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0.0'))  # Default: 0.0 (HPO ile aynı)
```

### 2. Evaluation Spec Kontrolü
- HPO JSON'da `evaluation_spec` olup olmadığını kontrol et
- Eğer yoksa, HPO'daki split'leri training'e aktar
- Split'lerin aynı olduğundan emin ol

### 3. Model State Temizliği
- Evaluation öncesi tüm global state'leri temizle
- ConfigManager cache'i temizle
- Singleton instance'ları sıfırla

### 4. Seed Bagging Kontrolü
- HPO best trial'da seed bagging açık/kapalı durumunu kontrol et
- Training evaluation'da aynı durumu kullan

## 📈 Örnek Vakalar

### Vaka 1: ADEL_1d (HPO: 85.42% → Training: 42.21%)
- **Fark**: -43.21 puan
- **Olası Neden**: Low support gating veya split farkı

### Vaka 2: BRSAN_1d (HPO: 100.00% → Training: 64.41%)
- **Fark**: -35.59 puan
- **Olası Neden**: Low support gating (HPO'da tüm split'ler dahil, training'de bazıları exclude)

### Vaka 3: EKGYO_1d (HPO: 100.00% → Training: 58.18%)
- **Fark**: -41.82 puan
- **Olası Neden**: Benzer - low support veya split farkı

## ✅ Doğrulama Adımları

1. **Low Support Kontrolü**: HPO ve training'de aynı threshold'ları kullan
2. **Split Kontrolü**: HPO JSON'dan split'leri al ve training'de kullan
3. **Seed Kontrolü**: Best trial'ın seed'ini doğru kullan
4. **Model State Kontrolü**: Evaluation öncesi tüm state'leri temizle
5. **Logging**: Her adımda detaylı log tut

## 🎯 Sonuç

En muhtemel kök neden: **Low Support Gating farkı**. Training'de daha sıkı kontrol (min_mask_count=10, min_mask_pct=5.0) var, HPO'da ise default (0, 0). Bu, training'de bazı split'lerin exclude edilmesine ve DirHit'in düşmesine neden olabilir.

**Öneri**: Low support gating'i HPO ile aynı yap (default: 0, 0) ve sonuçları gözlemle.

