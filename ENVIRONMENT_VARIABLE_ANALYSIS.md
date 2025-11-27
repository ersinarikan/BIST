# Environment Variable Analizi Raporu

## Özet

99-consolidated.conf dosyasında **141 environment variable** tanımlı.

## Kategoriler

### 1. NEWS & RSS (4 variable)
- `NEWS_SOURCES`: ✅ Kullanılıyor (rss_news_async.py, news_provider.py, config.py)
- `NEWS_CACHE_TTL`: ✅ Kullanılıyor (rss_news_async.py, news_provider.py)
- `NEWS_LOOKBACK_HOURS`: ✅ Kullanılıyor (rss_news_async.py, news_provider.py, config.py)
- `NEWS_MAX_ITEMS`: ✅ Kullanılıyor (rss_news_async.py, news_provider.py, config.py)

### 2. ML & Training (Çok sayıda)
- `ML_MIN_DATA_DAYS`: ✅ Kullanılıyor (enhanced_ml_system.py, ConfigManager)
- `ML_USE_ADAPTIVE_LEARNING`: ✅ Kullanılıyor (enhanced_ml_system.py, continuous_hpo_training_pipeline.py)
- `ML_USE_DIRECTIONAL_LOSS`: ✅ Kullanılıyor (enhanced_ml_system.py, continuous_hpo_training_pipeline.py)
- `ML_USE_SMART_ENSEMBLE`: ✅ Kullanılıyor (enhanced_ml_system.py)
- `ML_TRAIN_THREADS`: ⚠️ Kullanım kontrol edilmeli
- `OMP_NUM_THREADS`: ✅ Kullanılıyor (systemd tarafından)

### 3. HPO (3 variable)
- `HPO_MAX_WORKERS`: ✅ Kullanılıyor (continuous_hpo_training_pipeline.py)
- `HPO_MAX_SLOTS`: ✅ Kullanılıyor (continuous_hpo_training_pipeline.py)
- `HPO_TRIALS`: ❌ **HARDCODED** - Environment variable olmalı!

### 4. Pattern Detection (Çok sayıda)
- `PATTERN_CACHE_TTL`: ✅ Kullanılıyor (pattern_detector.py)
- `PATTERN_BASIC_WEIGHT`: ⚠️ Kullanım kontrol edilmeli
- `PATTERN_ADVANCED_WEIGHT`: ⚠️ Kullanım kontrol edilmeli
- `PATTERN_YOLO_WEIGHT`: ⚠️ Kullanım kontrol edilmeli
- `YOLO_MIN_CONF`: ✅ Kullanılıyor (config.py)

## ⚠️ SORUNLAR

### 1. HPO_TRIALS Hardcoded
**Sorun:** `HPO_TRIALS = 1500` hardcoded olarak 3 yerde tanımlı:
- `scripts/continuous_hpo_training_pipeline.py:228`
- `scripts/show_hpo_progress.py:28`
- `scripts/optuna_hpo_with_feature_flags.py:1026`

**Çözüm:** Environment variable olarak ekle:
```python
HPO_TRIALS = int(os.getenv('HPO_TRIALS', '1500'))
```

### 2. Kullanılmayan Environment Variable'lar
Çoğu environment variable kullanılıyor, ancak bazıları hiç kullanılmıyor olabilir. Detaylı kontrol gerekli.

### 3. ConfigManager vs os.getenv
Bazı yerlerde `ConfigManager.get()` kullanılıyor, bazı yerlerde `os.getenv()`. Tutarlılık sağlanmalı.

## 💡 ÖNERİLER

1. **HPO_TRIALS environment variable ekle**
   - 99-consolidated.conf'a ekle: `Environment="HPO_TRIALS=1500"`
   - Tüm hardcoded 1500 değerlerini `os.getenv('HPO_TRIALS', '1500')` ile değiştir

2. **Kullanılmayan variable'ları temizle**
   - Hiç kullanılmayan environment variable'ları 99-consolidated.conf'dan kaldır

3. **ConfigManager kullanımını standardize et**
   - Tüm environment variable okumaları için `ConfigManager.get()` kullan

4. **Hardcoded değerleri kontrol et**
   - 100, 1500, 24, 3600 gibi değerler environment variable olmalı mı kontrol et

