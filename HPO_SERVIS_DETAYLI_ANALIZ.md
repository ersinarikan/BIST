# HPO Servisi Detaylı Analiz

## 📋 Genel Bakış

HPO (Hyperparameter Optimization) servisi, BIST hisse senetleri için makine öğrenmesi modellerinin optimizasyonunu otomatikleştiren kapsamlı bir sistemdir. Sistem, **feature flags**, **feature internal parameters** ve **model hyperparameters**'ı birlikte optimize ederek en iyi performansı hedefler.

---

## 🏗️ Mimari Yapı

### 1. Ana Bileşenler

#### 1.1. `optuna_hpo_with_feature_flags.py` - HPO Objective Function
**Rol:** Optuna ile hyperparameter optimization yapan ana script

**Temel İşlevler:**
- **Feature Flag Optimization:** 11 adet feature flag'i optimize eder
  - `ENABLE_EXTERNAL_FEATURES`, `ENABLE_FINGPT_FEATURES`, `ENABLE_YOLO_FEATURES`
  - `ML_USE_DIRECTIONAL_LOSS`, `ENABLE_SEED_BAGGING`, `ENABLE_TALIB_PATTERNS`
  - `ML_USE_SMART_ENSEMBLE`, `ML_USE_STACKED_SHORT`, `ENABLE_META_STACKING`
  - `ML_USE_REGIME_DETECTION`, `ENABLE_FINGPT`
  
- **Feature Internal Parameters:** Feature'lar açıkken optimize edilen iç parametreler
  - Directional Loss: `ml_loss_mse_weight`, `ml_loss_threshold`, `ml_dir_penalty`
  - Seed Bagging: `n_seeds`
  - Meta Stacking: `meta_stacking_alpha`
  - Adaptive Learning: `ml_adaptive_k_{horizon}d`, `ml_pattern_weight_scale_{horizon}d`
  - YOLO: `yolo_min_conf`
  - Smart Ensemble: `smart_consensus_weight`, `smart_performance_weight`, `smart_sigma`, `smart_weight_xgb/lgbm/cat`
  - FinGPT: `fingpt_confidence_threshold`
  - External Features: `external_min_days`, `external_smooth_alpha`
  - Regime Detection: `regime_scale_low`, `regime_scale_high`

- **Model Hyperparameters:** XGBoost, LightGBM, CatBoost parametreleri
  - XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`, `gamma`, `grow_policy`, `tree_method`, `max_bin`
  - LightGBM: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_data_in_leaf`, `feature_fraction_bynode`, `bagging_freq`, `min_gain_to_split`
  - CatBoost: `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `subsample`, `rsm`, `border_count`, `random_strength`, `leaf_estimation_iterations`, `bootstrap_type`

**Optimizasyon Metrikleri:**
- **Primary Metric:** DirHit (Directional Hit Rate) - Yön tahmin doğruluğu
- **Secondary Metric:** nRMSE (normalized RMSE) - Normalize edilmiş hata
- **Combined Score:** `0.7 * DirHit - k * nRMSE` (k=6.0 for short horizons, k=4.0 for long)

**Walk-Forward Validation:**
- 4 split walk-forward validation kullanır
- Her split için train/test ayrımı yapılır
- Tüm split'lerin DirHit ortalaması alınır

**Data Leakage Önleme:**
- `ML_USE_ADAPTIVE_LEARNING = '0'` (HPO sırasında her zaman kapalı)
- `ML_SKIP_ADAPTIVE_PHASE2 = '1'` (Phase 2 skip)
- Model her split için sıfırdan eğitilir (test verisi görülmez)

**Study Management:**
- SQLite database'de study dosyası saklanır
- Cycle number ile study isimlendirilir: `hpo_with_features_{symbol}_h{horizon}_c{cycle}`
- WAL mode aktif (concurrent read/write için)
- Warm-start: Önceki cycle'ların best params'ları enqueue edilir

#### 1.2. `continuous_hpo_training_pipeline.py` - Pipeline Orchestrator
**Rol:** HPO ve training süreçlerini koordine eden ana orchestrator

**Temel İşlevler:**

**Cycle Management:**
- Her cycle: Tüm semboller için tüm horizonlar (1d, 3d, 7d, 14d, 30d)
- Cycle tamamlandıktan sonra yeni verilerle tekrar başlar (incremental learning)
- State file ile progress tracking (`continuous_hpo_state.json`)

**Processing Strategy:**
- **Horizon-First Approach:** Tüm semboller için bir horizon bitirilir, sonra diğer horizon'a geçilir
  - Phase 1: Tüm semboller için 1d
  - Phase 2: Tüm semboller için 3d
  - Phase 3: Tüm semboller için 7d
  - Phase 4: Tüm semboller için 14d
  - Phase 5: Tüm semboller için 30d
- **Parallelism:** `MAX_WORKERS` (default: 4) sembol paralel işlenir
- **Sequential per Symbol:** Her sembol için horizonlar sırayla işlenir (1d→3d→7d→14d→30d)

**Task States:**
- `pending`: Henüz başlamamış
- `hpo_in_progress`: HPO çalışıyor
- `training_in_progress`: Training çalışıyor
- `completed`: Başarıyla tamamlandı
- `failed`: Başarısız (retry mekanizması var)
- `skipped`: Yetersiz veri nedeniyle atlandı

**HPO Execution:**
- Subprocess olarak `optuna_hpo_with_feature_flags.py` çalıştırılır
- HPO slot locking (fcntl) ile concurrency kontrolü
- CPU affinity optimization (NUMA-aware)
- Timeout: 72 saat (1500 trials için)
- JSON file validation ve recovery mekanizması

**Training Execution:**
- Best params ile full training
- Adaptive learning KAPALI (HPO ile tutarlılık)
- Model kaydetme ve doğrulama
- DirHit evaluation (WFV ve online)

**DirHit Evaluation:**
- **WFV (Walk-Forward Validation):** Adaptive OFF, best params ile yeniden eğitim
- **Online:** Adaptive OFF, best params ile prediction
- HPO DirHit ile karşılaştırma (alignment check)

**Recovery Mechanisms:**
- HPO completed ama JSON missing: Study file'dan recovery
- State file corruption: Rebuild from study files
- Stale in-progress tasks: Reset to pending after restart

#### 1.3. `train_completed_hpo_with_best_params.py` - Training Script
**Rol:** HPO tamamlanmış semboller için best params ile training

**Temel İşlevler:**
- Completed HPO JSON dosyalarını okur
- Best params'ı environment variables'a set eder
- Full training yapar (tüm feature'lar açık)
- Walk-forward validation ile DirHit hesaplar
- Model kaydetme

**CatBoost Bootstrap Type Normalization:**
- `_normalize_cat_bootstrap_type()` helper function
- Optuna'dan gelen bootstrap_type değerlerini CatBoost enum'larına normalize eder
- Mapping: `'Bayesian'`, `'Bernoulli'`, `'MVS'`, `'No'`

#### 1.4. `enhanced_ml_system.py` - ML System
**Rol:** HPO params'ları kullanan ML training ve prediction sistemi

**HPO Param Integration:**
- Environment variables'dan `OPTUNA_XGB_*`, `OPTUNA_LGB_*`, `OPTUNA_CAT_*` okur
- `ConfigManager.get()` ile parametreleri alır
- Default değerler override edilir

**Feature Flag Integration:**
- `ENABLE_*` flags ile feature'lar açık/kapalı
- `ML_USE_*` flags ile ML özellikleri kontrol edilir
- Feature internal parameters environment variables'dan okunur

---

## 🔄 İş Akışı (Workflow)

### 2.1. HPO Workflow

```
1. Pipeline başlatılır
   ↓
2. Active symbols listesi alınır (database'den)
   ↓
3. Her symbol-horizon çifti için:
   a. Data quality check (minimum 100 days)
   b. HPO slot acquire (concurrency control)
   c. Subprocess: optuna_hpo_with_feature_flags.py
      - Study file oluştur/load (cycle-aware)
      - Warm-start: Önceki best params enqueue
      - 1500 trial optimization
        * Feature flags suggest
        * Feature params suggest (conditional)
        * Hyperparameters suggest
        * Environment variables set
        * EnhancedMLSystem instance create
        * Walk-forward validation (4 splits)
        * DirHit + nRMSE calculate
        * Score = 0.7 * DirHit - k * nRMSE
      - Best trial seçilir
      - JSON file save
   d. HPO slot release
   ↓
4. Best params ile training
   a. Best params environment'a set
   b. Feature flags set
   c. Adaptive learning OFF (HPO ile tutarlılık)
   d. EnhancedMLSystem.train_enhanced_models()
   e. Model save
   f. DirHit evaluation (WFV + online)
   ↓
5. State update (completed)
```

### 2.2. Cycle Management

```
Cycle 1:
  - Tüm semboller için tüm horizonlar HPO + Training
  - State file'da cycle=1 olarak kaydedilir
  
Cycle 2 (Yeni veriler eklendikten sonra):
  - Cycle number increment (cycle=2)
  - Tüm semboller için yeni HPO (yeni verilerle)
  - Study file: hpo_with_features_{symbol}_h{horizon}_c2.db
  - Önceki cycle'ın best params'ları warm-start olarak kullanılır
  - Training yeni best params ile yapılır
  
Cycle N:
  - Sürekli incremental learning
  - Her cycle'da yeni verilerle HPO
  - Model performance iyileşmesi
```

---

## 🎯 Mantık ve Tasarım Kararları

### 3.1. Neden Feature Flags + Hyperparameters Birlikte?

**Problem:** Feature'ların açık/kapalı durumu model performansını etkiler. Örneğin:
- Directional Loss açıkken: `ml_loss_mse_weight`, `ml_loss_threshold` optimize edilmeli
- Seed Bagging açıkken: `n_seeds` optimize edilmeli
- Smart Ensemble açıkken: `smart_consensus_weight`, `smart_performance_weight` optimize edilmeli

**Çözüm:** Feature flags ve hyperparameters birlikte optimize edilir. Bu sayede:
- Feature'ın etkisi doğru ölçülür
- Feature açıkken optimize edilen parametreler kullanılır
- Feature kapalıyken gereksiz parametreler optimize edilmez

### 3.2. Neden Adaptive Learning HPO'da Kapalı?

**Problem:** Adaptive learning, model'in test verisini görmesine izin verir. Bu data leakage'dır.

**Çözüm:** HPO sırasında adaptive learning her zaman kapalı:
- `ML_USE_ADAPTIVE_LEARNING = '0'`
- Model sadece train verisi ile eğitilir
- Test verisi sadece evaluation için kullanılır
- HPO DirHit gerçekçi bir metrik olur

**Training'de de Kapalı (Hibrit Yaklaşım):**
- Plan'a göre: HPO ve Training aynı veri miktarını kullanmalı
- Cycle zaten incremental learning etkisi yaratıyor (yeni verilerle yeniden HPO)
- Adaptive learning yerine cycle-based incremental learning kullanılıyor

### 3.3. Neden Walk-Forward Validation?

**Problem:** Single split validation overfitting riski taşır.

**Çözüm:** 4 split walk-forward validation:
- Her split için train/test ayrımı
- Expanding window approach
- Ortalama DirHit daha güvenilir
- Overfitting riski azalır

### 3.4. Neden Cycle-Based Study Files?

**Problem:** Yeni veriler eklendiğinde, eski study file'a yazmak karışıklığa neden olur.

**Çözüm:** Cycle number ile study file isimlendirme:
- `hpo_with_features_{symbol}_h{horizon}_c{cycle}.db`
- Her cycle kendi study file'ına sahip
- Önceki cycle'lar korunur (analiz için)
- Warm-start: Önceki cycle'ın best params'ları kullanılır

### 3.5. Neden Horizon-First Processing?

**Problem:** Symbol-first processing'de:
- 1d için tüm semboller bitene kadar 3d başlamaz
- Kullanıcı 1d sonuçlarını bekler

**Çözüm:** Horizon-first processing:
- Tüm semboller için 1d bitirilir → 1d sonuçları hazır
- Sonra tüm semboller için 3d → 3d sonuçları hazır
- Incremental value delivery: Kısa horizonlar önce hazır olur

### 3.6. Neden Symbol-Based Sequential (Her Symbol İçin)?

**Problem:** Her horizon için tüm semboller paralel işlenirse:
- Database yükü artar (aynı sembol için veri birden fazla kez çekilir)
- SQLite çakışmaları olur (aynı study file'a yazma)

**Çözüm:** Symbol-based sequential:
- Her sembol için tüm horizonlar sırayla işlenir (1d→3d→7d→14d→30d)
- Aynı sembol için veri bir kez çekilir
- SQLite çakışmaları azalır (bir sembol at a time)
- MAX_WORKERS: Semboller paralel, her biri sequential

### 3.7. Neden DirHit + nRMSE Combined Score?

**Problem:** Sadece DirHit optimize edilirse:
- Model yüksek confidence ile yanlış tahmin yapabilir
- RMSE yüksek olabilir (büyük hatalar)

**Çözüm:** Combined score:
- `score = 0.7 * DirHit - k * nRMSE`
- DirHit: Yön doğruluğu (primary)
- nRMSE: Normalize edilmiş hata (secondary)
- k=6.0 (short horizons), k=4.0 (long horizons)
- Hem yön doğruluğu hem de hata miktarı optimize edilir

### 3.8. Neden Seed Matching (HPO vs Training)?

**Problem:** HPO'da farklı seed, training'de farklı seed kullanılırsa:
- DirHit farklılıkları seed'den kaynaklanabilir
- HPO DirHit ile Training DirHit karşılaştırılamaz

**Çözüm:** Best trial'ın seed'i kullanılır:
- HPO: `ml.base_seeds = [42 + trial.number]`
- Training: `ml.base_seeds = [42 + best_trial_number]`
- Evaluation: `ml_eval.base_seeds = [42 + best_trial_number]`
- Seed matching ile DirHit karşılaştırması güvenilir olur

### 3.9. Neden CatBoost Bootstrap Type Normalization?

**Problem:** Optuna `suggest_categorical()` string döner, CatBoost enum bekler.

**Çözüm:** Normalization helper:
- `'Bayesian'` → `'Bayesian'`
- `'Bernoulli'` → `'Bernoulli'`
- `'MVS'` → `'MVS'`
- `'No'` → `'No'`
- Invalid değerler skip edilir (model default kullanır)

---

## 🔧 Teknik Detaylar

### 4.1. Concurrency Control

**HPO Slot Locking:**
- `acquire_hpo_slot()`: fcntl ile file-based locking
- `HPO_MAX_SLOTS` (default: 3) slot mevcut
- Her HPO process bir slot acquire eder
- Slot dolduğunda blocking wait

**State File Locking:**
- Read: Shared lock (`LOCK_SH`)
- Write: Exclusive lock (`LOCK_EX`)
- Atomic write: Temp file + `os.replace()`
- Merge-aware: Concurrent processes'in state'lerini merge eder

### 4.2. CPU Affinity Optimization

**NUMA-Aware:**
- 4 NUMA node, her biri 32 CPU
- Round-robin NUMA node assignment
- `taskset` ile CPU affinity binding
- Process priority: `nice(-5)` (higher priority)

### 4.3. Memory Management

**Memory Leak Prevention:**
- Her trial sonrası: `ml.models.clear()`
- Her 5 trial'da bir: `gc.collect()`
- Feature cache clearing
- Horizon features clearing

### 4.4. Error Handling

**Retry Mechanism:**
- HPO failed: 3 retry hakkı
- Permanent failures: `skipped` (retry yok)
  - Insufficient data
  - Symbol not found
  - Delisted
- Temporary failures: Retry
  - Timeout
  - Network errors
  - Subprocess errors

**Recovery Mechanisms:**
- HPO completed ama JSON missing: Study file'dan recovery
- State file corruption: Rebuild from study files
- Stale in-progress: Reset to pending

### 4.5. Data Quality Gates

**Minimum Data Requirements:**
- Tüm horizonlar için: 100 days minimum
- Test set için: `horizon + 10` days minimum
- Walk-forward splits için: Yeterli test data

**Data Validation:**
- Duplicate date kontrolü
- NaN/INF temizleme
- Cache bypass (HPO için fresh data)

---

## 📊 Metrikler ve Değerlendirme

### 5.1. HPO Metrics

**Primary:**
- DirHit: Yön tahmin doğruluğu (%)
- nRMSE: Normalize edilmiş hata
- Score: `0.7 * DirHit - k * nRMSE`

**Secondary:**
- RMSE: Root mean squared error
- MAPE: Mean absolute percentage error
- Valid predictions count
- Threshold mask statistics

### 5.2. Training Metrics

**WFV DirHit:**
- Adaptive OFF
- Best params ile yeniden eğitim
- Walk-forward validation
- HPO DirHit ile karşılaştırma

**Online DirHit:**
- Adaptive OFF
- Best params ile prediction
- Full dataset üzerinde

**Alignment Check:**
- WFV DirHit vs HPO DirHit
- Fark < 1%: ✅ Aligned
- Fark >= 1%: ⚠️ Warning

---

## 🎯 Hedefler ve Beklentiler

### 6.1. Optimizasyon Hedefleri

**Feature Flag Coverage:**
- 1500 trials → ~73% feature flag combination coverage (1500/2048)
- 11 feature flag → 2^11 = 2048 kombinasyon
- TPE sampler ile intelligent exploration

**Hyperparameter Space:**
- ~36-43 parametre optimize edilir
- 11 feature flag + 10-12 feature param + 15-20 hyperparam
- Conditional optimization (feature açıkken optimize et)

### 6.2. Performance Hedefleri

**DirHit Improvement:**
- Her cycle'da DirHit artışı beklenir
- Yeni verilerle incremental learning
- Best params ile training DirHit > HPO DirHit (adaptive learning etkisi)

**Training Time:**
- HPO: 72 saat (1500 trials)
- Training: ~5-10 dakika (best params ile)
- Total per symbol-horizon: ~72 saat

### 6.3. Scalability

**Parallel Processing:**
- MAX_WORKERS=4: 4 sembol paralel
- Her sembol sequential (horizonlar sırayla)
- HPO slot limiting: 3 concurrent HPO

**Resource Usage:**
- CPU: NUMA-aware binding
- Memory: Leak prevention
- Disk: Study files (SQLite), JSON results
- Database: PgBouncer connection pooling

---

## 🔍 Kritik Noktalar ve Dikkat Edilmesi Gerekenler

### 7.1. Data Leakage Prevention

**HPO:**
- ✅ Adaptive learning OFF
- ✅ Phase 2 skip
- ✅ Walk-forward validation
- ✅ Test verisi sadece evaluation için

**Training:**
- ✅ Adaptive learning OFF (HPO ile tutarlılık)
- ✅ Best params kullanımı
- ✅ Seed matching

### 7.2. State Management

**State File:**
- Merge-aware writes (concurrent processes)
- Atomic writes (temp file + replace)
- Cycle preservation
- Recovery mechanisms

**Study Files:**
- Cycle-aware naming
- WAL mode (concurrent access)
- Recovery from study files

### 7.3. Parameter Alignment

**HPO → Training:**
- Best params environment'a set
- Feature flags alignment
- Feature params alignment
- Seed matching

**Training → Evaluation:**
- Best params kullanımı
- Feature flags alignment
- Seed matching
- Adaptive learning OFF

### 7.4. Error Recovery

**HPO Failures:**
- Retry mechanism (3 attempts)
- Permanent vs temporary failures
- Study file recovery
- JSON file recovery

**Training Failures:**
- Model save verification
- Insufficient data handling
- Error logging and tracking

---

## 📈 Süreç Akışı Özeti

```
┌─────────────────────────────────────────────────────────────┐
│                    HPO SERVİSİ AKIŞI                        │
└─────────────────────────────────────────────────────────────┘

1. Pipeline Başlatma
   ├─ Active symbols listesi (database)
   ├─ State file load
   └─ Cycle number belirleme

2. Horizon-First Processing
   ├─ Phase 1: Tüm semboller için 1d
   ├─ Phase 2: Tüm semboller için 3d
   ├─ Phase 3: Tüm semboller için 7d
   ├─ Phase 4: Tüm semboller için 14d
   └─ Phase 5: Tüm semboller için 30d

3. Her Symbol-Horizon Çifti İçin:
   ├─ Data Quality Check (min 100 days)
   ├─ HPO Slot Acquire
   ├─ HPO Execution
   │  ├─ Study file create/load (cycle-aware)
   │  ├─ Warm-start (önceki best params)
   │  ├─ 1500 Trial Optimization
   │  │  ├─ Feature flags suggest
   │  │  ├─ Feature params suggest (conditional)
   │  │  ├─ Hyperparameters suggest
   │  │  ├─ Environment variables set
   │  │  ├─ EnhancedMLSystem create
   │  │  ├─ Walk-forward validation (4 splits)
   │  │  ├─ DirHit + nRMSE calculate
   │  │  └─ Score = 0.7 * DirHit - k * nRMSE
   │  ├─ Best trial select
   │  └─ JSON file save
   ├─ HPO Slot Release
   ├─ Training Execution
   │  ├─ Best params set (env vars)
   │  ├─ Feature flags set
   │  ├─ Adaptive learning OFF
   │  ├─ EnhancedMLSystem.train_enhanced_models()
   │  ├─ Model save
   │  └─ DirHit evaluation (WFV + online)
   └─ State Update (completed)

4. Cycle Completion
   ├─ Tüm semboller için tüm horizonlar tamamlandı
   ├─ Cycle number increment
   └─ Yeni verilerle tekrar başla (incremental learning)
```

---

## 🎓 Öğrenilen Dersler ve Best Practices

### 8.1. Data Leakage Prevention
- Adaptive learning HPO'da her zaman kapalı
- Walk-forward validation kullan
- Test verisi sadece evaluation için

### 8.2. Parameter Alignment
- HPO ve Training aynı seed kullanmalı
- Feature flags alignment kritik
- Best params environment'a doğru set edilmeli

### 8.3. State Management
- Merge-aware writes (concurrent processes)
- Atomic writes (temp file + replace)
- Recovery mechanisms (study file, JSON file)

### 8.4. Error Handling
- Retry mechanism (temporary failures)
- Permanent failures → skipped
- Recovery from study files

### 8.5. Performance Optimization
- CPU affinity (NUMA-aware)
- Memory leak prevention
- Concurrency control (HPO slots)

---

## 🔮 Gelecek İyileştirmeler

### 9.1. Potansiyel İyileştirmeler

**HPO Efficiency:**
- Early stopping (pruning) iyileştirmesi
- Parallel trials (Optuna distributed)
- Bayesian optimization tuning

**Training Efficiency:**
- Model caching
- Incremental training (sadece yeni verilerle)
- Distributed training

**Monitoring:**
- Real-time progress tracking
- DirHit trend analysis
- Resource usage monitoring

**Recovery:**
- Automatic recovery from failures
- State file backup
- Study file backup

---

## 📝 Sonuç

HPO servisi, BIST hisse senetleri için makine öğrenmesi modellerinin optimizasyonunu otomatikleştiren kapsamlı bir sistemdir. Sistem:

1. **Feature flags, feature internal parameters ve hyperparameters**'ı birlikte optimize eder
2. **Walk-forward validation** ile güvenilir metrikler üretir
3. **Data leakage** önleme mekanizmaları ile gerçekçi değerlendirme yapar
4. **Cycle-based incremental learning** ile sürekli iyileşme sağlar
5. **Horizon-first processing** ile incremental value delivery yapar
6. **Robust error handling** ve recovery mekanizmaları ile güvenilir çalışır

Sistem, production-ready, scalable ve maintainable bir yapıya sahiptir.

