# Karmaşıklık Analizi Raporu

## Genel Bakış

İki fonksiyon karmaşıklık uyarısı veriyor:
1. `train_enhanced_models` (Satır 2183-5855, ~3672 satır)
2. `predict_enhanced` (Satır 5857-8978, ~3121 satır)

## 1. train_enhanced_models Fonksiyonu

### Boyut
- **Toplam Satır**: ~3672 satır
- **Başlangıç**: Satır 2183
- **Bitiş**: Satır 5855

### Karmaşıklık Kaynakları

#### A. Çoklu İç İçe Döngüler
- Ana döngü: `for horizon in self.prediction_horizons:` (5 horizon: 1d, 3d, 7d, 14d, 30d)
- Her horizon için:
  - Model eğitimi döngüleri (XGBoost, LightGBM, CatBoost)
  - Seed bagging döngüleri (`for seed in self.base_seeds`)
  - Cross-validation döngüleri (`for fold in range(n_splits)`)
  - Feature engineering döngüleri

#### B. Çoklu Koşullu Dallanma
- Adaptive learning kontrolü (`if use_adaptive and not skip_phase2`)
- Phase 1, Phase 1.5, Phase 1.6, Phase 2 kontrolleri
- Model enable/disable kontrolleri (XGBoost, LightGBM, CatBoost)
- Feature flag kontrolleri (15+ farklı feature flag)
- Regime detection kontrolleri
- Meta-stacking kontrolleri

#### C. Çoklu Try-Except Blokları
- Her model eğitimi için ayrı try-except
- Her horizon için ayrı try-except
- Feature engineering için try-except
- Model kaydetme için try-except

#### D. İç İçe Geçmiş Mantık
- Adaptive learning Phase 1 → Phase 1.5 → Phase 1.6 → Phase 2
- Her phase içinde model eğitimi
- Her model eğitimi içinde seed bagging
- Her seed içinde cross-validation

### Örnek Karmaşıklık Yapısı
```
train_enhanced_models
├── if use_adaptive and not skip_phase2:
│   ├── if total_days >= 240:
│   ├── elif total_days >= 180:
│   └── else:
├── for horizon in self.prediction_horizons:  # 5 horizon
│   ├── if enable_xgb:
│   │   ├── if use_directional:
│   │   │   ├── for seed in self.base_seeds:  # N seeds
│   │   │   │   ├── for fold in range(n_splits):  # CV folds
│   │   │   │   └── try-except
│   │   │   └── try-except
│   │   └── else:
│   ├── if enable_lgb:
│   │   └── (benzer yapı)
│   └── if enable_cat:
│       └── (benzer yapı)
├── if use_adaptive:
│   ├── Phase 1.5 (test ile disiplin)
│   ├── Phase 1.6 (tekrar 300 günlük eğitim)
│   └── Phase 2 (400 günlük tam eğitim)
└── Meta-stacking eğitimi
```

## 2. predict_enhanced Fonksiyonu

### Boyut
- **Toplam Satır**: ~3121 satır
- **Başlangıç**: Satır 5857
- **Bitiş**: Satır 8978

### Karmaşıklık Kaynakları

#### A. Çoklu İç İçe Döngüler
- Ana döngü: `for horizon in self.prediction_horizons:` (5 horizon)
- Her horizon için:
  - Model prediction döngüleri (`for model_name, model_info in horizon_models.items()`)
  - Seed model ensemble döngüleri (`for seed_model in model_info['models']`)
  - Feature alignment döngüleri

#### B. Çoklu Koşullu Dallanma
- Model yükleme kontrolleri (memory vs disk)
- Feature guard kontrolleri
- Horizon-specific feature kontrolleri
- Model enable/disable kontrolleri
- Feature alignment kontrolleri
- Prediction validation kontrolleri (NaN/Inf/extreme values)
- Meta-stacking kontrolleri
- Smart ensemble kontrolleri

#### C. Çoklu Try-Except Blokları
- Model yükleme için try-except
- Feature engineering için try-except
- Her model prediction için try-except
- Feature alignment için try-except
- Prediction validation için try-except
- Meta-stacking için try-except

#### D. İç İçe Geçmiş Mantık
- Model yükleme → Feature engineering → Feature validation
- Her horizon için:
  - Model seçimi → Feature alignment → Prediction → Validation
  - Ensemble → Meta-stacking → Final prediction

### Örnek Karmaşıklık Yapısı
```
predict_enhanced
├── if not models_in_memory:
│   └── load_trained_models
├── Feature engineering
├── Feature validation
│   ├── if missing_cols:
│   │   ├── if enable_pred_feature_guard:
│   │   │   ├── for col in missing_cols:
│   │   │   └── if disallowed_missing:
│   │   └── else:
├── for horizon in self.prediction_horizons:  # 5 horizon
│   ├── if model_key in self.models:
│   │   ├── if horizon_feature_key in self.models:
│   │   └── else:
│   │       └── try-except (load from disk)
│   ├── for model_name, model_info in horizon_models.items():
│   │   ├── if model_name == 'xgboost':
│   │   │   └── if allow:
│   │   ├── if 'models' in model_info:  # Ensemble
│   │   │   ├── for seed_model in model_info['models']:
│   │   │   │   ├── if hasattr(seed_model, 'get_score'):
│   │   │   │   │   └── try-except
│   │   │   │   └── else:
│   │   │   │       └── try-except
│   │   │   └── Validation (NaN/Inf/extreme)
│   │   └── else:  # Single model
│   │       ├── if hasattr(model, 'get_score'):
│   │       │   ├── Feature alignment
│   │       │   │   ├── if expected_feature_names:
│   │       │   │   │   ├── for feat_name in expected_feature_names:
│   │       │   │   │   └── Validation
│   │       │   │   └── else:
│   │       │   └── Validation (NaN/Inf/extreme)
│   │       └── else:
│   │           └── Validation
│   ├── Prediction validation
│   │   ├── if math.isinf(pred_ret):
│   │   ├── if abs(pred_ret) > 1.0:
│   │   └── if pred <= 0:
│   └── Ensemble/Meta-stacking
│       ├── if meta_model exists:
│       │   ├── if training_model_order:
│       │   └── else:
│       └── else:
│           └── if use_smart_ensemble:
└── Horizon consistency check
```

## Karmaşıklık Metrikleri (Tahmini)

### train_enhanced_models
- **Cyclomatic Complexity**: ~150-200 (çok yüksek, ideal <10)
- **Nested Depth**: 8-10 seviye
- **If/Elif/Else**: ~200+ adet
- **For/While Loops**: ~50+ adet
- **Try-Except Blocks**: ~100+ adet

### predict_enhanced
- **Cyclomatic Complexity**: ~120-150 (çok yüksek, ideal <10)
- **Nested Depth**: 8-10 seviye
- **If/Elif/Else**: ~150+ adet
- **For/While Loops**: ~30+ adet
- **Try-Except Blocks**: ~80+ adet

## Refaktör Önerileri

### 1. train_enhanced_models için

#### A. Phase'leri Ayrı Metodlara Ayır
- `_train_phase_1()` - İlk eğitim
- `_train_phase_1_5()` - Test ile disiplin
- `_train_phase_1_6()` - Tekrar 300 günlük eğitim
- `_train_phase_2()` - 400 günlük tam eğitim

#### B. Model Eğitimini Ayrı Metodlara Ayır
- `_train_xgboost_model()` - XGBoost eğitimi
- `_train_lightgbm_model()` - LightGBM eğitimi
- `_train_catboost_model()` - CatBoost eğitimi
- `_train_meta_learner()` - Meta-stacking eğitimi

#### C. Seed Bagging'i Ayrı Metoda Ayır
- `_train_with_seed_bagging()` - Seed bagging mantığı

#### D. Feature Engineering'i Zaten Ayrı
- `create_advanced_features()` - Zaten ayrı metod

### 2. predict_enhanced için

#### A. Model Yükleme ve Validasyonu Ayrı Metodlara Ayır
- `_load_models_if_needed()` - Model yükleme mantığı
- `_validate_features()` - Feature validation mantığı

#### B. Prediction'i Ayrı Metodlara Ayır
- `_predict_single_model()` - Tek model prediction
- `_predict_ensemble()` - Ensemble prediction
- `_predict_with_meta_stacking()` - Meta-stacking prediction

#### C. Feature Alignment'i Ayrı Metoda Ayır
- `_align_features()` - Feature alignment mantığı

#### D. Validation'ı Ayrı Metodlara Ayır
- `_validate_prediction()` - Prediction validation
- `_validate_features_for_prediction()` - Feature validation

## Sonuç

Her iki fonksiyon da:
1. **Çok büyük** (3000+ satır)
2. **Çok karmaşık** (150+ cyclomatic complexity)
3. **Çok fazla sorumluluk** (SRP ihlali)
4. **Test edilmesi zor** (çok fazla branch)
5. **Bakımı zor** (değişiklik yapmak riskli)

**Önerilen Yaklaşım**: Adım adım refaktör, küçük metodlara böl, her metod tek bir sorumluluğa sahip olsun.

