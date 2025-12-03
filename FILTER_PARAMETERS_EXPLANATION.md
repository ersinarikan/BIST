# Filtreye Takılan Durumlarda Model Eğitimi - Parametreler

## 🎯 Soru: Filtreye Takılan Durumlarda Model Hangi Parametrelerle Eğitilir?

## 1. HPO Sürecinde (optuna_hpo_with_feature_flags.py)

### Senaryo: Bir sembol için tüm split'ler filtreye takılırsa

**Kod Akışı:**
1. Her split için model eğitilir (satır 630):
   ```python
   result = ml.train_enhanced_models(sym, train_df)
   ```

2. Model eğitimi **trial'ın önerdiği parametrelerle** yapılır:
   - Trial parametreleri (satır 302-600): `trial.suggest_*`
   - Feature flags (satır 303-316): `trial.suggest_categorical('enable_*', ...)`
   - Model choice (satır 318-340): `trial.suggest_categorical('model_choice', ...)`
   - Hyperparameters (satır 350-600): `trial.suggest_*` (xgb_*, lgb_*, cat_*)

3. Eğer bir sembol için tüm split'ler filtreye takılırsa:
   - O sembol score'a dahil edilmez (satır 884-885)
   - **Ama model yine de eğitilir** (her split için)
   - **Parametreler**: Trial'ın önerdiği parametreler (trial.suggest_*)

**Sonuç**: HPO sırasında filtreye takılan semboller için model **trial parametreleriyle** eğitilir, ama score'a dahil edilmez.

### Best Params Seçimi (satır 1135-1250)

**Kod:**
```python
best_params = study.best_params  # Tüm sembollerin ortalaması üzerinden seçilir
best_trial = study.best_trial
```

**Mantık:**
- Best params **tüm sembollerin ortalaması** üzerinden seçilir
- Eğer bir sembol için tüm split'ler filtreye takılırsa:
  - O sembol score'a dahil edilmez
  - Ama diğer semboller varsa → Onların ortalaması alınır
  - Best params **diğer semboller için** optimal olur

**⚠️  Sorun**: Eğer bir sembol için tüm split'ler filtreye takılırsa, o sembol için best params **optimal olmayabilir** (çünkü o sembol score'a dahil edilmemiş).

## 2. Training Sürecinde (continuous_hpo_training_pipeline.py)

### Senaryo: Tüm split'ler filtreye takılırsa

**Kod Akışı:**
1. Best params JSON dosyasından okunur (satır 3651):
   ```python
   best_params_with_trial = hpo_result['best_params'].copy()
   ```

2. Best params environment variable'larına set edilir (satır 3015):
   ```python
   set_hpo_params_as_env(best_params, horizon)
   ```

3. Model eğitimi (satır 3213):
   ```python
   result = ml.train_enhanced_models(symbol, df)
   ```

4. Eğer tüm split'ler filtreye takılırsa:
   - Model yine de eğitilir ✅
   - **Parametreler**: Best params (HPO'dan gelen)
   - DirHit None olur (hesaplanamaz)

**Sonuç**: Training sırasında filtreye takılan semboller için model **best params ile** eğitilir, ama DirHit hesaplanamaz.

## 🔍 Detaylı Parametre Setleme

### set_hpo_params_as_env (train_completed_hpo_with_best_params.py, satır 80-210)

**Yapılan İşlemler:**
1. **Feature Flags** (satır 85-100):
   ```python
   os.environ['ENABLE_EXTERNAL_FEATURES'] = str(params.get('enable_external_features', True))
   os.environ['ENABLE_FINGPT_FEATURES'] = str(params.get('enable_fingpt_features', True))
   # ... diğer feature flags
   ```

2. **Model Choice** (satır 102-110):
   ```python
   model_choice = params.get('model_choice', 'all')
   os.environ['ENABLE_XGBOOST'] = '1' if model_choice in ('xgb', 'all') else '0'
   os.environ['ENABLE_LIGHTGBM'] = '1' if model_choice in ('lgbm', 'all') else '0'
   os.environ['ENABLE_CATBOOST'] = '1' if model_choice in ('cat', 'all') else '0'
   ```

3. **Hyperparameters** (satır 112-200):
   ```python
   # XGBoost params
   for key, value in params.items():
       if key.startswith('xgb_'):
           os.environ[f'OPTUNA_XGB_{key[4:].upper()}'] = str(value)
   # LightGBM params
   for key, value in params.items():
       if key.startswith('lgb_'):
           os.environ[f'OPTUNA_LGB_{key[4:].upper()}'] = str(value)
   # CatBoost params
   for key, value in params.items():
       if key.startswith('cat_'):
           os.environ[f'OPTUNA_CAT_{key[4:].upper()}'] = str(value)
   ```

4. **Feature Parameters** (satır 3026-3043):
   ```python
   # Smart-ensemble params
   if 'smart_consensus_weight' in fp:
       os.environ['ML_SMART_CONSENSUS_WEIGHT'] = str(fp['smart_consensus_weight'])
   # ... diğer feature params
   ```

## ⚠️  Kritik Sorun

### Senaryo: Bir sembol için tüm split'ler filtreye takılırsa

**HPO'da:**
- Model **trial parametreleriyle** eğitilir
- O sembol score'a dahil edilmez
- Best params **diğer semboller için** optimal olur

**Training'de:**
- Model **best params ile** eğitilir (diğer semboller için optimal)
- **Ama bu sembol için optimal olmayabilir!** ⚠️

**Sonuç**: Eğer bir sembol için tüm split'ler filtreye takılırsa, o sembol için best params **optimal olmayabilir** (çünkü o sembol HPO score hesaplamasına dahil edilmemiş).

## ✅ Çözüm Önerileri

1. **Filtreyi gevşetmek**: 10/5.0 → 5/3.0 veya 0/0.0
2. **Sembol-spesifik filtre**: Bazı semboller için farklı filtre değerleri
3. **Best params seçimini değiştirmek**: Sadece geçerli semboller için best params seçmek
4. **Uyarı mekanizması**: Filtreye takılan semboller için uyarı vermek

## 📊 Özet Tablo

| Durum | HPO'da Parametreler | Training'de Parametreler |
|-------|---------------------|--------------------------|
| Tüm split'ler geçer | Trial params → Best params | Best params ✅ |
| Bazı split'ler geçer | Trial params → Best params | Best params ✅ |
| Hiçbir split geçemez | Trial params (score'a dahil değil) | Best params (optimal olmayabilir) ⚠️ |

