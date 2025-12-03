## HPO Pipeline Detaylı Dokümantasyon

Bu doküman, **HPO (Hyperparameter Optimization) + Training** sürecini uçtan uca, kod referanslarıyla birlikte açıklar.  
Amaç, süreci hem **teknik/mantıksal** açıdan hem de **iş/kural** bakış açısından tamamen şeffaf ve denetlenebilir hale getirmektir.

---

### 1. Genel Mimari Özeti

- **Amaç**: Her sembol–ufuk (`symbol_horizon`) çifti için:
  - Optuna ile **feature flags + feature iç parametreleri + model hyperparameter** optimizasyonu (HPO),
  - Aynı parametrelerle **WFV (walk‑forward validation) temelli eğitim**,
  - Sonuçların **DirHit, nRMSE, skor** ve mask filtreleri ile tutarlı şekilde kaydedilmesi.
- **Temel bileşenler**:
  - **HPO Objective**: `scripts/optuna_hpo_with_feature_flags.py`
  - **Orkestrasyon (HPO + Training + Cycle)**: `scripts/continuous_hpo_training_pipeline.py`
  - **Study dosyaları (Optuna)**: `hpo_studies/hpo_with_features_{SYMBOL}_h{H}_c{CYCLE}.db`
  - **HPO JSON çıktı dosyaları**: `results/optuna_pilot_features_on_h{H}_c{CYCLE}_*.json`
  - **Durum dosyası (state)**: `results/continuous_hpo_state.json`
  - **Filtreli JSON tekrar üretimi**: `scripts/recreate_all_json_from_study_with_filter.py`
  - **Düşük destek fallback best params**: `scripts/find_fallback_best_params.py`
  - **İlerleme görünümü**: `scripts/show_hpo_progress.py`

---

### 2. Ana Dosyalar ve Roller

- **`scripts/optuna_hpo_with_feature_flags.py`**  
  - Optuna **objective** fonksiyonunu tanımlar (`objective`), trial bazında:
    - Feature flag kombinasyonlarını,
    - Feature iç parametrelerini,
    - Model hyperparametrelerini,
    - Model seçimini (`model_choice`) optimize eder.
  - Her trial için **çoklu WFV split** üzerinde DirHit ve nRMSE hesaplar.
  - Sonuçları **`trial.user_attrs`** içine yazar: `avg_dirhit`, `avg_nrmse`, `symbol_metrics`, `features_enabled`, `feature_params` vb.

- **`scripts/continuous_hpo_training_pipeline.py`**  
  - `ContinuousHPOPipeline` sınıfı:
    - Tüm sembol–ufuk işleri için **HPO + Training** sürecini yönetir.
    - `continuous_hpo_state.json` durum dosyasını okur/yazar.
    - **Cycle yönetimi** yapar (Cycle 1, 2, 3...).
    - Paralel işlem (ProcessPoolExecutor) ile sembolleri horizon bazlı işler.
    - HPO sonucunu JSON + study dosyalarından **filtre kurallarıyla birlikte** seçer.
    - Eğitim sonrası WFV ve online DirHit ölçümlerini hesaplar.

- **`scripts/recreate_all_json_from_study_with_filter.py`**  
  - Study dosyalarından, belirtilen filtre ile (`min_mask_count`, `min_mask_pct`, `min_valid_splits`) **JSON yeniden üretir**.
  - Önce 5/2.5, gerekirse fallback 0/0.0 filtresi ile çalışır.

- **`scripts/find_fallback_best_params.py`**  
  - Bir sembol–ufuk için study içindeki trial’ları dolaşır,
  - **0/0.0 filtre** (mask filtresi yok) ile split DirHit ortalamasına göre en iyi trial’ı bulur,
  - Düşük destekli semboller için **fallback best params** sağlar.

- **`scripts/show_hpo_progress.py`**  
  - `continuous_hpo_state.json` + study dosyalarından:
    - HPO trial sayısı, en iyi trial, DirHit,
    - Eğitim DirHit (WFV/online) durumlarını gösterir.
  - DirHit’i **`symbol_metrics[symbol_key]['avg_dirhit']`** üzerinden **sembol spesifik** okur.

---

### 3. Optuna HPO Objective (optuna_hpo_with_feature_flags.py)

#### 3.1. Parametre Uzayı (Feature Flags + Feature Parametreleri + Hyperparametreler)

Objective fonksiyonu:

```288:367:scripts/optuna_hpo_with_feature_flags.py
def objective(trial: optuna.Trial, symbols, horizon: int, engine, db_url: str, study=None, max_trials: Optional[int] = None) -> float:
    """Optuna objective function - Feature flags + Hyperparameters birlikte optimize edilir."""
    # ✅ FIX: Check trial limit at the start of each trial to prevent exceeding n_trials
    # This provides a second layer of protection against race conditions in parallel execution
    # Count all trials except the current one (which is just starting)
    if study is not None and max_trials is not None:
        # Count all trials except the current one
        other_trials = [t for t in study.trials if t.number != trial.number]
        if len(other_trials) >= max_trials:
            # Skip this trial if we've already reached the limit
            raise optuna.TrialPruned(f"Trial limit reached ({len(other_trials)}/{max_trials})")
    
    ConfigManager.clear_cache()
    
    # ⚡ NEW: Feature flag'leri optimize et (12 feature - test script ile aynı)
    feature_flags = {
        'enable_external_features': trial.suggest_categorical('enable_external_features', [True, False]),
        'enable_fingpt_features': trial.suggest_categorical('enable_fingpt_features', [True, False]),
        'enable_yolo_features': trial.suggest_categorical('enable_yolo_features', [True, False]),
        'enable_directional_loss': trial.suggest_categorical('enable_directional_loss', [True, False]),
        'enable_seed_bagging': trial.suggest_categorical('enable_seed_bagging', [True, False]),
        'enable_talib_patterns': trial.suggest_categorical('enable_talib_patterns', [True, False]),
        'enable_smart_ensemble': trial.suggest_categorical('enable_smart_ensemble', [True, False]),
        'enable_stacked_short': trial.suggest_categorical('enable_stacked_short', [True, False]),
        'enable_meta_stacking': trial.suggest_categorical('enable_meta_stacking', [True, False]),
        'enable_regime_detection': trial.suggest_categorical('enable_regime_detection', [True, False]),
        'enable_fingpt': trial.suggest_categorical('enable_fingpt', [True, False]),
        # ML_USE_ADAPTIVE_LEARNING: HPO'da her zaman kapalı (data leakage önleme)
    }
    ...
    # Adaptive Learning parametreleri (her zaman optimize et, çünkü Phase 2 skip ediliyor ama Phase 1'de kullanılıyor)
    horizon_key = f'{horizon}d'
    feature_params[f'ml_adaptive_k_{horizon_key}'] = trial.suggest_float(f'ml_adaptive_k_{horizon_key}', 1.0, 3.0)
    feature_params[f'ml_pattern_weight_scale_{horizon_key}'] = trial.suggest_float(f'ml_pattern_weight_scale_{horizon_key}', 0.5, 2.0)
```

Özet:
- **Feature flags**: `enable_*` bayrakları ile hangi feature set’lerinin açık/kapalı olacağı optimize edilir.
- **Feature parametreleri**: Directional loss, seed bagging, meta stacking, adaptive learning, YOLO, smart ensemble vb.
- **Model seçimi**: `model_choice` ile `xgb`, `lgbm`, `cat`, `all` seçenekleri; sadece ortamda mevcut modeller arasından.
- **Hyperparametreler**: `xgb_*`, `lgb_*`, `cat_*` ana model parametreleri.

#### 3.2. Veri Alma ve Walk-Forward Split Üretimi

Her trial için her sembol üzerinde:

```580:607:scripts/optuna_hpo_with_feature_flags.py
    trial_symbol_metrics: Dict[str, Dict[str, Any]] = {}
    print(f"[hpo] Trial {trial.number}: Processing {len(symbols)} symbols: {symbols}", file=sys.stderr, flush=True)
    
    for sym in symbols:
        symbol_key = f"{sym}_{horizon}d"
        symbol_metric_entry: Dict[str, Any] = {
            'symbol': sym,
            'horizon': horizon,
            'split_metrics': []
        }
        trial_symbol_metrics[symbol_key] = symbol_metric_entry
        print(f"[hpo] Trial {trial.number}: Fetching prices for {sym}...", file=sys.stderr, flush=True)
        df = fetch_prices(engine, sym)
        ...
        # ⚡ FIX: Minimum data requirement - all horizons require 100 days
        min_required_days = 100  # All horizons require minimum 100 days
        if len(df) < min_required_days:
            print(f"[hpo] Trial {trial.number}: {sym} - len(df)={len(df)} < {min_required_days} (min required for {horizon}d), skipping", file=sys.stderr, flush=True)
            continue
        ...
        # ⚡ NEW: Generate multiple splits for walk-forward validation
        total_days = len(df)
        wfv_splits = generate_walkforward_splits(total_days, horizon, n_splits=4)
```

- Her sembol için **en az 100 gün veri zorunluluğu** var.
- Veri yeterliyse, 4 adet **WFV split** üretiliyor (`generate_walkforward_splits`).
- Split’ler üzerinde model eğitimi + tahmin yapılıyor; DirHit ve nRMSE hesaplanıyor.

#### 3.3. DirHit, nRMSE ve Score Hesabı + symbol_metrics

Split bazlı metriklerden sembol bazlı ve trial bazlı özetler:

```874:915:scripts/optuna_hpo_with_feature_flags.py
        # Average DirHit across all splits
        # ✅ FIX: Require at least 2 splits for reliable DirHit calculation
        # Single split DirHit is statistically unreliable
        avg_dirhit_value = None
        if len(split_dirhits) >= 2:
            avg_dirhit_value = float(np.mean(split_dirhits))
            print(
                f"[hpo] {sym} {horizon}d: Average DirHit across {len(split_dirhits)} splits: {avg_dirhit_value:.2f}% "
                f"(splits: {split_dirhits})",
                file=sys.stderr, flush=True
            )
            dirhits.append(avg_dirhit_value)
        elif len(split_dirhits) == 1:
            ...
            symbol_metric_entry['low_support_warning'] = True
        else:
            ...
            symbol_metric_entry['low_support_warning'] = True
        # Compute per-symbol nRMSE as the average across split nRMSE values
        avg_nrmse_value = None
        if split_nrmses_local:
            try:
                avg_nrmse_local = float(np.mean(split_nrmses_local))
                nrmses.append(avg_nrmse_local)
                avg_nrmse_value = avg_nrmse_local
            except Exception:
                pass
        symbol_metric_entry['avg_dirhit'] = avg_dirhit_value
        symbol_metric_entry['avg_nrmse'] = avg_nrmse_value
        symbol_metric_entry['split_count'] = len(symbol_metric_entry['split_metrics'])
        symbol_metric_entry['avg_model_metrics'] = _aggregate_model_metrics(symbol_metric_entry['split_metrics'])
```

Trial seviyesi skor ve user_attrs:

```929:947:scripts/optuna_hpo_with_feature_flags.py
    avg_dirhit = float(np.mean(dirhits))
    avg_nrmse = float(np.mean(nrmses)) if nrmses else float('inf')
    k = 6.0 if horizon in (1, 3, 7) else 4.0
    score = float(0.7 * avg_dirhit - k * (avg_nrmse if np.isfinite(avg_nrmse) else 3.0))
    print(f"[hpo] Trial {trial.number}: Average DirHit={avg_dirhit:.2f}% (from {len(dirhits)} symbols), nRMSE={avg_nrmse:.3f}, score={score:.2f}", file=sys.stderr, flush=True)
    try:
        trial.set_user_attr('avg_dirhit', avg_dirhit)
        trial.set_user_attr('avg_nrmse', avg_nrmse)
        trial.set_user_attr('model_choice', model_choice)
        # ✅ FIX: Store symbol_metrics in trial user_attrs so it can be retrieved later for best_trial_metrics
        if trial_symbol_metrics:
            ...
            trial.set_user_attr('symbol_metrics', trial_symbol_metrics)
```

**Önemli noktalar:**
- **Split seviyesinde** DirHit hesaplanıyor, ardından en az **2 split varsa** ortalaması alınıyor.
- `symbol_metrics[symbol_key]['avg_dirhit']` her sembol–ufuk için **sembol spesifik** DirHit’i tutuyor.
- `trial.user_attrs['avg_dirhit']` ise bir trial’da kullanılan **tüm sembollerin ortalaması**.  
  - Biz **sembol bazında** karar verirken **symbol_metrics**’i kullanıyoruz (bu, son fix’lerle garanti altına alındı).

---

### 4. Durum Yönetimi ve Cycle Mantığı (continuous_hpo_training_pipeline.py)

#### 4.1. TaskState ve State Dosyası

Her sembol–ufuk çifti için durum:

```605:621:scripts/continuous_hpo_training_pipeline.py
@dataclass
class TaskState:
    """Task state for a symbol-horizon pair"""
    symbol: str
    horizon: int
    status: str  # 'pending', 'hpo_in_progress', 'training_in_progress', 'completed', 'failed', 'skipped'
    hpo_completed_at: Optional[str] = None
    training_completed_at: Optional[str] = None
    best_params_file: Optional[str] = None
    hpo_dirhit: Optional[float] = None
    training_dirhit: Optional[float] = None  # backward compatibility (WFV)
    training_dirhit_wfv: Optional[float] = None
    training_dirhit_online: Optional[float] = None
    adaptive_dirhit: Optional[float] = None  # NEW: Adaptive learning DirHit (online DirHit with adaptive learning enabled)
    error: Optional[str] = None
    cycle: int = 0
    retry_count: int = 0  # ✅ FIX: Retry count for failed HPO tasks
```

- Durum dosyası: `STATE_FILE = Path('/opt/bist-pattern/results/continuous_hpo_state.json')`  
- Her entry key’i: `"{symbol}_{horizon}d"`.
- `status` alanı HPO ve training’in hangi aşamada olduğunu gösterir.

#### 4.2. Cycle Yönetimi (run_cycle)

Cycle mantığı:

```3990:4038:scripts/continuous_hpo_training_pipeline.py
    def run_cycle(self):
        """Run one complete cycle
        
        ✅ NEW APPROACH: Horizon-First processing (USER REQUEST)
        - Processes ALL symbols for each horizon before moving to next
        - Phase 1: All symbols for 1d → Phase 2: All symbols for 3d → ...
        - Incremental value delivery: 1d ready for all symbols first!
        - MAX_WORKERS: Symbols processed in parallel within each horizon phase
        """
        # ✅ CRITICAL FIX: Only increment cycle if current cycle is complete
        self.load_state()
        current_cycle = self.cycle
        ...
        if not has_incomplete and current_cycle > 0:
            ...
            self.cycle += 1
            ...
        else:
            ...
        logger.info(f"🔄 Starting cycle {self.cycle}")
        ...
        # ✅ FIX: Clean up old cycle files before starting new cycle
        if self.cycle > 1:  # Don't clean on first cycle
            logger.info("🧹 Cleaning up old cycle files (keeping only current cycle)...")
            self.cleanup_old_cycle_files(keep_cycles=1)
        ...
        # ✅ NEW: Horizon-First processing
        for horizon in HORIZON_ORDER:
            ...
            symbols_all = self.get_active_symbols()
            ...
            executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
            ...
            future = executor.submit(process_task_standalone, symbol, horizon, self.cycle)
```

Özet:
- **Cycle**, tüm sembol–ufuk işleri bittiğinde artar (tam cycle tamamlanmadan artmaz).
- Her cycle başında **eski cycle’a ait study/JSON dosyaları temizlenir** (current cycle hariç).
- Cycle içinde **horizon‑first**:
  - Önce tüm semboller için 1d,
  - sonra tüm semboller için 3d,
  - vb.  
  Böylece 1d ufku tüm semboller için önce hazır olur (iş açısından mantıklı).

---

### 5. HPO’nin Pipeline İçinde Çalıştırılması (run_hpo)

`ContinuousHPOPipeline.run_hpo` HPO script’ini subprocess ile çağırır:

```1180:1215:scripts/continuous_hpo_training_pipeline.py
    def run_hpo(self, symbol: str, horizon: int) -> Optional[Dict]:
        """Run HPO for a symbol-horizon pair"""
        try:
            logger.info(f"🔬 Starting HPO for {symbol} {horizon}d...")
            ...
            hpo_script = Path('/opt/bist-pattern/scripts/optuna_hpo_with_feature_flags.py')
            ...
            dry_run_trials = int(os.environ.get('DRY_RUN_TRIALS', '0'))
            trials_to_use = dry_run_trials if dry_run_trials > 0 else HPO_TRIALS
            timeout_to_use = 3600 if dry_run_trials > 0 else 900000
            ...
            cmd = [
                sys.executable,
                str(hpo_script),
                '--symbols', symbol,
                '--horizon', str(horizon),
                '--trials', str(trials_to_use),
                '--timeout', str(timeout_to_use)
            ]
            ...
            env = os.environ.copy()
            ...
            env['HPO_CYCLE'] = str(self.cycle)
```

Slot & CPU affinity, log dosyaları:

```1225:1270:scripts/continuous_hpo_training_pipeline.py
            # ✅ Acquire global HPO slot (limits cross-process concurrency)
            slot_context = HPOSlotContext()
            ...
            numa_node, cpu_list = _get_numa_node_and_cpus()
            numa_cmd, _, _ = _build_numa_cmd(cmd, numa_node, cpu_list)
            ...
            hpo_log_dir = Path('/opt/bist-pattern/logs/hpo_outputs')
            ...
            stdout_file = hpo_log_dir / f"{symbol}_{horizon}d_stdout.log"
            stderr_file = hpo_log_dir / f"{symbol}_{horizon}d_stderr.log"
            ...
            process = subprocess.Popen(
                numa_cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                text=True,
                cwd='/opt/bist-pattern',
                env=env,
                start_new_session=True,
                preexec_fn=lambda: os.nice(-5) if hasattr(os, 'nice') else None
            )
```

HPO tamamlandıktan sonra:
- Çıkış kodu kontrol ediliyor.
- `results/optuna_pilot_features_on_h{H}_c{CYCLE}_*.json` içinden ilgili HPO sonucu seçiliyor.
- Eğer JSON yoksa / bozuksa, **study dosyasından recovery** yapılmaya çalışılıyor.
- Son olarak, seçilen JSON’dan:
  - `best_params`,
  - `best_dirhit`,
  - `features_enabled`,
  - `feature_params` vb. okunup `run_hpo` return değeri olarak dönüyor.

JSON seçim ve filtreleme mantığı:

```1596:1719:scripts/continuous_hpo_training_pipeline.py
            # ✅ CRITICAL FIX: Check recent files for our symbol with timestamp validation
            ...
            valid_json_candidates = []
            
            for json_file in json_files[:50]:
                ...
                with open(json_file, 'r') as f:
                    hpo_data = json.load(f)
                ...
                # HPO tamamlanmış mı?
                n_trials = hpo_data.get('n_trials', 0)
                if not isinstance(n_trials, int) or n_trials < 10:
                    ...
                    continue
                ...
                best_dirhit = hpo_data.get('best_dirhit')
                candidate_best_score = hpo_data.get('best_value', 0)
                ...
                # LOW SUPPORT kontrolü (mask_count, mask_pct)
                best_trial_metrics = hpo_data.get('best_trial_metrics', {})
                symbol_key_check = f"{symbol}_{horizon}d"
                if symbol_key_check in best_trial_metrics:
                    symbol_metrics = best_trial_metrics[symbol_key_check]
                    split_metrics = symbol_metrics.get('split_metrics', [])
                    ...
                    total_mask_count = sum(s.get('mask_count', 0) for s in split_metrics)
                    mask_pcts = [s.get('mask_pct', 0.0) for s in split_metrics if s.get('mask_pct') is not None]
                    avg_mask_pct = np.mean(mask_pcts) if mask_pcts else 0.0
                    ...
                    _min_mc = int(os.getenv('HPO_MIN_MASK_COUNT', '0'))
                    _min_mp = float(os.getenv('HPO_MIN_MASK_PCT', '0.0'))
                    if total_mask_count < _min_mc or avg_mask_pct < _min_mp:
                        has_low_support = True
                        ...
                valid_json_candidates.append({...})
```

Ve aday seçimi:

```1769:1815:scripts/continuous_hpo_training_pipeline.py
            # ✅ FIX: Select the best JSON from all valid candidates
            # ✅ CRITICAL FIX: Prioritize candidates WITHOUT LOW SUPPORT
            # Priority: 1) No LOW SUPPORT, 2) Highest DirHit, 3) Highest best_value, 4) Most recent
            if valid_json_candidates:
                # First, separate candidates by LOW SUPPORT status
                candidates_with_support = [c for c in valid_json_candidates if not c.get('has_low_support', False)]
                candidates_low_support = [c for c in valid_json_candidates if c.get('has_low_support', False)]
                ...
                if candidates_with_support:
                    candidates_with_support.sort(
                        key=lambda x: (
                            x['best_dirhit'] if x['best_dirhit'] is not None else -1,
                            x['best_value'],
                            x['json_mtime']
                        ),
                        reverse=True
                    )
                    best_candidate = candidates_with_support[0]
                    ...
                else:
                    # ⚠️ FALLBACK: No candidates with sufficient support, but we still want to train the model
                    ...
                    candidates_low_support.sort(...)
                    best_candidate = candidates_low_support[0]
```

**Sonuç**:  
HPO sonucunu seçerken:
- Önce **5/2.5 filtresine göre yeterli destekli** (mask_count, mask_pct) adaylar,
- Sonra düşük destekli adaylar (fallback),
- İçlerinde de önce **DirHit**, sonra **score (best_value)**, sonra **timestamp** kriterleri kullanılır.

---

### 6. Training Süreci (run_training + _evaluate_training_dirhits)

#### 6.1. run_training: Ortam Değişkenleri ve Feature Bayrakları

```3071:3095:scripts/continuous_hpo_training_pipeline.py
    def run_training(self, symbol: str, horizon: int, best_params: Dict, hpo_result: Optional[Dict] = None) -> Optional[Dict[str, Optional[float]]]:
        """Run training with best params for a symbol-horizon pair
        ...
        """
        try:
            logger.info(f"🎯 Starting training for {symbol} {horizon}d with best params...")
            
            # Set parameters as environment variables
            from scripts.train_completed_hpo_with_best_params import set_hpo_params_as_env
            set_hpo_params_as_env(best_params, horizon)
            
            # ✅ CRITICAL FIX: Set feature flags from hpo_result (if available)
            # hpo_result contains 'features_enabled' dict with feature flags from best trial
            if hpo_result and 'features_enabled' in hpo_result:
                features_enabled = hpo_result['features_enabled']
                for key, value in features_enabled.items():
                    os.environ[key] = str(value)
                logger.info(f"🔧 {symbol} {horizon}d: Feature flags set from hpo_result: {len(features_enabled)} flags")
                ...
```

Ardından:

```3145:3213:scripts/continuous_hpo_training_pipeline.py
            # Set horizon
            os.environ['ML_HORIZONS'] = str(horizon)
            
            # ✅ UPDATED: Set feature flags from HPO best_params (features_enabled dict)
            # Use HPO-optimized feature flags, but always enable adaptive learning for training
            features_enabled = best_params.get('features_enabled', {})
            if features_enabled:
                # Set feature flags from HPO results
                os.environ['ENABLE_EXTERNAL_FEATURES'] = features_enabled.get('ENABLE_EXTERNAL_FEATURES', '1')
                ...
                os.environ['ENABLE_CATBOOST'] = features_enabled.get('ENABLE_CATBOOST', '0')
                ...
            else:
                # Fallback: Enable all features if features_enabled not found (backward compatibility)
                ...
            # ✅ HİBRİT YAKLAŞIM: Training'de adaptive learning KAPALI (HPO ile tutarlılık)
            os.environ['ML_USE_ADAPTIVE_LEARNING'] = '0'
            os.environ['ML_SKIP_ADAPTIVE_PHASE2'] = '1'
```

**Özet:**
- HPO JSON’dan gelen **best_params + features_enabled + feature_params**, hem:
  - `set_hpo_params_as_env` ile,
  - hem de doğrudan `os.environ` üzerinden
  eğitim sırasında **birebir aynen** kullanılıyor.
- Adaptive learning **HPO ile aynı şekilde kapalı** tutuluyor; incremental öğrenme etkisi **cycle’lar** üzerinden sağlanıyor.

#### 6.2. Eğitim Sonrası DirHit Hesabı (_evaluate_training_dirhits)

WFV evaluation:

```1955:1982:scripts/continuous_hpo_training_pipeline.py
    def _evaluate_training_dirhits(self, symbol: str, horizon: int, df: pd.DataFrame, best_params: Optional[Dict] = None, hpo_result: Optional[Dict] = None) -> Dict[str, Optional[float]]:
        """Evaluate DirHit after training using two modes:
        - wfv: adaptive OFF (no leakage)
        - online: adaptive OFF (HPO ile tutarlılık - hibrit yaklaşım)
        """
        import os
        results: Dict[str, Optional[float]] = {'wfv': None, 'online': None}
        total_days = len(df)
        ...
        min_total_days = max(100, (horizon + 10) * 5)
        
        if total_days < min_total_days:
            logger.warning(f"⚠️ {symbol} {horizon}d: Insufficient data for evaluation ({total_days} days, need {min_total_days})")
            return results
```

HPO ile aynı WFV split üretimi ve `evaluation_spec` eşleştirmesi:

```1984:2009:scripts/continuous_hpo_training_pipeline.py
        # ⚡ NEW: Use multiple splits for walk-forward validation (same as HPO)
        from scripts.optuna_hpo_with_feature_flags import generate_walkforward_splits, calculate_dynamic_split
        wfv_splits = generate_walkforward_splits(total_days, horizon, n_splits=4)
        ...
        # ✅ NEW: If evaluation_spec present in HPO JSON, override splits and thresholds to ensure parity
        eval_spec = None
        ...
        if isinstance(eval_spec, dict):
            # Set DirHit threshold from spec (fallback to default)
            ...
            # Optionally mirror mask thresholds for any gating logic downstream
            try:
                if 'min_mask_count' in eval_spec:
                    os.environ['HPO_MIN_MASK_COUNT'] = str(int(eval_spec['min_mask_count']))
                if 'min_mask_pct' in eval_spec:
                    os.environ['HPO_MIN_MASK_PCT'] = str(float(eval_spec['min_mask_pct']))
            except Exception:
                pass
            # Override WFV splits using indices if provided
            ...
```

Ve WFV DirHit ortalaması:

```2487:2519:scripts/continuous_hpo_training_pipeline.py
            # Average DirHit, nRMSE, and Score across all splits
            # ✅ FIX: Require at least 2 splits for reliable DirHit calculation (same as HPO)
            # Single split DirHit is statistically unreliable
            if len(split_dirhits) >= 2:
                avg_dirhit = float(np.mean(split_dirhits))
                avg_nrmse = float(np.mean(split_nrmses)) if split_nrmses else float('inf')
                avg_score = float(np.mean(split_scores)) if split_scores else 0.0
                logger.info(f"✅ {symbol} {horizon}d WFV: Average across {len(split_dirhits)} splits: DirHit={avg_dirhit:.2f}%, nRMSE={avg_nrmse:.3f}, Score={avg_score:.2f}")
                results['wfv'] = avg_dirhit
                results['wfv_nrmse'] = avg_nrmse
                results['wfv_score'] = avg_score
            elif len(split_dirhits) == 1:
                ...
                results['wfv'] = None
                ...
                results['low_support_warning'] = True
            else:
                ...
                results['wfv'] = None
                ...
                results['low_support_warning'] = True
                # ✅ FALLBACK: Try to find best params with 0/0.0 filter (no filter)
                if hpo_result and 'json_file' in hpo_result:
                    ...
                    fallback_params = find_fallback_best_params(study_db, symbol, horizon)
                    if fallback_params:
                        ...
                        results['fallback_best_params'] = fallback_params
                        results['fallback_available'] = True
```

**Özet:**
- Eğitim sonrası DirHit, **HPO ile aynı WFV mantığıyla** (aynı split’ler, aynı filtreler) hesaplanıyor.
- En az **2 split** zorunluluğu var; aksi durumda DirHit **LOW_SUPPORT** olarak işaretleniyor.
- Eğer filtreye takılan semboller varsa, **0/0.0 fallback** ile study’den en iyi trial parametreleri ek olarak bulunup `results['fallback_best_params']` içinde raporlanıyor.

---

### 7. Düşük Destek Fallback Mantığı

#### 7.1. Study Tabanlı Fallback (find_fallback_best_params.py)

```23:93:scripts/find_fallback_best_params.py
def find_fallback_best_params(study_db: Path, symbol: str, horizon: int) -> Optional[Dict]:
    """Find best params using 0/0.0 filter (no filter) as fallback
    ...
    """
    try:
        study = optuna.load_study(
            study_name=None,
            storage=f"sqlite:///{study_db}"
        )
        
        symbol_key = f"{symbol}_{horizon}d"
        best_trial = None
        best_filtered_score = float('-inf')
        
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            # Get split metrics
            symbol_metrics = trial.user_attrs.get('symbol_metrics', {})
            if symbol_key not in symbol_metrics:
                continue
            split_metrics = symbol_metrics[symbol_key].get('split_metrics', [])
            if not split_metrics:
                continue
            # Apply 0/0.0 filter (no filter) - include all splits
            filtered_dirhits = []
            for split in split_metrics:
                dirhit = split.get('dirhit')
                if dirhit is not None:
                    filtered_dirhits.append(dirhit)
            # Need at least 1 split
            if len(filtered_dirhits) == 0:
                continue
            # Calculate filtered average DirHit
            filtered_score = sum(filtered_dirhits) / len(filtered_dirhits)
            if filtered_score > best_filtered_score:
                best_filtered_score = filtered_score
                best_trial = trial
        ...
        return {
            'best_params': best_params,
            'best_trial_number': best_trial.number,
            'best_value': best_filtered_score,
            'features_enabled': features_enabled,
            'feature_params': feature_params,
            'filter_used': {'min_mask_count': 0, 'min_mask_pct': 0.0},  # Fallback filter
            'is_fallback': True
        }
```

Bu fallback:
- **Hiçbir split 5/2.5 filtresini geçemeyen** semboller için,
- En az 1 split DirHit’i kullanarak 0/0.0 filtre ile **en iyi trial’ı** bulur,
- Bu parametreler **LOW_SUPPORT uyarısıyla** birlikte training değerlendirmesine eklenir.

#### 7.2. JSON Tabanlı Fallback (recreate_all_json_from_study_with_filter.py)

```280:331:scripts/recreate_all_json_from_study_with_filter.py
def create_json_from_filtered_trial(db_file: Path, symbol: str, horizon: int, cycle: int,
                                    min_mask_count: int = 5, min_mask_pct: float = 2.5,
                                    min_valid_splits: int = 2, dry_run: bool = False,
                                    timeout_seconds: int = 300) -> Optional[Path]:
    """Create JSON file from study database using filtered best trial"""
    ...
    filtered_trial, filtered_score = find_best_trial_with_timeout(
        db_file, symbol, horizon, min_mask_count, min_mask_pct, timeout_seconds
    )
    ...
    # Get symbol-specific avg_dirhit
    symbol_key = f"{symbol}_{horizon}d"
    symbol_metrics = filtered_trial.user_attrs.get('symbol_metrics', {})
    symbol_metric = symbol_metrics.get(symbol_key, {}) if isinstance(symbol_metrics, dict) else {}
    symbol_avg_dirhit = symbol_metric.get('avg_dirhit') if isinstance(symbol_metric, dict) else None
    ...
    if symbol_avg_dirhit is not None:
        best_dirhit = float(symbol_avg_dirhit)
    else:
        best_dirhit = filtered_score
    ...
```

Bu script:
- HPO study’den **5/2.5 ve minimum 2 split** şartlarını sağlayan trial’ı bularak JSON üretir.
- Eğer böyle bir trial yoksa, üst seviyede **0/0.0 fallback** ile tekrar denenir (komut seviyesinde).

---

### 8. Akış ve Zincir (Uçtan Uca Özet)

1. **Veri katmanı**:
   - Sembollerin OHLC verileri PostgreSQL’de tutulur.
   - HPO ve training bu veriyi `fetch_prices` vb. fonksiyonlarla çeker.
2. **HPO süreci (optuna_hpo_with_feature_flags.py)**:
   - Her trial için:
     - Feature flags + iç parametreler + hyperparametreler örneklenir.
     - En az 100 gün veri varsa 4 WFV split üretilir.
     - Her split için model eğitilir, DirHit ve nRMSE hesaplanır.
     - En az 2 split varsa DirHit ortalaması alınır; aksi durumda DirHit sembol seviyesinde **LOW_SUPPORT** olur.
     - Tüm sembollerin DirHit’leri `avg_dirhit` olarak, sembol bazlı detaylar `symbol_metrics` olarak kaydedilir.
3. **HPO orkestrasyonu (ContinuousHPOPipeline.run_hpo)**:
   - HPO script’i subprocess olarak başlatılır (CPU affinity, slot kontrolü, log dosyaları).
   - Çıktı JSON’ları:
     - **5/2.5 + min_valid_splits ≥ 2** filtrelerine göre,
     - LOW_SUPPORT adaylar ikinci planda olmak üzere,
     - DirHit → score → zaman önceliğiyle seçilir.
4. **Durum güncelleme (TaskState)**:
   - HPO tamamlanınca `hpo_completed_at`, `hpo_dirhit`, `best_params_file`, `cycle` güncellenir.
5. **Training (run_training)**:
   - HPO’dan gelen `best_params`, `features_enabled`, `feature_params` ortam değişkenlerine yazılır.
   - Adaptive learning **kapalı**, feature bayrakları HPO ile uyumlu.
   - Eğitim yapılır, model disk’e kaydedilir.
6. **Eğitim sonrası değerlendirme (_evaluate_training_dirhits)**:
   - HPO ile aynı WFV split’leri (`evaluation_spec` ile tam eşleştirilmiş) kullanılır.
   - En az 2 split varsa DirHit hesaplanır; değilse WFV DirHit `None` + LOW_SUPPORT flag.
   - Gerekirse `find_fallback_best_params` ile 0/0.0 fallback parametreler bulunur ve raporlanır.
7. **State & raporlama**:
   - `continuous_hpo_state.json` içinde:
     - `status`, `hpo_dirhit`, `training_dirhit_wfv`, `training_dirhit_online`, `adaptive_dirhit`,
     - `cycle`, `retry_count`, `error` gibi alanlar güncel tutulur.
   - `show_hpo_progress.py`:
     - Hem state hem de study dosyalarından,
     - **sembol spesifik** DirHit’i `symbol_metrics[symbol_key]['avg_dirhit']` üzerinden okur,
     - HPO vs Training DirHit karşılaştırmasını ekrana basar.
8. **Cycle döngüsü**:
   - Tüm semboller tüm horizon’lar için tamamlandığında cycle artar.
   - Eski cycle’a ait study/JSON dosyaları temizlenir.
   - Yeni cycle’da **güncellenmiş veriyle** HPO yeniden çalışır; bu da incremental öğrenme etkisi yaratır.

---

### 9. Mantıksal ve Kural Bazlı Değerlendirme

#### 9.1. Mantıksal Tutarlılık

- **Sembol bazlı HPO**:
  - Pipeline her HPO çağrısında **tek sembol** gönderiyor (`--symbols {SYMBOL}`), bu yüzden:
    - `study.best_trial` o sembol–ufuk için en iyi trial,
    - `symbol_metrics[symbol_key]['avg_dirhit']` de **doğrudan o sembole ait** DirHit.
  - `show_hpo_progress.py` ve `continuous_hpo_training_pipeline.py` artık DirHit’i **bu sembol spesifik metrikten** okuyor; bu, mantıksal olarak doğru.

- **Filtre ve split sayısı**:
  - HPO ve training’de **aynı DirHit tanımı** ve **aynı WFV split stratejisi** kullanılıyor.
  - Hem HPO, hem training için **en az 2 split** şartı var; tek split sonuçları istatistiksel olarak güvensiz olduğu için hariç tutuluyor.
  - Bu, kullanıcı talebiyle birebir uyumlu ve istatistiksel açıdan mantıklı.

- **LOW_SUPPORT semboller**:
  - 5/2.5 filtresine takılan (mask_count/mask_pct düşük) semboller:
    - Önce HPO JSON seçiminde **deprioritize** ediliyor (varsa destekli adaylar tercih ediliyor).
    - Hiç destekli aday yoksa:
      - HPO DirHit **uyarı ile** kabul ediliyor,
      - Training DirHit’in daha güvenilir olacağı explicit log mesajlarıyla belirtiliyor.
  - Ayrıca `find_fallback_best_params` ile 0/0.0 filtreli fallback parametreler hesaplanıyor; bu da **“hiç model kalmasın”** riskini azaltıyor.

#### 9.2. İş/Kural Perspektifi

- **Veri kalitesi**:
  - Hem HPO hem training tarafında minimum gün sayısı ve mask filtreleriyle veri kalitesi korunuyor.
  - **En az iki split** ile karar verilmesi, iş açısından **daha stabil ve güvenilir** bir performans ölçümü sağlıyor.

- **Süreklilik ve geri kazanım (resilience)**:
  - HPO yarım kaldığında:
    - Study dosyasından trial sayısı kontrol ediliyor.
    - Yeterli trial yoksa, aynı study’den **warm‑start** ile devam ediliyor.
    - Yeterli trial varsa, JSON olmasa bile study’den JSON **recovery** yapılıyor.
  - Bu, kurumsal ortamda beklenen **dayanıklılık ve otomatik toparlanma** davranışına uygun.

- **Cycle yönetimi**:
  - Cycle numarası sadece **mevcut cycle tamamen bittiğinde** artıyor; bu, raporlama ve versiyonlama için mantıklı.
  - Eski cycle dosyalarının otomatik temizlenmesi, disk kullanımını ve kafa karışıklığını azaltıyor.

- **Ölçeklenebilirlik ve kaynak yönetimi**:
  - Global HPO slot’ları, NUMA/CPU affinity, log yönlendirme gibi detaylar:
    - Yüksek sembol sayısında bile sistemi **stabil ve performanslı** tutmayı hedefliyor.

- **Şeffaflık ve denetlenebilirlik**:
  - Tüm kritik kararlar:
    - HPO JSON’ları,
    - Study dosyaları,
    - `continuous_hpo_state.json`,
    - `show_hpo_progress.py` çıktıları üzerinden izlenebilir.
  - LOW_SUPPORT durumu, fallback kullanımı ve filtre değerleri log’larda ve state’te açıkça işaretleniyor.

#### 9.3. Potansiyel Geliştirme Alanları

- LOW_SUPPORT durumunda:
  - Şu an fallback 0/0.0 DirHit ile çalışıyor; bu, veri çok zayıfsa halen riskli olabilir.  
    - Geliştirme: **min split sayısı ≥ 2** kuralını fallback’te de zorlamak veya DirHit yerine **nRMSE ağırlıklı** bir skor kullanmak.
- İş kuralı olarak:
  - Belirli bir eşik altındaki DirHit’ler için (örneğin < 45%) modelin **otomatik olarak “kullanılamaz” işaretlenmesi** (örneğin prediction sisteminde devre dışı bırakma) düşünülebilir.

Genel olarak, mevcut mimari:
- **Sembol bazlı**,  
- **WFV temelli**,  
- **filtre ve split sayısı açısından tutarlı**,  
- **cycle bazlı incremental iyileşme** sağlayan,  
mantıksal ve kurumsal açıdan güçlü bir HPO + Training zinciri sunuyor.


