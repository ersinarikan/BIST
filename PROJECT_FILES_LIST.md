# Projede Kullanılan Tüm Dosyalar (app.py'den başlayarak)

## Ana Giriş Noktası
- **app.py** - Flask uygulama factory ve ana entry point

## Core Modüller (Doğrudan app.py'den import edilenler)

### Config & Models
- **config.py** - Flask konfigürasyon
- **models.py** - SQLAlchemy modelleri (db, User, Stock, StockPrice)

### BIST Pattern Core
- **bist_pattern/core/config_manager.py** - Merkezi konfigürasyon yönetimi
- **bist_pattern/utils/error_handler.py** - Hata yönetimi

### Pattern Detection
- **pattern_detector.py** - Hibrit pattern detection sistemi
- **advanced_patterns.py** - Gelişmiş pattern detection (opsiyonel)
- **visual_pattern_detector.py** - YOLO tabanlı görsel pattern detection (opsiyonel)
- **bist_pattern/core/pattern_validator.py** - Pattern validation sistemi
- **bist_pattern/core/basic_pattern_detector.py** - Temel pattern detection

### ML Sistemleri
- **ml_prediction_system.py** - Temel ML prediction sistemi (opsiyonel)
- **enhanced_ml_system.py** - Gelişmiş ML sistemi
- **bist_pattern/core/ml_coordinator.py** - ML koordinasyonu

### Automation & Data Collection
- **working_automation.py** - Otomasyon pipeline
- **bist_pattern/core/unified_collector.py** - Birleşik veri toplayıcı
- **yahoo_finance_enhanced.py** - Yahoo Finance veri çekici
- **rss_news_async.py** - RSS haber sistemi
- **fingpt_analyzer.py** - FinGPT sentiment analizi

## BIST Pattern Package Yapısı

### Blueprints (API Endpoints)
- **bist_pattern/blueprints/register_all.py** - Tüm blueprint'leri kaydeden modül
- **bist_pattern/blueprints/api_internal.py** - Internal API endpoints
- **bist_pattern/blueprints/auth.py** - Authentication endpoints
- **bist_pattern/blueprints/web.py** - Web sayfaları
- **bist_pattern/blueprints/api_public.py** - Public API endpoints
- **bist_pattern/blueprints/api_metrics.py** - Metrics API
- **bist_pattern/blueprints/api_recent.py** - Recent data API
- **bist_pattern/blueprints/api_watchlist.py** - Watchlist API
- **bist_pattern/blueprints/api_simulation.py** - Simulation API
- **bist_pattern/blueprints/api_health.py** - Health check API
- **bist_pattern/blueprints/api_batch.py** - Batch API
- **bist_pattern/blueprints/admin_dashboard.py** - Admin dashboard

### Core Modüller
- **bist_pattern/core/broadcaster.py** - WebSocket broadcasting
- **bist_pattern/core/cache.py** - Cache yönetimi
- **bist_pattern/core/csrf_security.py** - CSRF güvenliği
- **bist_pattern/core/db_manager.py** - Database yönetimi
- **bist_pattern/core/news_sentiment_system.py** - Haber sentiment sistemi
- **bist_pattern/core/pattern_coordinator.py** - Pattern koordinasyonu

### Utils
- **bist_pattern/utils/debug_utils.py** - Debug yardımcıları
- **bist_pattern/utils/broadcast.py** - Broadcast yardımcıları
- **bist_pattern/utils/file_permissions.py** - Dosya izin yönetimi
- **bist_pattern/utils/log_tailer.py** - Log takip
- **bist_pattern/utils/oauth_setup.py** - OAuth kurulumu
- **bist_pattern/utils/param_store_lock.py** - Param store kilitleme
- **bist_pattern/utils/symbols.py** - Symbol yardımcıları

### Features
- **bist_pattern/features/cleaning.py** - Veri temizleme
- **bist_pattern/features/pattern_features.py** - Pattern özellikleri
- **bist_pattern/features/selection.py** - Feature seçimi

### Simulation
- **bist_pattern/simulation/forward_engine.py** - Forward simulation engine
- **simulation_engine.py** - Simulation engine (legacy)

### WebSocket
- **bist_pattern/websocket/events.py** - WebSocket event handlers

### Settings & Extensions
- **bist_pattern/settings.py** - Ayarlar
- **bist_pattern/extensions.py** - Flask extensions
- **bist_pattern/__init__.py** - Package initialization

### API Modules
- **bist_pattern/api_modules/automation.py** - Automation API modülü
- **bist_pattern/api_modules/__init__.py** - API modules init

## Scripts (HPO & Training)

### HPO Scripts
- **scripts/optuna_hpo_with_feature_flags.py** - Optuna HPO with feature flags
- **scripts/continuous_hpo_training_pipeline.py** - Sürekli HPO pipeline
- **scripts/show_hpo_progress.py** - HPO ilerleme gösterimi

### Training Scripts
- **scripts/train_completed_hpo_with_best_params.py** - Best params ile training
- **scripts/retrain_all_completed_symbols.py** - Tüm sembolleri yeniden eğit
- **scripts/retrain_high_discrepancy_symbols.py** - Yüksek farklılık sembolleri

### Analysis Scripts
- **scripts/comprehensive_hpo_training_consistency_analysis.py** - HPO tutarlılık analizi
- **scripts/create_json_from_study.py** - Study'den JSON oluştur
- **scripts/fix_features_enabled_in_json.py** - JSON feature flags düzelt
- **scripts/recreate_all_json_from_study_with_filter.py** - Filter ile JSON yeniden oluştur
- **scripts/find_fallback_best_params.py** - Fallback best params bul
- **scripts/ensemble_utils.py** - Ensemble yardımcıları

### Update Scripts
- **scripts/update_all_study_hpo_dirhit.py** - Study HPO dirhit güncelle
- **scripts/update_in_progress_hpo_dirhit_from_study.py** - In-progress dirhit güncelle
- **scripts/update_state_hpo_dirhit_from_json.py** - State dirhit güncelle
- **scripts/update_json_with_10_5_filter.py** - JSON filter güncelle
- **scripts/update_json_with_filtered_best_params.py** - Filtered best params güncelle
- **scripts/verify_and_update_all_completed_symbols.py** - Completed symbols doğrula ve güncelle

## Diğer Dosyalar

### Legacy/Alternate Implementations
- **enhanced_ml_async.py** - Async ML sistemi (alternatif)
- **visual_pattern_async.py** - Async visual pattern (alternatif)
- **yfinance_gevent_native.py** - Gevent native yfinance wrapper
- **news_provider.py** - Haber sağlayıcı (legacy)

### Configuration
- **gunicorn.conf.py** - Gunicorn konfigürasyonu

## Toplam Dosya Sayısı

### Ana Modüller: ~15 dosya
### BIST Pattern Package: ~40+ dosya
### Scripts: ~20+ dosya
### Toplam: ~75+ aktif kullanılan Python dosyası

## Notlar

1. **Opsiyonel Modüller**: Bazı modüller opsiyonel olarak yüklenir (advanced_patterns, visual_pattern_detector, ml_prediction_system)
2. **Scripts**: Scripts klasöründeki dosyalar genellikle CLI araçlarıdır ve doğrudan app.py'den import edilmez
3. **Unused Scripts**: `scripts/unused_scripts_backup/` klasöründeki dosyalar kullanılmıyor
4. **Migrations**: `migrations/` klasöründeki dosyalar Alembic migration dosyalarıdır

