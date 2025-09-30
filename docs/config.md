# Configuration Reference

## Systemd Overrides (single source of truth in production)
- ENABLE_YOLO: Enable visual pattern detection (True/False). Source: `50-yolo.conf`
- YOLO_MIN_CONF: Min confidence threshold (e.g., 0.12). Source: `50-yolo.conf`
- YOLO_MODEL_PATH: Model path. Source: `50-yolo.conf`
- YOLO_CONFIG_DIR: Ultralytics cache/config dir. Source: `50-yolo.conf`
- YF_MAX_RETRIES: Yahoo chart/download retries. Source: `60-collector-retries.conf`
- YF_YFINANCE_TRIES: yfinance session tries. Source: `60-collector-retries.conf`
- YF_TRY_ALT_PERIODS: Try alternative periods (0/1). Source: `60-collector-retries.conf`
- YF_SINGLE_SESSION: Force session (cffi/default). Source: `60-collector-retries.conf`
- ML_TRAIN_INTERVAL_CYCLES: Train interval cycles. Source: `45-ml-train.conf`
- ML_TRAIN_PER_CYCLE: Train batch per cycle. Source: `45-ml-train.conf`

## Application (read by code)
- DEBUG_VERBOSE: Verbose debug logs (0/1). Used in: `scheduler.py`, `bist_pattern/core/unified_collector.py`, `pattern_detector.py`
- API cache TTLs: `API_CACHE_TTL_WATCHLIST` (seconds). Used in `app.py`
- INTERNAL_ALLOW_LOCALHOST: Allow localhost access to internal endpoints (0/1). Used in `app.py`
- STATUS_GRACE_SECONDS: Cache grace for status endpoint. Used in `app.py`
- BIST_LOG_PATH: Log directory (default `/opt/bist-pattern/logs`). Used in `app.py`, scheduler, etc.
- PIPELINE_MODE: Automation mode (default `CONTINUOUS_FULL`). Set at runtime by app/automation endpoints.
- RUNNING_FLAG_KEY, RUNNING_FLAG_TTL, RUNNING_FLAG_HEARTBEAT_SECONDS: Automation watchdog. Used in `scheduler.py`.
- MAX_SYMBOLS_PER_CYCLE, SYMBOL_SLEEP_SECONDS, SYMBOL_DELAY_SECONDS: Collector/scheduler pacing. Used in `scheduler.py`, `unified_collector.py`.
- PATTERN_RESULT_CACHE_TTL, PATTERN_RESULT_CACHE_MAX_SIZE: Pattern result cache. Used in `pattern_detector.py`.
- DATA_CACHE_TTL, DF_CACHE_MAX_SIZE: DataFrame cache. Used in `pattern_detector.py`.
- CONSENSUS_DELTA, VOL_HIGH_THRESHOLD, VOL_LOW_THRESHOLD: Signal weighting. Used in `pattern_detector.py`.
- YF_*: Backoff and sessions (also can be overridden via systemd). Used in `unified_collector.py`.
- NEWS_CACHE_TTL, NEWS_MAX_ITEMS, NEWS_LOOKBACK_HOURS, RSS_TIMEOUT, NEWS_SOURCES: Async RSS. Used in `rss_news_async.py`.

## Notes
- Production values should be defined in systemd override files under `/etc/systemd/system/bist-pattern.service.d/`.
- Avoid duplicating the same setting in multiple places. Prefer override files over ad-hoc exports.
- For troubleshooting, set `DEBUG_VERBOSE=1` temporarily; admin live log düzeni değişmez.
