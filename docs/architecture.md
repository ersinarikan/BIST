Architecture Overview
======================

Scope
-----
This document captures the current architecture, key modules, runtime configuration, and known pain points for the BIST Pattern Detection system. It will be iteratively refined alongside code audits.

Runtime Topology
----------------
- Nginx → Gunicorn (worker_class=gevent) → Flask app (`app.py`) with Flask-SocketIO
- Systemd manages environment (override.d drop-ins); no .env loading at runtime
- PostgreSQL for persistence via SQLAlchemy
- Scheduler (in-process) drives continuous pipeline: collect → analyze → sleep 5m
- Socket.IO streams logs/health to `templates/dashboard.html` / `user_dashboard.html`

Key Components
--------------
- app.py: routes, auth, CSRF config, Socket.IO setup, REST/WS endpoints
- scheduler.py: continuous automation loop, watchdog, pipeline status/history
- advanced_collector.py: Yahoo Finance fetch, backfill, upsert, batching/parallelism
- pattern_detector.py: hybrid pattern analysis, progress/heartbeat logging
- models.py: ORM models for Stock, StockPrice, users, and related entities
- templates/: dashboard & user dashboard, badges, recent tasks, WS clients
- gunicorn.conf.py: worker class/timeouts, 1 worker, gevent-based
- systemd drop-ins: service env config; INTERNAL_API_TOKEN, DB, secrets, YF UA
- nginx site: WS upgrade headers, proxying

Endpoint Map (app.py)
---------------------
- Public pages: `/`, `/api`, `/dashboard`, `/login`, `/logout`, `/user`, `/stocks`, `/analysis`
- Data APIs: `/api/stocks`, `/api/stock-prices/<symbol>`, `/api/pattern-analysis/<symbol>`, `/api/pattern-summary`, `/api/stocks/search`, `/api/visual-analysis/<symbol>`, `/api/system-info`
- Watchlist: `GET/POST /api/watchlist`, `DELETE /api/watchlist/<symbol>` (CSRF exempt on mutating)
- ML: `/api/ml-prediction/<symbol>`, `/api/train-ml-model/<symbol>` (optional layers gated by env)
- Alerts: `/api/alerts/*` (configs/history/start/stop/test)
- Automation:
  - `POST /api/automation/start` (admin, CSRF exempt)
  - `POST /api/automation/stop` (admin, CSRF exempt)
  - `GET  /api/automation/status` (admin)
  - `GET  /api/automation/health` (admin)
  - `POST /api/automation/run-task/<task>` (admin, CSRF exempt)
  - `GET  /api/automation/pipeline-history`
  - `GET  /api/recent-tasks`
- Internal-only: `POST /api/internal/broadcast-log`, `POST /api/internal/broadcast-user-signal`, `POST /api/internal/automation/<action>` (require `X-Internal-Token`)

Socket.IO & CSRF
----------------
- Socket.IO initialized with gevent, `ping_timeout=30`, `ping_interval=20`, logging disabled
- CSRF exempts Socket.IO transport endpoint and mutating internal/admin endpoints
- Dashboard uses `socket = io({ path: '/socket.io' });` and polls `recent-tasks` and `pipeline-history`

Automation Flow (scheduler.py)
------------------------------
- Single mode: CONTINUOUS_FULL
  1) Collect all stocks (parallel batches)
  2) Run AI analysis over active symbols
  3) Sleep 300s with periodic heartbeats (every 30s)
- Watchdog: if thread dies or idle > MAX_IDLE_SECONDS (default 900), auto-restart
- History JSON (`logs/pipeline_status.json`) records phase start/end/errors; dashboard reads it

Collector Highlights (advanced_collector.py)
-------------------------------------------
- Normalizes symbols, optional validation, chooses Yahoo range by period (chart API → yfinance fallbacks)
- Backfill: if data < MIN_HISTORY_DAYS (default 365), request longer period (e.g., 2y)
- Upsert: creates or updates today's price, ensures integrity
- Config from env via `config.py`: workers, delay range, retries, backoff, batch sleep
- Environment knobs (read from systemd env via `config.py`):
  - COLLECTOR_MAX_WORKERS (int)
  - COLLECTOR_DELAY_RANGE ("min,max" seconds)
  - YF_MAX_RETRIES (int), YF_BACKOFF_BASE_SECONDS (float)
  - COLLECTOR_BATCH_SIZE (int), BATCH_SLEEP_SECONDS (int)
  - MIN_HISTORY_DAYS (int), COLLECTION_PERIOD/PRIORITY_PERIOD
- Fetch order and resilience:
  1) Yahoo chart API (query2 → query1) with JSON parse guards
  2) yfinance Ticker.history (rotating UA via headers)
  3) Fallback: yfinance.download
  4) Final chart API attempt (mapped range)
- Data quality:
  - Drops rows without Close; fills missing O/H/L from Close; zero-volume default
  - Skips outliers above DB numeric(10,4) range; daily upsert for same date
  - Broadcasts granular progress with `INTERNAL_API_TOKEN` if provided

Pattern Detector (pattern_detector.py)
-------------------------------------
- Hybrid approach: basic TA (SMA/EMA/RSI/MACD/BB), optional advanced patterns, optional visual patterns
- Caches per-minute analysis results with TTL; broadcasts progress via `app.broadcast_log`
- Generates overall signal from weighted sub-signals; returns JSON-safe primitives
- Heartbeat/logging: logs analysis start/end; progress broadcast per N symbols (currently ~50; target 25 for smoother UI)

Frontend Behavior (templates)
-----------------------------
- dashboard.html: WebSocket init `io({ path: '/socket.io' })`, logs under Live System Logs, metrics updated every 15–20s
- Recent Tasks: pulls `/api/recent-tasks` and shows a single running task from pipeline history; Current Phase badge from `/api/automation/pipeline-history`
- user_dashboard.html: kullanıcı tarafı izleme; WS ile `pattern_analysis` güncellemelerini anlık işler, tahminleri `/api/user/predictions/<symbol>` ile çeker

Security & CSRF
---------------
- Mutating admin/internal uçlar CSRF exempt; internal uçlar `X-Internal-Token` ile korunur
- Socket.IO polling CSRF’den muaf (400 hatalarını engeller)

Models (models.py)
------------------
- users, stocks, stock_prices (unique (stock_id,date), index idx_stock_date), watchlist
- Simulation: simulation_sessions, simulation_trades, portfolio_snapshots with basic metrics helpers
- All timestamps are UTC; numeric types sized for price/volume limits

Runtime Configs
---------------
- Gunicorn (gunicorn.conf.py): workers=1, worker_class=gevent, timeout=600, keepalive=30, max_requests=0; logs under `/opt/bist-pattern/logs`
- Nginx: WS upgrade headers required in `location /` (Upgrade/Connection headers, proxy_buffering off)
- Systemd drop-ins (authoritative env): INTERNAL_API_TOKEN, DATABASE_URL, FLASK_SECRET_KEY, YF_USER_AGENTS (quoted), PIPELINE_MODE, collector knobs, caches (MPLCONFIGDIR, TRANSFORMERS_CACHE, HF_HOME)
- Config matrix: `config.py` reads only OS env; no dotenv

Known Issues Tracked
--------------------
- Align `recent_tasks` and badge state with pipeline history (avoid stale UI)
- Reduce log noise on WS disconnects (gevent improved it)
- Verify env var parsing for multi-value UAs (handled by 31-yf.conf, quoted)
- Consider Yahoo throttle handling (shared Session + 429/5xx cooldown; UA rotation already; candidate iyileştirme)
- Health panel “WebSocket connected” ibaresi her zaman doğru değil; gerçek bağlantı durumunu yansıtması için UI/WS event senkronizasyonu değerlendirilmeli
- Improve AI heartbeat frequency to every ~25 symbols (smoother progress)
- Remove/deprecate old scheduler cron-like jobs from UI hints; single continuous loop is canonical
- Add baseline performance metrics (collector success ratio, retries, analysis/sec) to logs and dashboard

Önceliklendirilmiş Sorunlar ve Güvenli Çözüm Önerileri
------------------------------------------------------
1) UI durum senkronizasyonu (yüksek öncelik)
- Sorun: "Current Phase" rozeti ve "Recent Tasks" bazen geç kalıyor ya da IDLE görünüyor.
- Çözüm: `scheduler.py` faz başlangıçlarında tek bir source-of-truth yazma (yapıldı); dashboard `updatePipelineInfo()` zaten son "start" kaydını okuyor. İnterval 15–20 sn → 10–15 sn’a çekilebilir.

2) AI heartbeat sıklığı (orta-üst)
- Sorun: 50 sembolde bir yayın, panelde ilerleme hissini zayıflatıyor.
- Çözüm: `scheduler.py` içinde 25 sembolde bir `broadcast_log` gönderecek şekilde parametrize etmek.

3) Collector throttle/geri basınç (orta)
- Sorun: 429/5xx durumlarında yeniden denemeler artıyor.
- Çözüm (non-breaking): mevcut backoff=2.0 ile eşzamanlılığı 4 worker’da tutmak; sonraki adımda paylaşımlı Session + 429 cooldown eklemek (koddaki sıralama korunur).

4) Backfill gün sayımı (orta)
- Sorun: Kayıt sayısı ile gün farkı aynı şey değil.
- Çözüm: Son tarih ile ilk tarih arasında gün farkını ölçerek 252 eşiğini kontrol etmek; mevcut mantıkla uyumlu non-breaking ek doğrulama.

5) Log gürültüsü (orta)
- Sorun: WS kapanış mesajları azalsa da hâlâ zaman zaman görünüyor.
- Çözüm: Gunicorn log seviyesini info’da tutup, Socket.IO error handler’ında benign kapatma kodlarını filtrelemek.

6) Kullanılmayan modüller (düşük-orta)
- Sorun: Eski scheduler akış referansları ve opsiyonel ML uçları UI’da kafa karıştırabilir.
- Çözüm: UI buton yazıları ve açıklamalarını CONTINUOUS FULL akışına uyarlamak; ML uçlarını env ile kapalı tutmak (mevcut).

7) Gözlenebilirlik (düşük-orta)
- Sorun: Başarı/başarısız toplama oranları ve analiz hızları panelde yok.
- Çözüm: Collector sonunda basit metrikleri `/api/automation/health` içine eklemek ve grafiğe yansıtmak.

Configuration Sources
---------------------
- Systemd drop-ins (authoritative):
  - 10-internal-token.conf → INTERNAL_API_TOKEN
  - 30-db.conf → DATABASE_URL
  - 30-secrets.conf → FLASK_SECRET_KEY
  - 31-yf.conf → YF_USER_AGENTS (quoted)
  - override.conf → PIPELINE_MODE, collector/scheduler knobs, etc.
- config.py: reads from environment only; no dotenv

Data Flow (Happy Path)
----------------------
1. User starts automation (or watchdog restarts if needed)
2. Collector loads config; batches stocks; for each symbol: fetch → normalize → upsert
3. After collection completes, pattern detector analyzes symbols; logs progress
4. Sleep for configured interval (continuous loop), then repeat
5. Socket.IO delivers live logs to dashboard; REST endpoints provide health/tasks

Open Items / Risks
------------------
- Ensure consistent CSRF exemptions for Socket.IO polling and internal endpoints
- Align UI badges with backend status + recent activity
- Reduce WS-close stack traces (gevent switch mitigated)
- Collector throttling strategies (retries/backoff/period fallbacks)
- Measure baseline AI performance; optional ML deps gated by flags


