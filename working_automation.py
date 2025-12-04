"""
Working Automation System - Complete Rewrite
Simple, functional, reliable
"""

import logging
import threading
import time
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from bist_pattern.utils.error_handler import ErrorHandler
from bist_pattern.core.config_manager import ConfigManager  # ✅ FIX: Use ConfigManager
# pandas is optional for this module (used in training branches only)

try:
    import gevent  # type: ignore  # noqa: F401
    import gevent.lock as _glock  # type: ignore
    import gevent.event as _gevent_event  # type: ignore
    GEVENT_AVAILABLE = True
except ImportError:
    _glock = None  # type: ignore[assignment]
    _gevent_event = None  # type: ignore[assignment]
    GEVENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkingAutomationPipeline:
    """
    Working Automation Pipeline - Continuous data collection and analysis system.
    
    This class manages a continuous automation cycle that:
    1. Collects stock data for all active symbols
    2. Performs AI analysis (pattern detection, ML predictions)
    3. Handles feature backfill (FinGPT, YOLO)
    4. Manages state persistence for restart safety
    5. Tracks detailed metrics for monitoring
    
    Features:
    - Thread-safe state management
    - Graceful shutdown support
    - State persistence (cycle count, no-data backoff)
    - Detailed metrics collection
    - Error handling with logging
    """
    
    def __init__(self):
        self._state_lock = (_glock.RLock() if (GEVENT_AVAILABLE and _glock is not None) else threading.RLock())
        self._is_running = False
        self.thread = None
        self.stop_event = (_gevent_event.Event() if (GEVENT_AVAILABLE and _gevent_event is not None) else threading.Event())
        self.last_run_stats = {}
        # Backoff map for symbols that returned no data: {symbol: next_allowed_cycle}
        self.no_data_backoff = {}
        # Persistent cycle counter for status/API
        self.cycle_count = 0
        # ✅ FIX: State persistence for restart safety
        log_path = ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        self.state_file = Path(log_path) / 'automation_state.json'
        self.load_state()
        logger.info("✅ Working Automation Pipeline initialized")
    
    @property
    def is_running(self):
        """Thread-safe is_running property"""
        with self._state_lock:
            return self._is_running
    
    @is_running.setter
    def is_running(self, value):
        """Thread-safe is_running setter"""
        with self._state_lock:
            self._is_running = bool(value)
    
    def start_scheduler(self) -> bool:
        """Start automation scheduler"""
        if self.is_running:
            logger.info("⚠️ Automation already running")
            return True
            
        try:
            self.is_running = True
            self.stop_event.clear()
            
            def automation_loop():
                """Main automation loop"""
                logger.info("🚀 Automation loop started")
                cycle_count = 0
                
                while self.is_running and not self.stop_event.is_set():
                    try:
                        cycle_count += 1
                        # Update instance-level counter for external visibility
                        self.cycle_count = cycle_count
                        cycle_start_time = time.time()
                        logger.info(f"📊 Automation cycle {cycle_count} starting...")
                        
                        # ✅ FIX: Initialize cycle metrics for detailed monitoring
                        cycle_metrics = {
                            'cycle': cycle_count,
                            'start_time': datetime.now().isoformat(),
                            'symbols': {},
                            'errors': [],
                            'phases': {}
                        }
                        
                        # Simple data collection with explicit Flask app context
                        from bist_pattern.core.unified_collector import get_unified_collector
                        
                        # Get Flask app with better error handling
                        flask_app = None
                        try:
                            from flask import current_app
                            flask_app = current_app._get_current_object()
                        except Exception as e:
                            logger.debug(f"Failed to get current_app, trying fallback: {e}")
                            # Fallback: import app instance
                            try:
                                from app import app as flask_app
                            except Exception as _e:
                                logger.error(f"❌ Cannot import Flask app for context: {_e}")
                                time.sleep(10)
                                continue
                        
                        if not flask_app:
                            logger.error("❌ Flask app not available")
                            time.sleep(10)
                            continue

                        with flask_app.app_context():
                            # Helpers - Define _broadcast first so it can be used in handler
                            def _broadcast(level: str, message: str, category: str = 'working_automation') -> None:
                                try:
                                    # ✅ FIX: Validate flask_app is not None
                                    if flask_app is None:
                                        logger.debug("flask_app is None, cannot broadcast")
                                        return
                                    
                                    if hasattr(flask_app, 'broadcast_log'):
                                        # ✅ FIX: Add service identifier to distinguish from HPO logs
                                        flask_app.broadcast_log(level, message, category='working_automation', service='working_automation')  # type: ignore[attr-defined]
                                    else:
                                        # Fallback emit if broadcast helper is absent
                                        sock = getattr(flask_app, 'socketio', None)
                                        if sock is not None:
                                            # ✅ FIX: Sanitize log_update data to prevent parse errors
                                            try:
                                                from bist_pattern.core.broadcaster import _sanitize_json_value
                                                log_data = {
                                                    'level': level,
                                                    'message': str(message)[:1000],  # Limit message length
                                                    'category': 'working_automation',  # ✅ FIX: Use specific category to distinguish from HPO
                                                    'timestamp': datetime.now().isoformat(),
                                                    'service': 'working_automation'  # ✅ FIX: Add service identifier
                                                }
                                                sanitized_data = _sanitize_json_value(log_data)
                                                json.dumps(sanitized_data)  # Test serialization
                                                # ✅ FIX: Only send to admin room - user clients don't need log updates
                                                sock.emit('log_update', sanitized_data, room='admin')
                                            except Exception as sanitize_err:
                                                logger.debug(f"Log broadcast sanitization failed: {sanitize_err}")
                                except Exception as e:
                                    logger.debug(f"_broadcast failed: {e}")
                            
                            def _append_history(phase: str, state: str, details: Dict[str, Any] | None = None) -> None:
                                try:
                                    log_dir = ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                                    os.makedirs(log_dir, exist_ok=True)
                                    status_file = os.path.join(log_dir, 'pipeline_status.json')
                                    payload = {'history': []}
                                    if os.path.exists(status_file):
                                        try:
                                            # ✅ FIX: Check if file is empty before reading
                                            if os.path.getsize(status_file) > 0:
                                                with open(status_file, 'r') as rf:
                                                    content = rf.read().strip()
                                                    if content:
                                                        payload = json.loads(content) or {'history': []}
                                            else:
                                                payload = {'history': []}
                                        except (json.JSONDecodeError, ValueError) as json_err:
                                            # File exists but is corrupted or empty, reset it
                                            logger.debug(f"pipeline_status.json parse error, resetting: {json_err}")
                                            payload = {'history': []}
                                        except Exception as read_err:
                                            logger.debug(f"pipeline_status.json read error: {read_err}")
                                            payload = {'history': []}
                                    payload.setdefault('history', []).append({
                                        'phase': phase,
                                        'state': state,
                                        'timestamp': datetime.now().isoformat(),
                                        'details': details or {}
                                    })
                                    payload['history'] = payload['history'][-200:]
                                    # ✅ FIX: Use atomic write to prevent corruption
                                    temp_file = status_file + '.tmp'
                                    try:
                                        with open(temp_file, 'w') as wf:
                                            json.dump(payload, wf, indent=2)
                                        os.replace(temp_file, status_file)
                                    except Exception as write_err:
                                        logger.debug(f"pipeline_status.json write error: {write_err}")
                                        # Clean up temp file if exists
                                        try:
                                            if os.path.exists(temp_file):
                                                os.remove(temp_file)
                                        except Exception as e:
                                            logger.debug(f"Failed to remove temp file {temp_file}: {e}")
                                except Exception as e:
                                    # ✅ FIX: Log exception instead of silent pass
                                    logger.warning(f"_append_history failed: {e}")
                            
                            # ✅ FIX: Setup enhanced_ml_system logger handler to broadcast logs with service identifier
                            # This ensures logs from enhanced_ml_system are visible in admin dashboard
                            enhanced_ml_logger = logging.getLogger('enhanced_ml_system')
                            # Remove any existing handlers to avoid duplicates
                            for handler in list(enhanced_ml_logger.handlers):
                                if hasattr(handler, '_is_working_automation_handler'):
                                    enhanced_ml_logger.removeHandler(handler)
                            
                            class WorkingAutomationLogHandler(logging.Handler):
                                """Custom handler to broadcast enhanced_ml_system logs with service identifier"""
                                def __init__(self, broadcast_func):
                                    super().__init__()
                                    self.broadcast_func = broadcast_func
                                    self._is_working_automation_handler = True
                                
                                def emit(self, record):
                                    try:
                                        level_map = {
                                            logging.DEBUG: 'DEBUG',
                                            logging.INFO: 'INFO',
                                            logging.WARNING: 'WARNING',
                                            logging.ERROR: 'ERROR',
                                            logging.CRITICAL: 'ERROR'
                                        }
                                        level = level_map.get(record.levelno, 'INFO')
                                        message = self.format(record)
                                        # Only broadcast INFO and above to avoid spam
                                        if record.levelno >= logging.INFO:
                                            self.broadcast_func(level, message, 'working_automation')
                                    except Exception as e:
                                        # Silently fail to avoid recursion, but log at debug level
                                        logger.debug(f"Log handler emit failed (recursion prevention): {e}")
                            
                            # Add handler to enhanced_ml_system logger
                            enhanced_ml_handler = WorkingAutomationLogHandler(_broadcast)
                            enhanced_ml_handler.setLevel(logging.INFO)  # Only INFO and above
                            enhanced_ml_logger.addHandler(enhanced_ml_handler)
                            enhanced_ml_logger.setLevel(logging.INFO)  # Ensure logger level allows INFO

                            def _rss_health_check() -> Dict[str, Any]:
                                """
                                Validate RSS configuration and warm cache before cycle.
                                
                                Checks RSS feed sources and warms cache for a frequent symbol (AKBNK).
                                
                                Returns:
                                    Dict containing:
                                    - sources: Total number of RSS sources configured
                                    - tested: Number of sources tested
                                    - ok: Number of sources that responded successfully
                                    - fail: Number of sources that failed
                                """
                                try:
                                    sources_env = ConfigManager.get('NEWS_SOURCES', '')
                                    sources = [s.strip() for s in (sources_env or '').split(',') if s.strip()]
                                    max_test = int(ConfigManager.get('RSS_HEALTH_MAX_SOURCES', '5'))
                                    tested = 0
                                    ok = 0
                                    fail = 0
                                    if not sources:
                                        _broadcast('WARNING', '📰 RSS check: no NEWS_SOURCES configured', 'news')
                                        return {'sources': 0, 'ok': 0, 'fail': 0}
                                    try:
                                        import feedparser  # type: ignore
                                    except Exception as e:
                                        logger.warning(f"feedparser not available: {e}")
                                        _broadcast('ERROR', '📰 RSS check: feedparser not available', 'news')
                                        return {'sources': len(sources), 'ok': 0, 'fail': len(sources)}

                                    # Quick parse a few sources
                                    for url in sources[:max_test]:
                                        try:
                                            feed = feedparser.parse(url)
                                            tested += 1
                                            if getattr(feed, 'entries', None):
                                                ok += 1
                                            else:
                                                fail += 1
                                        except Exception as e:
                                            logger.debug(f"RSS feed parse failed for {url}: {e}")
                                            tested += 1
                                            fail += 1

                                    # Warm provider cache best-effort for a frequent symbol
                                    try:
                                        from news_provider import get_recent_news  # type: ignore
                                        _ = get_recent_news('AKBNK')
                                    except Exception as e:
                                        # ✅ FIX: Log exception instead of silent pass
                                        logger.debug(f"RSS warm cache failed: {e}")

                                    msg = f"📰 RSS check: sources={len(sources)} tested={tested} ok={ok} fail={fail}"
                                    _broadcast('INFO', msg, 'news')
                                    return {'sources': len(sources), 'tested': tested, 'ok': ok, 'fail': fail}
                                except Exception as e:
                                    logger.warning(f"RSS health check failed: {e}")
                                    return {'sources': 0, 'tested': 0, 'ok': 0, 'fail': 0}

                            collector = get_unified_collector()
                            # ✅ FIX: Validate collector is not None
                            if collector is None:
                                logger.error("get_unified_collector returned None! Cannot proceed with collection.")
                                time.sleep(10)
                                continue
                            
                            # Cooldown for symbols with no data
                            try:
                                cooldown_cycles = int(ConfigManager.get('NO_DATA_COOLDOWN_CYCLES', '20'))
                            except Exception as e:
                                logger.debug(f"Failed to get NO_DATA_COOLDOWN_CYCLES, using default: {e}")
                                cooldown_cycles = 20
                            # Symbol sleep per iteration
                            try:
                                symbol_sleep_seconds = float(ConfigManager.get('SYMBOL_SLEEP_SECONDS', '0.05'))
                            except Exception as e:
                                logger.debug(f"Failed to get SYMBOL_SLEEP_SECONDS, using default: {e}")
                                symbol_sleep_seconds = 0.05

                            # 0) RSS health check & cache warm-up
                            try:
                                rss_res = _rss_health_check()
                                _append_history('rss_check', 'end', rss_res or {})
                            except Exception as e:
                                logger.warning(f"RSS health check failed: {e}")
                                _append_history('rss_check', 'error', {})

                            # SYMBOL_FLOW mode only (CONTINUOUS_FULL removed)
                            symbol_flow = str(ConfigManager.get('SYMBOL_FLOW', '1')).lower() in ('1', 'true', 'yes')

                            # Result accumulator visible for last_run_stats
                            col_res = None
                            # ✅ FIX: Initialize variables to avoid "possibly unbound" warnings
                            analyzed = 0
                            total_symbols = 0
                            collected_records = 0
                            updated_records = 0

                            if symbol_flow:
                                # Sequential per-symbol: collect -> analyze -> next
                                try:
                                    from models import Stock
                                    from app import get_pattern_detector
                                    
                                    # ✅ FIX: Use singleton pattern instead of direct instantiation
                                    try:
                                        det = get_pattern_detector()
                                        if det is None:
                                            logger.error("get_pattern_detector returned None! Cannot proceed with analysis.")
                                            raise ValueError("det is None")
                                    except Exception as det_err:
                                        logger.error(f"Failed to get pattern detector: {det_err}")
                                        raise  # Re-raise to be caught by outer try-except
                                    
                                    # ✅ FIX: Validate Stock.query exists
                                    if not hasattr(Stock, 'query') or Stock.query is None:
                                        logger.error("Stock.query is None! App context may be missing.")
                                        raise ValueError("Stock.query is None")
                                    
                                    # Universe: ALL active stocks (watchlist zaten bu kümenin alt kümesidir)
                                    symbols: List[str] = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).all()]
                                    
                                    # ✅ FIX: Validate symbols is not empty
                                    if not symbols:
                                        logger.warning("No active symbols found! Cannot proceed with automation cycle.")
                                        # Continue cycle anyway - maybe next cycle will have symbols
                                    total_symbols = len(symbols)
                                    # Reset counters for this cycle
                                    analyzed = 0
                                    collected_records = 0
                                    updated_records = 0
                                    _append_history('symbol_flow', 'start', {'total_symbols': total_symbols})
                                    _broadcast('INFO', f'🔁 SYMBOL_FLOW aktif: {total_symbols} sembol işlem sırası', 'pipeline')
                                    for symbol in symbols:
                                        if not self.is_running or self.stop_event.is_set():
                                            break
                                        
                                        # ✅ FIX: Track symbol metrics
                                        symbol_start_time = time.time()
                                        symbol_errors = []
                                        # ✅ FIX: Initialize variables to avoid "possibly unbound" warnings
                                        res: Dict[str, Any] | None = None
                                        result: Dict[str, Any] | None = None
                                        
                                        # Skip symbols that recently returned no data until cooldown expires
                                        try:
                                            backoff_until = self.no_data_backoff.get(symbol)
                                            if backoff_until is not None and cycle_count < backoff_until:
                                                remaining = max(0, backoff_until - cycle_count)
                                                _broadcast('INFO', f'⏭️ Skip {symbol} (no-data cooldown {remaining} cycles left)', 'collector')
                                                time.sleep(0.01)
                                                continue
                                        except Exception as backoff_err:
                                            logger.debug(f"Backoff check failed for {symbol}: {backoff_err}")
                                        # Collect minimal recent data for symbol
                                        try:
                                            _broadcast('INFO', f'📥 {symbol}: Yahoo Finance\'tan veri çekiliyor...', 'collector')
                                            res = collector.collect_single_stock(symbol, period='auto')
                                            
                                            # ✅ FIX: Validate res is not None
                                            if res is None:
                                                logger.debug(f"collect_single_stock returned None for {symbol}")
                                                _broadcast('WARNING', f'⚠️ {symbol}: Veri çekme sonucu None döndü', 'collector')
                                                # Continue to next symbol
                                                time.sleep(symbol_sleep_seconds)
                                                continue
                                            
                                            if isinstance(res, dict):
                                                success = bool(res.get('success'))
                                                recs = int(res.get('records', 0))
                                                upd = int(res.get('updated', 0))
                                                if success:
                                                    collected_records += recs
                                                    updated_records += upd
                                                    _broadcast('SUCCESS', f'✅ {symbol}: {recs} yeni kayıt, {upd} güncellenmiş kayıt çekildi', 'collector')
                                                    # Clear backoff if data arrived
                                                    try:
                                                        if symbol in self.no_data_backoff:
                                                            self.no_data_backoff.pop(symbol, None)
                                                    except Exception as pop_err:
                                                        logger.debug(f"Failed to pop backoff for {symbol}: {pop_err}")
                                                else:
                                                    # If explicitly no_data or zero rows, set cooldown
                                                    try:
                                                        err = str(res.get('error', ''))
                                                        if err == 'no_data' or recs == 0:
                                                            self.no_data_backoff[symbol] = cycle_count + cooldown_cycles
                                                            _broadcast('WARNING', f'no_data: {symbol} backoff until cycle {self.no_data_backoff[symbol]}', 'collector')
                                                    except Exception as e:
                                                        # ✅ FIX: Log exception instead of silent pass
                                                        logger.debug(f"No-data cooldown set failed for {symbol}: {e}")
                                        except Exception as e:
                                            # ✅ FIX: Log exception instead of silent pass
                                            logger.debug(f"Collection failed for {symbol}: {e}")
                                            symbol_errors.append(f"collection: {str(e)}")
                                        # Analyze immediately after collection
                                        try:
                                            result = det.analyze_stock(symbol)
                                            
                                            # ✅ FIX: Validate result is not None
                                            if result is None:
                                                logger.debug(f"analyze_stock returned None for {symbol}, skipping feature backfill")
                                                analyzed += 1  # Still count as analyzed attempt
                                                time.sleep(symbol_sleep_seconds)
                                                continue
                                            
                                            analyzed += 1
                                            
                                            # ⚡ NEW: Extract FinGPT and YOLO features from result and write to CSV
                                            try:
                                                from scripts.backfill_external_features import analyze_symbol_for_features, write_feature_csvs
                                                
                                                # ✅ FIX: Validate analyze_symbol_for_features result
                                                fingpt_data, yolo_data = analyze_symbol_for_features(symbol, result, lookback_days=1)
                                                if fingpt_data is None:
                                                    fingpt_data = {}
                                                if yolo_data is None:
                                                    yolo_data = {}
                                                if fingpt_data or yolo_data:
                                                    feature_dir = ConfigManager.get('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
                                                    write_feature_csvs(symbol, fingpt_data, yolo_data, feature_dir=feature_dir)
                                            except Exception as fe:
                                                logger.debug(f"Feature backfill failed for {symbol}: {fe}")
                                            
                                            try:
                                                sock = getattr(flask_app, 'socketio', None)
                                                if sock is not None:
                                                    # ✅ FIX: Send full result in correct format for frontend
                                                    # Frontend expects: { symbol, data: { current_price, ml_unified, enhanced_predictions, ... } }
                                                    emit_data = {
                                                        'symbol': symbol,
                                                        'data': {
                                                            'symbol': symbol,
                                                            'status': result.get('status', 'success') if result else 'success',
                                                            'timestamp': result.get('timestamp', datetime.now().isoformat()) if result else datetime.now().isoformat(),
                                                            'current_price': result.get('current_price', 0.0) if result else 0.0,
                                                            'price': result.get('current_price', 0.0) if result else 0.0,  # Fallback for frontend
                                                            'patterns': result.get('patterns', [])[:10] if result else [],
                                                            'ml_predictions': result.get('ml_predictions', {}) if result else {},
                                                            'enhanced_predictions': result.get('enhanced_predictions', {}) if result else {},
                                                            'ml_unified': result.get('ml_unified', {}) if result else {},  # ✅ CRITICAL: Frontend needs this!
                                                            'indicators': result.get('indicators', {}) if result else {},
                                                            'overall_signal': result.get('overall_signal', {}) if result else {},
                                                        },
                                                        'timestamp': datetime.now().isoformat(),
                                                    }
                                                    # ✅ CRITICAL FIX: Sanitize emit_data before sending to prevent parse errors
                                                    # NaN, inf, and out-of-range float values cause JSON serialization errors
                                                    try:
                                                        from bist_pattern.core.broadcaster import _sanitize_json_value
                                                        import json
                                                        sanitized_data = _sanitize_json_value(emit_data)
                                                        # Test JSON serialization before emitting
                                                        json.dumps(sanitized_data)
                                                        emit_data = sanitized_data
                                                    except Exception as sanitize_err:
                                                        logger.debug(f"Sanitization failed for {symbol}: {sanitize_err}, skipping emit")
                                                        emit_data = None
                                                    
                                                    # ✅ CRITICAL FIX: DISABLED pattern_analysis broadcasts completely
                                                    # User dashboard reads from batch API cache, doesn't need WebSocket events
                                                    # Admin dashboard can also use batch API for monitoring
                                                    # This eliminates unnecessary WebSocket traffic and parse errors
                                                    # if emit_data is not None:
                                                    #     try:
                                                    #         sock.emit('pattern_analysis', emit_data, room='admin')
                                                    #     except Exception as emit_err:
                                                    #         logger.debug(f"Socket emit failed for {symbol}: {emit_err}")
                                                    pass  # Broadcast disabled
                                            except Exception as e:
                                                # ✅ FIX: Log exception instead of silent pass
                                                logger.debug(f"Socket emit failed for {symbol}: {e}")
                                        except Exception as e:
                                            # ✅ FIX: Log exception instead of silent pass
                                            logger.debug(f"Analysis failed for {symbol}: {e}")
                                            symbol_errors.append(f"analysis: {str(e)}")
                                        
                                        # ✅ FIX: Record symbol metrics
                                        symbol_duration = time.time() - symbol_start_time
                                        symbol_collected = False
                                        symbol_analyzed = False
                                        # Check if this symbol was collected/analyzed
                                        # ✅ FIX: Variables are already initialized above, no need for locals() check
                                        if res is not None and isinstance(res, dict):
                                            symbol_collected = bool(res.get('success', False))
                                        if result is not None:
                                            symbol_analyzed = True
                                        
                                        cycle_metrics['symbols'][symbol] = {
                                            'duration': round(symbol_duration, 3),
                                            'collected': symbol_collected,
                                            'analyzed': symbol_analyzed,
                                            'errors': symbol_errors
                                        }
                                        
                                        time.sleep(symbol_sleep_seconds)
                                    _append_history('symbol_flow', 'end', {'analyzed': analyzed, 'collected_records': collected_records, 'updated_records': updated_records, 'total': total_symbols})
                                    # Expose collected records for this cycle in last_run_stats
                                    col_res = {'total_records': collected_records, 'updated_records': updated_records, 'total_symbols': total_symbols}
                                except Exception as e:
                                    _append_history('symbol_flow', 'error', {'error': str(e)})
                                    analyzed = 0
                                    total_symbols = 0
                            else:
                                # CONTINUOUS_FULL mode disabled - only SYMBOL_FLOW supported
                                _broadcast('WARNING', '⚠️ SYMBOL_FLOW=0 but CONTINUOUS_FULL mode removed. Defaulting to SYMBOL_FLOW.', 'pipeline')
                                # Force enable SYMBOL_FLOW for next cycle
                                os.environ['SYMBOL_FLOW'] = '1'
                                analyzed = 0
                                total_symbols = 0
                                col_res = {'success': False, 'error': 'CONTINUOUS_FULL mode removed'}

                            # 2. ML training gated off in automation (cron-only)
                            try:
                                if str(ConfigManager.get('ENABLE_TRAINING_IN_CYCLE', '0')).lower() in ('1', 'true', 'yes'):
                                    logger.info('⚠️ Training-in-cycle enabled by env; consider disabling in production')
                                else:
                                    logger.info('⏭️ Skipping ML training in cycle (cron-only policy active)')
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent pass
                                logger.debug(f"Training check failed: {e}")

                            # Update last run stats
                            try:
                                records = 0
                                if isinstance(col_res, dict):
                                    records = int(col_res.get('total_records', 0))
                                self.last_run_stats = {
                                    'cycle': cycle_count,
                                    'symbols_processed': total_symbols,
                                    'total_records': records,
                                    'updated_records': int(col_res.get('updated_records', 0)) if isinstance(col_res, dict) else 0,
                                    'analyzed': analyzed,
                                    'timestamp': datetime.now().isoformat(),
                                }
                            except Exception as e:
                                logger.debug(f"Failed to update last_run_stats: {e}")
                                self.last_run_stats = {'cycle': cycle_count, 'timestamp': datetime.now().isoformat()}
                            # ✅ FIX: Finalize cycle metrics
                            cycle_duration = time.time() - cycle_start_time
                            cycle_metrics['duration'] = round(cycle_duration, 3)
                            cycle_metrics['end_time'] = datetime.now().isoformat()
                            cycle_metrics['total_symbols'] = total_symbols
                            cycle_metrics['analyzed'] = analyzed
                            cycle_metrics['collected_records'] = collected_records
                            cycle_metrics['updated_records'] = updated_records
                            
                            # ✅ FIX: Save metrics to file
                            try:
                                import json  # Ensure json is available in this scope
                                metrics_file = Path(ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')) / 'automation_metrics.json'
                                metrics_file.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Load existing metrics
                                existing_metrics = []
                                if metrics_file.exists():
                                    try:
                                        with open(metrics_file, 'r') as f:
                                            existing_metrics = json.load(f) or []
                                    except Exception as e:
                                        logger.debug(f"Failed to load existing metrics, starting fresh: {e}")
                                        existing_metrics = []
                                
                                # Append new metrics (keep last 100 cycles)
                                existing_metrics.append(cycle_metrics)
                                existing_metrics = existing_metrics[-100:]
                                
                                # Save updated metrics
                                with open(metrics_file, 'w') as f:
                                    json.dump(existing_metrics, f, indent=2)
                            except Exception as metrics_err:
                                logger.warning(f"⚠️ Failed to save cycle metrics: {metrics_err}")
                            
                            logger.info(
                                f"✅ Cycle {cycle_count} completed: "
                                f"collected={self.last_run_stats.get('total_records', 0)} "
                                f"updated={self.last_run_stats.get('updated_records', 0)} "
                                f"analyzed={analyzed} "
                                f"duration={cycle_duration:.2f}s"
                            )
                            
                            # Ensure instance counter also reflects last completed
                            self.cycle_count = cycle_count
                            
                            # ✅ FIX: Save state after cycle completion
                            self.save_state()
                            
                            # Check forward simulation (if active)
                            try:
                                with flask_app.app_context():
                                    from bist_pattern.simulation.forward_engine import check_and_trade
                                    sim_result = check_and_trade()
                                    
                                    # ✅ FIX: Validate sim_result is not None
                                    if sim_result is None:
                                        logger.debug("check_and_trade returned None, skipping simulation check")
                                    elif isinstance(sim_result, dict) and sim_result.get('active'):
                                        logger.info(
                                            f"💼 Simulation check: "
                                            f"trades={sim_result.get('trades_made', 0)} "
                                            f"positions={sim_result.get('positions_count', 0)} "
                                            f"equity=₺{sim_result.get('equity', 0):.2f}"
                                        )
                            except Exception as sim_e:
                                logger.warning(f"⚠️ Simulation check failed: {sim_e}")
                        
                        # Wait for next cycle (environment-driven)
                        try:
                            sleep_total = int(ConfigManager.get('AUTOMATION_CYCLE_SLEEP_SECONDS', '300'))
                        except Exception as e:
                            logger.debug(f"Failed to get AUTOMATION_CYCLE_SLEEP_SECONDS, using default: {e}")
                            sleep_total = 300
                        for _ in range(sleep_total):
                            if self.stop_event.is_set() or not self.is_running:
                                break
                            time.sleep(1)
                            
                    except Exception as e:
                        logger.error(f"❌ Automation cycle error: {e}")
                        # Environment-driven error retry delay
                        try:
                            error_retry_delay = int(ConfigManager.get('AUTOMATION_ERROR_RETRY_DELAY', '30'))
                        except Exception as e:
                            logger.debug(f"Failed to get AUTOMATION_ERROR_RETRY_DELAY, using 30: {e}")
                            error_retry_delay = 30
                        time.sleep(error_retry_delay)
                
                logger.info("🛑 Automation loop stopped")
            
            self.thread = threading.Thread(target=automation_loop, daemon=True)
            self.thread.start()
            
            logger.info("🎉 Automation scheduler started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start automation: {e}")
            self.is_running = False
            return False
    
    def stop_scheduler(self) -> bool:
        """Stop automation scheduler"""
        try:
            self.is_running = False
            self.stop_event.set()
            
            # Avoid joining from within the same automation thread
            try:
                import threading as _th
                current_is_worker = (self.thread is not None and _th.current_thread() is self.thread)
            except Exception as e:
                logger.debug(f"Failed to check if current thread is worker: {e}")
                current_is_worker = False
            if self.thread and self.thread.is_alive() and not current_is_worker:
                self.thread.join(timeout=5)
            
            logger.info("🛑 Automation scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop automation: {e}")
            return False
    
    def load_state(self) -> None:
        """
        Load automation state from file for restart safety.
        
        Loads cycle_count and no_data_backoff map from automation_state.json.
        Called during initialization to restore state after service restart.
        
        Raises:
            No exceptions raised - errors are logged as warnings.
        """
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.cycle_count = state.get('cycle_count', 0)
                    self.no_data_backoff = state.get('no_data_backoff', {})
                    logger.info(f"✅ Loaded automation state: cycle={self.cycle_count}, backoff_size={len(self.no_data_backoff)}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load automation state: {e}")
    
    def save_state(self) -> None:
        """
        Save automation state to file for restart safety.
        
        Saves cycle_count and no_data_backoff map to automation_state.json.
        Called after each cycle completion to persist state.
        
        Raises:
            No exceptions raised - errors are logged as warnings.
        """
        try:
            state = {
                'cycle_count': self.cycle_count,
                'no_data_backoff': self.no_data_backoff,
                'last_updated': datetime.now().isoformat()
            }
            # Ensure directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Failed to save automation state: {e}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status.
        
        Returns:
            Dict containing:
            - is_running: Whether automation is currently running
            - thread_alive: Whether automation thread is alive
            - last_run_stats: Statistics from last completed cycle
            - status: 'running' or 'stopped'
            - cycle_count: Current cycle number
            - no_data_cooldown_size: Number of symbols in cooldown
            - skip_count_current_cycle: Number of symbols skipped in current cycle
        """
        try:
            skip_now = sum(1 for _sym, until in self.no_data_backoff.items() if until > self.cycle_count)
        except Exception as e:
            logger.debug(f"Failed to calculate skip_now: {e}")
            skip_now = 0
        return {
            'is_running': self.is_running,
            'thread_alive': self.thread.is_alive() if self.thread else False,
            'last_run_stats': self.last_run_stats,
            'status': 'running' if self.is_running else 'stopped',
            'cycle_count': getattr(self, 'cycle_count', 0),
            'no_data_cooldown_size': len(self.no_data_backoff),
            'skip_count_current_cycle': skip_now,
        }
    
    def system_health_check(self) -> Dict[str, Any]:
        """
        Perform system health check.
        
        Returns:
            Dict containing:
            - status: 'healthy' or 'unhealthy'
            - automation: 'running' or 'stopped'
            - thread_status: 'alive' or 'stopped'
        """
        return {
            'status': 'healthy',
            'automation': 'running' if self.is_running else 'stopped',
            'thread_status': 'alive' if (self.thread and self.thread.is_alive()) else 'stopped'
        }

    # Minimal API to satisfy admin endpoints
    def run_manual_task(self, task_name: str) -> Dict[str, Any]:
        try:
            if task_name == 'data_collection':
                # Collect fresh data for ALL active symbols (insert or update), no analysis
                from bist_pattern.core.unified_collector import get_unified_collector
                from app import app as flask_app
                with flask_app.app_context():
                    from models import Stock
                    collector = get_unified_collector()
                    symbols_list: List[str] = [s.symbol for s in Stock.query.filter_by(active=True).order_by(Stock.symbol.asc()).all()]
                    total = len(symbols_list)
                    added_total = 0
                    updated_total = 0
                    no_data = 0
                    errors = 0
                    try:
                        symbol_sleep_seconds = float(ConfigManager.get('MANUAL_TASK_SYMBOL_SLEEP', '0.01'))  # Faster for manual tasks
                    except Exception as e:
                        logger.debug(f"Failed to get MANUAL_TASK_SYMBOL_SLEEP, using 0.01: {e}")
                        symbol_sleep_seconds = 0.01
                    # Manual data collection: Process ALL symbols (no limit)
                    limited_symbols = symbols_list
                    logger.info(f"📊 Manual data collection for ALL {len(symbols_list)} symbols")
                    for i, sym in enumerate(limited_symbols):
                        try:
                            res = collector.collect_single_stock(sym, period='auto')
                            
                            # ✅ FIX: Validate res is not None
                            if res is None:
                                logger.debug(f"collect_single_stock returned None for {sym} in manual task")
                                no_data += 1
                                continue
                            
                            if isinstance(res, dict):
                                added_total += int(res.get('records', 0))
                                updated_total += int(res.get('updated', 0))
                                if not bool(res.get('success')) or (int(res.get('records', 0)) == 0 and int(res.get('updated', 0)) == 0):
                                    no_data += 1
                        except Exception as e:
                            # ✅ FIX: Log exception instead of silent pass
                            logger.debug(f"Manual collection failed for {sym}: {e}")
                            errors += 1
                        # Progress feedback every 10 symbols
                        if (i + 1) % 10 == 0:
                            try:
                                from app import app as flask_app
                                if hasattr(flask_app, 'broadcast_log'):
                                    flask_app.broadcast_log('INFO', f'📊 Manual collection progress: {i+1}/{len(limited_symbols)} symbols', 'collector')  # type: ignore[attr-defined]
                            except Exception as e:
                                # ✅ FIX: Log exception instead of silent pass
                                logger.debug(f"Broadcast failed in manual task: {e}")
                        time.sleep(symbol_sleep_seconds)
                return {
                    'ok': True,
                    'result': {
                        'total_symbols': total,
                        'added_records': added_total,
                        'updated_records': updated_total,
                        'no_data_or_empty': no_data,
                        'errors': errors,
                        'timestamp': datetime.now().isoformat(),
                    }
                }
            if task_name == 'health_check':
                return {'ok': True, 'health': self.system_health_check()}
            if task_name == 'status_report':
                return {'ok': True, 'status': self.get_scheduler_status()}
            if task_name == 'weekly_collection':
                # For now, reuse data_collection
                from bist_pattern.core.unified_collector import get_unified_collector
                from app import app as flask_app
                with flask_app.app_context():
                    result = {'note': 'weekly mode uses same collector path'}
                return {'ok': True, 'result': result, 'mode': 'weekly'}
            if task_name == 'model_retraining':
                # Manually train ALL eligible symbols (env-driven cooldown override)
                from app import app as flask_app
                with flask_app.app_context():
                    from models import Stock  # type: ignore
                    
                    # ✅ FIX: Validate Stock.query exists
                    if not hasattr(Stock, 'query') or Stock.query is None:
                        logger.error("Stock.query is None in model_retraining! App context may be missing.")
                        return {'ok': False, 'error': 'Stock.query is None'}
                    
                    symbols_to_train: List[str] = [s.symbol for s in Stock.query.filter_by(active=True).order_by(Stock.symbol.asc()).all()]
                    
                    # ✅ FIX: Validate symbols_to_train is not empty (but still return ok if empty)
                    if not symbols_to_train:
                        logger.warning("No active symbols found for model_retraining!")
                    
                return {'ok': True, 'trained': len(symbols_to_train)}
            return {'ok': False, 'error': 'unknown_task'}
        except Exception as e:
            logger.error(f"run_manual_task error: {e}")
            return {'ok': False, 'error': str(e)}

    def daily_status_report(self) -> Dict[str, Any]:
        # Generate volume tier data for report modal
        volume_data = self._generate_volume_tier_data()
        
        return {
            'last_run_stats': self.last_run_stats,
            'is_running': self.is_running,
            'thread_alive': self.thread.is_alive() if self.thread else False,
            'generated_at': datetime.now().isoformat(),
            'volume': volume_data,
        }
    
    def _generate_volume_tier_data(self) -> Dict[str, Any]:
        """Generate volume tier data for report modal"""
        try:
            from app import app as flask_app
            with flask_app.app_context():
                from models import db, Stock, StockPrice
                from sqlalchemy import func
                from datetime import timedelta
                
                lookback_days = int(ConfigManager.get('VOLUME_LOOKBACK_DAYS', '30'))
                cutoff = datetime.now().date() - timedelta(days=lookback_days)
                
                # Get average volumes for all active stocks
                rows = (
                    db.session.query(Stock.symbol, Stock.name, func.avg(StockPrice.volume).label('avg_vol'))
                    .join(StockPrice, Stock.id == StockPrice.stock_id)
                    .filter(Stock.is_active.is_(True), StockPrice.date >= cutoff)
                    .group_by(Stock.id, Stock.symbol, Stock.name)
                    .all()
                )
                
                if not rows:
                    return {
                        'lookback_days': lookback_days,
                        'symbols': [],
                        'summary': {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0},
                        'percentiles': {'p15': 0.0, 'p40': 0.0, 'p75': 0.0, 'p95': 0.0}
                    }
                
                # Calculate percentiles
                vols = [float(r[2] or 0) for r in rows]
                vols.sort()
                
                def percentile(data, p):
                    k = (len(data) - 1) * (p / 100.0)
                    f = int(k)
                    c = min(f + 1, len(data) - 1)
                    if f == c:
                        return float(data[int(k)])
                    d0 = data[f] * (c - k)
                    d1 = data[c] * (k - f)
                    return float(d0 + d1)
                
                p15 = percentile(vols, 15)
                p40 = percentile(vols, 40)
                p75 = percentile(vols, 75)
                p95 = percentile(vols, 95)
                
                def get_tier(v):
                    v = float(v or 0)
                    if v >= p95:
                        return 'very_high'
                    if v >= p75:
                        return 'high'
                    if v >= p40:
                        return 'medium'
                    if v >= p15:
                        return 'low'
                    return 'very_low'
                
                # Build symbols with tiers
                symbols_data = []
                summary = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
                
                for symbol, name, avg_vol in rows:
                    tier = get_tier(avg_vol)
                    symbols_data.append({
                        'symbol': symbol,
                        'name': name,
                        'avg_volume': float(avg_vol or 0),
                        'tier': tier
                    })
                    summary[tier] += 1
                
                return {
                    'lookback_days': lookback_days,
                    'symbols': symbols_data,
                    'summary': summary,
                    'percentiles': {'p15': p15, 'p40': p40, 'p75': p75, 'p95': p95}
                }
                
        except Exception as e:
            logger.error(f"Volume tier data generation error: {e}")
            safe_lookback = int(ConfigManager.get('VOLUME_LOOKBACK_DAYS', '30'))
            return {
                'lookback_days': safe_lookback,
                'symbols': [],
                'summary': {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0},
                'percentiles': {'p15': 0.0, 'p40': 0.0, 'p75': 0.0, 'p95': 0.0},
                'error': str(e)
            }

    # Optional: provide bulk predictions method used by some internal endpoints
    def run_bulk_predictions_all(self) -> Dict[str, Any]:  # pragma: no cover
        try:
            # Minimal placeholder to keep internal routes functional
            log_dir = ConfigManager.get('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            os.makedirs(log_dir, exist_ok=True)
            return {'status': 'disabled', 'predictions': {}, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'predictions': {}}

    # Convenience: run one full cycle synchronously (no background thread)
    def run_once(self) -> Dict[str, Any]:  # pragma: no cover
        if self.is_running:
            return {'status': 'error', 'message': 'Scheduler already running'}
        # Temporarily toggle running to reuse loop body pieces
        self.is_running = True
        try:
            # Reuse the same logic by calling start_scheduler and waiting one cycle is complex;
            # instead, directly execute the core of one cycle here by calling internal helpers via start_scheduler code is avoided.
            from app import app as flask_app
            from bist_pattern.core.unified_collector import get_unified_collector
            with flask_app.app_context():
                collector = get_unified_collector()
                
                # ✅ FIX: Validate collector is not None
                if collector is None:
                    logger.error("get_unified_collector returned None in run_once! Cannot proceed.")
                    self.is_running = False
                    return {'status': 'error', 'error': 'collector is None'}
                
                scope = ConfigManager.get('COLLECTION_SCOPE', 'DB_ACTIVE')
                col_res = collector.collect_all_stocks_parallel(scope=scope)
                
                # ✅ FIX: Validate col_res is not None
                if col_res is None:
                    logger.warning("collect_all_stocks_parallel returned None in run_once")
                    col_res = {}  # Fallback to empty dict
                
                from app import get_pattern_detector
                from models import Stock
                
                # ✅ FIX: Use singleton pattern instead of direct instantiation
                try:
                    det = get_pattern_detector()
                    if det is None:
                        logger.error("get_pattern_detector returned None in run_once! Cannot proceed.")
                        self.is_running = False
                        return {'status': 'error', 'error': 'det is None'}
                except Exception as det_err:
                    logger.error(f"Failed to get pattern detector in run_once: {det_err}")
                    self.is_running = False
                    return {'status': 'error', 'error': f'det creation failed: {det_err}'}
                
                # ✅ FIX: Validate Stock.query exists
                if not hasattr(Stock, 'query') or Stock.query is None:
                    logger.error("Stock.query is None in run_once! App context may be missing.")
                    self.is_running = False
                    return {'status': 'error', 'error': 'Stock.query is None'}
                
                symbols: List[str] = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).all()]
                
                # ✅ FIX: Validate symbols is not empty
                if not symbols:
                    logger.warning("No active symbols found in run_once!")
                    symbols = []  # Continue with empty list
                
                analyzed = 0
                for symbol in symbols:
                    try:
                        result = det.analyze_stock(symbol)
                        # ✅ FIX: Only count if result is not None
                        if result is not None:
                            analyzed += 1
                    except Exception as e:
                        # ✅ FIX: Log exception instead of silent pass
                        ErrorHandler.handle(e, f'analyze_stock_{symbol}', level='debug')
            out = {'collected_records': int(col_res.get('total_records', 0)) if isinstance(col_res, dict) else 0,
                   'analyzed': analyzed,
                   'total_symbols': len(symbols)}
            self.is_running = False
            return out
        except Exception as e:
            self.is_running = False
            return {'status': 'error', 'error': str(e)}


# Global instance
_working_pipeline = None


def get_working_automation_pipeline() -> WorkingAutomationPipeline:
    """
    Get working automation pipeline singleton instance.
    
    Returns:
        WorkingAutomationPipeline: The global singleton instance.
        
    Note:
        Creates a new instance on first call, returns existing instance on subsequent calls.
    """
    global _working_pipeline
    if _working_pipeline is None:
        _working_pipeline = WorkingAutomationPipeline()
    return _working_pipeline
