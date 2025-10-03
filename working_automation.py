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
from typing import Dict, Any, List
# pandas is optional for this module (used in training branches only)

try:
    import gevent
    import gevent.lock
    import gevent.event
    GEVENT_AVAILABLE = True
except ImportError:
    GEVENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkingAutomationPipeline:
    """
    Simple, working automation pipeline
    """
    
    def __init__(self):
        self._state_lock = gevent.lock.RLock() if GEVENT_AVAILABLE else threading.RLock()
        self._is_running = False
        self.thread = None
        self.stop_event = gevent.event.Event() if GEVENT_AVAILABLE else threading.Event()
        self.last_run_stats = {}
        # Backoff map for symbols that returned no data: {symbol: next_allowed_cycle}
        self.no_data_backoff = {}
        # Persistent cycle counter for status/API
        self.cycle_count = 0
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
                        logger.info(f"📊 Automation cycle {cycle_count} starting...")
                        
                        # Simple data collection with explicit Flask app context
                        from bist_pattern.core.unified_collector import get_unified_collector
                        try:
                            # Try to import the global Flask app instance
                            from app import app as flask_app
                        except Exception as _e:
                            logger.error(f"❌ Cannot import Flask app for context: {_e}")
                            time.sleep(10)
                            continue

                        with flask_app.app_context():
                            # Helpers
                            def _append_history(phase: str, state: str, details: Dict[str, Any] | None = None) -> None:
                                try:
                                    log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                                    os.makedirs(log_dir, exist_ok=True)
                                    status_file = os.path.join(log_dir, 'pipeline_status.json')
                                    payload = {'history': []}
                                    if os.path.exists(status_file):
                                        try:
                                            with open(status_file, 'r') as rf:
                                                payload = json.load(rf) or {'history': []}
                                        except Exception:
                                            payload = {'history': []}
                                    payload.setdefault('history', []).append({
                                        'phase': phase,
                                        'state': state,
                                        'timestamp': datetime.now().isoformat(),
                                        'details': details or {}
                                    })
                                    payload['history'] = payload['history'][-200:]
                                    with open(status_file, 'w') as wf:
                                        json.dump(payload, wf)
                                except Exception:
                                    pass

                            def _broadcast(level: str, message: str, category: str = 'pipeline') -> None:
                                try:
                                    if hasattr(flask_app, 'broadcast_log'):
                                        flask_app.broadcast_log(level, message, category)
                                    else:
                                        # Fallback emit if broadcast helper is absent
                                        sock = getattr(flask_app, 'socketio', None)
                                        if sock is not None:
                                            sock.emit('log_update', {
                                                'level': level,
                                                'message': message,
                                                'category': category,
                                                'timestamp': datetime.now().isoformat(),
                                            })
                                except Exception:
                                    pass

                            def _rss_health_check() -> dict:
                                """Validate RSS configuration and warm cache before cycle."""
                                try:
                                    sources_env = os.getenv('NEWS_SOURCES', '')
                                    sources = [s.strip() for s in sources_env.split(',') if s.strip()]
                                    max_test = int(os.getenv('RSS_HEALTH_MAX_SOURCES', '5'))
                                    tested = 0
                                    ok = 0
                                    fail = 0
                                    if not sources:
                                        _broadcast('WARNING', '📰 RSS check: no NEWS_SOURCES configured', 'news')
                                        return {'sources': 0, 'ok': 0, 'fail': 0}
                                    try:
                                        import feedparser  # type: ignore
                                    except Exception:
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
                                        except Exception:
                                            tested += 1
                                            fail += 1

                                    # Warm provider cache best-effort for a frequent symbol
                                    try:
                                        from news_provider import get_recent_news  # type: ignore
                                        _ = get_recent_news('AKBNK')
                                    except Exception:
                                        pass

                                    msg = f"📰 RSS check: sources={len(sources)} tested={tested} ok={ok} fail={fail}"
                                    _broadcast('INFO', msg, 'news')
                                    return {'sources': len(sources), 'tested': tested, 'ok': ok, 'fail': fail}
                                except Exception:
                                    return {'sources': 0, 'tested': 0, 'ok': 0, 'fail': 0}

                            collector = get_unified_collector()
                            # Cooldown for symbols with no data
                            try:
                                cooldown_cycles = int(os.getenv('NO_DATA_COOLDOWN_CYCLES', '20'))
                            except Exception:
                                cooldown_cycles = 20
                            # Symbol sleep per iteration
                            try:
                                symbol_sleep_seconds = float(os.getenv('SYMBOL_SLEEP_SECONDS', '0.05'))
                            except Exception:
                                symbol_sleep_seconds = 0.05

                            # 0) RSS health check & cache warm-up
                            try:
                                rss_res = _rss_health_check()
                                _append_history('rss_check', 'end', rss_res or {})
                            except Exception:
                                _append_history('rss_check', 'error', {})

                            # SYMBOL_FLOW mode only (CONTINUOUS_FULL removed)
                            symbol_flow = str(os.getenv('SYMBOL_FLOW', '1')).lower() in ('1', 'true', 'yes')

                            # Result accumulator visible for last_run_stats
                            col_res = None

                            if symbol_flow:
                                # Sequential per-symbol: collect -> analyze -> next
                                try:
                                    from models import Stock
                                    from pattern_detector import HybridPatternDetector
                                    det = HybridPatternDetector()
                                    # Universe: ALL active stocks (watchlist zaten bu kümenin alt kümesidir)
                                    symbols: List[str] = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).all()]
                                    total_symbols = len(symbols)
                                    analyzed = 0
                                    collected_records = 0
                                    updated_records = 0
                                    _append_history('symbol_flow', 'start', {'total_symbols': total_symbols})
                                    _broadcast('INFO', f'🔁 SYMBOL_FLOW aktif: {total_symbols} sembol işlem sırası', 'pipeline')
                                    for symbol in symbols:
                                        if not self.is_running or self.stop_event.is_set():
                                            break
                                        # Skip symbols that recently returned no data until cooldown expires
                                        try:
                                            backoff_until = self.no_data_backoff.get(symbol)
                                            if backoff_until is not None and cycle_count < backoff_until:
                                                remaining = max(0, backoff_until - cycle_count)
                                                _broadcast('INFO', f'⏭️ Skip {symbol} (no-data cooldown {remaining} cycles left)', 'collector')
                                                time.sleep(0.01)
                                                continue
                                        except Exception:
                                            pass
                                        # Collect minimal recent data for symbol
                                        try:
                                            res = collector.collect_single_stock(symbol, period='auto')
                                            if isinstance(res, dict):
                                                success = bool(res.get('success'))
                                                recs = int(res.get('records', 0))
                                                upd = int(res.get('updated', 0))
                                                if success:
                                                    collected_records += recs
                                                    updated_records += upd
                                                    # Clear backoff if data arrived
                                                    try:
                                                        if symbol in self.no_data_backoff:
                                                            self.no_data_backoff.pop(symbol, None)
                                                    except Exception:
                                                        pass
                                                else:
                                                    # If explicitly no_data or zero rows, set cooldown
                                                    try:
                                                        err = str(res.get('error', ''))
                                                        if err == 'no_data' or recs == 0:
                                                            self.no_data_backoff[symbol] = cycle_count + cooldown_cycles
                                                            _broadcast('WARNING', f'no_data: {symbol} backoff until cycle {self.no_data_backoff[symbol]}', 'collector')
                                                    except Exception:
                                                        pass
                                        except Exception:
                                            pass
                                        # Analyze immediately after collection
                                        try:
                                            det.analyze_stock(symbol)
                                            analyzed += 1
                                            try:
                                                sock = getattr(flask_app, 'socketio', None)
                                                if sock is not None:
                                                    sock.emit('pattern_analysis', {
                                                        'symbol': symbol,
                                                        'data': None,
                                                        'timestamp': datetime.now().isoformat(),
                                                    })
                                            except Exception:
                                                pass
                                        except Exception:
                                            pass
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
                                if str(os.getenv('ENABLE_TRAINING_IN_CYCLE', '0')).lower() in ('1', 'true', 'yes'):
                                    logger.info('⚠️ Training-in-cycle enabled by env; consider disabling in production')
                                else:
                                    logger.info('⏭️ Skipping ML training in cycle (cron-only policy active)')
                            except Exception:
                                pass

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
                            except Exception:
                                self.last_run_stats = {'cycle': cycle_count, 'timestamp': datetime.now().isoformat()}
                            logger.info(
                                f"✅ Cycle {cycle_count} completed: "
                                f"collected={self.last_run_stats.get('total_records', 0)} "
                                f"updated={self.last_run_stats.get('updated_records', 0)} "
                                f"analyzed={analyzed}"
                            )
                            
                            # Ensure instance counter also reflects last completed
                            self.cycle_count = cycle_count
                        
                        # Wait for next cycle (environment-driven)
                        try:
                            sleep_total = int(os.getenv('AUTOMATION_CYCLE_SLEEP_SECONDS', '300'))
                        except Exception:
                            sleep_total = 300
                        for _ in range(sleep_total):
                            if self.stop_event.is_set() or not self.is_running:
                                break
                            time.sleep(1)
                            
                    except Exception as e:
                        logger.error(f"❌ Automation cycle error: {e}")
                        # Environment-driven error retry delay
                        try:
                            error_retry_delay = int(os.getenv('AUTOMATION_ERROR_RETRY_DELAY', '30'))
                        except Exception:
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
            except Exception:
                current_is_worker = False
            if self.thread and self.thread.is_alive() and not current_is_worker:
                self.thread.join(timeout=5)
            
            logger.info("🛑 Automation scheduler stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop automation: {e}")
            return False
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        try:
            skip_now = sum(1 for _sym, until in self.no_data_backoff.items() if until > self.cycle_count)
        except Exception:
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
        """System health check"""
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
                    symbols: List[str] = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).all()]
                    total = len(symbols)
                    added_total = 0
                    updated_total = 0
                    no_data = 0
                    errors = 0
                    try:
                        symbol_sleep_seconds = float(os.getenv('MANUAL_TASK_SYMBOL_SLEEP', '0.01'))  # Faster for manual tasks
                    except Exception:
                        symbol_sleep_seconds = 0.01
                    
                    # Manual data collection: Process ALL symbols (no limit)
                    limited_symbols = symbols
                    logger.info(f"📊 Manual data collection for ALL {len(symbols)} symbols")
                    
                    for i, sym in enumerate(limited_symbols):
                        try:
                            res = collector.collect_single_stock(sym, period='auto')
                            if isinstance(res, dict):
                                added_total += int(res.get('records', 0))
                                updated_total += int(res.get('updated', 0))
                                if not bool(res.get('success')) or (int(res.get('records', 0)) == 0 and int(res.get('updated', 0)) == 0):
                                    no_data += 1
                        except Exception:
                            errors += 1
                        
                        # Progress feedback every 10 symbols
                        if (i + 1) % 10 == 0:
                            try:
                                from app import app as flask_app
                                if hasattr(flask_app, 'broadcast_log'):
                                    flask_app.broadcast_log('INFO', f'📊 Manual collection progress: {i+1}/{len(limited_symbols)} symbols', 'collector')
                            except Exception:
                                pass
                        
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
                    from models import Stock, StockPrice  # type: ignore
                    from sqlalchemy import and_  # type: ignore
                    from datetime import timedelta
                    import pandas as pd
                    from bist_pattern.core.ml_coordinator import get_ml_coordinator
                    mlc = get_ml_coordinator()
                    try:
                        ignore_cooldown = str(os.getenv('MANUAL_IGNORE_COOLDOWN', '1')).lower() in ('1', 'true', 'yes')
                    except Exception:
                        ignore_cooldown = True
                    original_cd = getattr(mlc, 'training_cooldown_hours', 6)
                    if ignore_cooldown:
                        try:
                            mlc.training_cooldown_hours = 0
                        except Exception:
                            pass
                    symbols: List[str] = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).all()]
                    
                    # Manual model training: Process ALL symbols (no limit)
                    limited_symbols = symbols
                    logger.info(f"🧠 Manual training for ALL {len(symbols)} symbols")
                    attempts = 0
                    success = 0
                    skipped = 0
                    skip_breakdown: Dict[str, int] = {'insufficient_data': 0, 'enhanced_unavailable': 0, 'cooldown_active': 0, 'model_fresh_or_exists': 0, 'unknown': 0}
                    
                    for i, sym in enumerate(limited_symbols):
                        try:
                            stock_obj = Stock.query.filter_by(symbol=sym).first()
                            if not stock_obj:
                                skipped += 1
                                continue
                            cutoff = datetime.now() - timedelta(days=730)
                            rows = (
                                StockPrice.query
                                .filter(and_(StockPrice.stock_id == stock_obj.id, StockPrice.date >= cutoff))
                                .order_by(StockPrice.date.asc())
                                .all()
                            )
                            if not rows:
                                skip_breakdown['insufficient_data'] += 1
                                skipped += 1
                                continue
                            df_rows = [{
                                'date': r.date,
                                'open': float(r.open_price),
                                'high': float(r.high_price),
                                'low': float(r.low_price),
                                'close': float(r.close_price),
                                'volume': int(r.volume or 0),
                            } for r in rows]
                            df = pd.DataFrame(df_rows)
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                            ok, reason = mlc.evaluate_training_gate(sym, len(df))
                            if not ok and reason == 'cooldown_active' and ignore_cooldown:
                                ok = True
                            if not ok:
                                skip_breakdown[reason] = skip_breakdown.get(reason, 0) + 1
                                skipped += 1
                                continue
                            attempts += 1
                            if mlc.train_enhanced_model_if_needed(sym, df):
                                success += 1
                            
                            # Progress feedback every 5 symbols
                            if (i + 1) % 5 == 0:
                                try:
                                    from app import app as flask_app
                                    if hasattr(flask_app, 'broadcast_log'):
                                        flask_app.broadcast_log('INFO', f'🧠 Manual training progress: {i+1}/{len(limited_symbols)} symbols', 'ml_training')
                                except Exception:
                                    pass
                        except Exception:
                            skip_breakdown['unknown'] = skip_breakdown.get('unknown', 0) + 1
                            skipped += 1
                            continue
                    # restore cooldown
                    try:
                        mlc.training_cooldown_hours = original_cd
                    except Exception:
                        pass
                return {
                    'ok': True,
                    'result': {
                        'symbols_total': len(symbols),
                        'symbols_processed': len(limited_symbols),
                        'attempts': attempts,
                        'success': success,
                        'skipped': skipped,
                        'skip_breakdown': skip_breakdown,
                        'ignore_cooldown': bool(ignore_cooldown),
                        'timestamp': datetime.now().isoformat(),
                    }
                }
            return {'ok': False, 'error': f'Unknown task: {task_name}'}
        except Exception as e:
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
                
                lookback_days = int(os.getenv('VOLUME_LOOKBACK_DAYS', '30'))
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
                    return None
                
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
            return None

    # Optional: provide bulk predictions method used by some internal endpoints
    def run_bulk_predictions_all(self) -> Dict[str, Any]:  # pragma: no cover
        try:
            # Minimal placeholder to keep internal routes functional
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
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
                scope = os.getenv('COLLECTION_SCOPE', 'DB_ACTIVE')
                col_res = collector.collect_all_stocks_parallel(scope=scope)
                from pattern_detector import HybridPatternDetector
                from models import Stock
                det = HybridPatternDetector()
                symbols: List[str] = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).all()]
                analyzed = 0
                for symbol in symbols:
                    try:
                        det.analyze_stock(symbol)
                        analyzed += 1
                    except Exception:
                        pass
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


def get_working_automation_pipeline():
    """Get working automation pipeline singleton"""
    global _working_pipeline
    if _working_pipeline is None:
        _working_pipeline = WorkingAutomationPipeline()
    return _working_pipeline
