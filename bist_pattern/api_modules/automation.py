"""
Automation API Blueprint
Automated pipeline control and monitoring endpoints
"""

import os
import json
import logging
from datetime import datetime
from flask import Blueprint, jsonify, current_app, request
from flask_login import login_required
from bist_pattern.core.decorators import internal_route, admin_required

logger = logging.getLogger(__name__)

bp = Blueprint('automation_api', __name__, url_prefix='/api/automation')


def get_pipeline_with_context():
    """Get pipeline instance with app context (compat via working_automation)."""
    try:
        from working_automation import get_working_automation_pipeline  # type: ignore
        return get_working_automation_pipeline()
    except Exception:
        try:
            from scheduler import get_automated_pipeline  # fallback
            return get_automated_pipeline()
        except Exception:
            return None


@bp.route('/status')
def automation_status_simple():
    """Get automation status"""
    try:
        pipeline = get_pipeline_with_context()
        if not pipeline:
            return jsonify({
                'status': 'unavailable',
                'available': False,
                'automation': {'enabled': False, 'running': False},
                'scheduler_status': {'is_running': False, 'thread_alive': False, 'scheduled_jobs': 0},
                'mode': 'UNAVAILABLE',
                'message': 'Pipeline sistemi mevcut değil'
            })

        # Get scheduler status
        try:
            status_info = pipeline.get_scheduler_status() or {}
        except Exception:
            status_info = {}

        is_running = bool(status_info.get('is_running', False))
        thread_alive = bool(status_info.get('thread_alive', False))

        response = {
            'status': 'success',
            'available': True,
            'automation': {
                'enabled': True,
                'running': is_running,
                'thread_alive': thread_alive,
                'last_run_stats': status_info.get('last_run_stats', {}),
                'scheduled_jobs': status_info.get('scheduled_jobs', 0)
            },
            'scheduler_status': {
                'is_running': is_running,
                'thread_alive': thread_alive,
                'last_run_stats': status_info.get('last_run_stats', {}),
                'scheduled_jobs': status_info.get('scheduled_jobs', 0)
            },
            'mode': 'CONTINUOUS_FULL',
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(response)
    except Exception as e:
        current_app.logger.error(f"Automation status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@bp.route('/start', methods=['POST'])
@internal_route
def start_automation():
    """Start automated pipeline"""
    try:
        current_app.logger.info("🚀 Automation start request received")

        pipeline = get_pipeline_with_context()
        if not pipeline:
            current_app.logger.error("❌ Pipeline not available")
            return jsonify({
                'status': 'unavailable',
                'message': 'Automated Pipeline sistemi mevcut değil'
            }), 503

        # Check current status first
        try:
            current_status = pipeline.get_scheduler_status() or {}
            is_currently_running = bool(current_status.get('is_running', False))
        except Exception as status_err:
            current_app.logger.warning(f"⚠️ Status check error: {status_err}")
            is_currently_running = bool(getattr(pipeline, 'is_running', False))

        if is_currently_running:
            current_app.logger.info("ℹ️ Pipeline already running")
            return jsonify({
                'status': 'already_running',
                'message': 'Automated Pipeline zaten çalışıyor',
                'current_status': current_status
            })

        # Set continuous mode as default
        os.environ['PIPELINE_MODE'] = os.getenv('PIPELINE_MODE', 'CONTINUOUS_FULL')
        current_app.logger.info(f"🔧 Starting pipeline in mode: {os.environ['PIPELINE_MODE']}")

        # Persist automation start timestamp for reporting
        try:
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            os.makedirs(log_dir, exist_ok=True)
            started_path = os.path.join(log_dir, 'automation_started_at.json')
            with open(started_path, 'w') as f:
                json.dump({'started_at': datetime.now().isoformat()}, f)
        except Exception as e:
            current_app.logger.warning(f"Could not write automation_started_at.json: {e}")

        # Fire-and-forget başlatma: HTTP response hemen dön, işlem background'da devam etsin
        def start_pipeline_async(app_instance):
            # Use module logger to avoid context issues
            try:
                logger.info("🚀 Background thread starting pipeline...")
                success = pipeline.start_scheduler()
                if success:
                    logger.info("✅ Pipeline started successfully")
                    # Try WebSocket broadcast with app context
                    try:
                        with app_instance.app_context():
                            if hasattr(app_instance, 'broadcast_log'):
                                app_instance.broadcast_log('SUCCESS', '🚀 Automation started successfully', 'automation')
                    except Exception as e:
                        logger.warning(f"WebSocket broadcast failed: {e}")
                else:
                    logger.error("❌ Pipeline start failed")
                    try:
                        with app_instance.app_context():
                            if hasattr(app_instance, 'broadcast_log'):
                                app_instance.broadcast_log('ERROR', '❌ Automation start failed', 'automation')
                    except Exception as e:
                        logger.warning(f"WebSocket broadcast failed: {e}")
            except Exception as e:
                logger.error(f"❌ Background pipeline start error: {e}")
                try:
                    with app_instance.app_context():
                        if hasattr(app_instance, 'broadcast_log'):
                            app_instance.broadcast_log('ERROR', f'❌ Automation start error: {str(e)}', 'automation')
                except Exception as e2:
                    logger.warning(f"WebSocket broadcast failed: {e2}")

        # Start in background thread with app instance
        import threading
        thread = threading.Thread(target=start_pipeline_async, args=(current_app._get_current_object(),), daemon=True)
        thread.start()

        # Immediate response
        return jsonify({
            'status': 'starting',
            'message': 'Automated Pipeline başlatılıyor... Durum WebSocket üzerinden bildirilecek',
            'timestamp': datetime.now().isoformat(),
            'mode': os.environ.get('PIPELINE_MODE', 'CONTINUOUS_FULL')
        })

    except Exception as e:
        current_app.logger.error(f"❌ Automation start error: {e}")
        # Broadcast error to WebSocket
        try:
            if hasattr(current_app, 'broadcast_log'):
                current_app.broadcast_log('ERROR', f'❌ Automation start failed: {str(e)}', 'automation')
        except Exception:
            pass

        return jsonify({
            'status': 'error',
            'message': f'Automation başlatma hatası: {str(e)}'
        }), 500


@bp.route('/stop', methods=['POST'])
@internal_route
def stop_automation():
    """Stop automated pipeline"""
    try:
        current_app.logger.info("🛑 Automation stop request received")

        def _clear_pipeline_history_file():
            try:
                log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                os.makedirs(log_dir, exist_ok=True)
                status_file = os.path.join(log_dir, 'pipeline_status.json')
                with open(status_file, 'w') as f:
                    json.dump({'history': [], 'stopped_at': datetime.now().isoformat()}, f)
                return True
            except Exception as clear_err:
                current_app.logger.warning(f"⚠️ History clear error: {clear_err}")
                return False

        pipeline = get_pipeline_with_context()
        if not pipeline:
            current_app.logger.error("❌ Pipeline not available for stop")
            return jsonify({
                'status': 'unavailable',
                'message': 'Automated Pipeline sistemi mevcut değil'
            }), 503

        # Check current status first
        try:
            current_status = pipeline.get_scheduler_status() or {}
            is_currently_running = bool(current_status.get('is_running', False))
        except Exception as status_err:
            current_app.logger.warning(f"⚠️ Status check error: {status_err}")
            is_currently_running = bool(getattr(pipeline, 'is_running', False))

        if not is_currently_running:
            current_app.logger.info("ℹ️ Pipeline already stopped")
            _clear_pipeline_history_file()
            return jsonify({
                'status': 'already_stopped',
                'message': 'Automated Pipeline zaten durmuş',
                'current_status': current_status
            })

        current_app.logger.info("🔧 Stopping pipeline...")

        # Fire-and-forget durdurma: HTTP response hemen dön, işlem background'da devam etsin
        def stop_pipeline_async():
            try:
                success = pipeline.stop_scheduler()
                if success:
                    current_app.logger.info("✅ Pipeline stopped successfully")
                    _clear_pipeline_history_file()
                    # Broadcast to WebSocket
                    try:
                        if hasattr(current_app, 'broadcast_log'):
                            current_app.broadcast_log('SUCCESS', '🛑 Automation stopped successfully', 'automation')
                    except Exception:
                        pass
                else:
                    current_app.logger.error("❌ Pipeline stop failed")
                    _clear_pipeline_history_file()
                    try:
                        if hasattr(current_app, 'broadcast_log'):
                            current_app.broadcast_log('ERROR', '❌ Automation stop failed', 'automation')
                    except Exception:
                        pass
            except Exception as e:
                current_app.logger.error(f"❌ Background pipeline stop error: {e}")
                try:
                    if hasattr(current_app, 'broadcast_log'):
                        current_app.broadcast_log('ERROR', f'❌ Automation stop error: {str(e)}', 'automation')
                except Exception:
                    pass

        # Stop in background thread
        import threading
        thread = threading.Thread(target=stop_pipeline_async, daemon=True)
        thread.start()

        # Immediate response
        return jsonify({
            'status': 'stopping',
            'message': 'Automated Pipeline durduruluyor... Durum WebSocket üzerinden bildirilecek',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        current_app.logger.error(f"❌ Automation stop error: {e}")
        # Broadcast error to WebSocket
        try:
            if hasattr(current_app, 'broadcast_log'):
                current_app.broadcast_log('ERROR', f'❌ Automation stop failed: {str(e)}', 'automation')
        except Exception:
            pass

        return jsonify({
            'status': 'error',
            'message': f'Automation durdurma hatası: {str(e)}'
        }), 500


@bp.route('/health')
def automation_health():
    """Pipeline-first health (no file fallback)."""
    try:
        pipeline = get_pipeline_with_context()
        if not pipeline or not hasattr(pipeline, 'system_health_check'):
            return jsonify({'status': 'unavailable', 'message': 'Pipeline unavailable'}), 503
        try:
            health_data = pipeline.system_health_check() or {}
        except Exception as _e:
            return jsonify({'status': 'unavailable', 'message': f'Health provider error: {str(_e)}'}), 503

        # Enrich systems
        health_data.setdefault('systems', {})
        health_data['systems']['flask_api'] = {'status': 'healthy', 'details': 'API is responsive'}
        health_data['systems']['websocket'] = {'status': 'connected', 'details': 'Socket.IO is active'}
        try:
            from sqlalchemy import text
            from models import db
            db.session.execute(text('SELECT 1'))
            health_data['systems']['database'] = {'status': 'connected'}
        except Exception as db_err:
            health_data['systems']['database'] = {'status': 'error', 'details': str(db_err)}
        # Automation engine
        try:
            status_map = pipeline.get_scheduler_status() if hasattr(pipeline, 'get_scheduler_status') else {}
            is_running = bool(status_map.get('is_running', getattr(pipeline, 'is_running', False)))
            health_data['systems']['automation_engine'] = {'status': 'running' if is_running else 'stopped'}
        except Exception as auto_err:
            health_data['systems']['automation_engine'] = {'status': 'error', 'details': str(auto_err)}

        # System resources (CPU/RAM/Disk)
        try:
            import psutil  # type: ignore
            cpu_percent = float(psutil.cpu_percent(interval=0.1))
            vm = psutil.virtual_memory()
            mem_percent = float(getattr(vm, 'percent', 0.0))
            mem_total_mb = float(getattr(vm, 'total', 0.0)) / (1024 * 1024)
            mem_used_mb = float(getattr(vm, 'used', 0.0)) / (1024 * 1024)
            du = psutil.disk_usage('/')
            disk_percent = float(getattr(du, 'percent', 0.0))
            disk_free_gb = float(getattr(du, 'free', 0.0)) / (1024 * 1024 * 1024)
            disk_used_gb = float(getattr(du, 'used', 0.0)) / (1024 * 1024 * 1024)
        except Exception:
            cpu_percent = None
            mem_percent = None
            mem_total_mb = None
            mem_used_mb = None
            disk_percent = None
            disk_free_gb = None
            disk_used_gb = None
        try:
            load_avg = None
            if hasattr(os, 'getloadavg'):
                la = os.getloadavg()
                load_avg = [float(la[0]), float(la[1]), float(la[2])]
        except Exception:
            load_avg = None
        health_data['system_resources'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': mem_percent,
            'memory_total_mb': mem_total_mb,
            'memory_used_mb': mem_used_mb,
            'disk_percent': disk_percent,
            'disk_free_gb': disk_free_gb,
            'disk_used_gb': disk_used_gb,
            'load_avg': load_avg,
        }

        # Derive overall_status for UI (healthy/warning/critical)
        try:
            def _norm(val: str) -> str:
                return (str(val or '').strip().lower())

            s = health_data.get('systems', {}) or {}
            st_db = _norm(s.get('database', {}).get('status'))
            st_api = _norm(s.get('flask_api', {}).get('status'))
            st_ws = _norm(s.get('websocket', {}).get('status'))
            st_auto = _norm(s.get('automation_engine', {}).get('status'))

            critical = any(x in ('error', 'disconnected') for x in (st_db, st_api))
            if critical:
                overall = 'critical'
            else:
                healthy = (st_db in ('healthy', 'connected')) and (st_api == 'healthy') and (st_ws in ('connected', 'healthy'))
                if healthy and st_auto == 'running':
                    overall = 'healthy'
                elif healthy and st_auto != 'running':
                    overall = 'warning'
                else:
                    overall = 'warning'
            health_data['overall_status'] = overall
        except Exception:
            # Best-effort; default unknown if derivation fails
            health_data['overall_status'] = 'unknown'

        return jsonify({'status': 'success', 'health_check': health_data})
    except Exception as e:
        current_app.logger.error(f"Health endpoint error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# NOTE: /run-task endpoint moved to api_internal blueprint 
# This duplicate route removed to prevent conflicts


@bp.route('/report')
@login_required
@admin_required
def automation_report():
    """Daily system report"""
    try:
        pipeline = get_pipeline_with_context()
        if not pipeline:
            return jsonify({
                'status': 'unavailable',
                'message': 'Pipeline sistemi mevcut değil'
            }), 503

        report = pipeline.daily_status_report()
        # Add global metrics since last automation start
        try:
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            started_path = os.path.join(log_dir, 'automation_started_at.json')
            since_dt = None
            if os.path.exists(started_path):
                try:
                    with open(started_path, 'r') as f:
                        since_dt = json.load(f).get('started_at')
                except Exception:
                    since_dt = None
            # Load pipeline history for cycles and analyzed counts
            status_file = os.path.join(log_dir, 'pipeline_status.json')
            cycles = 0
            analyzed_total = 0
            updated_total = 0
            collected_total = 0
            earliest_ts = None
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        ph = json.load(f) or {}
                        hist = ph.get('history', [])
                        for e in hist:
                            if e.get('phase') == 'symbol_flow' and e.get('state') == 'end':
                                if since_dt and e.get('timestamp') and e['timestamp'] < since_dt:
                                    continue
                                d = e.get('details', {}) or {}
                                cycles += 1
                                analyzed_total += int(d.get('analyzed', 0) or 0)
                                updated_total += int(d.get('updated_records', 0) or 0)
                                collected_total += int(d.get('collected_records', 0) or 0)
                            # Track earliest timestamp for UI if since_dt unknown
                            try:
                                ts = e.get('timestamp')
                                if ts and (earliest_ts is None or ts < earliest_ts):
                                    earliest_ts = ts
                            except Exception:
                                pass
                except Exception:
                    pass
            # Load ML model status for training counts
            train_total = 0
            ml_status = os.path.join(log_dir, 'ml_model_status.json')
            if os.path.exists(ml_status):
                try:
                    with open(ml_status, 'r') as f:
                        ms = json.load(f) or {}
                        for _, v in (ms.items() if isinstance(ms, dict) else []):
                            ts = v.get('last_training_attempt')
                            if ts and (not since_dt or ts >= since_dt):
                                train_total += 1
                except Exception:
                    pass
            # Load signals snapshot for signal count
            signal_total = 0
            try:
                sig_path = os.path.join(log_dir, 'signals_last.json')
                if os.path.exists(sig_path):
                    with open(sig_path, 'r') as f:
                        ss = json.load(f) or {}
                        signal_total = len(ss.keys()) if isinstance(ss, dict) else 0
            except Exception:
                pass
            # YOLO/AI breakdown from recent broadcasts is not persisted; approximate via patterns in signals_last
            yolo_total = 0
            try:
                # Count VISUAL evidence as YOLO hits
                sig_path = os.path.join(log_dir, 'signals_last.json')
                if os.path.exists(sig_path):
                    with open(sig_path, 'r') as f:
                        ss = json.load(f) or {}
                        for _, sv in (ss.items() if isinstance(ss, dict) else []):
                            vis = sv.get('visual') or []
                            if vis:
                                yolo_total += len(vis)
            except Exception:
                pass
            # Attach to report
            report = report or {}
            report['since_started'] = since_dt or earliest_ts or None
            report['aggregates'] = {
                'cycles': cycles,
                'analyzed': analyzed_total,
                'collected': collected_total,
                'updated': updated_total,
                'train_total': train_total,
                'signals_total': signal_total,
                'yolo_detections': yolo_total,
                'enhanced_analyses': None,
                'basic_analyses': None,
            }
        except Exception as agg_err:
            logger.warning(f"Aggregate metrics build failed: {agg_err}")
            # Ensure aggregates key exists for UI fallback
            try:
                report = report or {}
                report.setdefault('since_started', None)
                report.setdefault('aggregates', {
                    'cycles': 0,
                    'analyzed': 0,
                    'collected': 0,
                    'updated': 0,
                    'train_total': 0,
                    'signals_total': 0,
                    'yolo_detections': 0,
                    'enhanced_analyses': None,
                    'basic_analyses': None,
                })
            except Exception:
                pass
        # Enrich report with volume tiers (20-day average volume based)
        try:
            from models import db, Stock, StockPrice  # type: ignore
            from sqlalchemy import func  # type: ignore
            from datetime import timedelta
            lookback_days = int(os.getenv('VOLUME_LOOKBACK_DAYS', '30'))
            cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).date()
            rows = (
                db.session.query(Stock.symbol, Stock.name, func.avg(StockPrice.volume).label('avg_vol'))
                .join(StockPrice, Stock.id == StockPrice.stock_id)
                .filter(Stock.is_active.is_(True), StockPrice.date >= cutoff_date)
                .group_by(Stock.id, Stock.symbol, Stock.name)
                .all()
            )
            vols = [float(r[2] or 0) for r in rows]

            def _pct(values, p):
                try:
                    if not values:
                        return 0.0
                    s = sorted(values)
                    k = (len(s) - 1) * (p / 100.0)
                    f = int(k)
                    c = min(f + 1, len(s) - 1)
                    if f == c:
                        return float(s[int(k)])
                    d0 = s[f] * (c - k)
                    d1 = s[c] * (k - f)
                    return float(d0 + d1)
                except Exception:
                    return 0.0
            p15 = _pct(vols, 15)
            p40 = _pct(vols, 40)
            p75 = _pct(vols, 75)
            p95 = _pct(vols, 95)

            def _tier(v):
                try:
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
                except Exception:
                    return 'very_low'
            sym_list = []
            summary = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
            for sym, name, avg in rows:
                t = _tier(avg)
                summary[t] = summary.get(t, 0) + 1
                sym_list.append({'symbol': sym, 'name': name, 'avg_volume': float(avg or 0), 'tier': t})
            report = report or {}
            report['volume'] = {
                'lookback_days': lookback_days,
                'summary': summary,
                'percentiles': {'p15': p15, 'p40': p40, 'p75': p75, 'p95': p95},
                'symbols': sym_list,
            }
        except Exception as vol_err:
            logger.warning(f"Volume tiers enrichment failed: {vol_err}")

        if report:
            resp = jsonify({
                'status': 'success',
                'report': report,
                'timestamp': datetime.now().isoformat()
            })
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return resp
        else:
            return jsonify({
                'status': 'error',
                'message': 'Rapor oluşturulamadı'
            }), 500

    except Exception as e:
        current_app.logger.error(f"Automation report error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Rapor hatası: {str(e)}'
        }), 500


@bp.route('/pipeline-history')
def automation_pipeline_history():
    """Get pipeline execution history"""
    try:
        log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        status_file = os.path.join(log_dir, 'pipeline_status.json')

        history = []
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f) or {}
                    history = data.get('history', []) if isinstance(data, dict) else []
            except Exception:
                history = []

        # Recent tasks from automation system
        tasks = []
        try:
            pipeline = get_pipeline_with_context()
            if pipeline and hasattr(pipeline, 'last_run_stats'):
                stats = pipeline.last_run_stats or {}
                for task_name, task_data in stats.items():
                    if isinstance(task_data, dict):
                        tasks.append({
                            'task': task_name,
                            'data': task_data,
                            'timestamp': task_data.get('timestamp', 'Unknown')
                        })
        except Exception:
            pass

        resp = jsonify({'status': 'success', 'history': history, 'tasks': tasks})
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


# Duplicate route removed - already exists above at line 309


def register(app):
    """Register automation API blueprint"""
    app.register_blueprint(bp)


@bp.route('/volume/tiers')
def volume_tiers():
    """Public endpoint for volume tiers and optional per-symbol tier.
    Returns percentiles across active stocks and, if ?symbol= is provided,
    the 30-day average volume and its tier for that symbol.
    """
    try:
        from models import db, Stock, StockPrice  # type: ignore
        from sqlalchemy import func  # type: ignore
        from datetime import timedelta
        lookback_days = int(os.getenv('VOLUME_LOOKBACK_DAYS', '30'))
        cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).date()

        # Aggregate averages for all active stocks
        rows = (
            db.session.query(Stock.symbol, func.avg(StockPrice.volume).label('avg_vol'))
            .join(StockPrice, Stock.id == StockPrice.stock_id)
            .filter(Stock.is_active.is_(True), StockPrice.date >= cutoff_date)
            .group_by(Stock.id, Stock.symbol)
            .all()
        )
        vols = [float(r[1] or 0) for r in rows]

        def _pct(values, p):
            try:
                if not values:
                    return 0.0
                s = sorted(values)
                k = (len(s) - 1) * (p / 100.0)
                f = int(k)
                c = min(f + 1, len(s) - 1)
                if f == c:
                    return float(s[int(k)])
                d0 = s[f] * (c - k)
                d1 = s[c] * (k - f)
                return float(d0 + d1)
            except Exception:
                return 0.0

        p15 = _pct(vols, 15)
        p40 = _pct(vols, 40)
        p75 = _pct(vols, 75)
        p95 = _pct(vols, 95)

        def _tier(v):
            try:
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
            except Exception:
                return 'very_low'

        resp = {
            'status': 'success',
            'lookback_days': lookback_days,
            'percentiles': {'p15': p15, 'p40': p40, 'p75': p75, 'p95': p95},
        }

        sym = (request.args.get('symbol') or '').upper().strip()
        if sym:
            try:
                sym_avg = (
                    db.session.query(func.avg(StockPrice.volume))
                    .join(Stock, Stock.id == StockPrice.stock_id)
                    .filter(Stock.symbol == sym, Stock.is_active.is_(True), StockPrice.date >= cutoff_date)
                    .scalar()
                )
                sym_avg = float(sym_avg or 0)
            except Exception:
                sym_avg = 0.0
            resp['symbol'] = sym
            resp['avg_volume'] = sym_avg
            resp['tier'] = _tier(sym_avg)
        else:
            # If no symbol is provided, also return summary distribution
            summary = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
            try:
                for s, avg in rows:
                    t = _tier(float(avg or 0))
                    summary[t] = summary.get(t, 0) + 1
            except Exception:
                pass
            resp['summary'] = summary

        # Disable caching to ensure fresh values in UI
        out = jsonify(resp)
        out.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return out
    except Exception as e:
        current_app.logger.error(f"volume_tiers error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
