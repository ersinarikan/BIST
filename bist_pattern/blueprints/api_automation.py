from flask import Blueprint, jsonify, request
from datetime import datetime
import os
import time

bp = Blueprint('api_automation', __name__, url_prefix='/api/automation')


def register(app):
    from ..extensions import csrf

    # Lazy helpers to avoid circular imports with the app factory
    def _is_pipeline_available() -> bool:
        try:
            from app import AUTOMATED_PIPELINE_AVAILABLE as _APA  # type: ignore
            return bool(_APA)
        except Exception:
            return True

    def _get_pipeline():
        try:
            from app import get_pipeline_with_context as _gpwc  # type: ignore
            return _gpwc()
        except Exception:
            return None
    try:
        from models import db
    except Exception:
        db = None

    @bp.route('/status')
    def automation_status():
        try:
            cache_ttl_seconds = int(os.getenv('STATUS_CACHE_TTL', '15'))
            use_cache = (request.args.get('cache') or '1') in ('1', 'true', 'yes')
            if not hasattr(app, '_status_cache'):
                app._status_cache = {'ts': 0.0, 'payload': None}
            if use_cache and app._status_cache.get('payload') is not None:
                if (time.time() - float(app._status_cache.get('ts') or 0)) < cache_ttl_seconds:
                    return jsonify(app._status_cache['payload'])
            internal_status = {
                'is_running': False,
                'scheduled_jobs': 0,
                'next_runs': [],
                'thread_alive': False,
            }
            if _is_pipeline_available():
                try:
                    pipeline = _get_pipeline()
                    if pipeline and hasattr(pipeline, 'get_scheduler_status'):
                        internal_status = pipeline.get_scheduler_status()
                except Exception:
                    pass
            # External running flag (Redis/file) + grace window
            try:
                effective_running = bool(internal_status.get('is_running'))
                if pipeline and hasattr(pipeline, '_is_running_flag_effective'):
                    try:
                        if pipeline._is_running_flag_effective():  # type: ignore[attr-defined]
                            effective_running = True
                    except Exception:
                        pass
                grace_seconds = int(os.getenv('STATUS_GRACE_SECONDS', '5'))
                if effective_running:
                    setattr(app, '_last_running_ts', time.time())
                else:
                    last_ts = getattr(app, '_last_running_ts', 0)
                    if last_ts and (time.time() - float(last_ts)) < grace_seconds:
                        effective_running = True
                internal_status['is_running'] = bool(effective_running)
            except Exception:
                pass
            payload = {
                'status': 'success',
                'available': True,
                'scheduler_status': internal_status,
                'external_scheduler': {'is_running': False, 'message': 'removed'},
                'timestamp': datetime.now().isoformat(),
                'mode': 'CONTINUOUS_FULL',
            }
            if use_cache:
                app._status_cache = {'ts': time.time(), 'payload': payload}
            return jsonify(payload)
        except Exception as e:
            app.logger.error(f"Automation status error: {e}")
            return jsonify({'status': 'error', 'message': f'Automation status hatası: {str(e)}'}), 500

    @bp.route('/health')
    def automation_health():
        try:
            import json
            health_status_path = os.path.join(
                os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs'),
                'health_status.json',
            )
            if not os.path.exists(health_status_path):
                return jsonify({
                    'status': 'unavailable',
                    'message': 'Health status not available yet. Scheduler may be starting.',
                }), 503
            with open(health_status_path, 'r') as f:
                health_data = json.load(f)
            health_data.setdefault('systems', {})
            health_data['systems']['flask_api'] = {
                'status': 'healthy',
                'details': 'API is responsive',
            }
            health_data['systems']['websocket'] = {
                'status': 'connected',
                'details': 'Socket.IO is active',
            }
            try:
                if db is not None:
                    from sqlalchemy import text
                    db.session.execute(text('SELECT 1'))
                    health_data['systems']['database'] = {'status': 'connected'}
                else:
                    health_data['systems']['database'] = {'status': 'unknown'}
            except Exception as _db_err:
                health_data['systems']['database'] = {
                    'status': 'error',
                    'details': str(_db_err),
                }
            try:
                pipeline = _get_pipeline()
                is_running = bool(pipeline and getattr(pipeline, 'is_running', False))
                health_data['systems']['automation_engine'] = {
                    'status': 'running' if is_running else 'stopped'
                }
            except Exception as _auto_err:
                health_data['systems']['automation_engine'] = {
                    'status': 'error',
                    'details': str(_auto_err),
                }
            return jsonify({'status': 'success', 'health_check': health_data})
        except Exception as e:
            app.logger.error(f"Health check read error: {e}")
            return jsonify({'status': 'error', 'message': f'Could not read health status file: {str(e)}'}), 500

    @bp.route('/pipeline-history')
    def pipeline_history():
        try:
            import json
            log_path = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            status_file = os.path.join(log_path, 'pipeline_status.json')
            if not os.path.exists(status_file):
                return jsonify({'status': 'success', 'history': []})
            with open(status_file, 'r') as rf:
                data = json.load(rf) or {}
            hist = (data.get('history') or [])[-50:]
            return jsonify({'status': 'success', 'history': hist})
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)})

    # NOTE: /run-task endpoint moved to api_internal blueprint to avoid conflicts
    # This duplicate route has been removed to prevent routing conflicts

    @bp.route('/report')
    def automation_report():
        try:
            if not _is_pipeline_available():
                return jsonify({'status': 'unavailable', 'message': 'Automated Pipeline sistemi mevcut değil'}), 503
            pipeline = _get_pipeline()
            if not pipeline or not hasattr(pipeline, 'daily_status_report'):
                return jsonify({'status': 'error', 'message': 'Pipeline unavailable'}), 503
            report = pipeline.daily_status_report()
            return jsonify({
                'status': 'success',
                'report': report,
                'timestamp': datetime.now().isoformat(),
                'last_run_stats': getattr(pipeline, 'last_run_stats', {}),
            })
        except Exception as e:
            app.logger.error(f"Report generation error: {e}")
            return jsonify({'status': 'error', 'message': f'Rapor oluşturma hatası: {str(e)}'}), 500

    @bp.route('/start', methods=['POST'])
    @csrf.exempt
    def start_automation():
        try:
            if not _is_pipeline_available():
                return jsonify({'status': 'unavailable', 'message': 'Automated Pipeline sistemi mevcut değil'}), 503
            pipeline = _get_pipeline()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline unavailable'}), 503
            if getattr(pipeline, 'is_running', False):
                return jsonify({'status': 'already_running', 'message': 'Automated Pipeline zaten çalışıyor'})
            os.environ['PIPELINE_MODE'] = os.getenv('PIPELINE_MODE', 'CONTINUOUS_FULL')
            ok = pipeline.start_scheduler()
            if ok:
                return jsonify({'status': 'started', 'message': 'Automated Pipeline başlatıldı', 'timestamp': datetime.now().isoformat()})
            return jsonify({'status': 'error', 'message': 'Automated Pipeline başlatılamadı'}), 500
        except Exception as e:
            app.logger.error(f"Automation start error: {e}")
            return jsonify({'status': 'error', 'message': f'Automation başlatma hatası: {str(e)}'}), 500

    @bp.route('/stop', methods=['POST'])
    @csrf.exempt
    def stop_automation():
        try:
            if not _is_pipeline_available():
                return jsonify({'status': 'unavailable', 'message': 'Automated Pipeline sistemi mevcut değil'}), 503
            pipeline = _get_pipeline()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline unavailable'}), 503
            if not getattr(pipeline, 'is_running', False):
                return jsonify({'status': 'already_stopped', 'message': 'Automated Pipeline zaten durmuş'})
            ok = pipeline.stop_scheduler()
            if ok:
                return jsonify({'status': 'stopped', 'message': 'Automated Pipeline durduruldu', 'timestamp': datetime.now().isoformat()})
            return jsonify({'status': 'error', 'message': 'Automated Pipeline durdurulamadı'}), 500
        except Exception as e:
            app.logger.error(f"Automation stop error: {e}")
            return jsonify({'status': 'error', 'message': f'Automation durdurma hatası: {str(e)}'}), 500

    @bp.route('/volume/tiers')
    def volume_tiers():
        """Get volume tier information for a specific symbol or distribution.

        - If query param `symbol` is provided, returns the 30-day average volume and
          its tier for that symbol along with global percentiles.
        - If not provided, returns the summary distribution counts across all symbols
          and global percentiles.
        """
        try:
            from models import db, Stock, StockPrice  # type: ignore
            from sqlalchemy import func  # type: ignore
            from datetime import timedelta

            lookback_days = int(os.getenv('VOLUME_LOOKBACK_DAYS', '30'))
            cutoff = datetime.utcnow().date() - timedelta(days=lookback_days)

            # Compute average volumes across all active symbols
            rows = (
                db.session.query(Stock.symbol, Stock.name, func.avg(StockPrice.volume).label('avg_vol'))
                .join(StockPrice, Stock.id == StockPrice.stock_id)
                .filter(Stock.is_active.is_(True), StockPrice.date >= cutoff)
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

            resp = {
                'status': 'success',
                'lookback_days': lookback_days,
                'percentiles': {'p15': p15, 'p40': p40, 'p75': p75, 'p95': p95},
            }

            sym = (request.args.get('symbol') or '').upper().strip()
            if sym:
                try:
                    sym_row = (
                        db.session.query(func.avg(StockPrice.volume).label('avg_vol'))
                        .join(Stock, Stock.id == StockPrice.stock_id)
                        .filter(Stock.symbol == sym, Stock.is_active.is_(True), StockPrice.date >= cutoff)
                        .one_or_none()
                    )
                    sym_avg = float((sym_row[0] if sym_row else 0) or 0)
                except Exception:
                    sym_avg = 0.0
                resp['symbol'] = sym
                resp['avg_volume'] = sym_avg
                resp['tier'] = _tier(sym_avg)
            else:
                # Return distribution summary across all symbols
                summary = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
                try:
                    for _sym, _name, avg in rows:
                        summary[_tier(avg)] = summary.get(_tier(avg), 0) + 1
                except Exception:
                    pass
                resp['summary'] = summary

            out = jsonify(resp)
            out.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return out
        except Exception as e:
            app.logger.error(f"Volume tiers API error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    app.register_blueprint(bp)
