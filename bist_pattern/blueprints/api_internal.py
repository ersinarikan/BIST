from flask import Blueprint, jsonify, request
import os
from datetime import datetime
from functools import wraps

bp = Blueprint('api_internal', __name__, url_prefix='/api/internal')


def _allow_or_token(app):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                configured_token = app.config.get('INTERNAL_API_TOKEN')
                header_token = request.headers.get('X-Internal-Token')
                remote_ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
                is_local = remote_ip in ('127.0.0.1', '::1', 'localhost')
                allow_localhost = str(os.getenv('INTERNAL_ALLOW_LOCALHOST', str(app.config.get('INTERNAL_ALLOW_LOCALHOST', 'True')))).lower() == 'true'
                
                # Debug logging
                match_token = (header_token == configured_token) if configured_token else 'N/A'
                app.logger.warning(
                    f"Auth Debug: token_config={bool(configured_token)}, header_token={bool(header_token)}, "
                    f"match={match_token}, remote_ip='{remote_ip}', is_local={is_local}, allow_localhost={allow_localhost}"
                )
                
                if configured_token and header_token == configured_token:
                    return func(*args, **kwargs)
                if allow_localhost and is_local:
                    return func(*args, **kwargs)
            except Exception as e:
                app.logger.error(f"Auth exception: {e}")
                pass
            return jsonify({'status': 'forbidden'}), 403
        return wrapper
    return decorator


def register(app):
    from ..extensions import csrf
    try:
        csrf.exempt(bp)
    except Exception:
        pass

    @bp.route('/broadcast-log', methods=['POST'])
    @csrf.exempt  # keep CSRF exempt
    def broadcast_log():
        try:
            data = request.get_json() or {}
            level = data.get('level', 'INFO')
            message = data.get('message', '')
            category = data.get('category', 'system')
            token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            try:
                remote_ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
                is_local = remote_ip in ('127.0.0.1', '::1', 'localhost')
                allow_localhost = str(os.getenv('INTERNAL_ALLOW_LOCALHOST', str(app.config.get('INTERNAL_ALLOW_LOCALHOST', 'True')))).lower() == 'true'
            except Exception:
                is_local, allow_localhost = False, True
            if expected:
                if token != expected and not (allow_localhost and is_local):
                    return jsonify({'status': 'unauthorized'}), 401
            elif not is_local:
                return jsonify({'status': 'unauthorized'}), 401
            app.broadcast_log(level, message, category)
            return jsonify({'status': 'success', 'message': 'Log broadcasted'})
        except Exception as e:
            app.logger.error(f"Internal broadcast error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/broadcast-user-signal', methods=['POST'])
    @_allow_or_token(app)
    def broadcast_user_signal():
        try:
            token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                return jsonify({'status': 'unauthorized'}), 401
            data = request.get_json() or {}
            user_id = data.get('user_id')
            signal_data = data.get('signal_data')
            if not user_id or not signal_data:
                return jsonify({'status': 'error', 'error': 'user_id and signal_data are required'}), 400
            room = f'user_{user_id}'
            app.socketio.emit('user_signal', {
                'user_id': user_id,
                'signal': signal_data,
                'timestamp': datetime.now().isoformat()
            }, to=room)
            return jsonify({'status': 'success', 'message': f'signal broadcasted to {room}'})
        except Exception as e:
            app.logger.error(f"Internal user signal broadcast error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/automation/<action>', methods=['POST'])
    @_allow_or_token(app)
    def automation_control(action):
        try:
            from app import AUTOMATED_PIPELINE_AVAILABLE, get_pipeline_with_context  # lazy
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({'status': 'unavailable', 'message': 'Automated Pipeline sistemi mevcut değil'}), 503
            pipeline = get_pipeline_with_context()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline not initialized'}), 500
            action = (action or '').lower()
            if action == 'start':
                if getattr(pipeline, 'is_running', False):
                    return jsonify({'status': 'already_running'})
                ok = pipeline.start_scheduler()
                return jsonify({'status': 'started' if ok else 'error'})
            if action == 'stop':
                if not getattr(pipeline, 'is_running', False):
                    return jsonify({'status': 'already_stopped'})
                ok = pipeline.stop_scheduler()
                return jsonify({'status': 'stopped' if ok else 'error'})
            return jsonify({'status': 'error', 'message': 'invalid action'}), 400
        except Exception as e:
            app.logger.error(f"Internal automation error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/automation/status')
    @_allow_or_token(app)
    def automation_status():
        try:
            from app import AUTOMATED_PIPELINE_AVAILABLE, get_pipeline_with_context  # lazy
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({'status': 'unavailable'}), 503
            pipeline = get_pipeline_with_context()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline not initialized'}), 500
            return jsonify({'status': 'success', 'scheduler_status': pipeline.get_scheduler_status()})
        except Exception as e:
            app.logger.error(f"Internal automation status error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Internal health endpoint (mirrors public /api/automation/health)
    @bp.route('/automation/health')
    @_allow_or_token(app)
    def automation_health_internal():
        try:
            # Prefer pipeline-provided health if available
            try:
                from app import get_pipeline_with_context  # type: ignore
                pipeline = get_pipeline_with_context()
            except Exception:
                pipeline = None

            health_data = {'systems': {}}
            # API/WebSocket assumed healthy if this endpoint responds
            health_data['systems']['flask_api'] = {'status': 'healthy'}
            health_data['systems']['websocket'] = {'status': 'connected'}

            # DB connectivity quick check
            try:
                from sqlalchemy import text  # type: ignore
                from models import db  # type: ignore
                db.session.execute(text('SELECT 1'))
                health_data['systems']['database'] = {'status': 'connected'}
            except Exception as db_err:
                health_data['systems']['database'] = {'status': 'error', 'details': str(db_err)}

            # Automation engine
            try:
                status_map = pipeline.get_scheduler_status() if (pipeline and hasattr(pipeline, 'get_scheduler_status')) else {}
                is_running = bool(status_map.get('is_running', getattr(pipeline, 'is_running', False)))
                health_data['systems']['automation_engine'] = {'status': 'running' if is_running else 'stopped'}
            except Exception as auto_err:
                health_data['systems']['automation_engine'] = {'status': 'error', 'details': str(auto_err)}

            # System resources via psutil (best-effort)
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
                import os as _os
                if hasattr(_os, 'getloadavg'):
                    la = _os.getloadavg()
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

            # Derive overall status similar to public endpoint
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
            except Exception:
                overall = 'unknown'
            health_data['overall_status'] = overall

            return jsonify({'status': 'success', 'health_check': health_data})
        except Exception as e:
            app.logger.error(f"Internal health endpoint error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Internal pipeline history (alias of public /api/automation/pipeline-history)
    @bp.route('/automation/pipeline-history')
    @_allow_or_token(app)
    def automation_pipeline_history_internal():
        try:
            import os as _os
            import json as _json
            log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            status_file = _os.path.join(log_dir, 'pipeline_status.json')

            history = []
            if _os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        data = _json.load(f) or {}
                        history = data.get('history', []) if isinstance(data, dict) else []
                except Exception:
                    history = []

            resp = jsonify({'status': 'success', 'history': history, 'tasks': []})
            resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return resp
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Internal alias for volume tiers (mirrors public /api/automation/volume/tiers)
    @bp.route('/automation/volume/tiers')
    @_allow_or_token(app)
    def automation_volume_tiers_internal():
        try:
            from models import db, Stock, StockPrice  # type: ignore
            from sqlalchemy import func  # type: ignore
            from datetime import timedelta
            import os as _os
            from datetime import datetime as _dt

            lookback_days = int(_os.getenv('VOLUME_LOOKBACK_DAYS', '30'))
            cutoff_date = (_dt.utcnow() - timedelta(days=lookback_days)).date()

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
                summary = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
                try:
                    for s, avg in rows:
                        t = _tier(float(avg or 0))
                        summary[t] = summary.get(t, 0) + 1
                except Exception:
                    pass
                resp['summary'] = summary

            out = jsonify(resp)
            out.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            return out
        except Exception as e:
            app.logger.error(f"internal volume_tiers error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @bp.route('/bulk-predictions/status')
    @_allow_or_token(app)
    def bulk_predictions_status():
        """Return freshness info about ml_bulk_predictions.json (internal only)."""
        try:
            import os as _os
            import json as _json
            import time as _time
            log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            path = _os.path.join(log_dir, 'ml_bulk_predictions.json')
            exists = _os.path.exists(path)
            size = int(_os.path.getsize(path)) if exists else 0
            mtime = float(_os.path.getmtime(path)) if exists else None
            age_seconds = (float(_time.time()) - mtime) if mtime else None
            count = None
            stale = None
            if exists:
                try:
                    with open(path, 'r') as rf:
                        data = _json.load(rf) or {}
                    preds = (data.get('predictions') or {}) if isinstance(data, dict) else {}
                    if isinstance(preds, dict):
                        count = len(preds)
                except Exception:
                    pass
                try:
                    ttl = int(_os.getenv('BULK_PREDICTIONS_TTL_SECONDS', '7200'))
                    if age_seconds is not None:
                        stale = bool(age_seconds > ttl)
                except Exception:
                    stale = None
            return jsonify({
                'status': 'success',
                'path': path,
                'exists': bool(exists),
                'size': size,
                'mtime': mtime,
                'age_seconds': age_seconds,
                'stale': stale,
                'predictions_count': count,
            })
        except Exception as e:
            app.logger.error(f"Internal bulk predictions status error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Internal helper to rebuild bulk predictions from pattern_cache
    def _rebuild_bulk_from_pattern_cache() -> dict:
        import os as _os
        import json as _json
        import glob as _glob
        import time as _time

        log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        pat_dir = _os.path.join(log_dir, 'pattern_cache')
        _os.makedirs(pat_dir, exist_ok=True)
        bulk_path = _os.path.join(log_dir, 'ml_bulk_predictions.json')

        def _extract_from_unified(u):
            out = {}
            try:
                for h in ('1d', '3d', '7d', '14d', '30d'):
                    node = u.get(h) if isinstance(u, dict) else None
                    price = None
                    if node and isinstance(node, dict):
                        if isinstance(node.get('enhanced'), dict) and isinstance(node['enhanced'].get('price'), (int, float)):
                            price = float(node['enhanced']['price'])
                        elif isinstance(node.get('basic'), dict) and isinstance(node['basic'].get('price'), (int, float)):
                            price = float(node['basic']['price'])
                    if isinstance(price, (int, float)):
                        out[h] = {'price': float(price)}
            except Exception:
                return {}
            return out

        def _extract_generic(d):
            out = {}
            try:
                for h in ('1d', '3d', '7d', '14d', '30d'):
                    node = d.get(h) if isinstance(d, dict) else None
                    if isinstance(node, dict):
                        if 'ensemble_prediction' in node and isinstance(node['ensemble_prediction'], (int, float)):
                            out[h] = {'ensemble_prediction': float(node['ensemble_prediction'])}
                        elif 'price' in node and isinstance(node['price'], (int, float)):
                            out[h] = {'price': float(node['price'])}
                    elif isinstance(node, (int, float)):
                        out[h] = {'price': float(node)}
            except Exception:
                return {}
            return out

        preds_map = {}
        files = sorted(_glob.glob(_os.path.join(pat_dir, '*.json')))
        for f in files:
            try:
                sym = _os.path.basename(f).split('.')[0].upper()
                with open(f, 'r') as rf:
                    data = _json.load(rf) or {}
                enriched = {}
                if isinstance(data.get('ml_unified'), dict):
                    enriched = _extract_from_unified(data['ml_unified'])
                if not enriched and isinstance(data.get('enhanced_predictions'), dict):
                    enriched = _extract_generic(data['enhanced_predictions'])
                if not enriched and isinstance(data.get('ml_predictions'), dict):
                    enriched = _extract_generic(data['ml_predictions'])
                if not enriched and isinstance(data.get('predictions'), dict):
                    enriched = _extract_generic(data['predictions'])
                if enriched:
                    preds_map[sym] = {'enhanced': enriched}
            except Exception:
                continue

        obj = {'predictions': preds_map, 'rebuilt_at': _time.time()}

        tmp_path = bulk_path + '.tmp'
        try:
            payload = _json.dumps(obj)
            with open(tmp_path, 'w') as wf:
                wf.write(payload)
                try:
                    wf.flush()
                    _os.fsync(wf.fileno())
                except Exception:
                    pass
            _os.replace(tmp_path, bulk_path)
        finally:
            try:
                if _os.path.exists(tmp_path):
                    _os.remove(tmp_path)
            except Exception:
                pass

        return {'path': bulk_path, 'predictions_count': len(preds_map)}

    @bp.route('/bulk-predictions/refresh', methods=['POST'])
    @_allow_or_token(app)
    def bulk_predictions_refresh():
        """Rebuild ml_bulk_predictions.json from pattern_cache (fast, cache-only)."""
        try:
            res = _rebuild_bulk_from_pattern_cache()
            return jsonify({'status': 'success', **res})
        except Exception as e:
            app.logger.error(f"Internal bulk predictions refresh error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/pattern-cache/coverage')
    @_allow_or_token(app)
    def pattern_cache_coverage():
        """Return coverage stats for logs/pattern_cache against active stocks."""
        try:
            import os as _os
            import time as _time
            from models import Stock
            log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            pat_dir = _os.path.join(log_dir, 'pattern_cache')
            _os.makedirs(pat_dir, exist_ok=True)
            total = Stock.query.filter_by(is_active=True).count()
            files = {fn.split('.')[0].upper() for fn in _os.listdir(pat_dir) if fn.endswith('.json')}
            with_cache = len(files)
            without_cache = max(0, total - with_cache)
            # Age stats (avg)
            ages = []
            now = _time.time()
            for fn in files:
                fp = _os.path.join(pat_dir, f'{fn}.json')
                try:
                    m = _os.path.getmtime(fp)
                    ages.append(now - m)
                except Exception:
                    continue
            avg_age = (sum(ages) / len(ages)) if ages else None
            max_age = max(ages) if ages else None
            return jsonify({'status': 'success', 'total_active': total, 'with_cache': with_cache, 'without_cache': without_cache, 'avg_age_seconds': avg_age, 'max_age_seconds': max_age})
        except Exception as e:
            app.logger.error(f"Internal pattern cache coverage error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/automation/run-task/<task_name>', methods=['POST'])
    @_allow_or_token(app)
    def run_automation_task(task_name):
        try:
            token_header = request.headers.get('Authorization', '') or ''
            token = None
            if token_header.lower().startswith('bearer '):
                token = token_header.split(' ', 1)[1].strip()
            if not token:
                token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                return jsonify({'status': 'unauthorized'}), 401
            from app import AUTOMATED_PIPELINE_AVAILABLE, get_pipeline_with_context  # lazy
            if not AUTOMATED_PIPELINE_AVAILABLE:
                return jsonify({'status': 'unavailable'}), 503
            pipeline = get_pipeline_with_context()
            if not pipeline:
                return jsonify({'status': 'error', 'message': 'Pipeline not initialized'}), 500
            result = pipeline.run_manual_task(task_name)
            ok = bool(result) or isinstance(result, dict)
            return jsonify({'status': 'success' if ok else 'error', 'result': result})
        except Exception as e:
            app.logger.error(f"Internal automation run task error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/watchlist/cache-report', methods=['GET'])
    def internal_watchlist_cache_report():
        """Internal version of cache report that uses X-Internal-Token and allows localhost.
        Query param/email is required to choose user (e.g., testuser2@lotlot.net).
        """
        try:
            token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                # Allow localhost without token
                remote_ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
                is_local = remote_ip in ('127.0.0.1', '::1', 'localhost')
                allow_localhost = str(os.getenv('INTERNAL_ALLOW_LOCALHOST', str(app.config.get('INTERNAL_ALLOW_LOCALHOST', 'True')))).lower() == 'true'
                if not (is_local and allow_localhost):
                    return jsonify({'status': 'unauthorized'}), 401

            email = request.args.get('email')
            if not email:
                return jsonify({'status': 'error', 'error': 'email query parameter required'}), 400

            from models import User, Watchlist
            from sqlalchemy.orm import joinedload
            user = User.query.filter_by(email=email).first()
            if not user:
                return jsonify({'status': 'error', 'error': 'user not found'}), 404
            items = (
                Watchlist.query.options(joinedload(Watchlist.stock))
                .filter_by(user_id=user.id)
                .all()
            )
            symbols = [it.stock.symbol for it in items if getattr(it, 'stock', None)]

            import os
            import json
            import time
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            pat_dir = os.path.join(log_dir, 'pattern_cache')
            os.makedirs(pat_dir, exist_ok=True)
            bulk_path = os.path.join(log_dir, 'ml_bulk_predictions.json')
            bulk = {}
            bulk_mtime = None
            if os.path.exists(bulk_path):
                with open(bulk_path, 'r') as rf:
                    data = json.load(rf) or {}
                    bulk = (data.get('predictions') or {}) if isinstance(data, dict) else {}
                bulk_mtime = os.path.getmtime(bulk_path)
            now = time.time()

            report_items = []
            for sym in symbols:
                sp = os.path.join(pat_dir, f'{sym}.json')
                exists = os.path.exists(sp)
                age = None
                if exists:
                    try:
                        age = float(now - os.path.getmtime(sp))
                    except Exception:
                        age = None
                report_items.append({
                    'symbol': sym,
                    'pattern_cache': {'exists': bool(exists), 'age_seconds': age},
                    'predictions': {'exists': bool(sym in bulk)},
                })

            return jsonify({
                'status': 'success',
                'email': email,
                'count': len(report_items),
                'bulk_predictions_mtime': bulk_mtime,
                'missing_pattern': [it['symbol'] for it in report_items if not it['pattern_cache']['exists']],
                'missing_predictions': [it['symbol'] for it in report_items if not it['predictions']['exists']],
                'items': report_items,
            })
        except Exception as e:
            app.logger.error(f"Internal cache report error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/automation/full-cycle', methods=['POST'])
    @_allow_or_token(app)
    def automation_full_cycle():
        try:
            token_header = request.headers.get('Authorization', '') or ''
            token = None
            if token_header.lower().startswith('bearer '):
                token = token_header.split(' ', 1)[1].strip()
            if not token:
                token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                return jsonify({'status': 'unauthorized'}), 401

            def _append_status(phase: str, state: str, details: dict | None = None):
                try:
                    import json
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
                    entry = {
                        'phase': phase,
                        'state': state,
                        'timestamp': datetime.now().isoformat(),
                        'details': details or {}
                    }
                    payload.setdefault('history', []).append(entry)
                    payload['history'] = payload['history'][-200:]
                    with open(status_file, 'w') as wf:
                        import json as _json
                        _json.dump(payload, wf)
                except Exception:
                    pass

            def _broadcast(level, message, category='pipeline'):
                try:
                    if hasattr(app, 'broadcast_log'):
                        app.broadcast_log(level, message, category)
                except Exception:
                    pass

            def _run_cycle():
                _append_status('data_collection', 'start', {})
                _broadcast('INFO', '📊 Tam veri toplama başlıyor', 'collector')
                try:
                    col_res = None
                    collector = None
                    try:
                        # Primary path (if module present)
                        from advanced_collector import AdvancedBISTCollector  # type: ignore
                        collector = AdvancedBISTCollector()
                    except Exception:
                        # Fallback to unified collector (always available)
                        try:
                            from bist_pattern.core.unified_collector import get_unified_collector  # type: ignore
                            collector = get_unified_collector()
                        except Exception as ce:
                            _append_status('data_collection', 'error', {'error': f'unified_collector failed: {ce}'})
                            collector = None
                    if collector is not None:
                        try:
                            # Some collectors support scope; call safely
                            col_res = collector.collect_all_stocks_parallel()  # type: ignore[attr-defined]
                        except Exception as ie:
                            _append_status('data_collection', 'error', {'error': f'collector run failed: {ie}'})
                            col_res = None
                    _append_status('data_collection', 'end', col_res or {})
                except Exception as e:
                    _append_status('data_collection', 'error', {'error': str(e)})

                _append_status('ai_analysis', 'start', {})
                _broadcast('INFO', '🧠 AI analizi başlıyor', 'ai_analysis')
                analyzed = 0
                total = 0
                try:
                    from pattern_detector import HybridPatternDetector
                    det = HybridPatternDetector()
                    with app.app_context():
                        from models import Stock
                        symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
                    total = len(symbols)
                    for sym in symbols[:600]:
                        try:
                            det.analyze_stock(sym)
                            analyzed += 1
                        except Exception:
                            continue
                    _append_status('ai_analysis', 'end', {'analyzed': analyzed, 'total': total})
                except Exception as e:
                    _append_status('ai_analysis', 'error', {'error': str(e)})

                _append_status('bulk_predictions', 'start', {})
                _broadcast('INFO', '🤖 ML bulk predictions starting...', 'ml')
                try:
                    from app import get_pipeline_with_context
                    pipeline = get_pipeline_with_context()
                    bulk = pipeline.run_bulk_predictions_all() if pipeline else False
                    count = 0
                    try:
                        if isinstance(bulk, dict):
                            count = len(bulk.get('predictions') or {})
                    except Exception:
                        pass
                    _append_status('bulk_predictions', 'end', {'symbols': count})
                except Exception as e:
                    _append_status('bulk_predictions', 'error', {'error': str(e)})

                # Deterministic bulk rebuild from pattern_cache
                try:
                    rebuild = _rebuild_bulk_from_pattern_cache()
                    _append_status('bulk_predictions', 'end', {'symbols': rebuild.get('predictions_count', 0)})
                except Exception as _reb_err:
                    _append_status('bulk_predictions', 'error', {'error': str(_reb_err)})

                _broadcast('SUCCESS', '✅ Full cycle tamamlandı', 'pipeline')
                return {'analyzed': analyzed, 'total': total}

            do_async = (request.args.get('async') or '').lower() in ('1', 'true', 'yes')
            if do_async:
                import threading
                t = threading.Thread(target=_run_cycle, daemon=True)
                t.start()
                return jsonify({'status': 'accepted', 'mode': 'async'}), 202
            result = _run_cycle()
            return jsonify({'status': 'success', 'mode': 'sync', 'result': result})
        except Exception as e:
            app.logger.error(f"Internal full cycle error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/cycle/run-once', methods=['POST'])
    @_allow_or_token(app)
    def cycle_run_once():
        try:
            token_header = request.headers.get('Authorization', '') or ''
            token = None
            if token_header.lower().startswith('bearer '):
                token = token_header.split(' ', 1)[1].strip()
            if not token:
                token = request.headers.get('X-Internal-Token')
            expected = app.config.get('INTERNAL_API_TOKEN')
            if expected and token != expected:
                return jsonify({'status': 'unauthorized'}), 401

            try:
                body = request.get_json() or {}
            except Exception:
                body = {}
            symbols = body.get('symbols') or request.args.get('symbols')
            per_symbol_sleep = body.get('per_symbol_sleep') or request.args.get('per_symbol_sleep')

            backup_env = {}
            try:
                if symbols:
                    backup_env['CYCLE_SYMBOLS'] = os.getenv('CYCLE_SYMBOLS')
                    os.environ['CYCLE_SYMBOLS'] = symbols
                if per_symbol_sleep is not None:
                    backup_env['CYCLE_PER_SYMBOL_SLEEP'] = os.getenv('CYCLE_PER_SYMBOL_SLEEP')
                    os.environ['CYCLE_PER_SYMBOL_SLEEP'] = str(per_symbol_sleep)

                # Status file append helper (same format as pipeline history)
                def _append_status(phase: str, state: str, details: dict | None = None):
                    try:
                        import json
                        import os
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
                        from datetime import datetime as _dt
                        entry = {
                            'phase': phase,
                            'state': state,
                            'timestamp': _dt.now().isoformat(),
                            'details': details or {}
                        }
                        payload.setdefault('history', []).append(entry)
                        payload['history'] = payload['history'][-200:]
                        with open(status_file, 'w') as wf:
                            json.dump(payload, wf)
                    except Exception:
                        pass

                # Broadcast helper
                def _broadcast(level, message, category='pipeline'):
                    try:
                        if hasattr(app, 'broadcast_log'):
                            app.broadcast_log(level, message, category)
                    except Exception:
                        pass

                try:
                    from importlib import import_module
                    cycle_mod = import_module('cycle_runner')
                    get_cycle_runner = getattr(cycle_mod, 'get_cycle_runner', None)
                    if not callable(get_cycle_runner):
                        return jsonify({'status': 'error', 'error': 'cycle_runner unavailable'}), 503
                except Exception:
                    return jsonify({'status': 'error', 'error': 'cycle_runner unavailable'}), 503
                runner = get_cycle_runner()
                if not hasattr(runner, 'run_once'):
                    return jsonify({'status': 'error', 'error': 'runner missing run_once'}), 503
                _append_status('incremental_cycle', 'start', {'mode': 'manual'})
                _broadcast('INFO', '🧩 Manual incremental cycle started', 'pipeline')
                import time as _t
                _t0 = _t.time()
                run_once_fn = getattr(runner, 'run_once', None)
                if not callable(run_once_fn):
                    return jsonify({'status': 'error', 'error': 'runner missing run_once'}), 503
                result = run_once_fn()
                duration_s = int(_t.time() - _t0)
                processed = 0
                try:
                    processed = int((result or {}).get('processed') or 0)  # type: ignore[call-arg,operator]
                except Exception:
                    processed = 0
                _append_status('incremental_cycle', 'end', {'processed': processed, 'duration_s': duration_s})
                _broadcast('SUCCESS', f'✅ Manual incremental cycle finished ({processed} symbols, {duration_s}s)', 'pipeline')
            finally:
                if 'CYCLE_SYMBOLS' in backup_env:
                    if backup_env['CYCLE_SYMBOLS'] is None:
                        os.environ.pop('CYCLE_SYMBOLS', None)
                    else:
                        os.environ['CYCLE_SYMBOLS'] = backup_env['CYCLE_SYMBOLS']
                if 'CYCLE_PER_SYMBOL_SLEEP' in backup_env:
                    if backup_env['CYCLE_PER_SYMBOL_SLEEP'] is None:
                        os.environ.pop('CYCLE_PER_SYMBOL_SLEEP', None)
                    else:
                        os.environ['CYCLE_PER_SYMBOL_SLEEP'] = backup_env['CYCLE_PER_SYMBOL_SLEEP']

            return jsonify({'status': 'success', 'result': result})
        except Exception as e:
            app.logger.error(f"Internal cycle run-once error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/cycle/status')
    @_allow_or_token(app)
    def cycle_status():
        try:
            try:
                from importlib import import_module
                cycle_mod = import_module('cycle_runner')
                get_cycle_runner = getattr(cycle_mod, 'get_cycle_runner', None)
                if not callable(get_cycle_runner):
                    return jsonify({'status': 'error', 'error': 'cycle_runner unavailable'}), 503
            except Exception:
                return jsonify({'status': 'error', 'error': 'cycle_runner unavailable'}), 503
            runner = get_cycle_runner()
            status_fn = getattr(runner, 'status', None)
            status_data = status_fn() if callable(status_fn) else {}
            return jsonify({'status': 'success', 'runner': status_data})
        except Exception as e:
            app.logger.error(f"Internal cycle status error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Single-symbol collector trigger (uses same collector as automation)
    @bp.route('/collector/single/<symbol>', methods=['POST'])
    @_allow_or_token(app)
    def collector_single(symbol: str):
        try:
            try:
                body = request.get_json() or {}
            except Exception:
                body = {}
            # Determine period preference
            period = body.get('period') or request.args.get('period') or 'auto'
            from bist_pattern.core.unified_collector import get_unified_collector
            with app.app_context():
                col = get_unified_collector()
                res = col.collect_single_stock(symbol, period=period)
            status_ok = bool(isinstance(res, dict) and res.get('success'))
            payload = {
                'status': 'success' if status_ok else 'error',
                'result': res,
            }
            return jsonify(payload)
        except Exception as e:
            app.logger.error(f"Internal single collector error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    app.register_blueprint(bp)
