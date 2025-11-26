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
                # ✅ FIX: Use ConfigManager for consistent config access
                from bist_pattern.core.config_manager import ConfigManager
                configured_token = ConfigManager.get('INTERNAL_API_TOKEN')
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
            
            # ✅ DEBUG: Log what we're about to emit
            symbol = signal_data.get('symbol', 'UNKNOWN')
            app.logger.debug(f"🔔 broadcast_user_signal called for {symbol}, will emit 'user_signal' to room user_{user_id}")
            
            room = f'user_{user_id}'
            app.socketio.emit('user_signal', {
                'user_id': user_id,
                'signal': signal_data,
                'timestamp': datetime.now().isoformat()
            }, to=room)
            
            app.logger.debug(f"✅ user_signal emitted for {symbol} to {room}")
            
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
            # overall_status is a primitive string value; set directly
            # Assign directly
            health_data['overall_status'] = overall  # type: ignore[index]

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
                import os
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
            # Harmonized auth check: allow localhost if enabled, else require INTERNAL_API_TOKEN
            try:
                token_header = request.headers.get('Authorization', '') or ''
                token = None
                if token_header.lower().startswith('bearer '):
                    token = token_header.split(' ', 1)[1].strip()
                if not token:
                    token = request.headers.get('X-Internal-Token')
                expected = app.config.get('INTERNAL_API_TOKEN')
                remote_ip = (request.headers.get('X-Forwarded-For') or request.remote_addr or '').split(',')[0].strip()
                is_local = remote_ip in ('127.0.0.1', '::1', 'localhost')
                allow_localhost = str(os.getenv('INTERNAL_ALLOW_LOCALHOST', str(app.config.get('INTERNAL_ALLOW_LOCALHOST', 'True')))).lower() == 'true'
                if expected and token != expected and not (allow_localhost and is_local):
                    return jsonify({'status': 'unauthorized'}), 401
            except Exception:
                pass

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
                        # ✅ FIX: Use specific category to distinguish from HPO logs
                        app.broadcast_log(level, message, category='working_automation')
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
                            # Symbol-flow data collection: iterate active symbols and collect recent data
                            import time
                            from app import app as flask_app
                            with flask_app.app_context():
                                from models import Stock
                                symbols_list = [s.symbol for s in Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).all()]
                            total = len(symbols_list)
                            added_total = 0
                            updated_total = 0
                            no_data = 0
                            errors = 0
                            try:
                                symbol_sleep_seconds = float(os.getenv('MANUAL_TASK_SYMBOL_SLEEP', '0.01'))
                            except Exception:
                                symbol_sleep_seconds = 0.01
                            for sym in symbols_list:
                                try:
                                    res = None
                                    # Prefer collector.collect_single_stock if available
                                    if hasattr(collector, 'collect_single_stock'):
                                        res = collector.collect_single_stock(sym, period='auto')  # type: ignore[attr-defined]
                                    else:
                                        # Fallback to unified collector for single symbol
                                        try:
                                            from bist_pattern.core.unified_collector import get_unified_collector  # type: ignore
                                            res = get_unified_collector().collect_single_stock(sym, period='auto')
                                        except Exception:
                                            res = None
                                    if isinstance(res, dict):
                                        added_total += int(res.get('records', 0))
                                        updated_total += int(res.get('updated', 0))
                                        if not bool(res.get('success')) or (int(res.get('records', 0)) == 0 and int(res.get('updated', 0)) == 0):
                                            no_data += 1
                                except Exception:
                                    errors += 1
                                time.sleep(symbol_sleep_seconds)
                            col_res = {
                                'success': True,
                                'total_symbols': total,
                                'added_records': added_total,
                                'updated_records': updated_total,
                                'no_data_or_empty': no_data,
                                'errors': errors,
                            }
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
                    from app import get_pattern_detector
                    det = get_pattern_detector()  # ✅ FIX: Use singleton pattern
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

    # ------------------------
    # RSS NEWS INTERNAL ENDPOINTS
    # ------------------------
    @bp.route('/rss/health', methods=['GET'])
    @_allow_or_token(app)
    def rss_health():
        try:
            from rss_news_async import get_async_rss_news_provider  # lazy import in service context
            p = get_async_rss_news_provider()
            info = p.get_system_info()
            return jsonify({'status': 'success', 'info': info})
        except Exception as e:
            app.logger.error(f"RSS health error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/rss/refresh', methods=['POST'])
    @_allow_or_token(app)
    def rss_refresh():
        try:
            from rss_news_async import get_async_rss_news_provider
            p = get_async_rss_news_provider()
            ok = p.force_refresh_async()
            return jsonify({'status': 'success' if ok else 'busy'})
        except Exception as e:
            app.logger.error(f"RSS refresh error: {e}")
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

    # ------------------------
    # Forward Simulation (Real-time Trading Simulation)
    # ------------------------
    @bp.route('/simulation/forward-start', methods=['POST'])
    @_allow_or_token(app)
    def simulation_forward_start():
        """Start forward simulation (real-time trading simulation)."""
        from ..simulation.forward_engine import start_simulation
        try:
            payload = request.get_json(force=True) or {}
            with app.app_context():
                result = start_simulation(payload)
            app.logger.info(f"✅ Forward simulation started: {result.get('duration_days')}d horizon")
            return jsonify(result), 200
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            app.logger.error(f"Forward simulation start error: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/simulation/forward-stop', methods=['POST'])
    @_allow_or_token(app)
    def simulation_forward_stop():
        """Stop forward simulation and return summary."""
        from ..simulation.forward_engine import stop_simulation
        try:
            with app.app_context():
                result = stop_simulation()
            app.logger.info(f"✅ Forward simulation stopped: P&L={result.get('summary', {}).get('pnl', 0):.2f}")
            return jsonify(result), 200
        except Exception as e:
            app.logger.error(f"Forward simulation stop error: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/simulation/status', methods=['GET'])
    @_allow_or_token(app)
    def simulation_status():
        """Get current simulation status."""
        from ..simulation.forward_engine import get_simulation_status
        try:
            result = get_simulation_status()
            return jsonify(result), 200
        except Exception as e:
            app.logger.error(f"Simulation status error: {e}")
            return jsonify({'error': str(e)}), 500

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

    # Calibration readiness (counts-based gate)
    @bp.route('/calibration/readiness')
    @_allow_or_token(app)
    def calibration_readiness_internal():
        """Summarize matured outcomes per horizon and readiness state.
        Read thresholds from env; persist snapshot to logs/calibration_readiness.json.
        """
        try:
            import os as _os
            import json as _json
            from datetime import datetime as _dt
            from models import db, PredictionsLog, OutcomesLog  # type: ignore

            # Thresholds
            try:
                min_total = int(_os.getenv('MIN_OUTCOMES_PER_HORIZON', '250'))
            except Exception:
                min_total = 250
            try:
                min_per_decile = int(_os.getenv('MIN_SAMPLES_PER_DECILE', '50'))
            except Exception:
                min_per_decile = 50
            needed = max(min_total, min_per_decile * 10)

            # Gather counts per horizon (all time)
            rows = (
                db.session.query(PredictionsLog.horizon, db.func.count(OutcomesLog.id))
                .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                .group_by(PredictionsLog.horizon)
                .all()
            )
            by_h = {str(h or '').strip(): int(c or 0) for h, c in rows if (h or '')}
            horizons = ['1d', '3d', '7d', '14d', '30d']
            readiness = {}
            for h in horizons:
                cnt = int(by_h.get(h, 0))
                readiness[h] = {
                    'count': cnt,
                    'needed': needed,
                    'ready': bool(cnt >= needed),
                }

            payload = {
                'status': 'success',
                'thresholds': {
                    'min_total': min_total,
                    'min_per_decile': min_per_decile,
                    'needed': needed,
                },
                'readiness': readiness,
                'timestamp': _dt.now().isoformat(),
            }

            # Persist snapshot
            try:
                base_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                _os.makedirs(base_dir, exist_ok=True)
                fpath = _os.path.join(base_dir, 'calibration_readiness.json')
                with open(fpath, 'w') as wf:
                    _json.dump(payload, wf)
            except Exception:
                pass

            return jsonify(payload)
        except Exception as e:
            app.logger.error(f"Calibration readiness error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Calibration readiness history (from file)
    @bp.route('/calibration/readiness-history')
    @_allow_or_token(app)
    def calibration_readiness_history_internal():
        try:
            import os as _os
            import json as _json
            base_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            fpath = _os.path.join(base_dir, 'calibration_readiness.json')
            data = {}
            if _os.path.exists(fpath):
                with open(fpath, 'r') as rf:
                    data = _json.load(rf) or {}
            return jsonify({'status': 'success', 'data': data})
        except Exception as e:
            app.logger.error(f"Calibration readiness history error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Horizon metrics history (from file, last N entries)
    @bp.route('/metrics/horizon-history')
    @_allow_or_token(app)
    def metrics_horizon_history_internal():
        try:
            import os as _os
            import json as _json
            from flask import request as _req
            limit = 100
            try:
                q = _req.args.get('limit', '')
                if q:
                    limit = max(1, min(2000, int(q)))
            except Exception:
                limit = 100
            base_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            fpath = _os.path.join(base_dir, 'metrics_horizon.json')
            rows = []
            if _os.path.exists(fpath):
                try:
                    with open(fpath, 'r') as rf:
                        rows = _json.load(rf) or []
                except Exception:
                    rows = []
            if isinstance(rows, list) and len(rows) > limit:
                rows = rows[-limit:]
            return jsonify({'status': 'success', 'data': rows, 'limit': limit})
        except Exception as e:
            app.logger.error(f"Horizon metrics history error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Training summary (internal): parse latest train_*.log for ok/fail and nRMSE/Hit-rate aggregates
    @bp.route('/reports/training-summary')
    @_allow_or_token(app)
    def training_summary_internal():
        try:
            import os as _os
            import re as _re
            import json as _json
            from math import isfinite as _isfinite

            base_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            latest_log = None
            try:
                # Find most recent non-empty train_*.log
                candidates = [
                    _os.path.join(base_dir, fn)
                    for fn in _os.listdir(base_dir)
                    if fn.startswith('train_') and fn.endswith('.log')
                ]
                # Filter out empty files and sort by mtime (newest first)
                non_empty = [p for p in candidates if _os.path.exists(p) and _os.path.getsize(p) > 0]
                if non_empty:
                    latest_log = max(non_empty, key=lambda p: _os.path.getmtime(p))
            except Exception:
                latest_log = None

            ok_cnt = None
            fail_cnt = None
            nrmse_vals: list[float] = []
            hit_vals: list[float] = []

            by_h = {h: {'nrmse': [], 'hit': []} for h in ('1d', '3d', '7d', '14d', '30d')}
            if latest_log and _os.path.exists(latest_log):
                try:
                    # Read last ~10k lines to keep fast while likely catching summary
                    with open(latest_log, 'r', errors='ignore') as f:
                        lines = f.readlines()[-10000:]
                    # Extract ok/fail from JSON-like summary line
                    for ln in reversed(lines):
                        if '{' in ln and '"ok"' in ln and '"fail"' in ln:
                            try:
                                obj = _json.loads(ln.strip())
                                if isinstance(obj, dict):
                                    ok_cnt = int(obj.get('ok')) if obj.get('ok') is not None else ok_cnt
                                    fail_cnt = int(obj.get('fail')) if obj.get('fail') is not None else fail_cnt
                                    break
                            except Exception:
                                # Not strict JSON; best-effort regex fallback
                                try:
                                    m_ok = _re.search(r'"ok"\s*:\s*(\d+)', ln)
                                    m_fail = _re.search(r'"fail"\s*:\s*(\d+)', ln)
                                    if m_ok:
                                        ok_cnt = int(m_ok.group(1))
                                    if m_fail:
                                        fail_cnt = int(m_fail.group(1))
                                        break
                                except Exception:
                                    pass
                    # Collect nRMSE and Hit values across lines
                    for ln in lines:
                        try:
                            # detect horizon token like 1D, 3D, 7D, 14D, 30D
                            htok = None
                            m_h = _re.search(r'\b(1D|3D|7D|14D|30D)\b', ln)
                            if m_h:
                                htok = m_h.group(1).lower()
                            m1 = _re.search(r'nRMSE\s*:\s*([0-9]+(?:\.[0-9]+)?)', ln)
                            if m1:
                                v = float(m1.group(1))
                                if _isfinite(v):
                                    nrmse_vals.append(v)
                                    if htok and htok in by_h:
                                        by_h[htok]['nrmse'].append(v)
                            m2 = _re.search(r'Hit\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%', ln)
                            if m2:
                                v = float(m2.group(1))
                                if _isfinite(v):
                                    hit_vals.append(v)
                                    if htok and htok in by_h:
                                        by_h[htok]['hit'].append(v)
                        except Exception:
                            continue
                except Exception:
                    pass

            def _avg(xs: list[float]) -> float | None:
                try:
                    return (sum(xs) / len(xs)) if xs else None
                except Exception:
                    return None

            # Optional: metrics_horizon.json aggregation (best-effort)
            metrics_file = _os.path.join(base_dir, 'metrics_horizon.json')
            mh_nrmse: list[float] = []
            mh_hit: list[float] = []
            mh_by_h = {h: {'nrmse': [], 'hit': []} for h in ('1d', '3d', '7d', '14d', '30d')}
            if _os.path.exists(metrics_file):
                try:
                    data = None
                    with open(metrics_file, 'r') as rf:
                        data = _json.load(rf)
                    # Accept list of records; collect numeric fields by common keys
                    if isinstance(data, list):
                        for it in data[-2000:]:
                            if isinstance(it, dict):
                                # horizon field detection (common keys)
                                htok = None
                                for hk in ('h', 'horizon', 'hzn', 'days'):
                                    if hk in it and it.get(hk) is not None:
                                        try:
                                            # Accept like '7d' or 7 -> '7d'
                                            raw = str(it.get(hk)).strip().lower()
                                            if raw.endswith('d'):
                                                htok = raw
                                            else:
                                                htok = f"{int(float(raw))}d"
                                        except Exception:
                                            pass
                                        break
                                for k, v in it.items():
                                    kl = str(k).strip().lower()
                                    if kl in ('nrmse', 'n_rmse', 'n-rmse'):
                                        try:
                                            vv = float(v)
                                            if _isfinite(vv):
                                                mh_nrmse.append(vv)
                                                if htok in mh_by_h:
                                                    mh_by_h[htok]['nrmse'].append(vv)
                                        except Exception:
                                            pass
                                    if kl in ('hit_rate', 'hitrate', 'hit-rate'):
                                        try:
                                            vv = float(v)
                                            if _isfinite(vv):
                                                mh_hit.append(vv)
                                                if htok in mh_by_h:
                                                    mh_by_h[htok]['hit'].append(vv)
                                        except Exception:
                                            pass
                except Exception:
                    pass

            out = {
                'status': 'success',
                'log_path': latest_log,
                'ok': ok_cnt,
                'fail': fail_cnt,
                'aggregates': {
                    'nrmse': {
                        'count': len(nrmse_vals),
                        'mean': _avg(nrmse_vals)
                    },
                    'hit_rate': {
                        'count': len(hit_vals),
                        'mean': _avg(hit_vals)
                    },
                    'by_horizon': {
                        k: {
                            'nrmse_mean': _avg(v['nrmse']) if v['nrmse'] else None,
                            'hit_rate_mean': _avg(v['hit']) if v['hit'] else None,
                            'nrmse_count': len(v['nrmse']),
                            'hit_rate_count': len(v['hit'])
                        }
                        for k, v in by_h.items()
                    }
                },
                'metrics_file': {
                    'path': metrics_file if _os.path.exists(metrics_file) else None,
                    'nrmse': {
                        'count': len(mh_nrmse),
                        'mean': _avg(mh_nrmse)
                    },
                    'hit_rate': {
                        'count': len(mh_hit),
                        'mean': _avg(mh_hit)
                    },
                    'by_horizon': {
                        k: {
                            'nrmse_mean': _avg(v['nrmse']) if v['nrmse'] else None,
                            'hit_rate_mean': _avg(v['hit']) if v['hit'] else None,
                            'nrmse_count': len(v['nrmse']),
                            'hit_rate_count': len(v['hit'])
                        }
                        for k, v in mh_by_h.items()
                    }
                }
            }
            return jsonify(out)
        except Exception as e:
            app.logger.error(f"Training summary error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Calibration summary (internal, token or localhost)
    @bp.route('/calibration/summary')
    @_allow_or_token(app)
    def calibration_summary_internal():
        try:
            from datetime import date, timedelta
            import os as _os
            import json as _json
            from models import db, MetricsDaily

            # Prefer persisted calibration_state.json if present, else environment

            def _load_calibration_state():
                try:
                    base_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                    cpath = _os.path.join(base_dir, 'calibration_state.json')
                    if _os.path.exists(cpath):
                        with open(cpath, 'r') as cf:
                            data = _json.load(cf) or {}
                            bypass = bool(data.get('bypass', False))
                            pf = data.get('penalty_factor', None)
                            pf = float(pf) if (pf is not None) else None
                            return {
                                'bypass': bypass,
                                'status': ('bypass' if bypass else 'active'),
                                'penalty_factor': pf,
                                'source': 'file'
                            }
                except Exception:
                    pass
                try:
                    _raw = str(_os.getenv('BYPASS_ISOTONIC_CALIBRATION', '1')).strip().lower()
                    _bypass = _raw in ('1', 'true', 'yes', 'on')
                except Exception:
                    _bypass = True
                try:
                    _pf_env = _os.getenv('THRESHOLD_PENALTY_FACTOR', '')
                    _penalty = float(_pf_env) if (_pf_env is not None and str(_pf_env).strip() != '') else None
                except Exception:
                    _penalty = None
                return {
                    'bypass': bool(_bypass),
                    'status': ('bypass' if _bypass else 'active'),
                    'penalty_factor': _penalty,
                    'source': 'env'
                }
            # Load param_store.json
            base = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            ppath = _os.path.join(base, 'param_store.json')
            pstore = {}
            if _os.path.exists(ppath):
                try:
                    with open(ppath, 'r') as rf:
                        pstore = _json.load(rf) or {}
                except Exception:
                    pstore = {}

            # Helper to aggregate metrics for the last N days
            def _aggregate_last_ndays(n_days: int):
                _today = date.today()
                _start = _today - timedelta(days=n_days - 1)
                _rows = (
                    db.session.query(MetricsDaily)
                    .filter(MetricsDaily.date >= _start)
                    .filter(MetricsDaily.date <= _today)
                    .all()
                )
                _by_h = {}
                for _r in _rows:
                    _h = (_r.horizon or '').strip()
                    if not _h:
                        continue
                    _by_h.setdefault(_h, []).append(_r)
                _metrics = {}
                _acc_vals = []
                for _h, _items in _by_h.items():
                    _acc_list = [float(getattr(_it, 'acc')) for _it in _items if getattr(_it, 'acc') is not None]
                    _acc = (sum(_acc_list) / len(_acc_list)) if _acc_list else None
                    _metrics[_h] = {'acc': _acc, 'count': len(_items)}
                    if _acc is not None:
                        _acc_vals.append(_acc)
                _overall = (sum(_acc_vals) / len(_acc_vals)) if _acc_vals else None
                return _metrics, _overall

            metrics_7d, overall_acc_7d = _aggregate_last_ndays(7)
            metrics_30d, overall_acc_30d = _aggregate_last_ndays(30)
            metrics_90d, overall_acc_90d = _aggregate_last_ndays(90)  # ⚡ NEW: 90-day metrics

            # A/B summary 7d
            try:
                from models import PredictionsLog, OutcomesLog  # type: ignore
                hkeys = ['1d', '3d', '7d', '14d', '30d']
                ab_7d = {h: {'prod': {'acc': None, 'n': 0}, 'chall': {'acc': None, 'n': 0}} for h in hkeys}
                from datetime import date as _date, timedelta as _timedelta
                dt_today = _date.today()
                dt_start = dt_today - _timedelta(days=6)
                rows = (
                    db.session.query(PredictionsLog, OutcomesLog)
                    .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                    .filter(OutcomesLog.ts_eval >= dt_start)
                    .all()
                )
                tmp = {h: {'prod': [], 'chall': []} for h in hkeys}
                for p, o in rows:
                    try:
                        h = (p.horizon or '').strip()
                        if h not in tmp:
                            continue
                        pv = str(getattr(p, 'param_version', '') or '')
                        grp = 'chall' if 'ab:chall' in pv else 'prod'
                        tmp[h][grp].append(1.0 if bool(getattr(o, 'dir_hit')) else 0.0)
                    except Exception:
                        continue
                for h in hkeys:
                    for grp in ('prod', 'chall'):
                        vals = tmp[h][grp]
                        n = len(vals)
                        ab_7d[h][grp]['n'] = n
                        ab_7d[h][grp]['acc'] = (sum(vals) / n) if n else None
            except Exception:
                ab_7d = {}

            # ⚡ NEW: Calculate magnitude-based metrics for A/B test
            # This provides additional insight beyond dir_hit
            ab_7d_magnitude = {}
            try:
                hkeys2 = ['1d', '3d', '7d', '14d', '30d']
                ab_7d_magnitude = {h: {'prod': {'acc': None, 'n': 0}, 'chall': {'acc': None, 'n': 0}} for h in hkeys2}
                from datetime import date as _date2, timedelta as _timedelta2
                dt_today2 = _date2.today()
                dt_start2 = dt_today2 - _timedelta2(days=6)
                rows2 = (
                    db.session.query(PredictionsLog, OutcomesLog)
                    .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
                    .filter(OutcomesLog.ts_eval >= dt_start2)
                    .all()
                )
                tmp2 = {h: {'prod': [], 'chall': []} for h in hkeys2}
                magnitude_tol = float(_os.getenv('MAGNITUDE_HIT_TOLERANCE', '0.05'))
                for p2, o2 in rows2:
                    try:
                        h2 = (p2.horizon or '').strip()
                        if h2 not in tmp2:
                            continue
                        pv2 = str(getattr(p2, 'param_version', '') or '')
                        grp2 = 'chall' if 'ab:chall' in pv2 else 'prod'
                        # Calculate magnitude hit on-the-fly
                        delta_real = float(getattr(o2, 'delta_real') or 0.0)
                        delta_pred = float(getattr(p2, 'delta_pred') or 0.0)
                        dir_hit_val = bool(getattr(o2, 'dir_hit'))
                        abs_err_val = float(getattr(o2, 'abs_err') or 0.0)
                        price_eval_val = float(getattr(o2, 'price_eval') or 0.0)
                        threshold_val = float(_os.getenv('DIRECTION_HIT_THRESHOLD', '0.005'))
                        
                        if dir_hit_val and price_eval_val > 0:
                            if abs(delta_real) < threshold_val and abs(delta_pred) < threshold_val:
                                mag_hit = 1.0
                            elif abs_err_val / price_eval_val <= magnitude_tol:
                                mag_hit = 1.0
                            else:
                                mag_hit = 0.0
                        else:
                            mag_hit = 0.0
                        tmp2[h2][grp2].append(mag_hit)
                    except Exception:
                        continue
                for h2 in hkeys2:
                    for grp2 in ('prod', 'chall'):
                        vals2 = tmp2[h2][grp2]
                        n2 = len(vals2)
                        ab_7d_magnitude[h2][grp2]['n'] = n2
                        ab_7d_magnitude[h2][grp2]['acc'] = (sum(vals2) / n2) if n2 else None
            except Exception:
                ab_7d_magnitude = {}

            calibration_state = _load_calibration_state()
            
            # ⚡ NEW: Check online confidence adjustment status
            # Online adjustment is enabled if calibration is NOT bypassed AND env var allows it
            try:
                import os as _os2
                # Global enable check
                global_online_adj_enabled = (
                    not calibration_state.get('bypass', True) and
                    str(_os2.getenv('ENABLE_ONLINE_CONFIDENCE_ADJUSTMENT', '1')).lower() in ('1', 'true', 'yes')
                )
                online_adj_min_samples = int(_os2.getenv('ONLINE_CALIB_MIN_SAMPLES', '10'))
                online_adj_window_days = int(_os2.getenv('ONLINE_CALIB_WINDOW_DAYS', '45'))
                online_adj_alpha = float(_os2.getenv('ONLINE_CALIB_ALPHA', '0.5'))
                
                # ✅ NEW: Check per-horizon data availability
                # Build a map of which horizons have sufficient data
                skipped_horizons_list = pstore.get('skipped_horizons', [])
                skipped_map = {s.get('horizon'): s.get('reason') for s in skipped_horizons_list if isinstance(s, dict)}
                
                # Per-horizon online adjustment status
                online_adj_by_horizon = {}
                for h in ['1d', '3d', '7d', '14d', '30d']:
                    # Check if this horizon has sufficient data (not in skipped list)
                    has_data = h not in skipped_map
                    # Online adjustment enabled for this horizon if global enabled AND has data
                    online_adj_by_horizon[h] = {
                        'enabled': global_online_adj_enabled and has_data,
                        'has_data': has_data,
                        'reason': skipped_map.get(h, None) if not has_data else None
                    }
            except Exception:
                global_online_adj_enabled = False
                online_adj_min_samples = 10
                online_adj_window_days = 45
                online_adj_alpha = 0.5
                online_adj_by_horizon = {}

            return jsonify({
                'param_store': pstore,
                'metrics_7d': metrics_7d,
                'overall_acc_7d': overall_acc_7d,
                'metrics_30d': metrics_30d,
                'overall_acc_30d': overall_acc_30d,
                'metrics_90d': metrics_90d,  # ⚡ NEW: 90-day metrics
                'overall_acc_90d': overall_acc_90d,  # ⚡ NEW: 90-day overall accuracy
                'ab_7d': ab_7d,
                'ab_7d_magnitude': ab_7d_magnitude,  # ⚡ NEW: Magnitude-based A/B test metrics
                'calibration_state': calibration_state,
                'online_adjustment': {
                    'enabled': global_online_adj_enabled if 'global_online_adj_enabled' in locals() else False,
                    'min_samples': online_adj_min_samples,
                    'window_days': online_adj_window_days,
                    'alpha': online_adj_alpha,
                    'by_horizon': online_adj_by_horizon if 'online_adj_by_horizon' in locals() else {},  # ⚡ NEW: Per-horizon status
                    'note': 'Online adjustment disabled when calibration is bypassed or insufficient data for horizon'
                },
                'threshold_config': {  # ⚡ NEW: Threshold configuration
                    'direction_hit_threshold': float(_os.getenv('DIRECTION_HIT_THRESHOLD', '0.005')),
                    'magnitude_hit_tolerance': float(_os.getenv('MAGNITUDE_HIT_TOLERANCE', '0.05')),
                },
            })
        except Exception as e:
            app.logger.error(f"Internal calibration summary error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Tradable candidates (internal): filter by hit/nrmse heuristics and volume tiers
    @bp.route('/reports/tradable')
    @_allow_or_token(app)
    def tradable_candidates_internal():
        try:
            import os as _os
            import json as _json
            from typing import Dict
            # Load latest training summary (aggregates.by_horizon) and signals for confidence
            log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            mh_path = _os.path.join(log_dir, 'metrics_horizon.json')
            sig_path = _os.path.join(log_dir, 'signals_last.json')
            # Volume tiers via existing internal endpoint helper
            # Heuristics
            hkeys = ['1d', '3d', '7d', '14d', '30d']
            horizon = (request.args.get('horizon') or '7d').lower()
            if horizon not in hkeys:
                horizon = '7d'
            min_hit = float(request.args.get('min_hit', '58'))
            max_nrmse = float(request.args.get('max_nrmse', '1.00'))
            max_count = int(request.args.get('limit', '50'))

            # Build a simple score from signals_last (confidence) and thresholds (fallback only)
            conf_map: Dict[str, float] = {}
            if _os.path.exists(sig_path):
                try:
                    with open(sig_path, 'r') as rf:
                        ss = _json.load(rf) or {}
                    for sym, sv in (ss.items() if isinstance(ss, dict) else []):
                        try:
                            conf = float(sv.get('confidence') or 0.0)
                            conf_map[str(sym).upper()] = conf
                        except Exception:
                            continue
                except Exception:
                    pass

            # Aggregate metrics by horizon (best-effort)
            # Expect a list of rows with keys like horizon, hit_rate, nrmse
            metrics: Dict[str, Dict[str, float]] = {}
            if _os.path.exists(mh_path):
                try:
                    with open(mh_path, 'r') as rf:
                        rows = _json.load(rf) or []
                    if isinstance(rows, list):
                        for it in rows[-2000:]:
                            if not isinstance(it, dict):
                                continue
                            hk = str(it.get('horizon') or it.get('h') or '').strip().lower()
                            if not hk:
                                continue
                            try:
                                hr = float(it.get('hit_rate')) if it.get('hit_rate') is not None else None
                            except Exception:
                                hr = None
                            try:
                                nr = float(it.get('nrmse')) if it.get('nrmse') is not None else None
                            except Exception:
                                nr = None
                            if hk not in metrics:
                                metrics[hk] = {'hit_rate': hr or 0.0, 'nrmse': nr or 0.0}
                            else:
                                # Keep last value (already ordered) or average; here we keep last
                                metrics[hk]['hit_rate'] = hr or metrics[hk]['hit_rate']
                                metrics[hk]['nrmse'] = nr or metrics[hk]['nrmse']
                except Exception:
                    metrics = {}

            # Choose target horizon thresholds
            hr_ok = metrics.get(horizon, {})
            # Use global defaults if missing
            thr_hit = max(min_hit, float(hr_ok.get('hit_rate') or min_hit))
            thr_nrmse = min(max_nrmse, float(hr_ok.get('nrmse') or max_nrmse))

            # Pull volume tiers data using existing internal volume endpoint logic
            try:
                from models import db, Stock, StockPrice  # type: ignore
                from sqlalchemy import func  # type: ignore
                from datetime import timedelta
                lookback_days = int(_os.getenv('VOLUME_LOOKBACK_DAYS', '30'))
                cutoff_date = (__import__('datetime').datetime.utcnow() - timedelta(days=lookback_days)).date()
                rows = (
                    db.session.query(Stock.symbol, func.avg(StockPrice.volume).label('avg_vol'))
                    .join(StockPrice, Stock.id == StockPrice.stock_id)
                    .filter(Stock.is_active.is_(True), StockPrice.date >= cutoff_date)
                    .group_by(Stock.id, Stock.symbol)
                    .all()
                )
                vols = {str(s).upper(): float(v or 0) for s, v in rows}
            except Exception:
                vols = {}

            # Build candidates: use signals confidence as tie-breaker; thresholds from horizon metrics
            # In absence of per-symbol metrics, we return symbols with recent signals and acceptable confidence
            items = []
            for sym, conf in conf_map.items():
                items.append({
                    'symbol': sym,
                    'confidence': conf,
                    'horizon': horizon,
                    'hit_rate_ok': True if thr_hit else True,
                    'nrmse_ok': True if thr_nrmse else True,
                    'avg_volume': vols.get(sym)
                })
            # Sort by confidence desc and volume desc
            items.sort(key=lambda x: (float(x.get('confidence') or 0), float(x.get('avg_volume') or 0)), reverse=True)
            items = items[:max_count]

            return jsonify({'status': 'success', 'horizon': horizon, 'thresholds': {'min_hit': thr_hit, 'max_nrmse': thr_nrmse}, 'count': len(items), 'items': items})
        except Exception as e:
            app.logger.error(f"Tradable candidates error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Force YOLO visual analysis for a symbol and persist snapshot (internal)
    @bp.route('/visual-analysis/<symbol>')
    @_allow_or_token(app)
    def internal_visual_analysis(symbol: str):
        try:
            from visual_pattern_detector import get_visual_pattern_system  # type: ignore
            from app import get_pattern_detector  # type: ignore
            sym = (symbol or '').upper()
            if not sym:
                return jsonify({'status': 'error', 'message': 'Symbol required'}), 400
            # Fetch data
            stock_data = get_pattern_detector().get_stock_data(sym)
            if stock_data is None or len(stock_data) < 20:
                return jsonify({'status': 'error', 'message': f'{sym} için yeterli veri bulunamadı (min 20)'}), 400
            # Run analysis (best-effort; backend is async-aware)
            vsys = get_visual_pattern_system()
            res = vsys.analyze_stock_visual(sym, stock_data)
            # Persist compact snapshot visual evidence if available
            try:
                import os as _os
                import json as _json
                log_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                snap_path = _os.path.join(log_dir, 'signals_last.json')
                snap = {}
                try:
                    if _os.path.exists(snap_path):
                        with open(snap_path, 'r') as rf:
                            snap = _json.load(rf) or {}
                except Exception:
                    snap = {}
                visual_evidence = []
                try:
                    vis = [p for p in (res.get('visual_analysis', {}).get('patterns') or [])]
                    for p in vis:
                        v = {
                            'pattern': p.get('pattern'),
                            'confidence': float(p.get('confidence', 0.0) or 0.0),
                            'source': 'VISUAL_YOLO'
                        }
                        visual_evidence.append(v)
                except Exception:
                    visual_evidence = []
                if sym in snap:
                    snap[sym]['visual'] = visual_evidence
                else:
                    snap[sym] = {
                        'timestamp': res.get('timestamp'),
                        'signal': res.get('status', 'pending'),
                        'confidence': 0.0,
                        'strength': 0,
                        'visual': visual_evidence,
                    }
                with open(snap_path, 'w') as wf:
                    _json.dump(snap, wf)
            except Exception:
                pass
            return jsonify({'status': 'success', 'result': res})
        except Exception as e:
            app.logger.error(f"internal_visual_analysis error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # Toggle calibration (persisted soft toggle)
    @bp.route('/calibration/toggle', methods=['POST'])
    @_allow_or_token(app)
    def calibration_toggle_internal():
        try:
            import os as _os
            import json as _json
            from datetime import datetime as _dt
            try:
                body = request.get_json(force=True) or {}
            except Exception:
                body = {}
            base_dir = _os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            _os.makedirs(base_dir, exist_ok=True)
            cpath = _os.path.join(base_dir, 'calibration_state.json')
            # Load current state
            cur = {}
            if _os.path.exists(cpath):
                try:
                    with open(cpath, 'r') as cf:
                        cur = _json.load(cf) or {}
                except Exception:
                    cur = {}
            # Determine new state
            desired = body.get('bypass', None)
            toggle = body.get('toggle', None)
            if desired is None:
                if isinstance(toggle, bool) and toggle:
                    desired = (not bool(cur.get('bypass', True)))
                else:
                    # Default: invert if file exists, else keep True (bypass) as safe default
                    desired = (not bool(cur.get('bypass', True)))
            try:
                pf = body.get('penalty_factor', None)
                pf = float(pf) if (pf is not None and str(pf).strip() != '') else None
            except Exception:
                pf = None
            # If penalty not provided, pick defaults
            if pf is None:
                pf = 0.95 if bool(desired) else 0.85
            new_state = {
                'bypass': bool(desired),
                'penalty_factor': float(pf),
                'updated_at': _dt.now().isoformat()
            }
            # Persist atomically with file lock
            try:
                from bist_pattern.utils.param_store_lock import file_lock  # type: ignore
            except Exception:
                file_lock = None  # type: ignore
            tmp_path = cpath + '.tmp'
            content = _json.dumps(new_state, ensure_ascii=False, indent=2)
            if file_lock is not None:
                with file_lock(cpath):
                    with open(tmp_path, 'w') as wf:
                        wf.write(content)
                    _os.replace(tmp_path, cpath)
            else:
                with open(tmp_path, 'w') as wf:
                    wf.write(content)
                _os.replace(tmp_path, cpath)
            # Respond with fresh state
            return jsonify({'status': 'success', 'calibration_state': {
                'bypass': new_state['bypass'],
                'status': ('bypass' if new_state['bypass'] else 'active'),
                'penalty_factor': new_state['penalty_factor'],
                'updated_at': new_state['updated_at'],
                'source': 'file',
            }})
        except Exception as e:
            app.logger.error(f"Internal calibration toggle error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    app.register_blueprint(bp)
