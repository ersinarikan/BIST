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
                    from advanced_collector import AdvancedBISTCollector
                    collector = AdvancedBISTCollector()
                    col_res = collector.collect_all_stocks_parallel()
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
