from flask import Blueprint, jsonify, request
from datetime import datetime

bp = Blueprint('api_public', __name__, url_prefix='/api')


def register(app):
    try:
        from models import db, Stock, StockPrice
    except Exception:
        db = None  # noqa: F841 - kept for conditional branches
        Stock = None
        StockPrice = None
    from app import get_pattern_detector

    @bp.route('/')
    def api_info():
        return jsonify({
            "message": "BIST Pattern Detection API",
            "status": "running",
            "version": "2.2.0",
            "database": "PostgreSQL",
            "features": ["Real-time Data", "Yahoo Finance", "Scheduler", "Dashboard", "Automation"]
        })

    @bp.route('/stocks')
    def api_stocks():
        try:
            if not Stock:
                return jsonify({'status': 'success', 'stocks': []})
            stocks = Stock.query.limit(1000).all()
            out = [
                {'id': s.id, 'symbol': s.symbol, 'name': s.name or s.symbol}
                for s in stocks
            ]
            return jsonify({'status': 'success', 'stocks': out})
        except Exception as e:
            try:
                app.logger.warning(f"/api/stocks error: {e}")
            except Exception:
                pass
            return jsonify({'status': 'success', 'stocks': [], 'error': str(e)})

    @bp.route('/stocks/search')
    def search_stocks():
        try:
            if not Stock:
                return jsonify({'status': 'success', 'query': '', 'stocks': [], 'total': 0})
            query = (request.args.get('q') or '').strip()
            limit = int(request.args.get('limit', 50))
            if not query:
                stocks = Stock.query.limit(limit).all()
            else:
                search_pattern = f"%{query.upper()}%"
                try:
                    from sqlalchemy import or_  # local import for linter/type clarity
                except Exception:
                    or_ = None  # type: ignore[assignment]
                if or_ is None:
                    return jsonify({'status': 'success', 'query': query, 'stocks': [], 'total': 0})
                stocks = (
                    Stock.query.filter(
                        or_(
                            Stock.symbol.ilike(search_pattern),
                            Stock.name.ilike(search_pattern),
                            Stock.sector.ilike(search_pattern),
                        )
                    )
                    .limit(limit)
                    .all()
                )
            result = []
            for s in stocks:
                if not StockPrice:
                    latest = None
                else:
                    latest = (
                        StockPrice.query.filter_by(stock_id=s.id)
                        .order_by(StockPrice.date.desc())
                        .first()
                    )
                result.append({
                    'id': s.id,
                    'symbol': s.symbol,
                    'name': s.name or s.symbol,
                    'sector': s.sector or 'N/A',
                    'price': float(latest.close_price) if latest else None,
                    'last_update': latest.date.isoformat() if latest else None,
                })
            return jsonify({
                'status': 'success',
                'query': query,
                'stocks': result,
                'total': len(result),
            })
        except Exception as e:
            return jsonify({'status': 'success', 'query': '', 'stocks': [], 'total': 0, 'error': str(e)})

    @bp.route('/pattern-analysis/<symbol>')
    def pattern_analysis(symbol):
        try:
            # Fast/cached path: honor ?fast=1 to avoid recomputation on page refresh
            use_fast = (request.args.get('fast') or '').lower() in ('1', 'true', 'yes')
            try:
                from bist_pattern.core.cache import cache_get as _cache_get, cache_set as _cache_set
            except Exception:
                _cache_get = _cache_set = None  # type: ignore

            sym = symbol.upper()
            cache_key = f"pattern_analysis:{sym}"

            # Layer 1: in-memory cache (per worker)
            if use_fast and callable(_cache_get):
                cached = _cache_get(cache_key)
                if cached:
                    return jsonify(cached)

            # Layer 2: file cache shared across workers
            file_cache_hit = None
            file_cache_path = '/opt/bist-pattern/logs/pattern_cache'
            try:
                import os as _os
                import json as _json
                import time as _time
                ttl = float(_os.getenv('PATTERN_FILE_CACHE_TTL', '300'))
                _os.makedirs(file_cache_path, exist_ok=True)
                fpath = _os.path.join(file_cache_path, f'{sym}.json')
                if use_fast and _os.path.exists(fpath):
                    st = _os.stat(fpath)
                    if (_time.time() - float(getattr(st, 'st_mtime', 0))) < ttl:
                        with open(fpath, 'r') as rf:
                            file_cache_hit = _json.load(rf)
            except Exception:
                file_cache_hit = None
            if file_cache_hit:
                return jsonify(file_cache_hit)

            if use_fast and callable(_cache_get):
                cached = _cache_get(cache_key)
                if cached:
                    return jsonify(cached)

            # Compute fresh analysis
            result = get_pattern_detector().analyze_stock(sym)

            # Store to short-lived API cache + file cache (shared between workers)
            try:
                if callable(_cache_set):
                    import os as _os  # local import to avoid top-level side effects
                    try:
                        ttl = float(_os.getenv('PATTERN_CACHE_TTL', '30'))
                    except Exception:
                        ttl = 30.0
                    _cache_set(cache_key, result, ttl_seconds=ttl)
            except Exception:
                pass

            # Persist file cache (best-effort)
            try:
                import os as _os
                import json as _json
                _os.makedirs(file_cache_path, exist_ok=True)
                with open(_os.path.join(file_cache_path, f'{sym}.json'), 'w') as wf:
                    _json.dump(result, wf)
            except Exception:
                pass

            # Only process simulation side-effects on non-fast path to avoid duplicate trades
            if not use_fast:
                try:
                    from simulation_engine import get_simulation_engine
                    from models import SimulationSession
                    active_sessions = SimulationSession.query.filter_by(status='active').all()
                    if active_sessions and result.get('status') == 'success':
                        simulation_engine = get_simulation_engine()
                        for session in active_sessions:
                            trade = simulation_engine.process_signal(
                                session_id=session.id,
                                symbol=sym,
                                signal_data=result,
                            )
                            if trade and hasattr(app, 'socketio'):
                                app.socketio.emit(
                                    'simulation_trade',
                                    {
                                        'session_id': session.id,
                                        'trade': trade.to_dict(),
                                        'timestamp': datetime.now().isoformat(),
                                    },
                                    to='admin',
                                )
                except Exception as sim_error:
                    app.logger.warning(f"Simulation processing failed: {sim_error}")

            return jsonify(result)
        except Exception as e:
            return jsonify({'symbol': symbol, 'status': 'error', 'error': str(e)}), 500

    @bp.route('/pattern-summary')
    def pattern_summary():
        try:
            priority_stocks = [
                'THYAO',
                'AKBNK',
                'GARAN',
                'EREGL',
                'ASELS',
                'VAKBN',
                'MGROS',
                'FROTO',
            ]
            summary = get_pattern_detector().get_pattern_summary(priority_stocks)
            return jsonify(summary)
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/user/predictions/<symbol>')
    def user_predictions(symbol):
        try:
            symbol = symbol.upper()
            stock_data = get_pattern_detector().get_stock_data(symbol, days=365)
            if stock_data is None or len(stock_data) < 50:
                return jsonify({'status': 'error', 'message': f'{symbol} için yeterli veri bulunamadı', 'symbol': symbol}), 404

            def _pick_num(x):
                if isinstance(x, (int, float)):
                    return float(x)
                if isinstance(x, dict):
                    for cand in ('price', 'prediction', 'value', 'target', 'y'):
                        if cand in x and isinstance(x[cand], (int, float)):
                            return float(x[cand])
                return None

            def _normalize_predictions(raw, current):
                out = {}
                if not raw:
                    return out
                if isinstance(raw, dict):
                    for key, val in raw.items():
                        k = str(key).lower()
                        if k in ('1d', 'd1', 'one_day', 'day1', '1day'):
                            out['1d'] = _pick_num(val)
                        elif k in ('3d', 'd3', 'three_day', 'day3', '3day'):
                            out['3d'] = _pick_num(val)
                        elif k in ('7d', 'd7', 'seven_day', 'day7', '7day'):
                            out['7d'] = _pick_num(val)
                        elif k in ('14d', 'd14', 'fourteen_day', 'day14', '14day'):
                            out['14d'] = _pick_num(val)
                        elif k in ('30d', 'd30', 'thirty_day', 'day30', '30day'):
                            out['30d'] = _pick_num(val)
                return out
            
            # ✅ FIX: Indent corrected - this was outside _normalize_predictions!
            detector = get_pattern_detector()
            basic = {}
            enhanced = {}
            try:
                basic_fn = getattr(detector, 'get_basic_predictions', None)
                if callable(basic_fn):
                    basic = basic_fn(symbol, stock_data) or {}
            except Exception:
                basic = {}
            try:
                enhanced_fn = getattr(detector, 'get_enhanced_predictions', None)
                if callable(enhanced_fn):
                    enhanced = enhanced_fn(symbol, stock_data) or {}
            except Exception:
                enhanced = {}
            
            # ✅ FIX: Actually use basic and enhanced predictions!
            merged = {}
            merged.update(_normalize_predictions(basic, stock_data['close'].iloc[-1] if hasattr(stock_data, 'iloc') else 0))
            merged.update(_normalize_predictions(enhanced, stock_data['close'].iloc[-1] if hasattr(stock_data, 'iloc') else 0))
            
            return jsonify({'status': 'success', 'symbol': symbol, 'predictions': merged})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    app.register_blueprint(bp)
