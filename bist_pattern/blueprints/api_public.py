import logging
from flask import Blueprint, jsonify, request
from datetime import datetime  # noqa: F401 (kept for consistency)

logger = logging.getLogger(__name__)

bp = Blueprint('api_public', __name__, url_prefix='/api')


def register(app):
    try:
        from models import db, Stock, StockPrice
    except Exception as e:
        logger.debug(f"Failed to import models: {e}")
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
            except Exception as e2:
                logger.debug(f"Failed to log /api/stocks error: {e2}")
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
                except Exception as e:
                    logger.debug(f"Failed to import sqlalchemy.or_: {e}")
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
        """Cache-only pattern analysis endpoint.

        - Never triggers fresh analysis.
        - Returns cached result from memory/redis or file cache.
        - Accepts stale file cache and annotates it with staleness metadata so UI can still render overlays.
        - If no cached result, returns {status: 'pending'}
        """
        try:
            try:
                from bist_pattern.core.cache import cache_get as _cache_get
            except Exception as e:
                logger.debug(f"Failed to import cache_get: {e}")
                _cache_get = None  # type: ignore

            sym = symbol.upper()
            cache_key = f"pattern_analysis:{sym}"

            # Layer 1: in-memory/Redis cache
            if callable(_cache_get):
                try:
                    cached = _cache_get(cache_key)
                except Exception as e:
                    logger.debug(f"Failed to get cache for {sym}: {e}")
                    cached = None
                if cached:
                    return jsonify(cached)

            # Layer 2: file cache shared across workers (accept even if stale)
            file_cache_path = '/opt/bist-pattern/logs/pattern_cache'
            try:
                import os as _os
                import json as _json
                import time as _time
                from bist_pattern.core.broadcaster import _sanitize_json_value
                ttl = float(_os.getenv('PATTERN_FILE_CACHE_TTL', '300'))
                _os.makedirs(file_cache_path, exist_ok=True)
                fpath = _os.path.join(file_cache_path, f'{sym}.json')
                if _os.path.exists(fpath):
                    st = _os.stat(fpath)
                    age = (_time.time() - float(getattr(st, 'st_mtime', 0)))
                    with open(fpath, 'r') as rf:
                        file_cache_hit = _json.load(rf)
                        # ✅ FIX: Sanitize loaded JSON to handle NaN/Infinity from file cache
                        file_cache_hit = _sanitize_json_value(file_cache_hit)
                    if isinstance(file_cache_hit, dict):
                        file_cache_hit.setdefault('symbol', sym)
                        file_cache_hit.setdefault('status', 'success')
                        file_cache_hit['stale_seconds'] = float(age)
                        file_cache_hit['stale'] = bool(age >= ttl)
                    return jsonify(file_cache_hit)
            except Exception as e:
                logger.debug(f"Failed to read file cache for {sym}: {e}")

            # No compute – return pending
            return jsonify({'symbol': sym, 'status': 'pending'})
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
            except Exception as e:
                logger.debug(f"Failed to get basic predictions for {symbol}: {e}")
                basic = {}
            try:
                enhanced_fn = getattr(detector, 'get_enhanced_predictions', None)
                if callable(enhanced_fn):
                    enhanced = enhanced_fn(symbol, stock_data) or {}
            except Exception as e:
                logger.debug(f"Failed to get enhanced predictions for {symbol}: {e}")
                enhanced = {}
            
            # ✅ FIX: Actually use basic and enhanced predictions!
            merged = {}
            merged.update(_normalize_predictions(basic, stock_data['close'].iloc[-1] if hasattr(stock_data, 'iloc') else 0))
            merged.update(_normalize_predictions(enhanced, stock_data['close'].iloc[-1] if hasattr(stock_data, 'iloc') else 0))
            
            return jsonify({'status': 'success', 'symbol': symbol, 'predictions': merged})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    app.register_blueprint(bp)
