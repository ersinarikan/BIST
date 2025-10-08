"""
Stocks API Blueprint
Stock data, prices, search and pattern analysis endpoints
"""

import os
from flask import Blueprint, jsonify, request, current_app
from models import Stock, StockPrice, db
from bist_pattern.core.cache import cache_get as _cache_get, cache_set as _cache_set

bp = Blueprint('stocks_api', __name__, url_prefix='/api')


@bp.route('/stocks')
def api_stocks():
    """Get list of stocks"""
    try:
        stocks = Stock.query.limit(50).all()
        return jsonify([{
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector
        } for stock in stocks])
    except Exception as e:
        current_app.logger.error(f"API stocks error: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route('/stock-prices/<symbol>')
def get_stock_prices(symbol):
    """Get price data for specific stock"""
    try:
        from sqlalchemy import desc
        stock = Stock.query.filter_by(symbol=symbol.upper()).first()
        if not stock:
            return jsonify({'error': 'Hisse bulunamadı'}), 404
        
        # Son 60 günlük veri
        prices = StockPrice.query.filter_by(stock_id=stock.id)\
                    .order_by(desc(StockPrice.date))\
                    .limit(60).all()
        
        if not prices:
            return jsonify({'error': 'Fiyat verisi bulunamadı'}), 404
        
        # JSON formatına çevir
        price_data = []
        for price in reversed(prices):  # Tarihe göre sırala
            price_data.append({
                'date': price.date.strftime('%Y-%m-%d'),
                'open': float(price.open_price),
                'high': float(price.high_price),
                'low': float(price.low_price),
                'close': float(price.close_price),
                'volume': int(price.volume)
            })
        
        return jsonify({
            'symbol': symbol.upper(),
            'name': stock.name,
            'sector': stock.sector,
            'data': price_data,
            'total_records': len(price_data)
        })
        
    except Exception as e:
        current_app.logger.error(f"Stock prices error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/stocks/search')
def api_stocks_search():
    """Search stocks by symbol or name"""
    try:
        q = (request.args.get('q') or '').strip()
        if not q or len(q) < 2:
            return jsonify({'status': 'success', 'stocks': []})
        
        # Cache by query
        cache_key = f"fts:{q.lower()}"
        try:
            cached = _cache_get(cache_key)
            if cached:
                return jsonify({'status': 'success', 'stocks': cached})
        except Exception:
            pass
        
        from sqlalchemy import func, or_, text as _text
        results = []
        
        try:
            # PostgreSQL Full-Text Search
            cfg = os.getenv('PG_FTS_CONFIG', 'turkish')
            sql = _text("""
                SELECT symbol, name, sector
                FROM stocks
                WHERE to_tsvector(:cfg, coalesce(symbol,'') || ' ' || coalesce(name,''))
                      @@ websearch_to_tsquery(:cfg, :q)
                ORDER BY ts_rank_cd(
                    to_tsvector(:cfg, coalesce(symbol,'') || ' ' || coalesce(name,'')),
                    websearch_to_tsquery(:cfg, :q)
                ) DESC
                LIMIT 20
            """)
            rows = db.session.execute(sql, {'cfg': cfg, 'q': q}).mappings().all()
            results = [Stock(symbol=r['symbol'], name=r['name'], sector=r['sector']) for r in rows]
        except Exception:
            # Fallback: ILIKE search
            pat = f"%{q.upper()}%"
            results = (db.session.query(Stock)
                       .filter(or_(func.upper(Stock.symbol).like(pat), func.upper(Stock.name).like(pat)))
                       .limit(20).all())
        
        out = [
            {'symbol': s.symbol, 'name': s.name, 'sector': s.sector}
            for s in results
        ]
        
        try:
            _cache_set(cache_key, out, ttl_seconds=float(os.getenv('API_CACHE_TTL_SEARCH', '10')))
        except Exception:
            pass
        
        return jsonify({'status': 'success', 'stocks': out})
    except Exception as e:
        current_app.logger.error(f"Stock search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@bp.route('/pattern-analysis/<symbol>')
def api_pattern_analysis(symbol):
    """Get pattern analysis for specific stock"""
    try:
        sym = (symbol or '').upper().strip()
        if not sym:
            return jsonify({'status': 'error', 'message': 'symbol required'}), 400
        
        # Cache check
        use_fast = (request.args.get('fast') or '').lower() in ('1', 'true', 'yes')
        cache_key = f"pattern_analysis:{sym}"
        cached = _cache_get(cache_key)
        if cached and (use_fast or True):
            return jsonify(cached)
        
        # Get pattern detector (this might need adjustment)
        try:
            from app import get_pattern_detector
            result = get_pattern_detector().analyze_stock(sym)
        except Exception as e:
            current_app.logger.warning(f"Pattern detector error for {sym}: {e}")
            result = {'status': 'error', 'message': 'Pattern analysis unavailable'}
        
        # Ensure minimal schema
        if isinstance(result, dict):
            result.setdefault('symbol', sym)
            result.setdefault('status', 'success')
        
        # Cache result for short period
        try:
            cache_ttl = float(os.getenv('PATTERN_CACHE_TTL', '30'))
            _cache_set(cache_key, result, cache_ttl)
        except Exception:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Pattern analysis error for {symbol}: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


def register(app):
    """Register stocks API blueprint"""
    app.register_blueprint(bp)
