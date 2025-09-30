"""
Dashboard API Blueprint
System stats and recent tasks for dashboard
"""

import os
import json
from datetime import datetime
from flask import Blueprint, jsonify, current_app
from models import Stock, StockPrice, db
from bist_pattern.core.cache import cache_get as _cache_get, cache_set as _cache_set

bp = Blueprint('dashboard_api', __name__, url_prefix='/api')


@bp.route('/dashboard-stats')
def dashboard_stats():
    """Dashboard için istatistikler"""
    try:
        # Cache kontrolü
        cache_key = 'dashboard_stats'
        cached = _cache_get(cache_key)
        if cached:
            return jsonify(cached)
        
        from sqlalchemy import func, desc
        
        # Temel istatistikler
        total_stocks = Stock.query.count()
        total_prices = StockPrice.query.count()
        
        # En çok veri olan hisseler
        stock_with_most_data = db.session.query(
            Stock.symbol,
            func.count(StockPrice.id).label('price_count')
        ).join(StockPrice).group_by(Stock.symbol)\
        .order_by(desc('price_count')).limit(5).all()
        
        # Sektör dağılımı
        sector_stats = db.session.query(
            Stock.sector,
            func.count(Stock.id).label('stock_count')
        ).group_by(Stock.sector)\
        .order_by(desc('stock_count')).limit(10).all()
        
        top_stocks_data = [{'symbol': s, 'count': c} for s, c in stock_with_most_data]
        sector_data = [{'sector': s, 'count': c} for s, c in sector_stats]

        result = {
            'total_stocks': total_stocks,
            'total_prices': total_prices,
            'top_stocks': top_stocks_data,
            'sectors': sector_data,
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 60 saniye cache
        _cache_set(cache_key, result, 60)
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Dashboard stats error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/data-collection/status')
def data_collection_status():
    """Veri toplama durumu"""
    try:
        # Basit durum bilgisi
        from sqlalchemy import func
        
        # En son veri tarihi
        latest_date = db.session.query(func.max(StockPrice.date)).scalar()
        
        # Günlük veri sayısı
        latest_count = 0
        if latest_date:
            latest_count = StockPrice.query.filter_by(date=latest_date).count()
        
        return jsonify({
            'status': 'success',
            'latest_date': latest_date.isoformat() if latest_date else None,
            'latest_count': latest_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Data collection status error: {e}")
        return jsonify({'error': str(e)}), 500


@bp.route('/recent-tasks')
def recent_tasks_simple():
    """Dashboard için son görevler"""
    try:
        log_path = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        status_file = os.path.join(log_path, 'pipeline_status.json')
        history = []
        
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as rf:
                    data = json.load(rf) or {}
                    history = (data.get('history') or [])[-50:]
            except Exception:
                history = []
        
        # Compute durations between start/end and compress duplicates
        def _parse(ts):
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                try:
                    return datetime.strptime((ts or '').split('.')[0], '%Y-%m-%dT%H:%M:%S')
                except Exception:
                    return None
        
        hist_sorted = sorted(history, key=lambda h: h.get('timestamp') or '')
        open_phase = {}
        durations = {}
        
        for ev in hist_sorted:
            ph = ev.get('phase')
            st = ev.get('state')
            ts = _parse(ev.get('timestamp', ''))
            if not ph or not st or ts is None:
                continue
            if st == 'start':
                open_phase[ph] = ts
            elif st == 'end' and ph in open_phase:
                try:
                    durations[ph] = max(0, int((ts - open_phase.pop(ph)).total_seconds()))
                except Exception:
                    pass
        
        tasks = []
        last_key = None
        for h in reversed(hist_sorted):
            ph = h.get('phase', 'pipeline')
            st = h.get('state', 'pending')
            key = f"{ph}|{st}"
            desc = st
            if st == 'end' and ph in durations and durations[ph] > 0:
                desc = f"end (duration={durations[ph]}s)"
            if tasks and last_key == key:
                tasks[0]['repeat'] = tasks[0].get('repeat', 1) + 1
            else:
                tasks.insert(0, {
                    'task': ph,
                    'description': desc,
                    'status': st,
                    'timestamp': h.get('timestamp', ''),
                    'icon': '🧩',
                    'repeat': 1
                })
                last_key = key
        
        tasks = list(reversed(tasks))
        
        resp = jsonify({'status': 'success', 'history': history, 'tasks': tasks})
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp
        
    except Exception as e:
        current_app.logger.error(f"Recent tasks error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


def register(app):
    """Register dashboard blueprint with app"""
    app.register_blueprint(bp)
