import logging
from flask import Blueprint, jsonify
from datetime import datetime

from ..extensions import csrf

logger = logging.getLogger(__name__)

bp = Blueprint('api_metrics', __name__, url_prefix='/api')


def register(app):
    from models import db, Stock, StockPrice

    @bp.route('/system-info')
    def system_info():
        try:
            stocks = Stock.query.count()
            prices = StockPrice.query.count()
            return jsonify({
                'status': 'success',
                'database': {
                    'stocks': stocks,
                    'price_records': prices,
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @bp.route('/stock-prices/<symbol>')
    def get_stock_prices(symbol):
        try:
            from sqlalchemy import desc
            stock = Stock.query.filter_by(symbol=symbol.upper()).first()
            if not stock:
                return jsonify({'error': 'Hisse bulunamadı'}), 404
            prices = (
                StockPrice.query.filter_by(stock_id=stock.id)
                .order_by(desc(StockPrice.date))
                .limit(60)
                .all()
            )
            if not prices:
                return jsonify({'error': 'Fiyat verisi bulunamadı'}), 404
            price_data = [
                {
                    'date': p.date.strftime('%Y-%m-%d'),
                    'open': float(p.open_price),
                    'high': float(p.high_price),
                    'low': float(p.low_price),
                    'close': float(p.close_price),
                    'volume': int(p.volume),
                }
                for p in reversed(prices)
            ]
            return jsonify({
                'symbol': symbol.upper(),
                'name': stock.name,
                'sector': stock.sector,
                'data': price_data,
                'total_records': len(price_data),
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @bp.route('/dashboard-stats')
    def dashboard_stats():
        try:
            from sqlalchemy import func, desc
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            stock_with_most_data = (
                db.session.query(
                    Stock.symbol,
                    func.count(StockPrice.id).label('price_count'),
                )
                .join(StockPrice)
                .group_by(Stock.symbol)
                .order_by(desc('price_count'))
                .limit(5)
                .all()
            )
            sector_stats = (
                db.session.query(
                    Stock.sector,
                    func.count(Stock.id).label('stock_count'),
                )
                .group_by(Stock.sector)
                .order_by(desc('stock_count'))
                .limit(10)
                .all()
            )
            top_stocks_data = [{'symbol': s, 'count': c} for s, c in stock_with_most_data]
            sector_data = [{'sector': s, 'count': c} for s, c in sector_stats]
            return jsonify({
                'total_stocks': total_stocks,
                'total_prices': total_prices,
                'top_stocks': top_stocks_data,
                'sectors': sector_data,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        except Exception as e:
            app.logger.error(f"Dashboard stats error: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/system-ml-info')
    def system_ml_info():
        """Model/YOLO/scheduler görünürlüğü; front-end için güvenli, opsiyonel alanlar."""
        out = {
            'xgboost_available': None,  # type: ignore[assignment]
            'lightgbm_available': None,  # type: ignore[assignment]
            'catboost_available': None,  # type: ignore[assignment]
            'base_ml_available': None,  # type: ignore[assignment]
            'enhanced_models_trained': None,  # type: ignore[assignment]
            'prediction_horizons': None,  # type: ignore[assignment]
            'yolo': None,  # type: ignore[assignment]
            'scheduler': None,  # type: ignore[assignment]
        }
        # Enhanced ML
        try:
            from enhanced_ml_system import get_enhanced_ml_system
            info = get_enhanced_ml_system().get_system_info()
            out['xgboost_available'] = bool(info.get('xgboost_available'))  # type: ignore[index]
            out['lightgbm_available'] = bool(info.get('lightgbm_available'))  # type: ignore[index]
            out['catboost_available'] = bool(info.get('catboost_available'))  # type: ignore[index]
            out['base_ml_available'] = bool(info.get('base_ml_available'))  # type: ignore[index]
            out['enhanced_models_trained'] = int(info.get('models_trained', 0))  # type: ignore[index]
            out['prediction_horizons'] = info.get('prediction_horizons')  # type: ignore[index]
        except Exception as e:
            logger.debug(f"Failed to get enhanced ML system info: {e}")
        # YOLO (use async visual system for accurate runtime status)
        try:
            from visual_pattern_async import get_async_visual_pattern_system  # type: ignore
            y = get_async_visual_pattern_system().get_system_info()
            out['yolo'] = y  # type: ignore[index]
        except Exception as e:
            logger.debug(f"Failed to get YOLO system info: {e}")
        # Scheduler status via working_automation (fallback to legacy)
        try:
            from working_automation import get_working_automation_pipeline  # type: ignore
            s = get_working_automation_pipeline().get_scheduler_status()
            out['scheduler'] = s  # type: ignore[index]
        except Exception:
            try:
                from scheduler import get_automated_pipeline  # type: ignore
                s = get_automated_pipeline().get_scheduler_status()
                out['scheduler'] = s  # type: ignore[index]
            except Exception as e2:
                logger.debug(f"Failed to get scheduler status (fallback): {e2}")
        return jsonify({'status': 'success', 'system_ml_info': out, 'timestamp': datetime.now().isoformat()})

    @bp.route('/analysis/summary-24h')
    def analysis_summary_24h():
        """Son 24 saatte detailed vs simple analiz özeti (pipeline_status.json'dan)."""
        try:
            import os
            import json
            from datetime import datetime, timedelta
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            status_file = os.path.join(log_dir, 'pipeline_status.json')
            if not os.path.exists(status_file):
                return jsonify({'status': 'success', 'window_hours': 24, 'detailed': {'count': 0, 'symbols': []}, 'simple': {'count': 0, 'symbols': []}, 'timestamp': datetime.now().isoformat()})
            try:
                with open(status_file, 'r') as rf:
                    data = json.load(rf) or {}
            except Exception:
                data = {}
            history = data.get('history') or []
            since = datetime.now() - timedelta(hours=24)
            detailed_set = set()
            simple_set = set()
            for entry in history:
                try:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get('phase') != 'ai_analysis' or entry.get('state') != 'end':
                        continue
                    ts_str = entry.get('timestamp')
                    if not ts_str:
                        continue
                    ts = None
                    try:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    except Exception:
                        continue
                    if ts < since:
                        continue
                    details = entry.get('details') or {}
                    sym = details.get('symbol')
                    mode = details.get('mode')
                    if not sym or not mode:
                        continue
                    if mode == 'detailed':
                        detailed_set.add(sym)
                    elif mode == 'simple':
                        simple_set.add(sym)
                except Exception:
                    continue
            return jsonify({
                'status': 'success',
                'window_hours': 24,
                'detailed': {'count': len(detailed_set), 'symbols': sorted(list(detailed_set))[:200]},
                'simple': {'count': len(simple_set), 'symbols': sorted(list(simple_set))[:200]},
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/analysis/enhanced-usage')
    def analysis_enhanced_usage():
        """ml_bulk_predictions.json üzerinden ENH (enhanced) tahminleri olan sembolleri listele."""
        try:
            import os
            import json
            log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
            fp = os.path.join(log_dir, 'ml_bulk_predictions.json')
            if not os.path.exists(fp):
                return jsonify({'status': 'success', 'symbols_enhanced': [], 'file_timestamp': None})
            with open(fp, 'r') as rf:
                data = json.load(rf) or {}
            preds = data.get('predictions') or {}
            symbols_enh = []
            if isinstance(preds, dict):
                for sym, entry in preds.items():
                    if isinstance(entry, dict) and entry.get('enhanced'):
                        symbols_enh.append(sym)
            file_ts = None
            try:
                file_ts = datetime.fromtimestamp(os.path.getmtime(fp)).isoformat()
            except Exception:
                file_ts = None
            return jsonify({'status': 'success', 'symbols_enhanced': sorted(symbols_enh), 'file_timestamp': file_ts})
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/signals/last')
    def signals_last():
        """Son genel sinyal özetine hızlı erişim (log dosyasından)."""
        try:
            import os
            import json
            path = '/opt/bist-pattern/logs/signals_last.json'
            if not os.path.exists(path):
                return jsonify({'status': 'success', 'signals': {}, 'timestamp': datetime.now().isoformat()})
            with open(path, 'r') as f:
                data = json.load(f) or {}
            return jsonify({'status': 'success', 'signals': data, 'timestamp': datetime.now().isoformat()})
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/data-collection/status')
    def data_collection_status():
        try:
            from sqlalchemy import func
            latest_date = db.session.query(func.max(StockPrice.date)).scalar()
            latest_count = (
                StockPrice.query.filter_by(date=latest_date).count()
                if latest_date
                else 0
            )
            return jsonify({
                'latest_date': str(latest_date) if latest_date else None,
                'latest_count': latest_count,
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @bp.route('/data-collection/manual', methods=['POST'])
    @csrf.exempt
    def manual_data_collection():
        try:
            from advanced_collector import AdvancedBISTCollector
            collector = AdvancedBISTCollector()
            res = collector.collect_all_stocks_parallel()
            return jsonify({'status': 'success', 'result': res})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    app.register_blueprint(bp)
