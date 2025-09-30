from flask import Blueprint, jsonify, request
from datetime import datetime
from flask_login import login_required, current_user

from ..extensions import csrf

bp = Blueprint('api_watchlist', __name__, url_prefix='/api')


def register(app):
    from models import db, User, Stock, Watchlist

    def _get_effective_user():
        try:
            from flask_login import current_user
            if current_user.is_authenticated:
                return current_user
        except Exception:
            pass
        try:
            if app.config.get('DEV_AUTH_BYPASS'):
                user_id_header = request.headers.get('X-User-Id')
                user_id = int(user_id_header) if user_id_header else 1
                return User.query.get(user_id)
        except Exception:
            return None
        return None

    @bp.route('/watchlist', methods=['GET'])
    def get_watchlist():
        try:
            user = _get_effective_user()
            if not user:
                return jsonify({'status': 'unauthorized'}), 401
            from sqlalchemy.orm import joinedload
            items = (
                Watchlist.query.options(joinedload(Watchlist.stock))
                .filter_by(user_id=user.id)
                .all()
            )
            return jsonify({'status': 'success', 'user_id': user.id, 'watchlist': [i.to_dict() for i in items]})
        except Exception as e:
            app.logger.error(f"Watchlist get error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/watchlist', methods=['POST'])
    @csrf.exempt
    def add_watchlist():
        try:
            user = _get_effective_user()
            if not user:
                return jsonify({'status': 'unauthorized'}), 401
            data = request.get_json() or {}
            symbol = (data.get('symbol') or '').upper().strip()
            if not symbol:
                return jsonify({'status': 'error', 'error': 'symbol is required'}), 400
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                if app.config.get('AUTO_CREATE_STOCKS', True):
                    stock = Stock(symbol=symbol, name=f"{symbol} Hisse Senedi", sector=data.get('sector') or 'Unknown')
                    db.session.add(stock)
                    db.session.flush()
                else:
                    return jsonify({'status': 'error', 'error': 'stock not found'}), 404
            item = Watchlist.query.filter_by(user_id=user.id, stock_id=stock.id).first()
            if not item:
                item = Watchlist(user_id=user.id, stock_id=stock.id)
                db.session.add(item)
            item.notes = data.get('notes')
            if 'alert_enabled' in data:
                item.alert_enabled = bool(data.get('alert_enabled'))
            if 'alert_threshold_buy' in data and data.get('alert_threshold_buy') is not None:
                try:
                    buy_val = data.get('alert_threshold_buy')
                    if isinstance(buy_val, (int, float, str)) and str(buy_val).strip() != '':
                        item.alert_threshold_buy = float(buy_val)
                except Exception:
                    pass
            if 'alert_threshold_sell' in data and data.get('alert_threshold_sell') is not None:
                try:
                    sell_val = data.get('alert_threshold_sell')
                    if isinstance(sell_val, (int, float, str)) and str(sell_val).strip() != '':
                        item.alert_threshold_sell = float(sell_val)
                except Exception:
                    pass
            db.session.commit()
            try:
                from app import get_pattern_detector
                result = get_pattern_detector().analyze_stock(symbol)
                if hasattr(app, 'socketio') and result:
                    app.socketio.emit('pattern_analysis', {
                        'symbol': symbol,
                        'data': result,
                        'timestamp': datetime.now().isoformat()
                    }, to=f'stock_{symbol}')
            except Exception:
                pass
            return jsonify({'status': 'success', 'item': item.to_dict()})
        except Exception as e:
            app.logger.error(f"Watchlist add error: {e}")
            try:
                db.session.rollback()
            except Exception:
                pass
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/watchlist/<symbol>', methods=['DELETE'])
    @csrf.exempt
    def delete_watchlist(symbol):
        try:
            user = _get_effective_user()
            if not user:
                return jsonify({'status': 'unauthorized'}), 401
            if not symbol:
                return jsonify({'status': 'error', 'error': 'symbol is required'}), 400
            symbol = symbol.upper().strip()
            stock = Stock.query.filter_by(symbol=symbol).first()
            if not stock:
                return jsonify({'status': 'error', 'error': 'stock not found'}), 404
            item = Watchlist.query.filter_by(user_id=user.id, stock_id=stock.id).first()
            if not item:
                return jsonify({'status': 'error', 'error': 'watchlist item not found'}), 404
            db.session.delete(item)
            db.session.commit()
            return jsonify({'status': 'success', 'message': f'{symbol} removed'})
        except Exception as e:
            app.logger.error(f"Watchlist delete error: {e}")
            try:
                db.session.rollback()
            except Exception:
                pass
            return jsonify({'status': 'error', 'error': str(e)}), 500

    @bp.route('/watchlist/predictions')
    def watchlist_predictions():
        """Kullanıcının watchlist'indeki tüm hisseler için 1/3/7/14/30 günlük tahminleri döndürür.
        Öncelik sırası: Enhanced ML (varsa) → Basic ML (bulk dosyadan) → boş.
        Kaynak: /opt/bist-pattern/logs/ml_bulk_predictions.json
        """
        try:
            # ⚡ FIX: Use effective user (works for both authenticated and anonymous)
            user = _get_effective_user()
            user_id = user.id if user else 'anon'
            
            # Cache (per-user)
            try:
                cache_key = f"watchlist_predictions:{user_id}"
                cached = getattr(app, '_api_cache_get', None)
                if callable(cached):
                    hit = cached(cache_key)
                    if hit:
                        return jsonify(hit)
            except Exception:
                pass

            # Watchlist sembollerini al
            from models import StockPrice  # local import
            items = Watchlist.query.filter_by(user_id=user_id).all() if user_id != 'anon' else []
            symbols = [item.stock.symbol for item in items if getattr(item, 'stock', None)]

            # Bulk predictions dosyasını yükle
            predictions_map = {}
            try:
                import json as _json
                import os
                log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                fpath = os.path.join(log_dir, 'ml_bulk_predictions.json')
                if os.path.exists(fpath):
                    with open(fpath, 'r') as rf:
                        data = _json.load(rf) or {}
                        predictions_map = (data.get('predictions') or {}) if isinstance(data, dict) else {}
            except Exception:
                predictions_map = {}

            # Last signal snapshot dosyası
            last_signal_map = {}
            try:
                import json as _json
                import os
                log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                snap_path = os.path.join(log_dir, 'signals_last.json')
                if os.path.exists(snap_path):
                    with open(snap_path, 'r') as rf:
                        last_signal_map = _json.load(rf) or {}
            except Exception:
                last_signal_map = {}

            def _normalize(raw):
                out = {}
                if not raw:
                    return out
                try:
                    if isinstance(raw, dict) and 'predictions' in raw and isinstance(raw['predictions'], dict):
                        raw = raw['predictions']
                    for k, v in (raw.items() if isinstance(raw, dict) else []):
                        kk = str(k).lower()
                        if kk in ('1d', '3d', '7d', '14d', '30d'):
                            if isinstance(v, dict):
                                if 'ensemble_prediction' in v and isinstance(v['ensemble_prediction'], (int, float)):
                                    ep = v['ensemble_prediction']
                                    if ep is not None:
                                        out[kk] = float(ep)
                                elif 'price' in v and isinstance(v['price'], (int, float)):
                                    price_val = v['price']
                                    if price_val is not None:
                                        out[kk] = float(price_val)
                            elif isinstance(v, (int, float)):
                                out[kk] = float(v)
                except Exception:
                    return {}
                return out

            # DB'den toplu son fiyat, gün sayısı ve son tarih
            last_close_by_symbol = {}
            days_count_by_symbol = {}
            last_date_by_symbol = {}
            try:
                from sqlalchemy import func
                sym_set = set(symbols)
                if sym_set:
                    stocks = Stock.query.filter(Stock.symbol.in_(sym_set)).all()
                    id_to_sym = {s.id: s.symbol for s in stocks}
                    q = (
                        db.session.query(
                            StockPrice.stock_id.label('sid'),
                            func.count(StockPrice.id).label('cnt'),
                            func.max(StockPrice.date).label('maxd'),
                        )
                        .filter(StockPrice.stock_id.in_(list(id_to_sym.keys())))
                        .group_by(StockPrice.stock_id)
                    )
                    stats = q.all()
                    max_date_by_sid = {sid: maxd for sid, _cnt, maxd in stats}
                    if max_date_by_sid:
                        latest_rows = (
                            db.session.query(StockPrice)
                            .filter(
                                StockPrice.stock_id.in_(list(max_date_by_sid.keys())),
                                StockPrice.date.in_(list(set(max_date_by_sid.values()))),
                            )
                            .all()
                        )
                        for r in latest_rows:
                            sym = id_to_sym.get(r.stock_id)
                            if sym and max_date_by_sid.get(r.stock_id) == r.date:
                                last_close_by_symbol[sym] = float(r.close_price)
                    for sid, cnt, maxd in stats:
                        sym = id_to_sym.get(sid)
                        if not sym:
                            continue
                        days_count_by_symbol[sym] = int(cnt or 0)
                        last_date_by_symbol[sym] = str(maxd) if maxd else None
            except Exception:
                pass

            response_items = []
            for sym in symbols:
                pred_entry = predictions_map.get(sym) or {}
                model_used = None
                normalized = {}
                if isinstance(pred_entry, dict) and pred_entry.get('enhanced'):
                    normalized = _normalize(pred_entry.get('enhanced'))
                    model_used = 'enhanced'
                if not normalized and isinstance(pred_entry, dict) and pred_entry.get('basic'):
                    normalized = _normalize(pred_entry.get('basic'))
                    model_used = model_used or 'basic'
                s_obj = next((s for s in stocks if s.symbol == sym), None) if 'stocks' in locals() else None
                ls = last_signal_map.get(sym) if isinstance(last_signal_map, dict) else None
                response_items.append({
                    'symbol': sym,
                    'current_price': last_close_by_symbol.get(sym),
                    'predictions': normalized,
                    'model': model_used or 'none',
                    'data_days': days_count_by_symbol.get(sym),
                    'last_date': last_date_by_symbol.get(sym),
                    'name': getattr(s_obj, 'name', None) if s_obj else None,
                    'last_signal': ls if isinstance(ls, dict) else None,
                })

            payload = {'status': 'success', 'count': len(response_items), 'items': response_items}
            try:
                setter = getattr(app, '_api_cache_set', None)
                if callable(setter):
                    setter(cache_key, payload, ttl_seconds=float(os.getenv('API_CACHE_TTL_WATCHLIST', '5')))
            except Exception:
                pass
            return jsonify(payload)
        except Exception as e:
            app.logger.error(f"Watchlist predictions error: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    app.register_blueprint(bp)
