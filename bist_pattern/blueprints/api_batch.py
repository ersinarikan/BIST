"""
Batch API Blueprint
High-performance batch endpoints for multiple symbols
Reduces N+1 problem significantly
"""

from flask import Blueprint, jsonify, request
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

bp = Blueprint('api_batch', __name__, url_prefix='/api/batch')


def register(app):
    """Register batch API blueprint"""
    from ..extensions import csrf
    
    @bp.route('/pattern-analysis', methods=['POST'])
    @csrf.exempt  # ✅ FIX: Exempt from CSRF for frontend POST requests
    def batch_pattern_analysis():
        """
        Batch pattern analysis for multiple symbols (cache-only fast path)

        - DOES NOT trigger fresh analysis. Returns cached results if available.
        - If no cached result exists, returns {status: 'pending'} for that symbol.

        POST body: {symbols: ['THYAO', 'AKBNK', 'GARAN']}
        Returns: {THYAO: {...|pending}, AKBNK: {...|pending}, ...}
        """
        try:
            data = request.get_json() or {}
            symbols = data.get('symbols', [])

            if not symbols or len(symbols) > 50:  # Limit to 50 symbols
                return jsonify({
                    'status': 'error',
                    'message': 'Provide 1-50 symbols'
                }), 400

            # Cache helpers
            try:
                from bist_pattern.core.cache import cache_get as _cache_get  # type: ignore
            except Exception:
                _cache_get = None  # type: ignore

            import os as _os
            import time as _time
            import json as _json

            file_cache_dir = '/opt/bist-pattern/logs/pattern_cache'
            try:
                _os.makedirs(file_cache_dir, exist_ok=True)
            except Exception:
                pass
            try:
                file_ttl = float(_os.getenv('PATTERN_FILE_CACHE_TTL', '300'))
            except Exception:
                file_ttl = 300.0

            results = {}
            cache_hits = 0
            pendings = 0

            for symbol in symbols:
                sym = str(symbol or '').upper().strip()
                if not sym:
                    continue
                cache_key = f"pattern_analysis:{sym}"
                result = None
                # Memory/Redis cache
                try:
                    if callable(_cache_get):
                        result = _cache_get(cache_key)
                except Exception:
                    result = None
                # File cache (accept even if stale; mark with stale flag)
                if not result:
                    try:
                        import os as _os
                        from bist_pattern.core.broadcaster import _sanitize_json_value
                        fpath = _os.path.join(file_cache_dir, f'{sym}.json')
                        if _os.path.exists(fpath):
                            st = _os.stat(fpath)
                            age = (_time.time() - float(getattr(st, 'st_mtime', 0)))
                            with open(fpath, 'r') as rf:
                                result = _json.load(rf)
                                # ✅ FIX: Sanitize loaded JSON to handle NaN/Infinity from file cache
                                result = _sanitize_json_value(result)
                            # Attach staleness metadata
                            try:
                                if isinstance(result, dict):
                                    result.setdefault('symbol', sym)
                                    result.setdefault('status', 'success')
                                    result['stale_seconds'] = float(age)
                                    result['stale'] = bool(age >= file_ttl)
                            except Exception:
                                pass
                    except Exception:
                        result = None
                if result:
                    try:
                        result.setdefault('symbol', sym)
                        result.setdefault('status', 'success')
                        result['from_cache'] = True
                    except Exception:
                        pass
                    results[sym] = result
                    try:
                        cache_hits += 1
                    except Exception:
                        pass
                else:
                    results[sym] = {'symbol': sym, 'status': 'pending'}
                    pendings += 1

            logger.info(
                f"⚡ Batch pattern API (cache-only): {len(results)} symbols (mem_hits={cache_hits}, pending={pendings})"
            )

            return jsonify({
                'status': 'success',
                'results': results,
                'count': len(results),
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Batch pattern analysis error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @bp.route('/predictions', methods=['POST'])
    @csrf.exempt  # ✅ FIX: Exempt from CSRF for frontend POST requests
    def batch_predictions():
        """
        Batch predictions for multiple symbols (fast path, no fresh compute)

        - Reads precomputed predictions from logs/ml_bulk_predictions.json when available
        - Uses latest DB close price for current_price (single batched query)
        - Falls back to 'pending' for symbols not present in the bulk file

        POST body: {symbols: ['THYAO', 'AKBNK', 'GARAN']}
        Returns: {THYAO: {...}, AKBNK: {...}, GARAN: {...}}
        """
        try:
            data = request.get_json() or {}
            symbols = data.get('symbols', [])

            if not symbols or len(symbols) > 50:
                return jsonify({
                    'status': 'error',
                    'message': 'Provide 1-50 symbols'
                }), 400

            sym_list = [str(s or '').upper().strip() for s in symbols if str(s or '').strip()]
            sym_set = set(sym_list)

            # Load bulk predictions file (generated by automation cycle)
            predictions_map = {}
            bulk_mtime = None
            try:
                import os
                import json
                log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                fpath = os.path.join(log_dir, 'ml_bulk_predictions.json')
                if os.path.exists(fpath):
                    try:
                        bulk_mtime = float(os.path.getmtime(fpath))
                    except Exception:
                        bulk_mtime = None
                    with open(fpath, 'r') as rf:
                        data_json = json.load(rf) or {}
                        predictions_map = (data_json.get('predictions') or {}) if isinstance(data_json, dict) else {}
            except Exception as _pred_err:
                logger.warning(f"bulk predictions not available: {_pred_err}")

            # DB: last close price per symbol (batched)
            last_close_by_symbol = {}
            try:
                from models import db, Stock, StockPrice
                from sqlalchemy import func
                if sym_set:
                    stocks = db.session.query(Stock).filter(Stock.symbol.in_(sym_set)).all()
                    id_to_sym = {s.id: s.symbol for s in stocks}
                    q = (
                        db.session.query(
                            StockPrice.stock_id.label('sid'),
                            func.max(StockPrice.date).label('maxd'),
                        )
                        .filter(StockPrice.stock_id.in_(list(id_to_sym.keys())))
                        .group_by(StockPrice.stock_id)
                    )
                    stats = q.all()
                    max_date_by_sid = {sid: maxd for sid, maxd in stats}
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
            except Exception as _db_err:
                logger.warning(f"db last close fetch failed: {_db_err}")

            # Normalization helper (reused logic)
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
                                        # Store confidence separately (will be added to result later)
                                        conf_key = f'{kk}_conf'
                                        conf = v.get('confidence', 0.5)
                                        if isinstance(conf, (int, float)):
                                            out[conf_key] = float(conf)
                                elif 'price' in v and isinstance(v['price'], (int, float)):
                                    price_val = v['price']
                                    if price_val is not None:
                                        out[kk] = float(price_val)
                                        # ✅ FIX: Also extract confidence if available (for price-based predictions)
                                        conf_key = f'{kk}_conf'
                                        conf = v.get('confidence')
                                        if isinstance(conf, (int, float)):
                                            out[conf_key] = float(conf)
                            elif isinstance(v, (int, float)):
                                out[kk] = float(v)
                except Exception:
                    return {}
                return out

            results = {}
            for sym in sym_list:
                try:
                    # Per-symbol analysis timestamp from pattern_cache
                    analysis_ts = None
                    try:
                        import os
                        import json as _json2
                        log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                        pc_path = os.path.join(log_dir, 'pattern_cache', f'{sym}.json')
                        if os.path.exists(pc_path):
                            try:
                                with open(pc_path, 'r') as pr:
                                    pdata = _json2.load(pr) or {}
                                ts = pdata.get('timestamp')
                                if isinstance(ts, str) and ts:
                                    analysis_ts = ts
                                else:
                                    m = os.path.getmtime(pc_path)
                                    analysis_ts = datetime.fromtimestamp(m).isoformat()
                            except Exception:
                                try:
                                    m = os.path.getmtime(pc_path)
                                    analysis_ts = datetime.fromtimestamp(m).isoformat()
                                except Exception:
                                    analysis_ts = None
                    except Exception:
                        analysis_ts = None

                    pred_entry = predictions_map.get(sym)
                    normalized = {}
                    model_used = None
                    normalized_enhanced = {}
                    normalized_basic = {}
                    
                    # ✅ FIX: Normalize both enhanced and basic, then merge intelligently
                    if isinstance(pred_entry, dict):
                        if pred_entry.get('enhanced'):
                            normalized_enhanced = _normalize(pred_entry.get('enhanced'))
                        if pred_entry.get('basic'):
                            normalized_basic = _normalize(pred_entry.get('basic'))
                    
                    # Merge: Use enhanced where available, fallback to basic for missing horizons
                    if normalized_enhanced:
                        normalized = normalized_enhanced.copy()
                        model_used = 'enhanced'
                        # Fill missing horizons from basic
                        for h in ('1d', '3d', '7d', '14d', '30d'):
                            if h not in normalized and h in normalized_basic:
                                normalized[h] = normalized_basic[h]
                                # Also copy confidence if available
                                conf_key = f'{h}_conf'
                                if conf_key in normalized_basic and conf_key not in normalized:
                                    normalized[conf_key] = normalized_basic[conf_key]
                    elif normalized_basic:
                        normalized = normalized_basic.copy()
                        model_used = 'basic'

                    if normalized:
                        # Extract confidences from normalized (1d_conf, 3d_conf, etc.)
                        confidences = {}
                        predictions = {}
                        # ✅ FIX: Track which model was used for each horizon
                        models_by_horizon = {}
                        
                        for k, v in normalized.items():
                            if k.endswith('_conf'):
                                horizon = k.replace('_conf', '')
                                confidences[horizon] = v
                            else:
                                predictions[k] = v
                                # Determine model for this horizon
                                if normalized_enhanced and k in normalized_enhanced:
                                    models_by_horizon[k] = 'enhanced'
                                elif normalized_basic and k in normalized_basic:
                                    models_by_horizon[k] = 'basic'
                        
                        results[sym] = {
                            'status': 'success',
                            'predictions': predictions,
                            'confidences': confidences,
                            'model': model_used or 'basic',  # Overall model (enhanced if any horizon uses it)
                            'models_by_horizon': models_by_horizon,  # ✅ FIX: Per-horizon model info
                            'current_price': last_close_by_symbol.get(sym),
                            'source_timestamp': (datetime.fromtimestamp(bulk_mtime).isoformat() if bulk_mtime else None),
                            'analysis_timestamp': analysis_ts
                        }
                    else:
                        # No precomputed prediction present → pending (do not compute here)
                        results[sym] = {
                            'status': 'pending',
                            'predictions': {},
                            'current_price': last_close_by_symbol.get(sym),
                            'source_timestamp': (datetime.fromtimestamp(bulk_mtime).isoformat() if bulk_mtime else None),
                            'analysis_timestamp': analysis_ts
                        }
                except Exception as e:
                    logger.error(f"Batch prediction error for {sym}: {e}")
                    results[sym] = {'status': 'error', 'error': str(e)}

            return jsonify({
                'status': 'success',
                'results': results,
                'count': len(results),
                'timestamp': datetime.now().isoformat(),
                'source_timestamp': (datetime.fromtimestamp(bulk_mtime).isoformat() if bulk_mtime else None)
            })

        except Exception as e:
            logger.error(f"Batch predictions error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    app.register_blueprint(bp)
    logger.info("✅ Batch API blueprint registered")
