"""
Batch API Blueprint
High-performance batch endpoints for multiple symbols
Reduces N+1 problem significantly
"""

from flask import Blueprint, jsonify, request
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)

bp = Blueprint('api_batch', __name__, url_prefix='/api/batch')

# ⚡ Simple in-memory cache for batch results
_batch_cache = {}
_CACHE_TTL = 300  # 5 minutes


def register(app):
    """Register batch API blueprint"""
    
    @bp.route('/pattern-analysis', methods=['POST'])
    def batch_pattern_analysis():
        """
        Batch pattern analysis for multiple symbols
        
        POST body: {symbols: ['THYAO', 'AKBNK', 'GARAN']}
        Returns: {THYAO: {...}, AKBNK: {...}, GARAN: {...}}
        """
        try:
            data = request.get_json() or {}
            symbols = data.get('symbols', [])
            
            if not symbols or len(symbols) > 50:  # Limit to 50 symbols
                return jsonify({
                    'status': 'error',
                    'message': 'Provide 1-50 symbols'
                }), 400
            
            # Import pattern detector
            from app import get_pattern_detector
            detector = get_pattern_detector()
            
            results = {}
            cache_hits = 0
            cache_misses = 0
            
            for symbol in symbols:
                try:
                    sym = str(symbol).upper().strip()
                    if sym:
                        # ⚡ FIX: Use module-level cache
                        cache_key = f"pattern_{sym}"
                        now = time.time()
                        
                        # Check cache
                        if cache_key in _batch_cache:
                            entry = _batch_cache[cache_key]
                            age = now - entry.get('ts', 0)
                            if age < _CACHE_TTL:
                                results[sym] = entry['data']
                                cache_hits += 1
                                continue
                        
                        # Cache miss - do fresh analysis
                        analysis = detector.analyze_stock(sym)
                        results[sym] = analysis
                        cache_misses += 1
                        
                        # Store in cache
                        _batch_cache[cache_key] = {
                            'data': analysis,
                            'ts': now
                        }
                            
                except Exception as e:
                    logger.error(f"Batch analysis error for {symbol}: {e}")
                    results[str(symbol)] = {
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Log cache efficiency
            if cache_hits + cache_misses > 0:
                hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100
                logger.info(f"⚡ Batch pattern API: {len(results)} symbols, cache {cache_hits}/{cache_hits+cache_misses} ({hit_rate:.0f}%)")
            
            return jsonify({
                'status': 'success',
                'results': results,
                'count': len(results),
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Batch pattern analysis error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    @bp.route('/predictions', methods=['POST'])
    def batch_predictions():
        """
        Batch predictions for multiple symbols
        
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
            
            # Import dependencies
            from bist_pattern.core.ml_coordinator import get_ml_coordinator
            from app import get_pattern_detector
            
            ml_coord = get_ml_coordinator()
            detector = get_pattern_detector()
            
            results = {}
            for symbol in symbols:
                try:
                    sym = str(symbol).upper().strip()
                    if not sym:
                        continue
                    
                    # Get stock data
                    stock_data = detector.get_stock_data(sym)
                    if stock_data is None or len(stock_data) < 50:
                        results[sym] = {'status': 'insufficient_data'}
                        continue
                    
                    # Get ML predictions (coordinator handles caching)
                    pred_result = ml_coord.predict_with_coordination(sym, stock_data)
                    
                    if pred_result:
                        results[sym] = {
                            'status': 'success',
                            'predictions': pred_result
                        }
                    else:
                        results[sym] = {'status': 'no_predictions'}
                        
                except Exception as e:
                    logger.error(f"Batch prediction error for {symbol}: {e}")
                    results[str(symbol)] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            return jsonify({
                'status': 'success',
                'results': results,
                'count': len(results),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Batch predictions error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    app.register_blueprint(bp)
    logger.info("✅ Batch API blueprint registered")
