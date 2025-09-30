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
            cache_used = 0
            fresh_analysis = 0
            
            for symbol in symbols:
                try:
                    sym = str(symbol).upper().strip()
                    if sym:
                        # ⚡ OPTIMIZED: Get from cache (instant, no analysis!)
                        analysis = detector.analyze_stock(sym)  # Uses cache if available
                        results[sym] = analysis
                        
                        # Track cache usage
                        if analysis.get('from_cache'):
                            cache_used += 1
                        else:
                            fresh_analysis += 1
                            
                except Exception as e:
                    logger.error(f"Batch analysis error for {symbol}: {e}")
                    results[str(symbol)] = {
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    }
            
            # ✅ pattern_detector.analyze_stock() internally uses cache
            # If automation cycle analyzed these symbols, they will be cache hits!
            # No need for separate batch cache - automation results are reused
            logger.info(f"⚡ Batch pattern API: {len(results)} symbols analyzed (automation cache reused)")
            
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
            from app import get_pattern_detector
            
            detector = get_pattern_detector()
            
            results = {}
            for symbol in symbols:
                try:
                    sym = str(symbol).upper().strip()
                    if not sym:
                        continue
                    
                    # ⚡ OPTIMIZED: Get predictions from pattern analysis ml_unified!
                    # This is INSTANT (cache hit) and includes all predictions already computed!
                    # NO veri temizleme, NO prediction calculation - just extract!
                    analysis = detector.analyze_stock(sym)  # Cache hit if automation analyzed
                    
                    if not analysis or analysis.get('status') == 'error':
                        results[sym] = {'status': 'error'}
                        continue
                    
                    # Extract predictions from ml_unified (already computed!)
                    horizon_preds = {}
                    current_price = analysis.get('current_price', 0)
                    
                    try:
                        ml_unified = analysis.get('ml_unified', {})
                        if ml_unified and isinstance(ml_unified, dict):
                            for h in ['1d', '3d', '7d', '14d', '30d']:
                                if h in ml_unified:
                                    h_data = ml_unified[h]
                                    # Try enhanced first, then basic
                                    enhanced = h_data.get('enhanced', {}) if isinstance(h_data, dict) else {}
                                    basic = h_data.get('basic', {}) if isinstance(h_data, dict) else {}
                                    
                                    price = None
                                    if isinstance(enhanced, dict) and 'price' in enhanced:
                                        price = enhanced['price']
                                    elif isinstance(basic, dict) and 'price' in basic:
                                        price = basic['price']
                                    
                                    if price and isinstance(price, (int, float)):
                                        horizon_preds[h] = float(price)
                    except Exception as e:
                        logger.error(f"ml_unified extraction error for {sym}: {e}")
                    
                    results[sym] = {
                        'status': 'success',
                        'predictions': horizon_preds,
                        'current_price': current_price
                    }
                        
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
