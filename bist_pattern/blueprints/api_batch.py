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
            
            for symbol in symbols:
                try:
                    sym = str(symbol).upper().strip()
                    if sym:
                        # ⚡ FIX: Use pattern_detector's internal cache (automation results!)
                        # pattern_detector.analyze_stock() already has cache mechanism
                        # If automation analyzed this symbol, cache hit will be instant!
                        analysis = detector.analyze_stock(sym)
                        results[sym] = analysis
                            
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
                        # ⚡ FIX: Extract horizon-based predictions from enhanced ML
                        horizon_preds = {}
                        try:
                            enhanced = pred_result.get('enhanced', {})
                            if enhanced and isinstance(enhanced, dict):
                                # Enhanced ML format: {horizon: {price, confidence, ...}}
                                for h in ['1d', '3d', '7d', '14d', '30d']:
                                    if h in enhanced:
                                        price = enhanced[h].get('price') if isinstance(enhanced[h], dict) else None
                                        if price:
                                            horizon_preds[h] = price
                        except Exception:
                            pass
                        
                        results[sym] = {
                            'status': 'success',
                            'predictions': horizon_preds,  # Simple {1d: price, 3d: price, ...}
                            'current_price': float(stock_data['close'].iloc[-1]) if hasattr(stock_data, 'iloc') else 0
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
