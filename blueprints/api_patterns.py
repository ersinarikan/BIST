"""
Pattern Analysis API Blueprint
Pattern detection, analysis, and summary endpoints
"""

from flask import Blueprint, jsonify
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
api_patterns = Blueprint('api_patterns', __name__, url_prefix='/api')


@api_patterns.route('/pattern-analysis/<symbol>')
def pattern_analysis(symbol):
    """Hisse için pattern analizi"""
    try:
        # Import here to avoid circular dependencies
        from app import get_pattern_detector, socketio
        
        # Global singleton instance kullan - duplike instance oluşturma
        result = get_pattern_detector().analyze_stock(symbol.upper())
        
        # Simulation integration - aktif simulation varsa signal'i işle
        try:
            from simulation_engine import get_simulation_engine
            from models import SimulationSession
            
            # Aktif simulation session'ları bul
            active_sessions = SimulationSession.query.filter_by(status='active').all()
            
            if active_sessions and result.get('status') == 'success':
                simulation_engine = get_simulation_engine()
                
                for session in active_sessions:
                    # Her aktif session için signal'i işle
                    trade = simulation_engine.process_signal(
                        session_id=session.id,
                        symbol=symbol.upper(),
                        signal_data=result
                    )
                    
                    if trade:
                        logger.info(f"🤖 Simulation trade executed: {trade.trade_type} {trade.quantity}x{symbol} @ {trade.price}")
                        
                        # WebSocket ile simulation update broadcast
                        socketio.emit('simulation_trade', {
                            'session_id': session.id,
                            'trade': trade.to_dict(),
                            'timestamp': datetime.now().isoformat()
                        }, to='admin')
                    
        except Exception as sim_error:
            logger.warning(f"Simulation processing failed: {sim_error}")
            # Simulation hatası ana analizi etkilemesin
        
        # Optional: persist predictions via API when enabled
        try:
            truthy = ('1', 'true', 'yes', 'on')
            if str(os.getenv('API_WRITE_PREDICTIONS', '0')).lower() in truthy and isinstance(result, dict):
                from datetime import datetime as _dt
                from models import db, PredictionsLog, Stock  # type: ignore
                sym_up = symbol.upper()
                stock_id = None
                try:
                    st = Stock.query.filter_by(symbol=sym_up).first()
                    stock_id = getattr(st, 'id', None)
                except Exception:
                    stock_id = None
                enh_obj_any = result.get('enhanced_predictions')
                basic_obj_any = result.get('ml_predictions')
                enh = enh_obj_any if isinstance(enh_obj_any, dict) else {}
                basic = basic_obj_any if isinstance(basic_obj_any, dict) else {}
                try:
                    cp_any = result.get('current_price')
                    cur_price = float(cp_any) if isinstance(cp_any, (int, float)) else None
                except Exception:
                    cur_price = None
                keys = set()
                keys.update([k for k in list(enh.keys()) if isinstance(k, str)])
                keys.update([k for k in list(basic.keys()) if isinstance(k, str)])
                to_add = []
                for hk in keys:
                    try:
                        enh_obj = enh.get(hk) if isinstance(enh, dict) else None
                        bas_obj = basic.get(hk) if isinstance(basic, dict) else None
                        pred_px = None
                        conf = None
                        if isinstance(enh_obj, dict):
                            v = enh_obj.get('ensemble_prediction') if 'ensemble_prediction' in enh_obj else enh_obj.get('price')
                            if isinstance(v, (int, float)):
                                pred_px = float(v)
                            if isinstance(enh_obj.get('confidence'), (int, float)):
                                conf = float(enh_obj['confidence'])
                        if pred_px is None and isinstance(bas_obj, dict):
                            v = bas_obj.get('price')
                            if isinstance(v, (int, float)):
                                pred_px = float(v)
                        if cur_price is None:
                            continue
                        delta = (pred_px - cur_price) / cur_price if (pred_px is not None and cur_price) else None
                        if pred_px is None and delta is None:
                            continue
                        log = PredictionsLog(
                            stock_id=stock_id,
                            symbol=sym_up,
                            horizon=str(hk),
                            ts_pred=_dt.utcnow(),
                            price_now=cur_price,
                            pred_price=pred_px,
                            delta_pred=delta,
                            model='enhanced' if (isinstance(enh_obj, dict) and pred_px is not None) else 'basic',
                            unified_best='enhanced' if (isinstance(enh_obj, dict) and pred_px is not None) else 'basic',
                            confidence=conf,
                        )
                        to_add.append(log)
                    except Exception:
                        continue
                if to_add:
                    try:
                        for it in to_add:
                            db.session.add(it)
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
        except Exception as _persist_err:
            logger.warning(f"API prediction persist skipped: {_persist_err}")

        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Pattern analysis error for {symbol}: {e}")
        return jsonify({
            'symbol': symbol,
            'status': 'error',
            'error': str(e)
        }), 500


@api_patterns.route('/pattern-summary')
def pattern_summary():
    """Genel pattern özeti"""
    try:
        # Import here to avoid circular dependencies
        from app import get_pattern_detector
        
        # Öncelikli hisseler
        priority_stocks = ['THYAO', 'AKBNK', 'GARAN', 'EREGL', 'ASELS', 'VAKBN', 'MGROS', 'FROTO']
        
        # Global singleton instance kullan - duplike instance oluşturma
        summary = get_pattern_detector().get_pattern_summary(priority_stocks)
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Pattern summary error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@api_patterns.route('/visual-analysis/<symbol>')
def visual_pattern_analysis(symbol):
    """Visual pattern analysis endpoint"""
    try:
        from visual_pattern_detector import get_visual_pattern_system
        from app import get_pattern_detector
        
        # Visual pattern system'i al
        visual_system = get_visual_pattern_system()
        
        # Sistem bilgilerini kontrol et
        system_info = visual_system.get_system_info()
        if not system_info.get('yolo_available'):
            return jsonify({
                'status': 'unavailable',
                'message': 'YOLO visual pattern sistemi kullanılamıyor'
            })
        
        # Hisse verisini al
        stock_data = get_pattern_detector().get_stock_data(symbol)
        if stock_data is None or len(stock_data) < 20:
            return jsonify({
                'status': 'error',
                'message': f'{symbol} için yeterli veri bulunamadı (minimum 20 gün gerekli)'
            })
        
        # Visual pattern analizi yap
        analysis_result = visual_system.analyze_stock(symbol, stock_data)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Visual pattern analysis error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500
