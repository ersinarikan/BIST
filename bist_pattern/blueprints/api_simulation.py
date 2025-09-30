from flask import Blueprint, jsonify, request
from datetime import datetime
from decimal import Decimal

from ..extensions import csrf

bp = Blueprint('api_simulation', __name__, url_prefix='/api/simulation')


def register(app):
    try:
        from models import db, SimulationSession
    except Exception:
        db = SimulationSession = None

    @bp.route('/start', methods=['POST'])
    @csrf.exempt
    def start_simulation():
        try:
            from simulation_engine import get_simulation_engine
            data = request.get_json() or {}
            user_id = data.get('user_id', 1)
            initial_balance = float(data.get('initial_balance', 100.0))
            duration_hours = int(data.get('duration_hours', 48))
            session_name = data.get('session_name', 'AI Performance Test')
            trade_amount = float(data.get('trade_amount', 10000.0))
            simulation_engine = get_simulation_engine()
            simulation_engine.fixed_trade_amount = Decimal(str(trade_amount))
            session = simulation_engine.create_session(
                user_id=user_id,
                initial_balance=initial_balance,
                duration_hours=duration_hours,
                session_name=session_name,
            )
            app.logger.info(f"✅ New simulation started: {session.id}")
            return jsonify({
                'status': 'success',
                'message': 'Simulation başlatıldı',
                'session': session.to_dict(),
            })
        except Exception as e:
            app.logger.error(f"❌ Simulation start error: {e}")
            return jsonify({'status': 'error', 'message': f'Simulation başlatma hatası: {str(e)}'}), 500

    @bp.route('/<int:session_id>/status')
    def simulation_status(session_id):
        try:
            from simulation_engine import get_simulation_engine
            if SimulationSession is None:
                return jsonify({'status': 'error', 'message': 'DB unavailable'}), 500
            session = SimulationSession.query.get(session_id)
            if not session:
                return jsonify({'status': 'error', 'message': 'Simulation session bulunamadı'}), 404
            simulation_engine = get_simulation_engine()
            performance = simulation_engine.get_session_performance(session_id)
            return jsonify({'status': 'success', 'performance': performance})
        except Exception as e:
            app.logger.error(f"❌ Simulation status error: {e}")
            return jsonify({'status': 'error', 'message': f'Simulation status hatası: {str(e)}'}), 500

    @bp.route('/<int:session_id>/stop', methods=['POST'])
    @csrf.exempt
    def stop_simulation(session_id):
        try:
            if SimulationSession is None or db is None:
                return jsonify({'status': 'error', 'message': 'DB unavailable'}), 500
            session = SimulationSession.query.get(session_id)
            if not session:
                return jsonify({'status': 'error', 'message': 'Simulation session bulunamadı'}), 404
            session.status = 'completed'
            session.end_time = datetime.now()
            db.session.commit()
            app.logger.info(f"✅ Simulation stopped: {session_id}")
            return jsonify({
                'status': 'success',
                'message': 'Simulation durduruldu',
                'session': session.to_dict(),
            })
        except Exception as e:
            app.logger.error(f"❌ Simulation stop error: {e}")
            return jsonify({'status': 'error', 'message': f'Simulation durdurma hatası: {str(e)}'}), 500

    @bp.route('/list')
    def list_simulations():
        try:
            if SimulationSession is None:
                return jsonify({'status': 'success', 'sessions': []})
            from sqlalchemy import desc
            sessions = (
                SimulationSession.query.order_by(desc(SimulationSession.created_at))
                .limit(20)
                .all()
            )
            return jsonify({
                'status': 'success',
                'sessions': [s.to_dict() for s in sessions],
            })
        except Exception as e:
            app.logger.error(f"❌ Simulation list error: {e}")
            return jsonify({'status': 'error', 'message': f'Simulation listesi hatası: {str(e)}'}), 500

    @bp.route('/process-signal', methods=['POST'])
    def process_simulation_signal():
        try:
            from simulation_engine import get_simulation_engine
            data = request.get_json()
            if not data:
                return jsonify({'status': 'error', 'message': 'JSON data gerekli'}), 400
            session_id = data.get('session_id')
            symbol = data.get('symbol')
            signal_data = data.get('signal_data')
            if not all([session_id, symbol, signal_data]):
                return jsonify({'status': 'error', 'message': 'session_id, symbol ve signal_data gerekli'}), 400
            simulation_engine = get_simulation_engine()
            trade = simulation_engine.process_signal(session_id, symbol, signal_data)
            if trade:
                return jsonify({
                    'status': 'success',
                    'message': 'Signal işlendi, trade execute edildi',
                    'trade': trade.to_dict(),
                })
            return jsonify({
                'status': 'success',
                'message': 'Signal işlendi, trade execute edilmedi',
                'trade': None,
            })
        except Exception as e:
            app.logger.error(f"❌ Signal processing error: {e}")
            return jsonify({'status': 'error', 'message': f'Signal processing hatası: {str(e)}'}), 500

    app.register_blueprint(bp)
