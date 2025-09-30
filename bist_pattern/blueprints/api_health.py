from flask import Blueprint, jsonify
from datetime import datetime

bp = Blueprint('api_health', __name__)


def register(app):
    from models import db, Stock, StockPrice

    @bp.route('/health')
    def health():
        try:
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            total_stocks = Stock.query.count()
            total_prices = StockPrice.query.count()
            return jsonify({
                "status": "healthy",
                "database": "connected",
                "stocks": total_stocks,
                "price_records": total_prices,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
            }), 500

    app.register_blueprint(bp)
