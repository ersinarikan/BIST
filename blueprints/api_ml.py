"""
ML Prediction API Blueprint
ML training, prediction, and model management endpoints
"""

from flask import Blueprint, jsonify
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
api_ml = Blueprint('api_ml', __name__, url_prefix='/api')


@api_ml.route('/ml-prediction/<symbol>')
def ml_prediction_analysis(symbol):
    """ML tabanlı fiyat tahmini"""
    try:
        # Import here to avoid circular dependencies
        from app import get_pattern_detector, ML_PREDICTION_AVAILABLE
        
        if not ML_PREDICTION_AVAILABLE:
            return jsonify({
                'status': 'unavailable',
                'message': 'ML Prediction sistemi mevcut değil'
            })
        
        # Hisse verisini al
        stock_data = get_pattern_detector().get_stock_data(symbol, days=365)  # 1 yıllık veri
        if stock_data is None or len(stock_data) < 100:
            return jsonify({
                'status': 'error',
                'message': f'{symbol} için yeterli veri bulunamadı (minimum 100 gün gerekli)'
            })
        
        # ML prediction system'i al
        ml_system = get_pattern_detector().ml_predictor
        if not ml_system:
            return jsonify({
                'status': 'error',
                'message': 'ML prediction system mevcut değil'
            }), 503
        
        # Sentiment analizi ekle
        sentiment_score = None
        try:
            from fingpt_analyzer import get_fingpt_analyzer
            fingpt = get_fingpt_analyzer()
            if fingpt.model_loaded:
                # Basit sentiment analizi için dummy text
                test_text = f"{symbol} stock analysis"
                sentiment_result = fingpt.analyze_sentiment(test_text)
                if sentiment_result['status'] == 'success':
                    # positive: 1, negative: 0, neutral: 0.5
                    if sentiment_result['sentiment'] == 'positive':
                        sentiment_score = sentiment_result['scores']['positive']
                    elif sentiment_result['sentiment'] == 'negative':
                        sentiment_score = 1 - sentiment_result['scores']['negative']
                    else:
                        sentiment_score = 0.5
        except Exception as e:
            logger.warning(f"Sentiment analizi eklenemedi: {e}")
        
        # Tahmin yap
        predictions = ml_system.predict_prices(symbol, stock_data, sentiment_score)
        
        if predictions:
            result = {
                'symbol': symbol,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'current_price': float(stock_data['close'].iloc[-1]),
                'predictions': predictions,
                'sentiment_score': sentiment_score,
                'data_points': len(stock_data),
                'model_type': 'basic_ml'
            }
        else:
            result = {
                'symbol': symbol,
                'status': 'error',
                'message': 'ML tahmin yapılamadı',
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"ML prediction error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@api_ml.route('/train-ml-model/<symbol>')
def train_ml_model(symbol):
    """Belirli bir hisse için ML modelini eğit"""
    try:
        from app import get_pattern_detector, ML_PREDICTION_AVAILABLE
        
        if not ML_PREDICTION_AVAILABLE:
            return jsonify({
                'status': 'unavailable',
                'message': 'ML Training sistemi mevcut değil'
            })
        
        # Hisse verisini al (2 yıllık veri)
        stock_data = get_pattern_detector().get_stock_data(symbol, days=730)
        if stock_data is None or len(stock_data) < 200:
            return jsonify({
                'status': 'error',
                'message': f'{symbol} için yeterli veri bulunamadı (minimum 200 gün gerekli)'
            })
        
        # ML prediction system'i al
        ml_system = get_pattern_detector().ml_predictor
        if not ml_system:
            return jsonify({
                'status': 'error',
                'message': 'ML prediction system mevcut değil'
            }), 503
        
        # Model eğit
        training_result = ml_system.train_models(symbol, stock_data)
        
        return jsonify({
            'symbol': symbol,
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'message': 'ML model eğitimi tamamlandı',
            'training_result': training_result,
            'data_points': len(stock_data)
        })
        
    except Exception as e:
        logger.error(f"ML training error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@api_ml.route('/enhanced-ml/train/<symbol>')
def train_enhanced_ml(symbol):
    """Enhanced ML model eğitimi"""
    try:
        from enhanced_ml_system import get_enhanced_ml_system
        from app import get_pattern_detector
        
        # Hisse verisini al
        stock_data = get_pattern_detector().get_stock_data(symbol, days=730)
        if stock_data is None or len(stock_data) < 200:
            return jsonify({
                'status': 'error',
                'message': f'{symbol} için yeterli veri bulunamadı'
            })
        
        enhanced_ml = get_enhanced_ml_system()
        
        # Model eğit
        training_result = enhanced_ml.train_enhanced_models(symbol, stock_data)
        
        if training_result:
            return jsonify({
                'symbol': symbol,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'message': 'Enhanced ML model eğitimi tamamlandı',
                'models_trained': len(training_result),
                'data_points': len(stock_data)
            })
        else:
            return jsonify({
                'symbol': symbol,
                'status': 'error',
                'message': 'Enhanced ML eğitimi başarısız',
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Enhanced ML training error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


@api_ml.route('/enhanced-ml/predict/<symbol>')
def enhanced_ml_prediction(symbol):
    """Enhanced ML tahmin"""
    try:
        from enhanced_ml_system import get_enhanced_ml_system
        from app import get_pattern_detector
        
        # Hisse verisini al
        stock_data = get_pattern_detector().get_stock_data(symbol, days=365)
        if stock_data is None or len(stock_data) < 100:
            return jsonify({
                'status': 'error',
                'message': f'{symbol} için yeterli veri bulunamadı'
            })
        
        enhanced_ml = get_enhanced_ml_system()
        
        # Tahmin yap
        predictions = enhanced_ml.predict_enhanced(symbol, stock_data)
        
        if predictions:
            result = {
                'symbol': symbol,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'current_price': float(stock_data['close'].iloc[-1]),
                'predictions': predictions,
                'data_points': len(stock_data),
                'enhanced_models': True
            }
        else:
            result = {
                'symbol': symbol,
                'status': 'error',
                'message': 'Enhanced ML tahmin yapılamadı - model eğitimi gerekli',
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Enhanced ML prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@api_ml.route('/enhanced-ml/features/<symbol>')
def enhanced_ml_features(symbol):
    """Feature importance analizi"""
    try:
        from enhanced_ml_system import get_enhanced_ml_system
        
        enhanced_ml = get_enhanced_ml_system()
        
        # Feature importance
        features = enhanced_ml.get_top_features(symbol, model_type='xgboost', top_n=20)
        
        if features:
            return jsonify({
                'symbol': symbol,
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'features': features
            })
        else:
            return jsonify({
                'symbol': symbol,
                'status': 'error',
                'message': 'Feature importance bilgisi bulunamadı - model eğitimi gerekli',
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return jsonify({'error': str(e)}), 500


@api_ml.route('/enhanced-ml/info')
def enhanced_ml_info():
    """Enhanced ML sistem bilgileri"""
    try:
        from enhanced_ml_system import get_enhanced_ml_system
        
        enhanced_ml = get_enhanced_ml_system()
        info = enhanced_ml.get_system_info()
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'system_info': info
        })
        
    except Exception as e:
        logger.error(f"Enhanced ML info error: {e}")
        return jsonify({'error': str(e)}), 500
