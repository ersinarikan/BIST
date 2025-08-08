"""
BIST ML Prediction System
LSTM, Random Forest ve SVR ile fiyat tahmini
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLPredictionSystem:
    """Machine Learning tabanlı fiyat tahmin sistemi"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_directory = "ml_models"
        self.lookback_period = 60  # 60 günlük veri
        self.prediction_horizons = [1, 3, 7, 30]  # 1D, 3D, 7D, 30D
        
        # Model klasörünü oluştur
        os.makedirs(self.model_directory, exist_ok=True)
        
        logger.info("ML Prediction System başlatıldı")
    
    def create_technical_features(self, data):
        """Gelişmiş teknik indikatörler oluştur"""
        try:
            df = data.copy()
            
            # Sütun adlarını düzelt (yfinance büyük harf kullanır)
            if 'Close' in df.columns:
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high', 
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
            
            # Gerekli sütunları kontrol et
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Eksik sütunlar: {missing_cols}")
                return data
            
            # Temel fiyat features
            df['price_change'] = df['close'].pct_change()
            df['price_volatility'] = df['close'].rolling(20).std()
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            
            # Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            
            # RSI
            for period in [14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            for period in [20, 30]:
                sma = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                df[f'bb_upper_{period}'] = sma + (std * 2)
                df[f'bb_lower_{period}'] = sma - (std * 2)
                df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / df[f'bb_width_{period}']
            
            # Stochastic Oscillator
            for period in [14, 21]:
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
                df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
            
            # Williams %R
            for period in [14, 21]:
                high_max = df['high'].rolling(window=period).max()
                low_min = df['low'].rolling(window=period).min()
                df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
            
            # Volume indicators
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                df['price_volume'] = df['close'] * df['volume']
                df['vwap'] = df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # Momentum indicators
            for period in [10, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
                df[f'roc_{period}'] = df['close'].pct_change(periods=period)
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
            
            # Time-based features
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['month'] = pd.to_datetime(df.index).month
            df['quarter'] = pd.to_datetime(df.index).quarter
            
            # Trend features
            df['trend_5'] = df['close'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
            df['trend_10'] = df['close'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering hatası: {e}")
            return data
    
    def prepare_lstm_data(self, data, target_col='close', lookback=60):
        """LSTM için veri hazırlama"""
        try:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[[target_col]])
            
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i, 0])
                y.append(scaled_data[i, 0])
            
            return np.array(X), np.array(y), scaler
            
        except Exception as e:
            logger.error(f"LSTM veri hazırlama hatası: {e}")
            return None, None, None
    
    def build_lstm_model(self, input_shape):
        """LSTM model oluştur"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow mevcut değil, LSTM modeli atlanıyor")
                return None
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            return model
            
        except Exception as e:
            logger.error(f"LSTM model oluşturma hatası: {e}")
            return None
    
    def train_models(self, symbol, data):
        """Tüm modelleri eğit"""
        try:
            logger.info(f"{symbol} için model eğitimi başlatılıyor")
            
            # Feature engineering
            df_features = self.create_technical_features(data)
            df_features = df_features.dropna()
            
            if len(df_features) < 100:
                logger.warning(f"{symbol} için yeterli veri yok")
                return False
            
            # Feature columns (numeric only)
            feature_cols = [col for col in df_features.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume'] 
                          and df_features[col].dtype in ['float64', 'int64']]
            
            self.feature_columns = feature_cols
            results = {}
            
            # Her tahmin ufku için model eğit
            for horizon in self.prediction_horizons:
                logger.info(f"{symbol} - {horizon} gün tahmini için model eğitimi")
                
                # Target variable (gelecekteki fiyat)
                target = f'target_{horizon}d'
                df_features[target] = df_features['close'].shift(-horizon)
                
                # Son horizon kadar veriyi çıkar (target NaN olur)
                df_model = df_features[:-horizon].copy()
                
                X = df_model[feature_cols].values
                y = df_model[target].values
                
                # Veri setini böl
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, shuffle=False
                )
                
                # Scaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                models_horizon = {}
                
                # 1. Random Forest
                try:
                    rf_model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_model.fit(X_train_scaled, y_train)
                    rf_pred = rf_model.predict(X_test_scaled)
                    rf_score = r2_score(y_test, rf_pred)
                    
                    models_horizon['random_forest'] = {
                        'model': rf_model,
                        'score': rf_score,
                        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                        'mae': mean_absolute_error(y_test, rf_pred)
                    }
                    logger.info(f"Random Forest {horizon}D - R²: {rf_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Random Forest eğitim hatası: {e}")
                
                # 2. Support Vector Regression
                try:
                    svr_model = SVR(kernel='rbf', C=100, gamma='scale')
                    svr_model.fit(X_train_scaled, y_train)
                    svr_pred = svr_model.predict(X_test_scaled)
                    svr_score = r2_score(y_test, svr_pred)
                    
                    models_horizon['svr'] = {
                        'model': svr_model,
                        'score': svr_score,
                        'rmse': np.sqrt(mean_squared_error(y_test, svr_pred)),
                        'mae': mean_absolute_error(y_test, svr_pred)
                    }
                    logger.info(f"SVR {horizon}D - R²: {svr_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"SVR eğitim hatası: {e}")
                
                # 3. LSTM (sadece uzun vadeli tahminler için)
                if horizon >= 7 and TENSORFLOW_AVAILABLE:
                    try:
                        lstm_X, lstm_y, lstm_scaler = self.prepare_lstm_data(
                            df_model, 'close', self.lookback_period
                        )
                        
                        if lstm_X is not None and len(lstm_X) > 50:
                            # Train-test split for LSTM
                            split_idx = int(len(lstm_X) * 0.8)
                            X_train_lstm = lstm_X[:split_idx]
                            X_test_lstm = lstm_X[split_idx:]
                            y_train_lstm = lstm_y[:split_idx]
                            y_test_lstm = lstm_y[split_idx:]
                            
                            # Reshape for LSTM
                            X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
                            X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
                            
                            lstm_model = self.build_lstm_model((X_train_lstm.shape[1], 1))
                            
                            if lstm_model:
                                lstm_model.fit(
                                    X_train_lstm, y_train_lstm,
                                    batch_size=32, epochs=50,
                                    validation_split=0.1,
                                    verbose=0
                                )
                                
                                lstm_pred = lstm_model.predict(X_test_lstm)
                                # Scale back
                                lstm_pred = lstm_scaler.inverse_transform(lstm_pred)
                                y_test_lstm_scaled = lstm_scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
                                
                                lstm_score = r2_score(y_test_lstm_scaled, lstm_pred)
                                
                                models_horizon['lstm'] = {
                                    'model': lstm_model,
                                    'scaler': lstm_scaler,
                                    'score': lstm_score,
                                    'rmse': np.sqrt(mean_squared_error(y_test_lstm_scaled, lstm_pred)),
                                    'mae': mean_absolute_error(y_test_lstm_scaled, lstm_pred)
                                }
                                logger.info(f"LSTM {horizon}D - R²: {lstm_score:.3f}")
                        
                    except Exception as e:
                        logger.error(f"LSTM eğitim hatası: {e}")
                
                # Scaler'ı kaydet
                self.scalers[f"{symbol}_{horizon}d"] = scaler
                self.models[f"{symbol}_{horizon}d"] = models_horizon
                
                results[f"{horizon}d"] = models_horizon
            
            # Modelleri kaydet
            self.save_models(symbol)
            
            logger.info(f"{symbol} model eğitimi tamamlandı")
            return results
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
            return False
    
    def predict_prices(self, symbol, current_data, sentiment_score=None):
        """Fiyat tahminleri yap"""
        try:
            # Model yüklü mü kontrol et
            if not any(key.startswith(symbol) for key in self.models.keys()):
                logger.warning(f"{symbol} için eğitilmiş model bulunamadı")
                return None
            
            # Feature engineering
            df_features = self.create_technical_features(current_data)
            df_features = df_features.dropna()
            
            if len(df_features) == 0:
                return None
            
            # Son veri noktasını al
            latest_features = df_features[self.feature_columns].iloc[-1:].values
            
            predictions = {}
            
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.models and model_key in self.scalers:
                    # Feature'ları scale et
                    scaler = self.scalers[model_key]
                    features_scaled = scaler.transform(latest_features)
                    
                    horizon_models = self.models[model_key]
                    model_predictions = {}
                    
                    # Her model ile tahmin yap
                    for model_name, model_info in horizon_models.items():
                        try:
                            if model_name == 'lstm':
                                # LSTM için özel işlem gerekli
                                continue
                            else:
                                pred = model_info['model'].predict(features_scaled)[0]
                                confidence = model_info['score']
                                
                                model_predictions[model_name] = {
                                    'prediction': float(pred),
                                    'confidence': float(confidence),
                                    'rmse': float(model_info['rmse']),
                                    'mae': float(model_info['mae'])
                                }
                        except Exception as e:
                            logger.error(f"{model_name} tahmin hatası: {e}")
                    
                    # Ensemble prediction (weighted average)
                    if model_predictions:
                        weights = [info['confidence'] for info in model_predictions.values()]
                        predictions_list = [info['prediction'] for info in model_predictions.values()]
                        
                        if sum(weights) > 0:
                            ensemble_pred = np.average(predictions_list, weights=weights)
                            avg_confidence = np.mean(weights)
                            
                            predictions[f"{horizon}d"] = {
                                'ensemble_prediction': float(ensemble_pred),
                                'confidence': float(avg_confidence),
                                'models': model_predictions,
                                'current_price': float(current_data['close'].iloc[-1])
                            }
            
            # Sentiment entegrasyonu
            if sentiment_score:
                for horizon in predictions:
                    pred_info = predictions[horizon]
                    sentiment_adjustment = 1 + (sentiment_score - 0.5) * 0.1  # ±5% adjustment
                    pred_info['sentiment_adjusted'] = pred_info['ensemble_prediction'] * sentiment_adjustment
                    pred_info['sentiment_score'] = sentiment_score
            
            return predictions
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return None
    
    def save_models(self, symbol):
        """Modelleri dosyaya kaydet"""
        try:
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.models:
                    # Scaler kaydet
                    if model_key in self.scalers:
                        joblib.dump(
                            self.scalers[model_key],
                            f"{self.model_directory}/{model_key}_scaler.pkl"
                        )
                    
                    # Non-LSTM modelleri kaydet
                    models = self.models[model_key]
                    for model_name, model_info in models.items():
                        if model_name != 'lstm':
                            joblib.dump(
                                model_info['model'],
                                f"{self.model_directory}/{model_key}_{model_name}.pkl"
                            )
                        elif TENSORFLOW_AVAILABLE and 'model' in model_info:
                            # TensorFlow model kaydet
                            model_info['model'].save(f"{self.model_directory}/{model_key}_lstm.h5")
            
            logger.info(f"{symbol} modelleri kaydedildi")
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {e}")
    
    def load_models(self, symbol):
        """Modelleri dosyadan yükle"""
        try:
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                # Scaler yükle
                scaler_path = f"{self.model_directory}/{model_key}_scaler.pkl"
                if os.path.exists(scaler_path):
                    self.scalers[model_key] = joblib.load(scaler_path)
                
                # Modelleri yükle
                models = {}
                
                # Random Forest
                rf_path = f"{self.model_directory}/{model_key}_random_forest.pkl"
                if os.path.exists(rf_path):
                    models['random_forest'] = {'model': joblib.load(rf_path)}
                
                # SVR
                svr_path = f"{self.model_directory}/{model_key}_svr.pkl"
                if os.path.exists(svr_path):
                    models['svr'] = {'model': joblib.load(svr_path)}
                
                # LSTM
                lstm_path = f"{self.model_directory}/{model_key}_lstm.h5"
                if os.path.exists(lstm_path) and TENSORFLOW_AVAILABLE:
                    models['lstm'] = {'model': tf.keras.models.load_model(lstm_path)}
                
                if models:
                    self.models[model_key] = models
            
            logger.info(f"{symbol} modelleri yüklendi")
            return True
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            return False
    
    def get_system_info(self):
        """Sistem bilgilerini döndür"""
        return {
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'models_loaded': len(self.models),
            'scalers_loaded': len(self.scalers),
            'prediction_horizons': self.prediction_horizons,
            'feature_count': len(self.feature_columns),
            'model_directory': self.model_directory
        }

# Global singleton instance
_ml_prediction_system = None

def get_ml_prediction_system():
    """ML Prediction System singleton'ını döndür"""
    global _ml_prediction_system
    if _ml_prediction_system is None:
        _ml_prediction_system = MLPredictionSystem()
    return _ml_prediction_system

if __name__ == "__main__":
    # Test
    import yfinance as yf
    
    ml_system = get_ml_prediction_system()
    
    # Test verisi
    ticker = yf.Ticker("THYAO.IS")
    data = ticker.history(period="1y")
    
    if not data.empty:
        print("ML Prediction System Test:")
        
        # Model eğitimi
        results = ml_system.train_models("THYAO", data)
        if results:
            print("✅ Model eğitimi başarılı")
            
            # Tahmin
            predictions = ml_system.predict_prices("THYAO", data.tail(100))
            if predictions:
                print("✅ Tahminler:")
                for horizon, pred_info in predictions.items():
                    print(f"  {horizon}: {pred_info['ensemble_prediction']:.2f} TL (güven: {pred_info['confidence']:.3f})")
        else:
            print("❌ Model eğitimi başarısız")
    else:
        print("❌ Test verisi alınamadı")
