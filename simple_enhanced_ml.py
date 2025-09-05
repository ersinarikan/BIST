"""
Simple Enhanced ML System
Çalışan basit XGBoost tabanlı tahmin sistemi
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ML Models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleEnhancedML:
    """Basit ama çalışan Enhanced ML sistemi"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_directory = os.getenv('ML_MODEL_PATH', "simple_ml_models")
        self.prediction_horizons = [1, 3, 7, 30]
        
        os.makedirs(self.model_directory, exist_ok=True)
        logger.info("🚀 Simple Enhanced ML System başlatıldı")
    
    def create_simple_features(self, data):
        """Basit ama etkili feature'lar"""
        try:
            df = data.copy()
            
            # Sütun adlarını standardize et
            if 'Close' in df.columns:
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 
                    'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
            
            # Temel price features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            
            # Moving averages
            for period in [5, 10, 20]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            
            # Volatility
            df['volatility_10'] = df['close'].pct_change().rolling(10).std()
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            
            # Volume features (if available)
            if 'volume' in df.columns and df['volume'].sum() > 0:
                df['volume_sma'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Lag features
            for lag in [1, 2, 3]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
            
            # Time features
            if hasattr(df.index, 'dayofweek'):
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return data
    
    def clean_data(self, df):
        """Güvenli veri temizleme"""
        try:
            # INF ve NaN temizleme
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Outlier removal (simple)
            for col in numeric_cols:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue
                
                if df[col].std() > 0:
                    q1 = df[col].quantile(0.01)
                    q99 = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower=q1, upper=q99)
            
            # Fill NaN
            df = df.ffill().fillna(0)
            
            # Final INF check
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                df = df.replace([np.inf, -np.inf], 0)
            
            return df
            
        except Exception as e:
            logger.error(f"Data cleaning error: {e}")
            return df
    
    def train_simple_models(self, symbol, data):
        """Basit model eğitimi"""
        try:
            logger.info(f"🎯 {symbol} için simple enhanced ML eğitimi")
            
            # Features
            df_features = self.create_simple_features(data)
            df_features = self.clean_data(df_features)
            df_features = df_features.dropna()
            
            if len(df_features) < 100:
                logger.warning(f"Insufficient data: {len(df_features)}")
                return False
            
            # Feature selection
            feature_cols = []
            for col in df_features.columns:
                if col not in ['open', 'high', 'low', 'close', 'volume']:
                    if df_features[col].dtype in ['float64', 'int64']:
                        if not df_features[col].isnull().all():
                            if df_features[col].std() > 0:
                                feature_cols.append(col)
            
            self.feature_columns = feature_cols
            logger.info(f"📊 Using {len(feature_cols)} features")
            
            results = {}
            
            # Train for each horizon
            for horizon in self.prediction_horizons:
                logger.info(f"Training {horizon}D model...")
                
                # Create target
                target = f'target_{horizon}d'
                df_features[target] = df_features['close'].shift(-horizon)
                
                # Prepare data
                model_data = df_features[:-horizon].copy()
                X = model_data[feature_cols].values
                y = model_data[target].values
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, shuffle=False
                )
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                models = {}
                
                # XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        xgb_model = xgb.XGBRegressor(
                            n_estimators=50,
                            max_depth=4,
                            learning_rate=0.1,
                            random_state=42,
                            verbosity=0
                        )
                        
                        xgb_model.fit(X_train_scaled, y_train)
                        xgb_pred = xgb_model.predict(X_test_scaled)
                        xgb_score = r2_score(y_test, xgb_pred)
                        
                        models['xgboost'] = {
                            'model': xgb_model,
                            'score': xgb_score,
                            'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred))
                        }
                        
                        logger.info(f"XGBoost {horizon}D - R²: {xgb_score:.3f}")
                        
                    except Exception as e:
                        logger.error(f"XGBoost training error: {e}")
                
                # LightGBM
                if LIGHTGBM_AVAILABLE:
                    try:
                        lgb_model = lgb.LGBMRegressor(
                            n_estimators=50,
                            max_depth=4,
                            learning_rate=0.1,
                            random_state=42,
                            verbose=-1
                        )
                        
                        lgb_model.fit(X_train_scaled, y_train)
                        lgb_pred = lgb_model.predict(X_test_scaled)
                        lgb_score = r2_score(y_test, lgb_pred)
                        
                        models['lightgbm'] = {
                            'model': lgb_model,
                            'score': lgb_score,
                            'rmse': np.sqrt(mean_squared_error(y_test, lgb_pred))
                        }
                        
                        logger.info(f"LightGBM {horizon}D - R²: {lgb_score:.3f}")
                        
                    except Exception as e:
                        logger.error(f"LightGBM training error: {e}")
                
                # Store
                if models:
                    self.models[f"{symbol}_{horizon}d"] = models
                    self.scalers[f"{symbol}_{horizon}d"] = scaler
                    results[f"{horizon}d"] = models
            
            # Save
            self.save_models(symbol)
            
            logger.info(f"✅ {symbol} training completed")
            return results
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_simple(self, symbol, current_data):
        """Basit tahmin"""
        try:
            # Features
            df_features = self.create_simple_features(current_data)
            df_features = self.clean_data(df_features)
            
            if len(df_features) == 0:
                return None
            
            # Model yüklü değilse yükle
            if not self.feature_columns or not self.models:
                logger.info(f"Loading models for {symbol}...")
                if not self.load_models(symbol):
                    logger.error("No trained models found")
                    return None
            
            # Get latest features
            latest_features = df_features[self.feature_columns].iloc[-1:].values
            
            predictions = {}
            
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.models and model_key in self.scalers:
                    scaler = self.scalers[model_key]
                    features_scaled = scaler.transform(latest_features)
                    
                    models = self.models[model_key]
                    model_predictions = {}
                    
                    for model_name, model_info in models.items():
                        try:
                            pred = model_info['model'].predict(features_scaled)[0]
                            model_predictions[model_name] = {
                                'prediction': float(pred),
                                'confidence': float(model_info['score'])
                            }
                        except Exception as e:
                            logger.error(f"{model_name} prediction error: {e}")
                    
                    # Ensemble
                    if model_predictions:
                        predictions_list = [p['prediction'] for p in model_predictions.values()]
                        confidences = [p['confidence'] for p in model_predictions.values()]
                        
                        ensemble_pred = np.mean(predictions_list)
                        avg_confidence = np.mean(confidences)
                        
                        predictions[f"{horizon}d"] = {
                            'ensemble_prediction': float(ensemble_pred),
                            'confidence': float(avg_confidence),
                            'models': model_predictions,
                            'current_price': float(current_data['close'].iloc[-1])
                        }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def save_models(self, symbol):
        """Model kaydetme"""
        try:
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.scalers:
                    joblib.dump(
                        self.scalers[model_key],
                        f"{self.model_directory}/{model_key}_scaler.pkl"
                    )
                
                if model_key in self.models:
                    models = self.models[model_key]
                    for model_name, model_info in models.items():
                        joblib.dump(
                            model_info['model'],
                            f"{self.model_directory}/{model_key}_{model_name}.pkl"
                        )
            
            # Feature columns kaydet
            if self.feature_columns:
                joblib.dump(
                    self.feature_columns,
                    f"{self.model_directory}/{symbol}_features.pkl"
                )
            
            logger.info(f"Models saved for {symbol}")
            
        except Exception as e:
            logger.error(f"Save models error: {e}")
    
    def load_models(self, symbol):
        """Model yükleme"""
        try:
            loaded_count = 0
            
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                # Scaler yükle
                scaler_file = f"{self.model_directory}/{model_key}_scaler.pkl"
                if os.path.exists(scaler_file):
                    self.scalers[model_key] = joblib.load(scaler_file)
                
                # Modelleri yükle
                models = {}
                
                # XGBoost
                xgb_file = f"{self.model_directory}/{model_key}_xgboost.pkl"
                if os.path.exists(xgb_file):
                    try:
                        model = joblib.load(xgb_file)
                        models['xgboost'] = {'model': model, 'score': 0.5}  # Default score
                    except Exception as e:
                        logger.error(f"XGBoost load error: {e}")
                
                # LightGBM
                lgb_file = f"{self.model_directory}/{model_key}_lightgbm.pkl"
                if os.path.exists(lgb_file):
                    try:
                        model = joblib.load(lgb_file)
                        models['lightgbm'] = {'model': model, 'score': 0.5}  # Default score
                    except Exception as e:
                        logger.error(f"LightGBM load error: {e}")
                
                if models:
                    self.models[model_key] = models
                    loaded_count += 1
            
            # Feature columns dosyasını yükle
            feature_file = f"{self.model_directory}/{symbol}_features.pkl"
            if os.path.exists(feature_file):
                try:
                    self.feature_columns = joblib.load(feature_file)
                    logger.info(f"Feature columns loaded: {len(self.feature_columns)}")
                except Exception as e:
                    logger.error(f"Feature columns load error: {e}")
            
            logger.info(f"Models loaded for {symbol}: {loaded_count} horizons")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Load models error: {e}")
            return False
    
    def get_system_info(self):
        """System info"""
        return {
            'type': 'simple_enhanced_ml',
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
            'models_trained': len(self.models),
            'prediction_horizons': self.prediction_horizons,
            'feature_count': len(self.feature_columns)
        }

# Global instance
_simple_enhanced_ml = None

def get_simple_enhanced_ml():
    """Simple Enhanced ML singleton"""
    global _simple_enhanced_ml
    if _simple_enhanced_ml is None:
        _simple_enhanced_ml = SimpleEnhancedML()
    return _simple_enhanced_ml

if __name__ == "__main__":
    # Test
    ml_system = get_simple_enhanced_ml()
    info = ml_system.get_system_info()
    print("🚀 Simple Enhanced ML System Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
