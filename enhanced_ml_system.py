"""
Enhanced ML System
XGBoost, LightGBM, CatBoost ile gelişmiş tahmin modelleri
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Enhanced ML Models
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

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Base ML system
try:
    from ml_prediction_system import MLPredictionSystem
    BASE_ML_AVAILABLE = True
except ImportError:
    BASE_ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedMLSystem:
    """Gelişmiş ML tahmin sistemi"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.model_directory = os.getenv('ML_MODEL_PATH', "enhanced_ml_models")
        self.prediction_horizons = [1, 3, 7, 14, 30]  # 1D, 3D, 7D, 14D, 30D
        self.feature_columns = []  # Initialize feature columns list
        
        # Model klasörünü oluştur
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Base ML system
        if BASE_ML_AVAILABLE:
            self.base_ml = MLPredictionSystem()
        
        # CatBoost çalışma dizini (write permission hatalarını önlemek için)
        try:
            self.catboost_train_dir = os.getenv('CATBOOST_TRAIN_DIR', '/opt/bist-pattern/.cache/catboost')
            os.makedirs(self.catboost_train_dir, exist_ok=True)
        except Exception:
            # Son çare: tmp dizini
            self.catboost_train_dir = '/tmp/catboost_info'
            try:
                os.makedirs(self.catboost_train_dir, exist_ok=True)
            except Exception:
                pass

        # Model kayıt dizini (yazılabilir)
        try:
            self.model_directory = os.getenv('ML_MODEL_PATH', '/opt/bist-pattern/.cache/enhanced_ml_models')
            os.makedirs(self.model_directory, exist_ok=True)
        except Exception:
            try:
                self.model_directory = 'enhanced_ml_models'
                os.makedirs(self.model_directory, exist_ok=True)
            except Exception:
                pass

        logger.info("🧠 Enhanced ML System başlatıldı")
        logger.info(f"📊 XGBoost: {XGBOOST_AVAILABLE}")
        logger.info(f"📊 LightGBM: {LIGHTGBM_AVAILABLE}")
        logger.info(f"📊 CatBoost: {CATBOOST_AVAILABLE}")
    
    def create_advanced_features(self, data):
        """Gelişmiş feature engineering"""
        try:
            if BASE_ML_AVAILABLE:
                # Base features from original system
                df = self.base_ml.create_technical_features(data)
            else:
                df = data.copy()
                # Basic fallback features
                if 'Close' in df.columns:
                    df = df.rename(columns={
                        'Open': 'open', 'High': 'high', 
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    })
            
            # Advanced technical indicators
            self._add_advanced_indicators(df)
            
            # Market microstructure features
            self._add_microstructure_features(df)
            
            # Volatility features
            self._add_volatility_features(df)
            
            # Cyclical features
            self._add_cyclical_features(df)
            
            # Statistical features
            self._add_statistical_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Advanced feature engineering hatası: {e}")
            return data
    
    def _add_advanced_indicators(self, df):
        """Gelişmiş teknik indikatörler"""
        try:
            # ATR (Average True Range)
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift())
            df['low_close'] = np.abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            
            for period in [14, 21]:
                df[f'atr_{period}'] = df['true_range'].rolling(period).mean()
            
            # Commodity Channel Index (CCI)
            for period in [14, 20]:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                mean_typical = typical_price.rolling(period).mean()
                mean_deviation = typical_price.rolling(period).apply(
                    lambda x: np.mean(np.abs(x - x.mean()))
                )
                df[f'cci_{period}'] = (typical_price - mean_typical) / (0.015 * mean_deviation)
            
            # Money Flow Index (MFI)
            if 'volume' in df.columns:
                for period in [14, 21]:
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    money_flow = typical_price * df['volume']
                    
                    positive_flow = money_flow.where(df['close'] > df['close'].shift(), 0)
                    negative_flow = money_flow.where(df['close'] < df['close'].shift(), 0)
                    
                    positive_sum = positive_flow.rolling(period).sum()
                    negative_sum = negative_flow.rolling(period).sum()
                    
                    money_ratio = positive_sum / negative_sum
                    df[f'mfi_{period}'] = 100 - (100 / (1 + money_ratio))
            
            # Parabolic SAR (simplified)
            df['sar'] = df['close'].copy()  # Simplified implementation
            
            # Awesome Oscillator
            sma_5 = ((df['high'] + df['low']) / 2).rolling(5).mean()
            sma_34 = ((df['high'] + df['low']) / 2).rolling(34).mean()
            df['awesome_oscillator'] = sma_5 - sma_34
            
        except Exception as e:
            logger.error(f"Advanced indicators hatası: {e}")
    
    def _add_microstructure_features(self, df):
        """Market microstructure features"""
        try:
            # OHLC ratios
            df['body_ratio'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['shadow_ratio'] = df['upper_shadow'] / df['lower_shadow']
            
            # Gap analysis
            df['gap'] = df['open'] - df['close'].shift()
            df['gap_ratio'] = df['gap'] / df['close'].shift()
            
            # Intraday returns
            df['intraday_return'] = (df['close'] - df['open']) / df['open']
            df['overnight_return'] = (df['open'] - df['close'].shift()) / df['close'].shift()
            
            # Volume-price trend
            if 'volume' in df.columns:
                df['vpt'] = df['volume'] * ((df['close'] - df['close'].shift()) / df['close'].shift())
                df['vpt_sma'] = df['vpt'].rolling(10).mean()
            
        except Exception as e:
            logger.error(f"Microstructure features hatası: {e}")
    
    def _add_volatility_features(self, df):
        """Volatility features"""
        try:
            # Different volatility measures
            for window in [5, 10, 20, 30]:
                returns = df['close'].pct_change()
                df[f'volatility_{window}'] = returns.rolling(window).std()
                df[f'volatility_rank_{window}'] = df[f'volatility_{window}'].rolling(window*2).rank(pct=True)
            
            # GARCH-like features
            returns = df['close'].pct_change()
            df['return_squared'] = returns ** 2
            df['volatility_garch'] = df['return_squared'].ewm(alpha=0.1).mean()
            
            # Volatility regime
            vol_20 = df['close'].pct_change().rolling(20).std()
            vol_ma = vol_20.rolling(60).mean()
            df['vol_regime'] = (vol_20 / vol_ma - 1)
            
        except Exception as e:
            logger.error(f"Volatility features hatası: {e}")
    
    def _add_cyclical_features(self, df):
        """Cyclical time features"""
        try:
            # Assuming df.index is datetime
            dates = pd.to_datetime(df.index)
            
            # Day of week effects
            df['day_of_week'] = dates.dayofweek
            df['is_monday'] = (dates.dayofweek == 0).astype(int)
            df['is_friday'] = (dates.dayofweek == 4).astype(int)
            
            # Month effects
            df['month'] = dates.month
            df['quarter'] = dates.quarter
            df['is_month_end'] = dates.is_month_end.astype(int)
            df['is_quarter_end'] = dates.is_quarter_end.astype(int)
            
            # Cyclical encoding
            df['day_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
            df['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
            
        except Exception as e:
            logger.error(f"Cyclical features hatası: {e}")
    
    def _add_statistical_features(self, df):
        """Statistical features"""
        try:
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'skewness_{window}'] = df['close'].rolling(window).skew()
                df[f'kurtosis_{window}'] = df['close'].rolling(window).kurt()
                
                # Percentile features
                df[f'percentile_25_{window}'] = df['close'].rolling(window).quantile(0.25)
                df[f'percentile_75_{window}'] = df['close'].rolling(window).quantile(0.75)
                
                # Z-score
                mean = df['close'].rolling(window).mean()
                std = df['close'].rolling(window).std()
                df[f'zscore_{window}'] = (df['close'] - mean) / std
            
            # Entropy-like measures
            for window in [10, 20]:
                returns = df['close'].pct_change()
                abs_returns = np.abs(returns)
                df[f'entropy_{window}'] = abs_returns.rolling(window).apply(
                    lambda x: -np.sum(x * np.log(x + 1e-10)) if len(x) > 0 else 0
                )
            
        except Exception as e:
            logger.error(f"Statistical features hatası: {e}")
    
    def _clean_data(self, df):
        """Veri temizleme - INF, NaN ve aşırı değerleri temizle"""
        try:
            logger.info(f"🧹 Veri temizleme başlatılıyor - Shape: {df.shape}")
            
            # INF değerleri temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Numeric sütunları al
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # Her numeric sütun için outlier temizleme
            for col in numeric_columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue  # Ana price sütunlarını dokunma
                
                # Z-score ile outlier tespiti (3 sigma)
                if df[col].std() > 0:  # Std > 0 kontrolü
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df.loc[z_scores > 3, col] = np.nan
                
                # Çok büyük değerleri sınırla
                percentile_99 = df[col].quantile(0.99)
                percentile_1 = df[col].quantile(0.01)
                
                if not np.isnan(percentile_99) and not np.isnan(percentile_1):
                    df[col] = df[col].clip(lower=percentile_1, upper=percentile_99)
            
            # NaN değerleri forward fill ile doldur
            df = df.ffill()
            
            # Hala NaN varsa 0 ile doldur
            df = df.fillna(0)
            
            # Final check - hala INF var mı?
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                logger.warning(f"⚠️ {inf_count} INF değer hala mevcut, 0 ile değiştiriliyor")
                df = df.replace([np.inf, -np.inf], 0)
            
            logger.info(f"✅ Veri temizleme tamamlandı - Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Veri temizleme hatası: {e}")
            return df
    
    def train_enhanced_models(self, symbol, data):
        """Gelişmiş modelleri eğit"""
        try:
            logger.info(f"🧠 {symbol} için enhanced model eğitimi başlatılıyor")
            
            # Data validation
            if data is None or len(data) == 0:
                logger.error(f"{symbol} için veri bulunamadı")
                return False
            
            logger.info(f"📊 Veri boyutu: {data.shape}")
            
            # Feature engineering
            df_features = self.create_advanced_features(data)
            df_features = df_features.dropna()
            
            # Clean infinite and large values
            df_features = self._clean_data(df_features)
            
            if len(df_features) < 200:
                logger.warning(f"{symbol} için yeterli veri yok (200+ gerekli)")
                return False
            
            # Feature selection
            feature_cols = [col for col in df_features.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']
                          and df_features[col].dtype in ['float64', 'int64']
                          and not df_features[col].isnull().all()]
            
            logger.info(f"📊 {len(feature_cols)} feature kullanılacak")
            
            results = {}
            
            # Her tahmin ufku için model eğit
            for horizon in self.prediction_horizons:
                logger.info(f"📈 {symbol} - {horizon} gün tahmini için model eğitimi")
                
                # Target variable
                target = f'target_{horizon}d'
                df_features[target] = df_features['close'].shift(-horizon)
                
                # Remove last horizon rows
                df_model = df_features[:-horizon].copy()
                
                X = df_model[feature_cols].values
                y = df_model[target].values
                
                # Time series split
                tscv = TimeSeriesSplit(n_splits=3)
                
                # Train models
                horizon_models = {}
                
                # 1. XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        xgb_model = xgb.XGBRegressor(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        # Cross-validation
                        xgb_scores = []
                        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                            try:
                                X_train, X_val = X[train_idx], X[val_idx]
                                y_train, y_val = y[train_idx], y[val_idx]
                                
                                xgb_model.fit(X_train, y_train)
                                pred = xgb_model.predict(X_val)
                                score = r2_score(y_val, pred)
                                xgb_scores.append(score)
                                logger.info(f"XGBoost fold {fold}: R² = {score:.3f}")
                            except Exception as e:
                                logger.error(f"XGBoost fold {fold} error: {e}")
                                raise
                        
                        # Final training
                        xgb_model.fit(X, y)
                        xgb_pred = xgb_model.predict(X[-100:])  # Last 100 for validation
                        
                        horizon_models['xgboost'] = {
                            'model': xgb_model,
                            'score': np.mean(xgb_scores),
                            'rmse': np.sqrt(mean_squared_error(y[-100:], xgb_pred)),
                            'mape': mean_absolute_percentage_error(y[-100:], xgb_pred)
                        }
                        
                        # Feature importance
                        self.feature_importance[f"{symbol}_{horizon}d_xgb"] = dict(
                            zip(feature_cols, xgb_model.feature_importances_)
                        )
                        
                        logger.info(f"XGBoost {horizon}D - R²: {np.mean(xgb_scores):.3f}")
                        
                    except Exception as e:
                        logger.error(f"XGBoost eğitim hatası: {e}")
                
                # 2. LightGBM
                if LIGHTGBM_AVAILABLE:
                    try:
                        lgb_model = lgb.LGBMRegressor(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42,
                            n_jobs=-1,
                            verbose=-1
                        )
                        
                        # Cross-validation
                        lgb_scores = []
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            lgb_model.fit(X_train, y_train)
                            pred = lgb_model.predict(X_val)
                            score = r2_score(y_val, pred)
                            lgb_scores.append(score)
                        
                        # Final training
                        lgb_model.fit(X, y)
                        lgb_pred = lgb_model.predict(X[-100:])
                        
                        horizon_models['lightgbm'] = {
                            'model': lgb_model,
                            'score': np.mean(lgb_scores),
                            'rmse': np.sqrt(mean_squared_error(y[-100:], lgb_pred)),
                            'mape': mean_absolute_percentage_error(y[-100:], lgb_pred)
                        }
                        
                        # Feature importance
                        self.feature_importance[f"{symbol}_{horizon}d_lgb"] = dict(
                            zip(feature_cols, lgb_model.feature_importances_)
                        )
                        
                        logger.info(f"LightGBM {horizon}D - R²: {np.mean(lgb_scores):.3f}")
                        
                    except Exception as e:
                        logger.error(f"LightGBM eğitim hatası: {e}")
                
                # 3. CatBoost
                if CATBOOST_AVAILABLE:
                    try:
                        cat_model = cb.CatBoostRegressor(
                            iterations=100,
                            depth=6,
                            learning_rate=0.1,
                            random_seed=42,
                            allow_writing_files=False,
                            train_dir=self.catboost_train_dir,
                            logging_level='Silent'
                        )
                        
                        # Cross-validation
                        cat_scores = []
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            cat_model.fit(X_train, y_train)
                            pred = cat_model.predict(X_val)
                            score = r2_score(y_val, pred)
                            cat_scores.append(score)
                        
                        # Final training
                        cat_model.fit(X, y)
                        cat_pred = cat_model.predict(X[-100:])
                        
                        horizon_models['catboost'] = {
                            'model': cat_model,
                            'score': np.mean(cat_scores),
                            'rmse': np.sqrt(mean_squared_error(y[-100:], cat_pred)),
                            'mape': mean_absolute_percentage_error(y[-100:], cat_pred)
                        }
                        
                        # Feature importance
                        self.feature_importance[f"{symbol}_{horizon}d_cat"] = dict(
                            zip(feature_cols, cat_model.feature_importances_)
                        )
                        
                        logger.info(f"CatBoost {horizon}D - R²: {np.mean(cat_scores):.3f}")
                        
                    except Exception as e:
                        logger.error(f"CatBoost eğitim hatası: {e}")
                
                # Store models and results
                self.models[f"{symbol}_{horizon}d"] = horizon_models
                results[f"{horizon}d"] = horizon_models
                
                # Store feature columns
                self.feature_columns = feature_cols
            
            # Save models
            self.save_enhanced_models(symbol)
            
            # Store performance
            self.model_performance[symbol] = results
            
            logger.info(f"✅ {symbol} enhanced model eğitimi tamamlandı")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced model eğitim hatası: {e}")
            return False
    
    def predict_enhanced(self, symbol, current_data):
        """Enhanced predictions"""
        try:
            # Feature engineering
            df_features = self.create_advanced_features(current_data)
            df_features = df_features.dropna()
            
            # Clean data
            df_features = self._clean_data(df_features)
            
            if len(df_features) == 0:
                return None
            
            # Get latest features
            if not self.feature_columns:
                logger.error("Feature columns not set. Model training required.")
                return None
                
            # Check if all feature columns exist
            missing_cols = [col for col in self.feature_columns if col not in df_features.columns]
            if missing_cols:
                logger.error(f"Missing feature columns: {missing_cols}")
                return None
                
            latest_features = df_features[self.feature_columns].iloc[-1:].values
            
            predictions = {}
            
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.models:
                    horizon_models = self.models[model_key]
                    model_predictions = {}
                    
                    for model_name, model_info in horizon_models.items():
                        try:
                            pred = model_info['model'].predict(latest_features)[0]
                            
                            model_predictions[model_name] = {
                                'prediction': float(pred),
                                'confidence': float(model_info['score']),
                                'rmse': float(model_info['rmse']),
                                'mape': float(model_info['mape'])
                            }
                        except Exception as e:
                            logger.error(f"{model_name} prediction error: {e}")
                    
                    # Ensemble prediction (weighted by performance)
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
                                'current_price': float(current_data['close'].iloc[-1]),
                                'model_count': len(model_predictions)
                            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Enhanced prediction error: {e}")
            return None
    
    def save_enhanced_models(self, symbol):
        """Enhanced modelleri kaydet"""
        try:
            for horizon in self.prediction_horizons:
                model_key = f"{symbol}_{horizon}d"
                
                if model_key in self.models:
                    models = self.models[model_key]
                    
                    for model_name, model_info in models.items():
                        filename = f"{self.model_directory}/{model_key}_{model_name}.pkl"
                        joblib.dump(model_info['model'], filename)
            
            # Feature importance kaydet
            importance_file = f"{self.model_directory}/{symbol}_feature_importance.pkl"
            symbol_importance = {k: v for k, v in self.feature_importance.items() if k.startswith(symbol)}
            joblib.dump(symbol_importance, importance_file)
            
            logger.info(f"💾 {symbol} enhanced modelleri kaydedildi")
            
        except Exception as e:
            logger.error(f"Enhanced model kaydetme hatası: {e}")
    
    def get_top_features(self, symbol, model_type='xgboost', top_n=20):
        """En önemli feature'ları döndür"""
        try:
            top_features = {}
            
            for horizon in self.prediction_horizons:
                # Normalize model_type mapping: 'xgboost'->'xgb', 'lightgbm'->'lgb', 'catboost'->'cat'
                mt = model_type.lower()
                if mt.startswith('xgboost') or mt == 'xgb':
                    short = 'xgb'
                elif mt.startswith('lightgbm') or mt == 'lgb':
                    short = 'lgb'
                elif mt.startswith('catboost') or mt == 'cat':
                    short = 'cat'
                else:
                    short = mt[:3]
                key = f"{symbol}_{horizon}d_{short}"
                if key in self.feature_importance:
                    importance = self.feature_importance[key]
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    top_features[f"{horizon}d"] = sorted_features[:top_n]
            
            return top_features
            
        except Exception as e:
            logger.error(f"Feature importance error: {e}")
            return {}
    
    def get_system_info(self):
        """Enhanced ML system bilgileri"""
        return {
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
            'catboost_available': CATBOOST_AVAILABLE,
            'base_ml_available': BASE_ML_AVAILABLE,
            'models_trained': len(self.models),
            'prediction_horizons': self.prediction_horizons,
            'feature_count': len(getattr(self, 'feature_columns', [])),
            'performance_tracked': len(self.model_performance)
        }

# Global singleton instance
_enhanced_ml_system = None

def get_enhanced_ml_system():
    """Enhanced ML System singleton'ını döndür"""
    global _enhanced_ml_system
    if _enhanced_ml_system is None:
        _enhanced_ml_system = EnhancedMLSystem()
    return _enhanced_ml_system

if __name__ == "__main__":
    # Test
    enhanced_ml = get_enhanced_ml_system()
    info = enhanced_ml.get_system_info()
    
    print("🧠 Enhanced ML System Test:")
    print(f"📊 XGBoost: {info['xgboost_available']}")
    print(f"📊 LightGBM: {info['lightgbm_available']}")
    print(f"📊 CatBoost: {info['catboost_available']}")
    print(f"🎯 Prediction Horizons: {info['prediction_horizons']}")
    print(f"✅ Enhanced ML System ready!")
