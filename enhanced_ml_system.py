"""
Enhanced ML System
XGBoost, LightGBM, CatBoost ile gelişmiş tahmin modelleri
"""

import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.linear_model import Ridge  # TODO: Uncomment when meta-learner training is added
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ⚡ PURGED TIME-SERIES SPLIT - Data Leakage Prevention
class PurgedTimeSeriesSplit:
    """
    Time-series cross-validation with purging and embargo.
    
    Purging: Remove samples from training set that are too close to test set
    Embargo: Add gap between train and test to prevent lookahead bias
    
    Based on "Advances in Financial Machine Learning" by Marcos López de Prado
    """
    
    def __init__(self, n_splits=3, purge_gap=5, embargo_td=2):
        """
        Args:
            n_splits: Number of splits
            purge_gap: Number of samples to purge before test (removes overlap)
            embargo_td: Number of samples to embargo after train (future data gap)
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_td = embargo_td
    
    def split(self, X, y=None, groups=None):
        """Generate purged train/test indices."""
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Test set
            test_start = (i + 1) * fold_size
            test_end = test_start + fold_size
            test_indices = indices[test_start:test_end]
            
            # Train set (before test, with purging)
            train_end = test_start - self.purge_gap  # Purge gap
            if train_end <= 0:
                continue
            
            train_indices = indices[:train_end]
            
            # Apply embargo (remove recent samples that overlap with test timing)
            if self.embargo_td > 0 and len(train_indices) > self.embargo_td:
                train_indices = train_indices[:-self.embargo_td]
            
            if len(train_indices) > 10 and len(test_indices) > 3:
                yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# Enhanced ML Models
try:
    import xgboost as xgb  # type: ignore[import]
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None  # type: ignore[assignment]

try:
    import lightgbm as lgb  # type: ignore[import]
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore[assignment]

try:
    import catboost as cb  # type: ignore[import]
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None  # type: ignore[assignment]

# Base ML system
try:
    from ml_prediction_system import MLPredictionSystem  # type: ignore[import]
    BASE_ML_AVAILABLE = True
except ImportError:
    BASE_ML_AVAILABLE = False
    MLPredictionSystem = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class EnhancedMLSystem:
    """Gelişmiş ML tahmin sistemi"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        # ⚡ NEW: Meta-learners storage (Ridge for stacking)
        self.meta_learners = {}  # {symbol_horizon: Ridge model}
        self.model_directory = os.getenv('ML_MODEL_PATH', "enhanced_ml_models")
        self.prediction_horizons = [1, 3, 7, 14, 30]  # 1D, 3D, 7D, 14D, 30D
        self.feature_columns = []  # Initialize feature columns list
        # Optional feature flags
        try:
            self.enable_talib_patterns = str(os.getenv('ENABLE_TALIB_PATTERNS', 'True')).lower() in ('1', 'true', 'yes')
        except Exception:
            self.enable_talib_patterns = True
        try:
            self.enable_external_features = str(os.getenv('ENABLE_EXTERNAL_FEATURES', 'True')).lower() in ('1', 'true', 'yes')
        except Exception:
            self.enable_external_features = True
        try:
            self.enable_fingpt_features = str(os.getenv('ENABLE_FINGPT_FEATURES', 'True')).lower() in ('1', 'true', 'yes')
        except Exception:
            self.enable_fingpt_features = True
        try:
            # ⚡ NEW: Meta-stacking with Ridge learner OOF (default: enabled!)
            self.enable_meta_stacking = str(os.getenv('ENABLE_META_STACKING', 'True')).lower() in ('1', 'true', 'yes')
        except Exception:
            self.enable_meta_stacking = False
        try:
            # ⚡ NEW: Seed bagging (multiple random seeds for variance reduction)
            self.enable_seed_bagging = str(os.getenv('ENABLE_SEED_BAGGING', 'True')).lower() in ('1', 'true', 'yes')
            self.n_seeds = int(os.getenv('N_SEEDS', '3'))  # 3 seeds by default
            self.base_seeds = [42, 123, 456, 789, 999][:self.n_seeds]  # Use first N seeds
        except Exception:
            self.enable_seed_bagging = True  # Enable by default!
            self.n_seeds = 3
            self.base_seeds = [42, 123, 456]
        try:
            self.enable_yolo_features = str(os.getenv('ENABLE_YOLO_FEATURES', 'True')).lower() in ('1', 'true', 'yes')
        except Exception:
            self.enable_yolo_features = True
        # External features directory (backfilled files live here)
        try:
            self.external_feature_dir = os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
        except Exception:
            self.external_feature_dir = '/opt/bist-pattern/logs/feature_backfill'
        # Training parallelism and early stopping configuration (ENV-driven)
        try:
            self.train_threads = int(os.getenv('ML_TRAIN_THREADS', '2'))
            if self.train_threads <= 0:
                # Fallback to CPU count if invalid
                cpu_count = os.cpu_count() or 2
                self.train_threads = max(1, int(cpu_count))
        except Exception:
            cpu_count = os.cpu_count() or 2
            self.train_threads = max(1, int(cpu_count))
        try:
            self.early_stop_rounds = int(os.getenv('ML_EARLY_STOP_ROUNDS', '50'))
        except Exception:
            self.early_stop_rounds = 50
        try:
            self.early_stop_min_val = int(os.getenv('ML_EARLY_STOP_MIN_VAL', '10'))
        except Exception:
            self.early_stop_min_val = 10
        # Optional stop-sentinel file path (for graceful halt)
        self.stop_file_path = os.getenv('TRAIN_STOP_FILE', '/opt/bist-pattern/.cache/STOP_TRAIN')
        
        # Model klasörünü oluştur
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Base ML system
        if BASE_ML_AVAILABLE:
            self.base_ml = MLPredictionSystem()  # type: ignore[misc]
        
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
    
    @staticmethod
    def _smape(y_true, y_pred, eps: float = 1e-8) -> float:
        try:
            y_true_arr = np.asarray(y_true, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            denom = np.abs(y_true_arr) + np.abs(y_pred_arr) + eps
            return float(np.mean(2.0 * np.abs(y_pred_arr - y_true_arr) / denom))
        except Exception:
            return float('nan')
    
    @staticmethod
    def _r2_to_confidence(r2_score):
        """
        Convert R² score to confidence [0-1]
        R² can be negative (model worse than mean baseline)
        
        CRITICAL FIX: R² directly as confidence was wrong!
        - R² < 0 → poor model, but still give some confidence (0.3-0.4)
        - R² = 0 → baseline (0.45)
        - R² = 0.5 → moderate (0.7)
        - R² = 0.8 → good (0.85)
        - R² = 1.0 → excellent (0.95, never 1.0 for humility)
        
        Formula: conf = 0.3 + 0.65 / (1 + exp(-5*r2))
        """
        try:
            # Sigmoid transformation shifted and scaled
            exponent = -5.0 * float(r2_score)
            confidence = 0.3 + (0.65 / (1.0 + np.exp(exponent)))
            # Clamp to [0.2, 0.95] for safety
            return float(np.clip(confidence, 0.2, 0.95))
        except Exception:
            return 0.5  # Fallback

    def create_advanced_features(self, data, symbol: str | None = None):
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
            
            # ⚡ NEW: Liquidity/Volume tier features
            self._add_liquidity_features(df)
            
            # ⚡ NEW: Macro economic features (USDTRY, CDS, TCMB Rate)
            if symbol:  # Only if symbol provided
                self._add_macro_features(df)
            
            # Optional: TA-Lib candlestick pattern features (lightweight subset)
            if self.enable_talib_patterns:
                self._add_candlestick_features(df)
            
            # Optional: Merge external backfilled features (FinGPT / YOLO)
            if symbol and self.enable_external_features:
                self._merge_external_features(symbol, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Advanced feature engineering hatası: {e}")
            return data

    def _add_candlestick_features(self, df):
        """Lightweight TA-Lib candlestick features: last-3 day bull/bear counts and today's signal."""
        try:
            try:
                import talib  # type: ignore
            except Exception:
                return
            if not all(c in df.columns for c in ('open', 'high', 'low', 'close')):
                return
            op = df['open'].astype(float)
            hi = df['high'].astype(float)
            lo = df['low'].astype(float)
            cl = df['close'].astype(float)

            # Subset of robust patterns; TA-Lib returns 100/-100/0 values
            pat_series = []
            try:
                pat_series.append(talib.CDLENGULFING(op, hi, lo, cl))
            except Exception:
                pass
            try:
                pat_series.append(talib.CDLHAMMER(op, hi, lo, cl))
            except Exception:
                pass
            try:
                pat_series.append(talib.CDLSHOOTINGSTAR(op, hi, lo, cl))
            except Exception:
                pass
            try:
                pat_series.append(talib.CDLHARAMI(op, hi, lo, cl))
            except Exception:
                pass
            try:
                pat_series.append(talib.CDLDOJI(op, hi, lo, cl))
            except Exception:
                pass

            if not pat_series:
                return

            pats = None
            try:
                import pandas as _pd  # local alias
                pats = sum(pat_series)
                # Ensure pandas Series
                if not hasattr(pats, 'tail'):
                    pats = _pd.Series(pats, index=df.index)
            except Exception:
                return

            last3 = pats.tail(3)
            bull3 = int((last3 > 0).sum())
            bear3 = int((last3 < 0).sum())
            df['pat_bull3'] = float(bull3)
            df['pat_bear3'] = float(bear3)
            df['pat_net3'] = float(bull3 - bear3)
            # Normalize today's raw signal to [-1, 1]
            try:
                if hasattr(pats, 'iloc') and hasattr(pats, '__len__') and len(pats) > 0:  # type: ignore[arg-type]
                    today_raw = float(pats.iloc[-1])
                else:
                    today_raw = 0.0
            except Exception:
                today_raw = 0.0
            df['pat_today'] = float(np.clip(today_raw / 100.0, -1.0, 1.0))
        except Exception as e:
            logger.debug(f"TA-Lib candlestick features skipped: {e}")

    def _merge_external_features(self, symbol: str, df: pd.DataFrame) -> None:
        """Merge offline backfilled features (FinGPT sentiment and YOLO pattern density) if available.

        Expected files (CSV) under EXTERNAL_FEATURE_DIR:
          - fingpt/{SYMBOL}.csv with columns like: date, sentiment_score (or score/sentiment), news_count
          - yolo/{SYMBOL}.csv   with columns like: date, yolo_density (or density), yolo_bull, yolo_bear, yolo_score
        """
        try:
            if df is None or len(df) == 0:
                return
            dates = pd.to_datetime(df.index).normalize()

            def _load_csv_safe(path: str) -> pd.DataFrame | None:
                try:
                    if not os.path.exists(path):
                        return None
                    tmp = pd.read_csv(path)
                    # Flexible date parsing
                    if 'date' in tmp.columns:
                        tmp['date'] = pd.to_datetime(tmp['date']).dt.normalize()
                        tmp = tmp.set_index('date').sort_index()
                    elif 'timestamp' in tmp.columns:
                        tmp['timestamp'] = pd.to_datetime(tmp['timestamp']).dt.normalize()
                        tmp = tmp.set_index('timestamp').sort_index()
                    else:
                        # try to parse index if unnamed
                        tmp.index = pd.to_datetime(tmp.index).normalize()
                    return tmp
                except Exception as _e:
                    logger.debug(f"External feature load failed: {path} → {_e}")
                    return None

            # FinGPT features
            if self.enable_fingpt_features:
                f_csv = os.path.join(self.external_feature_dir, 'fingpt', f'{symbol}.csv')
                fdf = _load_csv_safe(f_csv)
                if fdf is not None and len(fdf) > 0:
                    # pick score column
                    score_cols = [c for c in (
                        'sentiment_score', 'score', 'avg_score', 'sentiment', 'sentiment_avg', 'polarity'
                    ) if c in fdf.columns]
                    count_cols = [c for c in ('news_count', 'count', 'n') if c in fdf.columns]
                    # Align to DF dates
                    fdf = fdf.reindex(dates)
                    if score_cols:
                        try:
                            df['fingpt_sent'] = pd.to_numeric(fdf[score_cols[0]], errors='coerce').fillna(0.0).astype(float)
                        except Exception:
                            df['fingpt_sent'] = 0.0
                    else:
                        df['fingpt_sent'] = 0.0
                    if count_cols:
                        try:
                            df['fingpt_news'] = pd.to_numeric(fdf[count_cols[0]], errors='coerce').fillna(0.0).astype(float)
                        except Exception:
                            df['fingpt_news'] = 0.0
                    else:
                        df['fingpt_news'] = 0.0

            # YOLO features
            if self.enable_yolo_features:
                y_csv = os.path.join(self.external_feature_dir, 'yolo', f'{symbol}.csv')
                ydf = _load_csv_safe(y_csv)
                if ydf is not None and len(ydf) > 0:
                    dens_cols = [c for c in ('yolo_density', 'density', 'det_density') if c in ydf.columns]
                    bull_cols = [c for c in ('yolo_bull', 'bull', 'bull_count') if c in ydf.columns]
                    bear_cols = [c for c in ('yolo_bear', 'bear', 'bear_count') if c in ydf.columns]
                    score_cols = [c for c in ('yolo_score', 'score', 'align') if c in ydf.columns]
                    ydf = ydf.reindex(dates)
                    if dens_cols:
                        try:
                            df['yolo_density'] = pd.to_numeric(ydf[dens_cols[0]], errors='coerce').fillna(0.0).astype(float)
                        except Exception:
                            df['yolo_density'] = 0.0
                    else:
                        df['yolo_density'] = 0.0
                    if bull_cols:
                        try:
                            df['yolo_bull'] = pd.to_numeric(ydf[bull_cols[0]], errors='coerce').fillna(0.0).astype(float)
                        except Exception:
                            df['yolo_bull'] = 0.0
                    else:
                        df['yolo_bull'] = 0.0
                    if bear_cols:
                        try:
                            df['yolo_bear'] = pd.to_numeric(ydf[bear_cols[0]], errors='coerce').fillna(0.0).astype(float)
                        except Exception:
                            df['yolo_bear'] = 0.0
                    else:
                        df['yolo_bear'] = 0.0
                    if score_cols:
                        try:
                            df['yolo_score'] = pd.to_numeric(ydf[score_cols[0]], errors='coerce').fillna(0.0).astype(float)
                        except Exception:
                            df['yolo_score'] = 0.0
                    else:
                        df['yolo_score'] = 0.0
        except Exception as e:
            logger.debug(f"External feature merge skipped: {e}")
    
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
            
            # ⚡ NEW: Bollinger Bands (3 features)
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_upper'] = sma_20 + (2 * std_20)
            df['bb_lower'] = sma_20 - (2 * std_20)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
            
            # ⚡ NEW: EMA (2 features)
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # ⚡ NEW: Stochastic Oscillator (2 features)
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # ⚡ NEW: ROC - Rate of Change (1 feature)
            df['roc'] = df['close'].pct_change(12) * 100  # 12-period ROC
            
            # ⚡ NEW: Williams %R (1 feature)
            df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)
            
            # ⚡ NEW: TRIX (1 feature)
            ema1 = df['close'].ewm(span=15).mean()
            ema2 = ema1.ewm(span=15).mean()
            ema3 = ema2.ewm(span=15).mean()
            df['trix'] = ema3.pct_change() * 100
            
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
            df['vol_regime'] = (vol_20 / vol_ma - 1).fillna(method='ffill').fillna(0)  # ⚡ FIX: Fill NaN!
            
            # ⚡ NEW: ADX (Average Directional Index) - Trend Strength
            try:
                high = df['high']
                low = df['low']
                close = df['close']
                
                # True Range
                tr1 = high - low
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                
                # Directional Movement
                up_move = high - high.shift()
                down_move = low.shift() - low
                
                # Convert to Series with proper index
                plus_dm = pd.Series(np.where((up_move > 0) & (up_move > down_move), up_move, 0), index=df.index)
                minus_dm = pd.Series(np.where((down_move > 0) & (down_move > up_move), down_move, 0), index=df.index)
                
                # Directional Indicators
                plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-10)
                minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-10)
                
                # ADX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
                df['adx'] = dx.rolling(14).mean()
                df['adx_trending'] = (df['adx'] > 25).astype(int)  # 1 if trending, 0 if ranging
                
            except Exception as e:
                logger.debug(f"ADX calculation error: {e}")
                df['adx'] = 0
                df['adx_trending'] = 0
            
            # ⚡ NEW: Realized Volatility (Annualized)
            try:
                returns = df['close'].pct_change()
                # Realized vol over different windows
                df['realized_vol_5d'] = returns.rolling(5).std() * np.sqrt(252)
                df['realized_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
                df['realized_vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
                
                # Volatility regime based on quantiles
                vol_5d = df['realized_vol_5d']
                df['vol_regime_high'] = (vol_5d > vol_5d.quantile(0.75)).astype(int)
                df['vol_regime_low'] = (vol_5d < vol_5d.quantile(0.25)).astype(int)
                
            except Exception as e:
                logger.debug(f"Realized vol calculation error: {e}")
            
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
    
    def _add_liquidity_features(self, df):
        """Liquidity and volume tier features"""
        try:
            volume = df['volume']
            close = df['close']
            
            # Volume statistics
            vol_mean = volume.mean()
            vol_std = volume.std()
            
            # Relative volume (vs rolling average)
            for window in [5, 20, 60]:
                vol_ma = volume.rolling(window).mean()
                df[f'relative_volume_{window}'] = volume / (vol_ma + 1e-10)
            
            # Volume tier classification (based on percentiles)
            vol_q25 = volume.quantile(0.25)
            vol_q75 = volume.quantile(0.75)
            
            # Tier features (one-hot style)
            df['volume_tier_high'] = (volume > vol_q75).astype(int)  # Top 25%
            df['volume_tier_low'] = (volume < vol_q25).astype(int)   # Bottom 25%
            df['volume_tier_mid'] = ((volume >= vol_q25) & (volume <= vol_q75)).astype(int)
            
            # Continuous volume score (normalized)
            if vol_std > 0:
                df['volume_zscore'] = (volume - vol_mean) / vol_std
            else:
                df['volume_zscore'] = 0
            
            # Volume regime (high activity vs low activity)
            vol_rank = volume.rolling(60).rank(pct=True)
            df['volume_regime'] = vol_rank  # 0-1, higher = more active lately
            
            # Dollar volume (proxy for liquidity)
            df['dollar_volume'] = volume * close
            dollar_vol_ma = df['dollar_volume'].rolling(20).mean()
            df['relative_dollar_volume'] = df['dollar_volume'] / (dollar_vol_ma + 1e-10)
            
            # Volume-price relationship
            # High volume + up move = strong momentum
            # High volume + down move = strong sell-off
            returns = close.pct_change()
            df['volume_price_corr_5'] = volume.rolling(5).corr(returns)
            df['volume_price_corr_20'] = volume.rolling(20).corr(returns)
            
            logger.debug("Liquidity/volume features added")
            
        except Exception as e:
            logger.error(f"Liquidity features hatası: {e}")
    
    def _add_macro_features(self, df):
        """Macro economic features from VT (USDTRY, CDS, TCMB Rate)"""
        try:
            logger.debug("_add_macro_features() called")
            from models import db
            
            # Get macro data from VT (SQL query)
            query = """
                SELECT date, usdtry_close, turkey_cds, tcmb_policy_rate
                FROM macro_indicators
                WHERE date >= :start_date AND date <= :end_date
                ORDER BY date
            """
            
            # Date range from df
            start_date = df.index.min().date() if hasattr(df.index.min(), 'date') else df.index.min()
            end_date = df.index.max().date() if hasattr(df.index.max(), 'date') else df.index.max()
            
            # Execute query with named parameters
            result = db.session.execute(db.text(query), {'start_date': start_date, 'end_date': end_date})
            macro_data = pd.DataFrame(result.fetchall(), columns=['date', 'usdtry', 'cds', 'rate'])
            
            if len(macro_data) > 0:
                # Convert date to datetime for merge
                macro_data['date'] = pd.to_datetime(macro_data['date'])
                macro_data = macro_data.set_index('date')
                
                # ⚡ FIX: Merge macro data (reindex to match df dates)
                # Don't use join - use reindex for alignment
                macro_data = macro_data.reindex(df.index, method='ffill')
                
                # Add columns directly (in-place!) + Convert to float64!
                df['usdtry'] = pd.to_numeric(macro_data['usdtry'], errors='coerce').ffill().bfill().fillna(0).astype('float64')
                df['cds'] = pd.to_numeric(macro_data['cds'], errors='coerce').ffill().bfill().fillna(0).astype('float64')
                df['rate'] = pd.to_numeric(macro_data['rate'], errors='coerce').ffill().bfill().fillna(0).astype('float64')
                logger.info("✅ Macro base features added: usdtry, cds, rate (dtype=float64)")
                
                # Create derivative features
                df['usdtry_change_1d'] = df['usdtry'].pct_change().fillna(0)
                df['usdtry_change_5d'] = df['usdtry'].pct_change(5).fillna(0)
                df['usdtry_change_20d'] = df['usdtry'].pct_change(20).fillna(0)
                df['cds_change_5d'] = df['cds'].pct_change(5).fillna(0)
                df['rate_change_20d'] = df['rate'].pct_change(20).fillna(0)
                logger.info("✅ Macro derivative features added (5 changes)")
                
                logger.info(f"✅ Macro features complete: {len(macro_data)} days merged, 8 features")
            else:
                logger.warning("No macro data found in VT")
                
        except Exception as e:
            import traceback
            logger.error(f"Macro features error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback: zero features
            for col in ['usdtry', 'usdtry_change_1d', 'usdtry_change_5d', 'usdtry_change_20d',
                       'cds', 'cds_change_5d', 'rate', 'rate_change_20d']:
                if col not in df.columns:
                    df[col] = 0.0
    
    def _clean_data(self, df):
        """Veri temizleme - INF, NaN ve aşırı değerleri temizle"""
        try:
            logger.info(f"🧹 Veri temizleme başlatılıyor - Shape: {df.shape}")
            
            # INF değerleri temizle
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Numeric sütunları al
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            # CRITICAL FIX: Softened outlier removal (was too aggressive)
            # Previous: 3 sigma + 1-99 percentile → Market shocks were removed!
            # New: 5 sigma + 0.5-99.5 percentile → Keep real market events
            for col in numeric_columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue  # Ana price sütunlarını dokunma
                
                # Z-score ile outlier tespiti (5 sigma - yumuşatıldı)
                if df[col].std() > 0:  # Std > 0 kontrolü
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df.loc[z_scores > 5, col] = np.nan  # 3 → 5 sigma
                
                # Çok büyük değerleri sınırla (yumuşatıldı)
                percentile_high = df[col].quantile(0.995)  # 0.99 → 0.995
                percentile_low = df[col].quantile(0.005)   # 0.01 → 0.005
                
                if not np.isnan(percentile_high) and not np.isnan(percentile_low):
                    df[col] = df[col].clip(lower=percentile_low, upper=percentile_high)
            
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
    
    def _should_halt(self) -> bool:
        """Eğitim sırasında dıştan durdurma talebi var mı kontrol et"""
        try:
            return bool(self.stop_file_path and os.path.exists(self.stop_file_path))
        except Exception:
            return False

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
            df_features = self.create_advanced_features(data, symbol=symbol)
            
            # ⚡ CRITICAL FIX: Don't drop all NaN rows!
            # Rolling features (ADX, realized_vol_60d) have NaN at start
            # dropna() would remove 60-100 days unnecessarily!
            # Instead: Fill NaN with 0 or forward fill for features
            # Target NaN will be removed later (per horizon)
            
            # Forward fill for features (conservative approach)
            for col in df_features.columns:
                if df_features[col].dtype in ['float64', 'float32', 'int64', 'int32']:  # ⚡ FIX: Include all numeric!
                    df_features[col] = df_features[col].fillna(method='ffill').fillna(0)
            
            # Clean infinite and large values
            df_features = self._clean_data(df_features)
            
            try:
                # Read minimum data days from environment; default 180 to align with override policy
                min_days = int(os.getenv('ML_MIN_DATA_DAYS', os.getenv('ML_MIN_DAYS', '180')))
            except Exception:
                min_days = 180
            if len(df_features) < min_days:
                logger.warning(f"{symbol} için yeterli veri yok ({min_days}+ gerekli)")
                return False
            
            # Feature selection
            feature_cols = [col for col in df_features.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume']
                          and df_features[col].dtype in ['float64', 'float32', 'int64', 'int32']  # ⚡ FIX: Include int32!
                          and not df_features[col].isnull().all()]
            
            logger.info(f"📊 {len(feature_cols)} feature kullanılacak")
            
            results = {}
            
            # Her tahmin ufku için model eğit
            for horizon in self.prediction_horizons:
                # Graceful stop kontrolü
                if self._should_halt():
                    logger.warning("⛔ Stop sentinel tespit edildi, eğitim durduruluyor")
                    # Kısmi sonuçları kaydetmeden çık
                    return False
                logger.info(f"📈 {symbol} - {horizon} gün tahmini için model eğitimi")
                
                # Target variable: forward return (percentage)
                target = f'target_ret_{horizon}d'
                df_features[target] = (
                    df_features['close'].shift(-horizon) / df_features['close'] - 1.0
                )
                
                # Remove last horizon rows
                df_model = df_features[:-horizon].copy()
                
                X = df_model[feature_cols].values
                y = df_model[target].values
                
                # Time series split
                # ⚡ USE PURGED CV: Prevents data leakage with purging + embargo
                # purge_gap=5: Remove 5 days before test set
                # embargo_td=2: Add 2-day gap after train set
                tscv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=5, embargo_td=2)
                logger.info("✅ Using Purged Time-Series CV (purge=5, embargo=2)")
                
                # Train models
                horizon_models = {}
                
                # 1. XGBoost
                if XGBOOST_AVAILABLE:
                    try:
                        # CRITICAL FIX: Improved XGBoost parameters for realistic predictions
                        # Previous: n_estimators=100 was too few, causing underfitting
                        # Added regularization to prevent overfitting
                        # Added early stopping for optimal model complexity
                        xgb_model = xgb.XGBRegressor(  # type: ignore[union-attr]
                            n_estimators=500,           # Increased from 100
                            max_depth=8,                # Increased from 6 for more expressiveness
                            learning_rate=0.05,         # Decreased from 0.1 for stability
                            subsample=0.8,              # NEW: Row sampling for generalization
                            colsample_bytree=0.8,       # NEW: Column sampling
                            n_jobs=self.train_threads,  # ENV: Parallelism
                            min_child_weight=5,         # NEW: Regularization
                            gamma=0.1,                  # NEW: Pruning parameter
                            reg_alpha=0.1,              # NEW: L1 regularization
                            reg_lambda=1.0,             # NEW: L2 regularization
                            random_state=42,
                            eval_metric='rmse'          # NEW: Explicit metric
                        )
                        
                        # Cross-validation (on returns) + OOF collection for meta-learner
                        xgb_scores = []
                        xgb_oof_preds = np.zeros(len(X))  # OOF predictions storage
                        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                            try:
                                X_train, X_val = X[train_idx], X[val_idx]
                                y_train, y_val = y[train_idx], y[val_idx]
                                
                                # ⚡ FIX: Skip early stopping if insufficient validation data
                                if len(val_idx) >= self.early_stop_min_val:
                                    # Use early stopping with eval_set
                                    xgb_model.set_params(early_stopping_rounds=self.early_stop_rounds)
                                    xgb_model.fit(
                                        X_train, y_train,
                                        eval_set=[(X_val, y_val)],
                                        verbose=False
                                    )
                                else:
                                    # No early stopping for insufficient data
                                    xgb_model.set_params(early_stopping_rounds=None)
                                    xgb_model.fit(X_train, y_train)
                                
                                pred = xgb_model.predict(X_val)
                                xgb_oof_preds[val_idx] = pred  # ⚡ Save OOF predictions!
                                score = r2_score(y_val, pred)
                                xgb_scores.append(score)
                                logger.info(f"XGBoost fold {fold}: R² = {score:.3f}")
                            except Exception as e:
                                logger.error(f"XGBoost fold {fold} error: {e}")
                                # Don't raise - continue with other folds
                        
                        # Final training with seed bagging (variance reduction)
                        if self.enable_seed_bagging and self.n_seeds > 1:
                            # ⚡ SEED BAGGING: Train with multiple seeds and average
                            seed_predictions = []
                            for seed in self.base_seeds:
                                try:
                                    xgb_model.set_params(random_state=seed, early_stopping_rounds=None)
                                    xgb_model.fit(X, y)
                                    pred = xgb_model.predict(X[-100:])
                                    seed_predictions.append(pred)
                                except Exception as e:
                                    logger.error(f"XGBoost seed {seed} error: {e}")
                            
                            if seed_predictions:
                                xgb_pred = np.mean(seed_predictions, axis=0)  # Average across seeds
                                logger.info(f"XGBoost: Seed bagging with {len(seed_predictions)} seeds")
                            else:
                                # Fallback to single seed
                                xgb_model.set_params(random_state=42, early_stopping_rounds=None)
                                xgb_model.fit(X, y)
                                xgb_pred = xgb_model.predict(X[-100:])
                        else:
                            # Original: Single seed (faster)
                            try:
                                xgb_model.set_params(early_stopping_rounds=None)
                            except Exception:
                                pass
                            xgb_model.fit(X, y)
                            xgb_pred = xgb_model.predict(X[-100:])  # Last 100 returns for validation
                        
                        # CRITICAL FIX: R² to confidence conversion
                        raw_r2 = np.mean(xgb_scores)
                        confidence = self._r2_to_confidence(raw_r2)
                        
                        horizon_models['xgboost'] = {
                            'model': xgb_model,
                            'score': confidence,  # Confidence [0-1]
                            'raw_r2': float(raw_r2),  # Keep raw R² for debugging
                            'rmse': float(np.sqrt(mean_squared_error(y[-100:], xgb_pred))),
                            'mape': float(self._smape(y[-100:], xgb_pred)),
                        }
                        
                        # Feature importance
                        self.feature_importance[f"{symbol}_{horizon}d_xgb"] = dict(
                            zip(feature_cols, xgb_model.feature_importances_)
                        )
                        
                        logger.info(f"XGBoost {horizon}D - R²: {raw_r2:.3f} → Confidence: {confidence:.3f}")
                        
                    except Exception as e:
                        logger.error(f"XGBoost eğitim hatası: {e}")
                
                # 2. LightGBM
                if LIGHTGBM_AVAILABLE:
                    try:
                        # ✨ IMPROVED: Optimized hyperparameters (matched with XGBoost quality)
                        lgb_model = lgb.LGBMRegressor(  # type: ignore[union-attr]
                            n_estimators=500,           # Increased from 100
                            max_depth=8,                # Increased from 6
                            learning_rate=0.05,         # Decreased from 0.1 for stability
                            num_leaves=63,              # ✨ TUNED: Increased from 31
                            min_data_in_leaf=15,        # ✨ TUNED: Decreased from 20
                            num_threads=self.train_threads,  # ENV: Parallelism
                            subsample=0.8,              # NEW: Row sampling
                            colsample_bytree=0.8,       # NEW: Feature sampling
                            reg_alpha=0.1,              # NEW: L1 regularization
                            reg_lambda=1.0,             # NEW: L2 regularization
                            random_state=42,
                            n_jobs=self.train_threads,
                            verbose=-1
                        )
                        
                        # Cross-validation (on returns) + OOF
                        lgb_scores = []
                        lgb_oof_preds = np.zeros(len(X))
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]

                            if len(val_idx) >= self.early_stop_min_val:
                                # LightGBM sklearn wrapper: use callbacks for early stopping
                                lgb_model.fit(
                                    X_train, y_train,
                                    eval_set=[(X_val, y_val)],
                                    eval_metric='rmse',
                                    callbacks=[lgb.early_stopping(self.early_stop_rounds, verbose=False)],  # type: ignore[union-attr]
                                )
                            else:
                                lgb_model.fit(X_train, y_train)

                            pred = lgb_model.predict(X_val)
                            lgb_oof_preds[val_idx] = pred  # Save OOF
                            score = r2_score(y_val, pred)
                            lgb_scores.append(score)
                        
                        # Final training with seed bagging
                        if self.enable_seed_bagging and self.n_seeds > 1:
                            # ⚡ SEED BAGGING: Train with multiple seeds and average
                            seed_predictions = []
                            for seed in self.base_seeds:
                                try:
                                    lgb_model.set_params(random_state=seed)
                                    lgb_model.fit(X, y)
                                    pred = lgb_model.predict(X[-100:])
                                    seed_predictions.append(pred)
                                except Exception as e:
                                    logger.error(f"LightGBM seed {seed} error: {e}")
                            
                            if seed_predictions:
                                lgb_pred = np.mean(seed_predictions, axis=0)
                                logger.info(f"LightGBM: Seed bagging with {len(seed_predictions)} seeds")
                            else:
                                lgb_model.set_params(random_state=42)
                                lgb_model.fit(X, y)
                                lgb_pred = lgb_model.predict(X[-100:])
                        else:
                            # Original: Single seed
                            lgb_model.fit(X, y)
                            lgb_pred = lgb_model.predict(X[-100:])
                        
                        # CRITICAL FIX: R² to confidence conversion
                        raw_r2 = np.mean(lgb_scores)
                        confidence = self._r2_to_confidence(raw_r2)
                        
                        horizon_models['lightgbm'] = {
                            'model': lgb_model,
                            'score': confidence,  # Confidence [0-1]
                            'raw_r2': float(raw_r2),
                            'rmse': float(np.sqrt(mean_squared_error(y[-100:], lgb_pred))),
                            'mape': float(self._smape(y[-100:], lgb_pred)),
                        }
                        
                        # Feature importance
                        self.feature_importance[f"{symbol}_{horizon}d_lgb"] = dict(
                            zip(feature_cols, lgb_model.feature_importances_)
                        )
                        
                        logger.info(f"LightGBM {horizon}D - R²: {raw_r2:.3f} → Confidence: {confidence:.3f}")
                        
                    except Exception as e:
                        logger.error(f"LightGBM eğitim hatası: {e}")
                
                # 3. CatBoost
                if CATBOOST_AVAILABLE:
                    try:
                        # ✨ IMPROVED: Optimized hyperparameters (matched with XGBoost quality)
                        cat_model = cb.CatBoostRegressor(  # type: ignore[union-attr]
                            iterations=500,             # Increased from 100
                            depth=8,                    # Increased from 6
                            learning_rate=0.05,         # Decreased from 0.1
                            l2_leaf_reg=2.0,            # ✨ TUNED: Decreased from 3.0
                            border_count=128,           # NEW: Optimal splits
                            thread_count=self.train_threads,  # ENV: Parallelism
                            subsample=0.8,              # NEW: Row sampling
                            rsm=0.8,                    # NEW: Feature sampling (Random Subspace Method)
                            random_seed=42,
                            allow_writing_files=False,
                            train_dir=self.catboost_train_dir,
                            logging_level='Silent',
                            od_type='Iter',
                            od_wait=self.early_stop_rounds
                        )
                        
                        # Cross-validation (on returns) + OOF
                        cat_scores = []
                        cat_oof_preds = np.zeros(len(X))
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]

                            if len(val_idx) >= self.early_stop_min_val:
                                cat_model.fit(
                                    X_train, y_train,
                                    eval_set=(X_val, y_val),
                                    use_best_model=True,
                                    verbose=False
                                )
                            else:
                                cat_model.fit(X_train, y_train, verbose=False)

                            pred = cat_model.predict(X_val)
                            cat_oof_preds[val_idx] = pred  # Save OOF
                            score = r2_score(y_val, pred)
                            cat_scores.append(score)
                        
                        # Final training with seed bagging
                        if self.enable_seed_bagging and self.n_seeds > 1:
                            # ⚡ SEED BAGGING: Train with multiple seeds and average
                            # CatBoost needs NEW model for each seed (can't change params after fit)
                            seed_predictions = []
                            for seed in self.base_seeds:
                                try:
                                    # Create NEW CatBoost model for each seed
                                    cat_seed_model = cb.CatBoostRegressor(  # type: ignore[union-attr]
                                        iterations=500,
                                        depth=8,
                                        learning_rate=0.05,
                                        l2_leaf_reg=2.0,
                                        border_count=128,
                                        thread_count=self.train_threads,
                                        random_seed=seed,  # Different seed!
                                        logging_level='Silent',  # Use only logging_level (not verbose!)
                                        allow_writing_files=False,
                                        train_dir=self.catboost_train_dir,
                                        od_type='Iter',
                                        od_wait=self.early_stop_rounds
                                    )
                                    cat_seed_model.fit(X, y)  # logging_level='Silent' already set
                                    pred = cat_seed_model.predict(X[-100:])
                                    seed_predictions.append(pred)
                                except Exception as e:
                                    logger.error(f"CatBoost seed {seed} error: {e}")
                            
                            if seed_predictions:
                                cat_pred = np.mean(seed_predictions, axis=0)
                                logger.info(f"CatBoost: Seed bagging with {len(seed_predictions)} seeds")
                            else:
                                cat_model.fit(X, y, verbose=False)
                                cat_pred = cat_model.predict(X[-100:])
                        else:
                            # Original: Single seed
                            cat_model.fit(X, y, verbose=False)
                            cat_pred = cat_model.predict(X[-100:])
                        
                        # CRITICAL FIX: R² to confidence conversion
                        raw_r2 = np.mean(cat_scores)
                        confidence = self._r2_to_confidence(raw_r2)
                        
                        horizon_models['catboost'] = {
                            'model': cat_model,
                            'score': confidence,  # Confidence [0-1]
                            'raw_r2': float(raw_r2),
                            'rmse': float(np.sqrt(mean_squared_error(y[-100:], cat_pred))),
                            'mape': float(self._smape(y[-100:], cat_pred)),
                        }
                        
                        # Feature importance
                        self.feature_importance[f"{symbol}_{horizon}d_cat"] = dict(
                            zip(feature_cols, cat_model.feature_importances_)
                        )
                        
                        logger.info(f"CatBoost {horizon}D - R²: {raw_r2:.3f} → Confidence: {confidence:.3f}")
                        
                    except Exception as e:
                        logger.error(f"CatBoost eğitim hatası: {e}")
                
                # ⚡ META-LEARNER: Train Ridge on OOF predictions
                if self.enable_meta_stacking and len(horizon_models) >= 2:
                    try:
                        # Collect OOF predictions from all models (check if exist)
                        oof_list = []
                        if 'xgboost' in horizon_models:
                            try:
                                oof_list.append(xgb_oof_preds)  # type: ignore[name-defined]
                            except NameError:
                                pass
                        if 'lightgbm' in horizon_models:
                            try:
                                oof_list.append(lgb_oof_preds)  # type: ignore[name-defined]
                            except NameError:
                                pass
                        if 'catboost' in horizon_models:
                            try:
                                oof_list.append(cat_oof_preds)  # type: ignore[name-defined]
                            except NameError:
                                pass
                        
                        if len(oof_list) >= 2:
                            # Stack OOF predictions as features
                            meta_X = np.column_stack(oof_list)  # Shape: (n_samples, n_models)
                            meta_y = y  # True targets
                            
                            # Train Ridge meta-learner
                            from sklearn.linear_model import Ridge
                            meta_model = Ridge(alpha=1.0)
                            meta_model.fit(meta_X, meta_y)
                            
                            # Store meta-learner
                            meta_key = f"{symbol}_{horizon}d_meta"
                            self.meta_learners[meta_key] = meta_model
                            
                            logger.info(f"✅ Meta-learner trained for {symbol} {horizon}d (OOF-based Ridge)")
                    except Exception as e:
                        logger.error(f"Meta-learner training error: {e}")
                
                # Store models and results
                self.models[f"{symbol}_{horizon}d"] = horizon_models
                results[f"{horizon}d"] = horizon_models
                
                # Store feature columns
                self.feature_columns = feature_cols
            
            # Save models
            self.save_enhanced_models(symbol)
            
            # Store performance
            self.model_performance[symbol] = results
            
            # Auto-backtest if enabled
            backtest_enabled = str(os.getenv('ENABLE_AUTO_BACKTEST', 'True')).lower() == 'true'
            if backtest_enabled:
                try:
                    from bist_pattern.ml.ml_backtester import get_ml_backtester
                    backtester = get_ml_backtester()
                    
                    backtest_results = backtester.backtest_model(
                        symbol=symbol,
                        model_predictor=self,  # self has predict_enhanced method
                        historical_data=data,
                        horizons=[f"{h}d" for h in self.prediction_horizons]  # Convert to string format
                    )
                    
                    # Store backtest results
                    if backtest_results.get('status') == 'success':
                        overall = backtest_results.get('overall', {})
                        results['backtest'] = {
                            'sharpe_ratio': overall.get('avg_sharpe_ratio', 0.0),
                            'mape': overall.get('avg_mape', 0.0),
                            'hit_rate': overall.get('avg_hit_rate', 0.0),
                            'quality': overall.get('quality', 'UNKNOWN')
                        }
                        
                        logger.info(
                            f"📊 Backtest {symbol}: Sharpe={overall.get('avg_sharpe_ratio', 0):.2f}, "
                            f"Hit Rate={overall.get('avg_hit_rate', 0):.1%}, "
                            f"Quality={overall.get('quality', 'UNKNOWN')}"
                        )
                        
                        # Warn if poor performance
                        min_sharpe = float(os.getenv('BACKTEST_MIN_SHARPE', '0.3'))
                        if overall.get('avg_sharpe_ratio', 0) < min_sharpe:
                            logger.warning(
                                f"⚠️ {symbol} model has low Sharpe ratio: "
                                f"{overall.get('avg_sharpe_ratio', 0):.2f} < {min_sharpe}"
                            )
                    
                except Exception as e:
                    logger.warning(f"Backtest error for {symbol}: {e}")
            
            logger.info(f"✅ {symbol} enhanced model eğitimi tamamlandı")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced model eğitim hatası: {e}")
            return False
    
    def predict_enhanced(self, symbol, current_data, sentiment_score=None):
        """Enhanced predictions with optional sentiment adjustment"""
        try:
            # Auto-load models for this symbol if not already loaded
            if not self.feature_columns or len(self.models) == 0:
                logger.info(f"🔄 {symbol}: Auto-loading trained models...")
                if self.has_trained_models(symbol):
                    loaded = self.load_trained_models(symbol)
                    if not loaded:
                        logger.warning(f"⚠️ {symbol}: Failed to load trained models")
                        return None
                    logger.info(f"✅ {symbol}: Models loaded successfully ({len(self.feature_columns)} features)")
                else:
                    logger.warning(f"⚠️ {symbol}: No trained models found")
                    return None
            
            # Feature engineering
            df_features = self.create_advanced_features(current_data, symbol=symbol)
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
                            # Predict forward return and map to forward price
                            pred_ret = float(model_info['model'].predict(latest_features)[0])
                            current_px = float(current_data['close'].iloc[-1])
                            pred = current_px * (1.0 + pred_ret)

                            model_predictions[model_name] = {
                                'prediction': float(pred),
                                'confidence': float(model_info['score']),
                                'rmse': float(model_info['rmse']),
                                'mape': float(model_info['mape'])
                            }
                        except Exception as e:
                            logger.error(f"{model_name} prediction error: {e}")
                    
                    # Ensemble prediction (weighted by performance OR meta-stacking)
                    if model_predictions:
                        weights = [info['confidence'] for info in model_predictions.values()]
                        predictions_list = [info['prediction'] for info in model_predictions.values()]
                        
                        # ⚡ NEW: Meta-Stacking with Ridge Learner (if enabled)
                        if self.enable_meta_stacking and len(predictions_list) >= 2:
                            try:
                                # Meta-features: base predictions as features
                                meta_key = f"{symbol}_{horizon}d_meta"
                                
                                # Check if meta-learner exists (trained during model training)
                                if meta_key in self.meta_learners:
                                    meta_model = self.meta_learners[meta_key]
                                    # Stack predictions as features
                                    meta_X = np.array(predictions_list).reshape(1, -1)
                                    ensemble_pred = float(meta_model.predict(meta_X)[0])
                                    avg_confidence = np.mean(weights) * 1.1  # Meta-stacking bonus
                                    logger.debug(f"Meta-stacking used for {symbol} {horizon}d")
                                else:
                                    # Fallback to weighted average if meta-learner not trained yet
                                    ensemble_pred = np.average(predictions_list, weights=weights) if sum(weights) > 0 else float(np.mean(predictions_list))
                                    avg_confidence = np.mean(weights) if sum(weights) > 0 else 0.55
                            except Exception as e:
                                logger.error(f"Meta-stacking error: {e}, falling back to weighted average")
                                ensemble_pred = np.average(predictions_list, weights=weights) if sum(weights) > 0 else float(np.mean(predictions_list))
                                avg_confidence = np.mean(weights) if sum(weights) > 0 else 0.55
                        
                        else:
                            # Original: Performance-based weighting + disagreement penalty
                            if sum(weights) > 0:
                                ensemble_pred = np.average(predictions_list, weights=weights)
                                avg_confidence = np.mean(weights)
                                
                                # ✨ NEW: Reduce confidence if models disagree significantly
                                if len(predictions_list) > 1:
                                    pred_std = np.std(predictions_list)
                                    pred_mean = np.mean(predictions_list)
                                    disagreement_ratio = pred_std / max(abs(pred_mean), 1e-8)
                                    
                                    # Penalize confidence if disagreement > 5%
                                    if disagreement_ratio > 0.05:
                                        disagreement_penalty = min(0.3, disagreement_ratio * 2)
                                        avg_confidence = max(0.25, avg_confidence * (1 - disagreement_penalty))
                                        logger.debug(f"{symbol} {horizon}d: Model disagreement {disagreement_ratio*100:.1f}%, confidence adjusted to {avg_confidence:.2f}")
                            else:
                                ensemble_pred = float(np.mean(predictions_list))
                                # conservative default confidence
                                avg_confidence = 0.55
                        predictions[f"{horizon}d"] = {
                            'ensemble_prediction': float(ensemble_pred),
                            'confidence': float(avg_confidence),
                            'models': model_predictions,
                            'current_price': float(current_data['close'].iloc[-1]),
                            'model_count': len(model_predictions)
                        }
            
            # ⚡ NEW: Sentiment-based prediction adjustment (optional)
            if sentiment_score is not None and isinstance(sentiment_score, (int, float)):
                try:
                    sent = float(sentiment_score)
                    # Dynamic adjustment based on sentiment strength (like Basic ML)
                    if sent > 0.7:  # Strong bullish
                        adjustment_factor = 1.10  # +10%
                    elif sent < 0.3:  # Strong bearish
                        adjustment_factor = 0.90  # -10%
                    elif sent > 0.6:  # Moderate bullish
                        adjustment_factor = 1.05  # +5%
                    elif sent < 0.4:  # Moderate bearish
                        adjustment_factor = 0.95  # -5%
                    else:  # Neutral (0.4-0.6)
                        adjustment_factor = 1.0  # No adjustment
                    
                    if adjustment_factor != 1.0:
                        for h_key in predictions:
                            if 'ensemble_prediction' in predictions[h_key]:
                                original = predictions[h_key]['ensemble_prediction']
                                adjusted = original * adjustment_factor
                                predictions[h_key]['ensemble_prediction'] = float(adjusted)
                                predictions[h_key]['sentiment_adjusted'] = True
                                predictions[h_key]['sentiment_score'] = float(sent)
                        logger.debug(f"Sentiment adjustment applied: {sent:.2f} → {adjustment_factor:.2f}x")
                except Exception as e:
                    logger.error(f"Sentiment adjustment error: {e}")
            
            # ⚡ NEW: Volatility-based calibration (tanh scaling)
            try:
                # Calculate recent volatility
                returns = current_data['close'].pct_change().tail(20)
                vol_20d = float(returns.std()) if len(returns) > 5 else 0.02
                
                # Calibration factor (higher vol → wider predictions)
                for h_key in predictions:
                    pred = predictions[h_key]['ensemble_prediction']
                    current_price = predictions[h_key]['current_price']
                    
                    # Delta in percentage
                    delta_pct = (pred - current_price) / current_price
                    
                    # Tanh calibration (compress extreme predictions)
                    # High vol → less compression
                    # Low vol → more compression
                    scale = 1.0 + vol_20d * 10  # vol 0.02 → scale 1.2, vol 0.05 → scale 1.5
                    calibrated_delta = np.tanh(delta_pct * scale) / scale
                    
                    # Apply calibrated delta
                    calibrated_pred = current_price * (1 + calibrated_delta)
                    predictions[h_key]['ensemble_prediction'] = float(calibrated_pred)
                    predictions[h_key]['calibrated'] = True
                    predictions[h_key]['vol_20d'] = float(vol_20d)
                    
                logger.debug(f"Volatility calibration applied (vol={vol_20d:.4f})")
            except Exception as e:
                logger.error(f"Calibration error: {e}")
            
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
            
            # ⚡ META-LEARNERS kaydet
            symbol_meta = {k: v for k, v in self.meta_learners.items() if k.startswith(symbol)}
            if symbol_meta:
                meta_file = f"{self.model_directory}/{symbol}_meta_learners.pkl"
                joblib.dump(symbol_meta, meta_file)
                logger.debug(f"Meta-learners saved: {len(symbol_meta)} models")
            
            # Feature columns'ı ayrı JSON olarak kaydet (prediction için gerekli)
            try:
                cols_file = f"{self.model_directory}/{symbol}_feature_columns.json"
                with open(cols_file, 'w') as wf:
                    json.dump(list(self.feature_columns or []), wf)
            except Exception:
                pass

            # Eğitim meta verisini kaydet (dashboard için)
            try:
                log_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
                os.makedirs(log_dir, exist_ok=True)
                meta_dir = os.path.join(log_dir, 'model_performance')
                os.makedirs(meta_dir, exist_ok=True)
                meta = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'horizons': self.prediction_horizons,
                    'models': {
                        f"{h}d": {
                            m: {
                                'score': float(info.get('score', 0.0)),
                                'rmse': float(info.get('rmse', 0.0)),
                                'mape': float(info.get('mape', 0.0)),
                            }
                            for m, info in (self.models.get(f"{symbol}_{h}d", {}) or {}).items()
                        }
                        for h in self.prediction_horizons
                    },
                    'feature_count': len(getattr(self, 'feature_columns', [])),
                }
                with open(os.path.join(meta_dir, f"{symbol}.json"), 'w') as wf:
                    json.dump(meta, wf)
            except Exception:
                pass

            logger.info(f"💾 {symbol} enhanced modelleri kaydedildi")
            
        except Exception as e:
            logger.error(f"Enhanced model kaydetme hatası: {e}")

    def has_trained_models(self, symbol: str) -> bool:
        """Diskte bu sembol için en az bir horizon model dosyası var mı?"""
        try:
            for h in self.prediction_horizons:
                for m in ('xgboost', 'lightgbm', 'catboost'):
                    fpath = f"{self.model_directory}/{symbol}_{h}d_{m}.pkl"
                    if os.path.exists(fpath):
                        return True
            return False
        except Exception:
            return False

    def load_trained_models(self, symbol: str) -> bool:
        """Diskten sembol modellerini ve feature kolonlarını yükle (varsa)."""
        try:
            loaded_any = False
            for h in self.prediction_horizons:
                model_key = f"{symbol}_{h}d"
                horizon_models = {}
                for m in ('xgboost', 'lightgbm', 'catboost'):
                    fpath = f"{self.model_directory}/{symbol}_{h}d_{m}.pkl"
                    if os.path.exists(fpath):
                        try:
                            model_obj = joblib.load(fpath)
                            horizon_models[m] = {
                                'model': model_obj,
                                'score': 0.0,
                                'rmse': 0.0,
                                'mape': 0.0,
                            }
                            loaded_any = True
                        except Exception:
                            continue
                if horizon_models:
                    self.models[model_key] = horizon_models

            # Feature columns: JSON → fallback importance → fallback empty
            cols_file = f"{self.model_directory}/{symbol}_feature_columns.json"
            cols = []
            try:
                if os.path.exists(cols_file):
                    with open(cols_file, 'r') as rf:
                        cols = json.load(rf) or []
            except Exception:
                cols = []
            if not cols:
                try:
                    importance_file = f"{self.model_directory}/{symbol}_feature_importance.pkl"
                    if os.path.exists(importance_file):
                        imp = joblib.load(importance_file) or {}
                        # Union of feature names (order fallback: sorted)
                        keys = set()
                        for _k, _v in (imp.items() if isinstance(imp, dict) else []):
                            try:
                                keys.update(list(_v.keys()))
                            except Exception:
                                continue
                        cols = sorted(keys)
                except Exception:
                    cols = []
            if cols:
                self.feature_columns = list(cols)
            
            # ⚡ Load meta-learners
            meta_file = f"{self.model_directory}/{symbol}_meta_learners.pkl"
            if os.path.exists(meta_file):
                try:
                    symbol_meta = joblib.load(meta_file)
                    self.meta_learners.update(symbol_meta)
                    logger.debug(f"Meta-learners loaded: {len(symbol_meta)} models")
                except Exception as e:
                    logger.error(f"Meta-learner load error: {e}")
            
            return loaded_any
        except Exception:
            return False
    
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
    print("✅ Enhanced ML System ready!")
