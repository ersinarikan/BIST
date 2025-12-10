"""
Basic ML Prediction System
- Provides MLPredictionSystem with simple technical features
- train_models(symbol, data) and predict_prices(symbol, data, sentiment_score)
- NOW WITH PERSISTENCE: Models saved to disk and reloaded
"""
from __future__ import annotations

from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd
import os
import logging

# Persistence support
joblib = None
JOBLIB_AVAILABLE = False
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Optional statsmodels for ETS/ARIMA baseline
ExponentialSmoothing = None
ARIMA = None
STATSMODELS_AVAILABLE = False
try:
    # type: ignore
    from statsmodels.tsa.holtwinters import (  # type: ignore
        ExponentialSmoothing
    )
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    STATSMODELS_AVAILABLE = True
except Exception as e:
    logger.debug(f"statsmodels not available: {e}")
    STATSMODELS_AVAILABLE = False


class MLPredictionSystem:
    def __init__(self) -> None:
        self.models: Dict[str, Dict] = {}
        self.prediction_horizons = [1, 3, 7, 14, 30]

        # Model persistence directory
        cache_dir = os.getenv(
            'ML_MODEL_PATH',
            '/opt/bist-pattern/.cache/enhanced_ml_models'
        )
        self.basic_model_dir = os.path.join(
            os.path.dirname(cache_dir), 'basic_ml_models'
        )
        os.makedirs(self.basic_model_dir, exist_ok=True)

        logger.info(
            f"📊 Basic ML System initialized "
            f"(Real ML: True, Persistence: {JOBLIB_AVAILABLE}, "
            f"ETS/ARIMA: {STATSMODELS_AVAILABLE})"
        )

    def _get_model_path(self, symbol: str) -> str:
        """Get file path for symbol's model"""
        return os.path.join(self.basic_model_dir, f"{symbol}_basic_model.pkl")

    def _load_model_from_disk(self, symbol: str) -> Optional[Dict]:
        """Load model from disk if exists and not too old"""
        if not JOBLIB_AVAILABLE:
            return None

        model_path = self._get_model_path(symbol)
        if not os.path.exists(model_path):
            return None

        try:
            # Check age
            max_age_days = int(os.getenv('ML_MAX_MODEL_AGE_DAYS', '7'))
            mtime = os.path.getmtime(model_path)
            age_days = (datetime.now().timestamp() - mtime) / 86400

            if age_days > max_age_days:
                logger.debug(
                    f"Basic ML model too old for {symbol}: "
                    f"{age_days:.1f} days"
                )
                return None

            # Load model
            models = joblib.load(model_path)  # type: ignore
            logger.debug(f"✅ Basic ML model loaded from disk for {symbol}")
            return models

        except Exception as e:
            logger.debug(f"Failed to load Basic ML model for {symbol}: {e}")
            return None

    def _save_model_to_disk(self, symbol: str, models: Dict) -> bool:
        """Save model to disk"""
        if not JOBLIB_AVAILABLE:
            return False

        try:
            # ✅ FIX: Ensure directory exists and has correct permissions
            os.makedirs(self.basic_model_dir, exist_ok=True)

            # ✅ FIX: Try to set permissions if possible (non-critical if fails)
            try:
                os.chmod(self.basic_model_dir, 0o775)
            except Exception as e:
                logger.debug(
                    f"Failed to chmod basic_model_dir (non-critical): {e}"
                )

            model_path = self._get_model_path(symbol)
            joblib.dump(models, model_path)  # type: ignore
            logger.debug(f"💾 Basic ML model saved to disk for {symbol}")
            return True
        except PermissionError as e:
            logger.warning(
                f"⚠️ Permission denied saving Basic ML model for {symbol}: "
                f"{e}. Model will work in memory only."
            )
            return False
        except Exception as e:
            logger.error(f"❌ Failed to save Basic ML model for {symbol}: {e}")
            return False

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'Close' in df.columns:
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
        # simple TA
        for p in [5, 10, 20]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
        df['rsi'] = self._rsi(df['close'], 14)
        df['macd'] = (
            df['close'].ewm(span=12).mean() -
            df['close'].ewm(span=26).mean()
        )
        df['volatility_10'] = df['close'].pct_change().rolling(10).std()
        return df

    def _rsi(self, series: Any, period: int = 14) -> pd.Series:
        s = pd.Series(series).astype(float)
        delta = s.diff().fillna(0.0)
        up = delta.where(delta > 0, 0.0)
        down = (-delta.where(delta < 0, 0.0)).abs()
        roll_up = up.rolling(int(period), min_periods=1).mean()
        roll_down = down.rolling(int(period), min_periods=1).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return pd.Series(rsi, index=s.index).astype(float)

    def train_models(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Train models for symbol and save to disk"""
        df = self.create_technical_features(data).dropna()
        if len(df) < 50:
            return {}

        models: Dict[str, Dict[str, Any]] = {}
        # Always include a naive mean fallback
        for h in self.prediction_horizons:
            models[str(h)] = {'type': 'naive_mean', 'window': max(10, h * 5)}

        # Train ETS (trend+seasonal) as lightweight baseline if
        # statsmodels available
        if STATSMODELS_AVAILABLE and len(df) >= 60:
            try:
                assert ExponentialSmoothing is not None  # type narrowing
                close = df['close'].astype(float)
                # Weekly-ish seasonal pattern ~5 trading days
                seasonal_periods = 5
                ets_model = ExponentialSmoothing(
                    close, trend='add', seasonal='add',
                    seasonal_periods=seasonal_periods
                )
                ets_fit = ets_model.fit(optimized=True, use_brute=True)
                # ✅ FIX: Calculate confidence from model fit quality
                # (AIC + in-sample metrics)
                # Lower AIC = better fit = higher confidence
                # Also consider in-sample prediction accuracy
                try:
                    aic = (
                        float(ets_fit.aic)
                        if hasattr(ets_fit, 'aic') else None
                    )

                    # Calculate in-sample prediction accuracy (RMSE)
                    try:
                        fitted_values = ets_fit.fittedvalues
                        if (
                            len(fitted_values) > 0 and
                            len(close) == len(fitted_values)
                        ):
                            residuals = close - fitted_values
                            rmse = float(
                                (residuals ** 2).mean() ** 0.5
                            )
                            # Normalize RMSE to confidence: lower RMSE →
                            # higher confidence
                            # Typical RMSE range: 0.1-2.0 (relative to price),
                            # normalize to 0.0-0.1 confidence impact
                            current_price = (
                                float(close.iloc[-1])
                                if len(close) > 0 else 1.0
                            )
                            # RMSE as percentage of price
                            rmse_pct = rmse / max(current_price, 1.0)
                            # Penalty up to 0.1
                            rmse_penalty = min(0.1, max(0.0, rmse_pct * 2))
                        else:
                            # Default penalty if can't calculate
                            rmse_penalty = 0.05
                    except Exception as e:
                        logger.debug(
                            f"Failed to calculate rmse_penalty for ETS: {e}"
                        )
                        rmse_penalty = 0.05  # Default penalty

                    # Combine AIC and RMSE for confidence
                    if aic is not None:
                        # AIC-based confidence: AIC < 1000 → 0.5,
                        # AIC > 2000 → 0.4
                        aic_conf = 0.5 - max(
                            0, min(0.1, (aic - 1000) / 10000)
                        )
                        aic_conf = max(0.4, min(0.5, aic_conf))
                        # Apply RMSE penalty
                        model_quality = max(
                            0.35, min(0.5, aic_conf - rmse_penalty)
                        )
                    else:
                        # No AIC, use RMSE-based confidence
                        model_quality = max(
                            0.35, min(0.45, 0.45 - rmse_penalty)
                        )
                except Exception as e:
                    logger.debug(
                        f"Failed to calculate model_quality for ETS: {e}"
                    )
                    model_quality = 0.45  # Default for ETS

                for h in self.prediction_horizons:
                    fc = float(ets_fit.forecast(steps=h).iloc[-1])
                    models[str(h)] = {
                        'type': 'ets',
                        'forecast': fc,
                        'seasonal_periods': seasonal_periods,
                        # ✅ FIX: Store calculated confidence
                        'confidence': model_quality,
                    }
                logger.debug(
                    f"✅ ETS trained for {symbol} "
                    f"(confidence: {model_quality:.2f})"
                )
            except Exception as e:
                logger.debug(f"ETS training failed for {symbol}: {e}")

        # Optional ARIMA as a secondary baseline for longer horizons
        if STATSMODELS_AVAILABLE and len(df) >= 80:
            try:
                assert ARIMA is not None  # type narrowing
                close = df['close'].astype(float)
                # Simple automatically selected (p,d,q) candidate;
                # conservative
                order = (1, 1, 1)
                arima_fit = ARIMA(close, order=order).fit()
                # ✅ FIX: Calculate confidence from model fit quality
                # (AIC + in-sample metrics)
                try:
                    aic = (
                        float(arima_fit.aic)
                        if hasattr(arima_fit, 'aic') else None
                    )

                    # Calculate in-sample prediction accuracy (RMSE)
                    try:
                        fitted_values = arima_fit.fittedvalues
                        if (
                            len(fitted_values) > 0 and
                            len(close) == len(fitted_values)
                        ):
                            residuals = close - fitted_values
                            rmse = float(
                                (residuals ** 2).mean() ** 0.5
                            )
                            # Normalize RMSE to confidence: lower RMSE →
                            # higher confidence
                            current_price = (
                                float(close.iloc[-1])
                                if len(close) > 0 else 1.0
                            )
                            # RMSE as percentage of price
                            rmse_pct = rmse / max(current_price, 1.0)
                            # Penalty up to 0.1
                            rmse_penalty = min(0.1, max(0.0, rmse_pct * 2))
                        else:
                            # Default penalty if can't calculate
                            rmse_penalty = 0.05
                    except Exception as e:
                        logger.debug(
                            f"Failed to calculate rmse_penalty for ETS: {e}"
                        )
                        rmse_penalty = 0.05  # Default penalty

                    # Combine AIC and RMSE for confidence
                    if aic is not None:
                        # AIC-based confidence: AIC < 1000 → 0.45,
                        # AIC > 2000 → 0.35
                        aic_conf = 0.45 - max(
                            0, min(0.1, (aic - 1000) / 10000)
                        )
                        aic_conf = max(0.35, min(0.45, aic_conf))
                        # Apply RMSE penalty
                        model_quality = max(
                            0.3, min(0.45, aic_conf - rmse_penalty)
                        )
                    else:
                        # No AIC, use RMSE-based confidence
                        model_quality = max(0.3, min(0.4, 0.4 - rmse_penalty))
                except Exception as e:
                    logger.debug(
                        f"Failed to calculate model_quality for ARIMA: {e}"
                    )
                    model_quality = 0.4  # Default for ARIMA

                for h in self.prediction_horizons:
                    fc = float(arima_fit.forecast(steps=h).iloc[-1])
                    # Prefer ETS for short horizons; ARIMA for long
                    if h >= 14:
                        models[str(h)] = {
                            'type': 'arima',
                            'forecast': fc,
                            'order': order,
                            # ✅ FIX: Store calculated confidence
                            'confidence': model_quality,
                        }
                logger.debug(
                    f"✅ ARIMA trained for {symbol} "
                    f"(confidence: {model_quality:.2f})"
                )
            except Exception as e:
                logger.debug(
                    f"ARIMA training failed for {symbol}: {e}"
                )

        # Store in memory
        self.models[symbol] = models

        # Save to disk
        self._save_model_to_disk(symbol, models)

        logger.info(
            f"🧠 Basic ML trained for {symbol}"
        )
        return models

    def predict_prices(
        self, symbol: str, data: pd.DataFrame,
        sentiment_score: Optional[float]
    ) -> Dict[str, Any]:
        """Predict prices - load from disk or train if needed"""
        df = self.create_technical_features(data).dropna()
        if len(df) == 0:
            return {}

        current = float(df['close'].iloc[-1])

        # Try to get models: memory -> disk -> train new
        models = self.models.get(symbol)
        if not models:
            models = self._load_model_from_disk(symbol)
            if models:
                self.models[symbol] = models  # Cache in memory
        if not models:
            models = self.train_models(symbol, df)
        out: Dict[str, Any] = {}
        # CRITICAL FIX: Dynamic sentiment impact based on signal strength
        # Strong sentiment (>0.7 bullish or <0.3 bearish) has 15% impact
        # Weak/neutral sentiment (0.3-0.7) has only 5% impact
        # Previous fixed 2% was ineffective
        alpha = 0.0
        if isinstance(sentiment_score, (int, float)):
            sent = float(sentiment_score)
            if sent > 0.7 or sent < 0.3:  # Strong signal
                alpha = 0.15
            elif 0.4 <= sent <= 0.6:  # Neutral zone
                alpha = 0.02
            else:  # Moderate signal
                alpha = 0.08
        for h in self.prediction_horizons:
            m = models.get(str(h), {})
            mtype = m.get('type')
            proj = None
            confidence = None

            if mtype == 'ets' and isinstance(m.get('forecast'), (int, float)):
                proj = float(m['forecast'])
                # ✅ FIX: Use calculated confidence from model training
                # Default 0.45 for ETS
                confidence = m.get('confidence', 0.45)
            elif mtype == 'arima' and isinstance(
                m.get('forecast'), (int, float)
            ):
                proj = float(m['forecast'])
                # ✅ FIX: Use calculated confidence from model training
                # Default 0.4 for ARIMA
                confidence = m.get('confidence', 0.4)
            else:
                # Naive mean fallback
                mw = m.get('window', max(10, h * 5))
                base = (
                    float(df['close'].tail(mw).mean())
                    if len(df) >= mw else current
                )
                proj = current + (base - current) * min(1.0, h / 30.0)

                # ✅ FIX: Calculate dynamic confidence for naive mean based on
                # data quality
                # Factors: volatility, trend strength, data amount
                try:
                    # Calculate volatility (lower volatility = higher
                    # confidence)
                    returns = df['close'].pct_change().dropna()
                    volatility = (
                        float(returns.std())
                        if len(returns) > 0 else 0.1
                    )
                    # Normalize volatility: 0.01 (1%) = high confidence,
                    # 0.05 (5%) = low confidence
                    # Penalty up to 0.15
                    vol_penalty = min(
                        0.15, max(0.0, (volatility - 0.01) * 3)
                    )

                    # Calculate trend strength (stronger trend = higher
                    # confidence)
                    if len(df) >= 20:
                        short_ma = float(df['close'].tail(10).mean())
                        long_ma = float(df['close'].tail(20).mean())
                        trend_strength = abs(short_ma - long_ma) / max(
                            long_ma, 1.0
                        )
                        # Strong trend (>5%) = bonus, weak trend (<1%) =
                        # penalty
                        trend_bonus = min(
                            0.05, max(-0.05, (trend_strength - 0.01) * 2)
                        )
                    else:
                        trend_bonus = 0.0

                    # Data amount factor (more data = higher confidence)
                    # Bonus up to 0.05 for 200+ days
                    data_factor = min(0.05, len(df) / 200.0)

                    # Base confidence for naive mean
                    base_conf = 0.35
                    # Apply adjustments
                    confidence = (
                        base_conf + data_factor + trend_bonus - vol_penalty
                    )
                    # Clamp to reasonable range [0.25, 0.45]
                    confidence = max(0.25, min(0.45, confidence))
                except Exception as e:
                    logger.debug(
                        f"Failed to calculate confidence: {e}"
                    )
                    # Fallback to default if calculation fails
                    confidence = 0.4

            if alpha:
                sent = (
                    float(sentiment_score)
                    if sentiment_score is not None else 0.5
                )
                proj = proj * (1 + alpha * (sent - 0.5))

            # ✅ FIX: Use calculated confidence from model, or fallback to
            # default
            out[f'{h}d'] = {
                'price': float(proj),
                'confidence': confidence or 0.4
            }
        out['timestamp'] = datetime.now().isoformat()
        out['model'] = (
            'basic_ets_arima' if STATSMODELS_AVAILABLE else 'basic_naive'
        )
        return out


# Module-level singleton
_ml_system = None


def get_ml_prediction_system():
    global _ml_system
    if _ml_system is None:
        _ml_system = MLPredictionSystem()
    return _ml_system
