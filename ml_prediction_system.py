"""
Basic ML Prediction System
- Provides MLPredictionSystem with simple technical features
- train_models(symbol, data) and predict_prices(symbol, data, sentiment_score)
"""
from __future__ import annotations

from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd


class MLPredictionSystem:
    def __init__(self) -> None:
        self.models: Dict[str, Dict] = {}
        self.prediction_horizons = [1, 3, 7, 14, 30]

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'Close' in df.columns:
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        # simple TA
        for p in [5, 10, 20]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
        df['rsi'] = self._rsi(df['close'], 14)
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
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
        df = self.create_technical_features(data).dropna()
        if len(df) < 50:
            return {}
        models = {}
        for h in self.prediction_horizons:
            models[str(h)] = {'type': 'naive_mean', 'window': max(10, h * 5)}
        self.models[symbol] = models
        return models

    def predict_prices(self, symbol: str, data: pd.DataFrame, sentiment_score: Optional[float]) -> Dict[str, Any]:
        df = self.create_technical_features(data).dropna()
        if len(df) == 0:
            return {}
        current = float(df['close'].iloc[-1])
        models = self.models.get(symbol) or self.train_models(symbol, df)
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
            mw = models.get(str(h), {}).get('window', max(10, h * 5))
            base = float(df['close'].tail(mw).mean()) if len(df) >= mw else current
            # horizon ile hafif trend projeksiyonu
            proj = current + (base - current) * min(1.0, h / 30.0)
            if alpha:
                sent = float(sentiment_score) if sentiment_score is not None else 0.5
                proj = proj * (1 + alpha * (sent - 0.5))
            out[f'{h}d'] = {'price': float(proj)}
        out['timestamp'] = datetime.now().isoformat()
        out['model'] = 'basic_naive'
        return out


# Module-level singleton
_ml_system = None


def get_ml_prediction_system():
    global _ml_system
    if _ml_system is None:
        _ml_system = MLPredictionSystem()
    return _ml_system
