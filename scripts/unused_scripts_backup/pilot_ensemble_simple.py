#!/usr/bin/env python3
"""
Simple Ensemble Pilot: XGBoost + LightGBM

Tests 3 strategies:
1. XGBoost only (baseline)
2. LightGBM only
3. Simple Average (XGBoost + LightGBM)

Output: CSV comparing performance
"""

import os
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, '/opt/bist-pattern')
from enhanced_ml_system import EnhancedMLSystem  # noqa: E402

# Try to import LightGBM
try:
    import lightgbm  # noqa: F401
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available, install with: pip install lightgbm")


SYMS = ['GARAN']
HORIZONS = [1, 3, 7]


def fetch_prices(engine, symbol: str) -> pd.DataFrame:
    """Fetch all available historical prices"""
    q = text(
        """
        SELECT p.date, p.open_price, p.high_price, p.low_price, p.close_price, p.volume
        FROM stock_prices p
        JOIN stocks s ON s.id = p.stock_id
        WHERE s.symbol = :sym
        ORDER BY p.date DESC
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"sym": symbol}).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([
        {
            'date': r[0],
            'open': float(r[1]),
            'high': float(r[2]),
            'low': float(r[3]),
            'close': float(r[4]),
            'volume': float(r[5]) if r[5] is not None else 0.0,
        }
        for r in rows
    ])
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'close', 'volume']]


def compute_returns(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Compute forward returns"""
    close = df['close'].values
    ret = np.full(len(close), np.nan, dtype=float)
    for i in range(len(close) - horizon):
        if close[i] > 0:
            ret[i] = (close[i + horizon] / close[i]) - 1.0
    return ret


def dirhit(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.005) -> float:
    """Directional hit rate"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    dir_true = np.where(np.abs(yt) < threshold, 0, np.sign(yt))
    dir_pred = np.where(np.abs(yp) < threshold, 0, np.sign(yp))
    return float(np.mean(dir_true == dir_pred) * 100.0)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan


def run() -> None:
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM required for this pilot")
        sys.exit(1)
    
    os.environ.setdefault('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url)
    
    # Simple test: Use XGBoost from EnhancedMLSystem
    # Add LightGBM as alternative
    # Compare performance
    
    print("🚀 Simple Ensemble Pilot: XGBoost vs LightGBM vs Average")
    print("="*80)
    print("Note: This is a proof-of-concept")
    print("Full implementation requires EnhancedMLSystem modification")
    print("="*80)
    
    for sym in SYMS:
        print(f"\n📊 {sym}")
        df = fetch_prices(engine, sym)
        if df is None or df.empty or len(df) < 400:
            print("❌ Insufficient data")
            continue
        
        # Split: 70% train, 30% test
        split = int(len(df) * 0.7)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]
        
        print(f"  Train: {len(train_df)} days, Test: {len(test_df)} days")
        
        # Compute returns (for future use)
        # y_true = {h: compute_returns(df, h) for h in HORIZONS}
        
        # Strategy 1: XGBoost (via EnhancedMLSystem)
        print("\n  🔵 Strategy 1: XGBoost (baseline)")
        ml = EnhancedMLSystem()
        ok = ml.train_enhanced_models(sym, train_df)
        
        if ok:
            for h in HORIZONS:
                # Simplified prediction (just for demo)
                # In real implementation, use walk-forward
                print(f"    {h}d: XGBoost trained ✅")
        
        # Strategy 2: LightGBM
        print("\n  🟢 Strategy 2: LightGBM")
        print("    Note: Requires EnhancedMLSystem modification")
        print("    Placeholder for now")
        
        # Strategy 3: Ensemble
        print("\n  🟣 Strategy 3: Simple Average (XGB + LGB)")
        print("    Note: Requires both models trained")
        print("    Placeholder for now")
    
    print("\n" + "="*80)
    print("✅ Pilot complete")
    print("Next step: Modify EnhancedMLSystem to support multiple model types")
    print("="*80)


if __name__ == '__main__':
    run()
