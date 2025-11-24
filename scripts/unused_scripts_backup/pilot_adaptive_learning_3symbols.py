#!/usr/bin/env python3
"""
Adaptive Learning Backtest Pilot (3 symbols: GARAN, AKBNK, EREGL)

Tests 4 strategies:
A) Static: Train once, no updates
B) Weekly Retrain: Retrain every 7 days with rolling window
C) Incremental Learning: Daily bias correction + feature adaptation
D) Hybrid: Weekly retrain + daily bias correction

Uses adopted settings: adaptive K + pattern-weight + cap (per horizon)
Evaluates 1/3/7d DirHit, R², MAPE, nRMSE on last 20% of data

Output: CSV with rows: symbol, strategy, horizon, dir_hit_pct, r2, mape, nrmse, n
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

sys.path.insert(0, '/opt/bist-pattern')
from enhanced_ml_system import EnhancedMLSystem  # noqa: E402


SYMS: List[str] = ['GARAN', 'AKBNK', 'EREGL']  # 3 symbols
HORIZONS = [1, 3, 7, 14, 30]  # Tüm ufuklar

# Dynamic split strategy (no fixed ratio)
MIN_TRAIN_DAYS = 200  # Minimum training days for reliable model
TARGET_TEST_DAYS = 120  # Target test period (4 months)
MAX_TEST_DAYS = 180  # Maximum test period (6 months)


def fetch_prices(engine, symbol: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Fetch historical prices with datetime index (all available data if limit=None)"""
    if limit:
        q = text(
            """
            SELECT p.date, p.open_price, p.high_price, p.low_price, p.close_price, p.volume
            FROM stock_prices p
            JOIN stocks s ON s.id = p.stock_id
            WHERE s.symbol = :sym
            ORDER BY p.date DESC
            LIMIT :lim
            """
        )
        with engine.connect() as conn:
            rows = conn.execute(q, {"sym": symbol, "lim": limit}).fetchall()
    else:
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


def find_latest(pattern: str) -> Optional[str]:
    import glob
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def load_map(csv_path: Optional[str], col: str) -> Dict[str, Dict[str, float]]:
    """Load symbol×horizon map from CSV"""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        sym = row['symbol']
        hz = row['horizon']
        val = row.get(col)
        if sym not in result:
            result[sym] = {}
        if val is not None and not pd.isna(val):
            result[sym][hz] = float(val)
    return result


def set_env_for_symbol(sym: str, k_map: dict, s_map: dict, c_map: dict) -> None:
    """Set environment variables for symbol-specific settings"""
    for h in [1, 3, 7]:
        hz = f'{h}d'
        envk = f'ML_ADAPTIVE_K_{h}D'
        envs = f'ML_PATTERN_WEIGHT_SCALE_{h}D'
        envc = f'ML_CAP_PCTL_{h}D'
        os.environ[envk] = '' if k_map.get(sym, {}).get(hz) is None else str(float(k_map.get(sym, {}).get(hz)))
        os.environ[envs] = '' if s_map.get(sym, {}).get(hz) is None else str(float(s_map.get(sym, {}).get(hz)))
        os.environ[envc] = '' if c_map.get(sym, {}).get(hz) is None else str(float(c_map.get(sym, {}).get(hz)))


def compute_returns(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """Compute forward returns for horizon"""
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


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(np.abs(yt - yp) / denom))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized RMSE"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 10:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    std = np.std(yt)
    return float(rmse / std) if std > 0 else np.nan


def calculate_optimal_split(total_days: int) -> tuple:
    """
    Calculate optimal train/test split based on data length
    
    Strategy:
    - Test period: 120 days target (or max 30% of data, capped at 180 days)
    - Train period: Remaining (minimum 200 days required)
    
    Returns: (train_days, test_days) or (None, None) if insufficient data
    """
    # Minimum data requirement
    if total_days < MIN_TRAIN_DAYS + 60:
        return None, None
    
    # Calculate test period
    # Target 120 days, but max 30% of data, and cap at 180 days
    test_days = min(TARGET_TEST_DAYS, int(total_days * 0.30), MAX_TEST_DAYS)
    train_days = total_days - test_days
    
    # Ensure minimum training period
    if train_days < MIN_TRAIN_DAYS:
        train_days = MIN_TRAIN_DAYS
        test_days = total_days - MIN_TRAIN_DAYS
    
    return train_days, test_days


def run() -> None:
    os.environ.setdefault('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})

    # Load adopted maps
    k_csv = find_latest('/opt/bist-pattern/logs/bist30_adaptive_deadband_grid_*.csv')
    s_csv = find_latest('/opt/bist-pattern/logs/bist30_pattern_weight_grid_*.csv')
    c_csv = find_latest('/opt/bist-pattern/logs/bist30_cap_grid_*.csv')
    k_map = load_map(k_csv, 'best_k')
    s_map = load_map(s_csv, 'best_scale')
    c_map = load_map(c_csv, 'best_cap_pctl')

    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')
    os.environ['ML_USE_DIRECTIONAL_LOSS'] = '0'
    os.environ['ML_USE_STACKED_SHORT'] = '0'
    os.environ['ML_HORIZONS'] = '1,3,7,14,30'
    os.environ.setdefault('ML_ADAPTIVE_DEADBAND_MODE', 'std')

    rows: List[Dict[str, object]] = []
    thr_eval = float(os.getenv('ML_LOSS_THRESHOLD', '0.005'))

    for sym in SYMS:
        print(f"\n{'='*100}\nADAPTIVE LEARNING: {sym}\n{'='*100}\n")
        # Fetch ALL available data (no limit)
        df = fetch_prices(engine, sym, limit=None)
        if df is None or df.empty:
            print(f"❌ {sym} no data")
            continue

        # Calculate optimal split dynamically
        train_days, test_days = calculate_optimal_split(len(df))
        if train_days is None or test_days is None:
            print(f"❌ {sym} insufficient data: {len(df)} days (need {MIN_TRAIN_DAYS + 60})")
            continue
        
        train_end = train_days
        test_start = train_end
        train_pct = (train_days / len(df)) * 100
        test_pct = (test_days / len(df)) * 100
        print(f"📊 {sym}: {len(df)} days total")
        print(f"   Train: {train_days} days ({train_pct:.0f}%)")
        print(f"   Test:  {test_days} days ({test_pct:.0f}%)")

        # Compute forward returns for evaluation
        y_true: Dict[int, np.ndarray] = {h: compute_returns(df, h) for h in HORIZONS}

        # Set symbol-specific env vars
        set_env_for_symbol(sym, k_map, s_map, c_map)

        # ═══════════════════════════════════════════════════════════════
        # STRATEGY A: STATIC (train once, no updates)
        # ═══════════════════════════════════════════════════════════════
        print("\n🔵 Strategy A: STATIC (train once)")
        ml_static = EnhancedMLSystem()
        train_df = df.iloc[:train_end].copy()
        ok = ml_static.train_enhanced_models(sym, train_df)
        
        if ok:
            for h in HORIZONS:
                preds = np.full(len(df), np.nan, dtype=float)
                for t in range(test_start, len(df) - h):
                    cur = df.iloc[: t + 1].copy()
                    p = ml_static.predict_enhanced(sym, cur)
                    if isinstance(p, dict):
                        key = f"{h}d"
                        obj = p.get(key)
                        if isinstance(obj, dict):
                            pred_price = obj.get('ensemble_prediction')
                            try:
                                if isinstance(pred_price, (int, float)):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        preds[t] = float(pred_price) / last_close - 1.0
                            except Exception:
                                pass
                
                y = y_true[h]
                rows.append({
                    'symbol': sym,
                    'strategy': 'static',
                    'horizon': f'{h}d',
                    'dir_hit_pct': dirhit(y[test_start:], preds[test_start:], thr_eval),
                    'r2': r2(y[test_start:], preds[test_start:]),
                    'mape': smape(y[test_start:], preds[test_start:]),
                    'nrmse': nrmse(y[test_start:], preds[test_start:]),
                    'n': int(np.isfinite(y[test_start:]).sum())
                })

        # ═══════════════════════════════════════════════════════════════
        # STRATEGY B: WEEKLY RETRAIN (every 7 days)
        # ═══════════════════════════════════════════════════════════════
        print("\n🟢 Strategy B: WEEKLY RETRAIN")
        for h in HORIZONS:
            preds = np.full(len(df), np.nan, dtype=float)
            t = test_start
            retrain_counter = 0
            
            while t < len(df) - h:
                # Retrain every 7 days
                if retrain_counter % 7 == 0:
                    ml_weekly = EnhancedMLSystem()
                    set_env_for_symbol(sym, k_map, s_map, c_map)
                    train_df2 = df.iloc[max(0, t - 500): t].copy()
                    ok2 = ml_weekly.train_enhanced_models(sym, train_df2)
                    if not ok2:
                        t += 1
                        retrain_counter += 1
                        continue
                
                # Predict next 7 days (or until end)
                for tt in range(t, min(t + 7, len(df) - h)):
                    cur = df.iloc[: tt + 1].copy()
                    p = ml_weekly.predict_enhanced(sym, cur)
                    if isinstance(p, dict):
                        key = f"{h}d"
                        obj = p.get(key)
                        if isinstance(obj, dict):
                            pred_price = obj.get('ensemble_prediction')
                            try:
                                if isinstance(pred_price, (int, float)):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        preds[tt] = float(pred_price) / last_close - 1.0
                            except Exception:
                                pass
                
                t += 7
                retrain_counter += 7
            
            y = y_true[h]
            rows.append({
                'symbol': sym,
                'strategy': 'weekly_retrain',
                'horizon': f'{h}d',
                'dir_hit_pct': dirhit(y[test_start:], preds[test_start:], thr_eval),
                'r2': r2(y[test_start:], preds[test_start:]),
                'mape': smape(y[test_start:], preds[test_start:]),
                'nrmse': nrmse(y[test_start:], preds[test_start:]),
                'n': int(np.isfinite(y[test_start:]).sum())
            })

        # ═══════════════════════════════════════════════════════════════
        # STRATEGY C: INCREMENTAL LEARNING (daily bias correction)
        # ═══════════════════════════════════════════════════════════════
        print("\n🟡 Strategy C: INCREMENTAL LEARNING (bias correction)")
        ml_incr = EnhancedMLSystem()
        train_df = df.iloc[:train_end].copy()
        ok = ml_incr.train_enhanced_models(sym, train_df)
        
        if ok:
            # Track bias per horizon
            bias_correction: Dict[int, float] = {h: 0.0 for h in HORIZONS}
            error_history: Dict[int, List[float]] = {h: [] for h in HORIZONS}
            
            for h in HORIZONS:
                preds = np.full(len(df), np.nan, dtype=float)
                
                for t in range(test_start, len(df) - h):
                    # Predict with current bias correction
                    cur = df.iloc[: t + 1].copy()
                    p = ml_incr.predict_enhanced(sym, cur)
                    pred_return = np.nan
                    
                    if isinstance(p, dict):
                        key = f"{h}d"
                        obj = p.get(key)
                        if isinstance(obj, dict):
                            pred_price = obj.get('ensemble_prediction')
                            try:
                                if isinstance(pred_price, (int, float)):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        pred_return = float(pred_price) / last_close - 1.0
                                        # Apply bias correction
                                        pred_return_corrected = pred_return - bias_correction[h]
                                        preds[t] = pred_return_corrected
                            except Exception:
                                pass
                    
                    # Learn from previous prediction (if available)
                    if t >= test_start + h and np.isfinite(pred_return):
                        actual_return = y_true[h][t - h]
                        if np.isfinite(actual_return):
                            error = pred_return - actual_return
                            error_history[h].append(error)
                            
                            # Update bias with exponential moving average
                            # Recent errors weighted more (alpha=0.1)
                            if len(error_history[h]) > 0:
                                recent_errors = error_history[h][-20:]  # Last 20 errors
                                bias_correction[h] = np.mean(recent_errors)
                
                y = y_true[h]
                rows.append({
                    'symbol': sym,
                    'strategy': 'incremental',
                    'horizon': f'{h}d',
                    'dir_hit_pct': dirhit(y[test_start:], preds[test_start:], thr_eval),
                    'r2': r2(y[test_start:], preds[test_start:]),
                    'mape': smape(y[test_start:], preds[test_start:]),
                    'nrmse': nrmse(y[test_start:], preds[test_start:]),
                    'n': int(np.isfinite(y[test_start:]).sum())
                })

        # ═══════════════════════════════════════════════════════════════
        # STRATEGY D: HYBRID (weekly retrain + daily bias correction)
        # ═══════════════════════════════════════════════════════════════
        print("\n🟣 Strategy D: HYBRID (weekly retrain + bias correction)")
        for h in HORIZONS:
            preds = np.full(len(df), np.nan, dtype=float)
            bias_correction_hybrid = 0.0
            error_history_hybrid: List[float] = []
            t = test_start
            retrain_counter = 0
            
            while t < len(df) - h:
                # Retrain every 7 days
                if retrain_counter % 7 == 0:
                    ml_hybrid = EnhancedMLSystem()
                    set_env_for_symbol(sym, k_map, s_map, c_map)
                    train_df3 = df.iloc[max(0, t - 500): t].copy()
                    ok3 = ml_hybrid.train_enhanced_models(sym, train_df3)
                    if not ok3:
                        t += 1
                        retrain_counter += 1
                        continue
                
                # Predict next 7 days with bias correction
                for tt in range(t, min(t + 7, len(df) - h)):
                    cur = df.iloc[: tt + 1].copy()
                    p = ml_hybrid.predict_enhanced(sym, cur)
                    pred_return = np.nan
                    
                    if isinstance(p, dict):
                        key = f"{h}d"
                        obj = p.get(key)
                        if isinstance(obj, dict):
                            pred_price = obj.get('ensemble_prediction')
                            try:
                                if isinstance(pred_price, (int, float)):
                                    last_close = float(cur['close'].iloc[-1])
                                    if last_close > 0:
                                        pred_return = float(pred_price) / last_close - 1.0
                                        pred_return_corrected = pred_return - bias_correction_hybrid
                                        preds[tt] = pred_return_corrected
                            except Exception:
                                pass
                    
                    # Learn from previous prediction
                    if tt >= test_start + h and np.isfinite(pred_return):
                        actual_return = y_true[h][tt - h]
                        if np.isfinite(actual_return):
                            error = pred_return - actual_return
                            error_history_hybrid.append(error)
                            if len(error_history_hybrid) > 0:
                                recent_errors = error_history_hybrid[-20:]
                                bias_correction_hybrid = np.mean(recent_errors)
                
                t += 7
                retrain_counter += 7
            
            y = y_true[h]
            rows.append({
                'symbol': sym,
                'strategy': 'hybrid',
                'horizon': f'{h}d',
                'dir_hit_pct': dirhit(y[test_start:], preds[test_start:], thr_eval),
                'r2': r2(y[test_start:], preds[test_start:]),
                'mape': smape(y[test_start:], preds[test_start:]),
                'nrmse': nrmse(y[test_start:], preds[test_start:]),
                'n': int(np.isfinite(y[test_start:]).sum())
            })

    if not rows:
        print('⚠️ No adaptive learning results.')
        return

    df_out = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = f'/opt/bist-pattern/logs/adaptive_learning_3sym_{ts}.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ Adaptive learning results saved: {out_csv}")

    # Print summary
    print("\n" + "="*100)
    print("📊 ADAPTIVE LEARNING SUMMARY (Tüm Ufuklar: 1/3/7/14/30d)")
    print("="*100)
    
    for strategy in ['static', 'weekly_retrain', 'incremental', 'hybrid']:
        subset = df_out[df_out['strategy'] == strategy]
        if len(subset) == 0:
            continue
        print(f"\n🔹 {strategy.upper()}")
        for h in [1, 3, 7, 14, 30]:
            hz = f'{h}d'
            row = subset[subset['horizon'] == hz]
            if len(row) == 0:
                continue
            dh = row['dir_hit_pct'].mean()
            r2_val = row['r2'].mean()
            mape_val = row['mape'].mean()
            nrmse_val = row['nrmse'].mean()
            print(f"  {hz}: DirHit={dh:.2f}%, R²={r2_val:.3f}, MAPE={mape_val:.4f}, nRMSE={nrmse_val:.3f}")


if __name__ == '__main__':
    run()
