#!/usr/bin/env python3
"""
Walk-forward pilot (single-anchor and weekly retrain) for 3 symbols (GARAN, AKBNK, EREGL).

Uses adopted settings: adaptive K + pattern-weight (+ optional cap if maps exist).
Evaluates 1/3/7d DirHit, R², MAPE(SMAPE), nRMSE on a 120-day out-of-sample window.

Output: CSV with rows: symbol, mode, horizon, dir_hit_pct, r2, mape, nrmse, n
Modes: single_anchor, weekly_retrain
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


SYMS: List[str] = [
    'AKBNK', 'ARCLK', 'ASELS', 'BIMAS', 'EKGYO', 'ENJSA', 'EREGL',
    'FROTO', 'GARAN', 'HEKTS', 'ISCTR', 'KCHOL', 'KOZAL', 'KOZAA',
    'KRDMD', 'PETKM', 'PGSUS', 'SAHOL', 'SASA', 'SISE', 'TAVHL',
    'TCELL', 'THYAO', 'TOASO', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK',
    'ODAS', 'SMRTG'
]
HORIZONS = [1, 3, 7, 14, 30]  # Tüm ufuklar


def fetch_prices(engine, symbol: str, limit: int = 1200) -> pd.DataFrame:
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
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_map(path: Optional[str], key: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not path or not os.path.exists(path):
        return out
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        out.setdefault(str(r['symbol']).strip(), {})[str(r['horizon']).strip()] = float(r[key])
    return out


def compute_returns(df: pd.DataFrame, horizon: int) -> np.ndarray:
    c = df['close'].values.astype(float)
    fwd = np.empty_like(c)
    fwd[:] = np.nan
    if len(c) > horizon:
        fwd[:-horizon] = (c[horizon:] / c[:-horizon]) - 1.0
    return fwd


def dirhit(y_true: np.ndarray, y_pred: np.ndarray, thr: float) -> float:
    yt = np.where(np.abs(y_true) < thr, 0, np.sign(y_true))
    yp = np.where(np.abs(y_pred) < thr, 0, np.sign(y_pred))
    m = ~np.isnan(yt) & ~np.isnan(yp)
    if m.sum() == 0:
        return float('nan')
    return float(np.mean(yt[m] == yp[m]) * 100.0)


def smape(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() == 0:
        return float('nan')
    denom = (np.abs(a[m]) + np.abs(b[m]))
    denom[denom == 0] = 1e-9
    return float(np.mean(2.0 * np.abs(a[m] - b[m]) / denom))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() == 0:
        return float('nan')
    rmse = np.sqrt(np.mean((a[m] - b[m]) ** 2))
    stdy = np.std(a[m])
    return float(rmse / stdy) if stdy > 0 else float('inf')


def r2(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() == 0:
        return float('nan')
    ss_res = np.sum((a[m] - b[m]) ** 2)
    ss_tot = np.sum((a[m] - np.mean(a[m])) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def set_env_for_symbol(sym: str, k_map, s_map, c_map) -> None:
    for hz, envk, envs, envc in [
        ('1d', 'ML_ADAPTIVE_K_1D', 'ML_PATTERN_WEIGHT_SCALE_1D', 'ML_CAP_PCTL_1D'),
        ('3d', 'ML_ADAPTIVE_K_3D', 'ML_PATTERN_WEIGHT_SCALE_3D', 'ML_CAP_PCTL_3D'),
        ('7d', 'ML_ADAPTIVE_K_7D', 'ML_PATTERN_WEIGHT_SCALE_7D', 'ML_CAP_PCTL_7D'),
    ]:
        os.environ[envk] = '' if k_map.get(sym, {}).get(hz) is None else str(float(k_map.get(sym, {}).get(hz)))
        os.environ[envs] = '' if s_map.get(sym, {}).get(hz) is None else str(float(s_map.get(sym, {}).get(hz)))
        os.environ[envc] = '' if c_map.get(sym, {}).get(hz) is None else str(float(c_map.get(sym, {}).get(hz)))


def run() -> None:
    os.environ.setdefault('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})

    # Adopted maps
    k_csv = find_latest('/opt/bist-pattern/logs/bist30_adaptive_deadband_grid_*.csv')
    s_csv = find_latest('/opt/bist-pattern/logs/bist30_pattern_weight_grid_*.csv')
    c_csv = find_latest('/opt/bist-pattern/logs/bist30_cap_grid_*.csv')
    k_map = load_map(k_csv, 'best_k')
    s_map = load_map(s_csv, 'best_scale')
    c_map = load_map(c_csv, 'best_cap_pctl')

    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')
    os.environ['ENABLE_EXTERNAL_FEATURES'] = '0'  # Kapalı (geçmiş veri yok)
    os.environ['ENABLE_FINGPT_FEATURES'] = '0'    # Kapalı (geçmiş veri yok)
    os.environ['ML_USE_DIRECTIONAL_LOSS'] = '0'
    os.environ['ML_USE_STACKED_SHORT'] = '0'
    os.environ['ML_HORIZONS'] = '1,3,7,14,30'
    os.environ.setdefault('ML_ADAPTIVE_DEADBAND_MODE', 'std')

    rows: List[Dict[str, object]] = []
    # Threshold for DirHit eval
    thr_eval = float(os.getenv('ML_LOSS_THRESHOLD', '0.005'))

    for sym in SYMS:
        print(f"\n{'='*100}\nWALKFWD: {sym}\n{'='*100}\n")
        df = fetch_prices(engine, sym, limit=1200)
        if df is None or df.empty or len(df) < 500:
            print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
            continue

        # Compute forward returns matrix for evaluation
        y_true: Dict[int, np.ndarray] = {h: compute_returns(df, h) for h in HORIZONS}

        # Anchor point T0 at 120 days from end
        T0 = max(0, len(df) - 120)
        train_end = T0  # exclusive

        # Single-anchor: train once using up to train_end (use last 500 bars if available)
        set_env_for_symbol(sym, k_map, s_map, c_map)
        ml = EnhancedMLSystem()
        train_df = df.iloc[max(0, train_end - 500): train_end].copy()
        ok = ml.train_enhanced_models(sym, train_df)
        if not ok:
            print(f"❌ Training failed (single_anchor): {sym}")
        else:
            for h in HORIZONS:
                preds = np.full(len(df), np.nan, dtype=float)
                for t in range(T0, len(df) - h):
                    cur = df.iloc[: t + 1].copy()
                    p = ml.predict_enhanced(sym, cur)
                    if not isinstance(p, dict):
                        continue
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
                    'mode': 'single_anchor',
                    'horizon': f'{h}d',
                    'dir_hit_pct': dirhit(y[T0:], preds[T0:], thr_eval),
                    'r2': r2(y[T0:], preds[T0:]),
                    'mape': smape(y[T0:], preds[T0:]),
                    'nrmse': nrmse(y[T0:], preds[T0:]),
                    'n': int(np.isfinite(y[T0:]).sum())
                })

        # Bi-weekly retrain: every 10 business days retrain using all available data up to that point
        for h in HORIZONS:
            preds = np.full(len(df), np.nan, dtype=float)
            t = T0
            while t < len(df) - h:
                set_env_for_symbol(sym, k_map, s_map, c_map)
                ml2 = EnhancedMLSystem()
                train_df2 = df.iloc[: t].copy()  # Use ALL data (expanding window)
                ok2 = ml2.train_enhanced_models(sym, train_df2)
                if not ok2:
                    t += 10  # Every 10 business days (bi-weekly)
                    continue
                # produce predictions for next 10 days or until end-h
                for tt in range(t, min(t + 10, len(df) - h)):
                    cur = df.iloc[: tt + 1].copy()
                    p = ml2.predict_enhanced(sym, cur)
                    if not isinstance(p, dict):
                        continue
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
                t += 10  # Every 10 business days (bi-weekly)
            y = y_true[h]
            rows.append({
                'symbol': sym,
                'mode': 'weekly_retrain',
                'horizon': f'{h}d',
                'dir_hit_pct': dirhit(y[T0:], preds[T0:], thr_eval),
                'r2': r2(y[T0:], preds[T0:]),
                'mape': smape(y[T0:], preds[T0:]),
                'nrmse': nrmse(y[T0:], preds[T0:]),
                'n': int(np.isfinite(y[T0:]).sum())
            })

    if not rows:
        print('⚠️ No walk-forward results.')
        return

    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"walkforward_bist30_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Wrote walk-forward summary: {out_path}")


if __name__ == '__main__':
    run()
