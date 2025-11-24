#!/usr/bin/env python3
"""
FAST Walk-forward backtest for BIST30.

OPTIMIZATION: Feature engineering once, then predict for each time step.
Expected time: ~2 hours (vs. 35.5 hours for naive version)
"""

import os
import sys
from datetime import datetime
from typing import Dict, List

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
HORIZONS = [1, 3, 7]


def fetch_prices(engine, symbol: str, limit: int = 1200) -> pd.DataFrame:
    # Cache to reduce DB load when WF iterates many times
    cache_dir = '/opt/bist-pattern/cache/prices'
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{symbol}.csv')
    try:
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df = df.set_index('date')
            if not df.empty:
                if limit > 0 and len(df) > limit:
                    df = df.iloc[-limit:].copy()
                return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception:
        pass
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
    try:
        df.to_csv(cache_path, index=False)
    except Exception:
        pass
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'close', 'volume']]


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
    correct = (yt[m] == yp[m]).sum()
    return 100.0 * float(correct) / float(m.sum())


def smape(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() == 0:
        return float('nan')
    num = np.abs(a[m] - b[m])
    den = (np.abs(a[m]) + np.abs(b[m])) / 2.0
    den = np.where(den < 1e-9, 1e-9, den)
    return float(np.mean(num / den))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() == 0:
        return float('nan')
    mse = np.mean((a[m] - b[m]) ** 2)
    rmse = np.sqrt(mse)
    rng = np.ptp(a[m])
    return float(rmse / rng) if rng > 0 else float('nan')


def r2(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() == 0:
        return float('nan')
    ss_res = np.sum((a[m] - b[m]) ** 2)
    ss_tot = np.sum((a[m] - np.mean(a[m])) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def run() -> None:
    os.environ.setdefault('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})

    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')
    os.environ['ENABLE_EXTERNAL_FEATURES'] = '0'
    os.environ['ENABLE_FINGPT_FEATURES'] = '0'
    os.environ['ML_USE_DIRECTIONAL_LOSS'] = '0'
    # Align evaluation with training: enable stacking/ensemble for short horizons
    os.environ['ML_USE_STACKED_SHORT'] = '1'
    os.environ['ML_USE_SMART_ENSEMBLE'] = '1'
    os.environ['ML_HORIZONS'] = '1,3,7'
    os.environ.setdefault('ML_ADAPTIVE_DEADBAND_MODE', 'std')
    os.environ.setdefault('ML_ADAPTIVE_K_1D', '2.0')
    os.environ.setdefault('ML_ADAPTIVE_K_3D', '1.8')
    os.environ.setdefault('ML_ADAPTIVE_K_7D', '1.6')
    os.environ.setdefault('ML_PATTERN_WEIGHT_SCALE_1D', '1.2')
    os.environ.setdefault('ML_PATTERN_WEIGHT_SCALE_3D', '1.15')
    os.environ.setdefault('ML_PATTERN_WEIGHT_SCALE_7D', '1.1')
    os.environ.setdefault('ML_CAP_PCTL_3D', '92.5')

    rows: List[Dict[str, object]] = []
    thr_eval = float(os.getenv('ML_LOSS_THRESHOLD', '0.005'))
    # Load per-symbol per-horizon thresholds if available
    thr_json_path = os.getenv('ML_DIR_THRESHOLDS_JSON', '/opt/bist-pattern/logs/threshold_grid_bist30_hp_result.json')
    thr_map_global: Dict[str, float] = {}
    thr_map_per_symbol: Dict[str, Dict[str, float]] = {}
    try:
        if os.path.exists(thr_json_path):
            obj = pd.read_json(thr_json_path, typ='series') if thr_json_path.endswith('.pkl') else None
    except Exception:
        obj = None
    import json as _json
    try:
        if os.path.exists(thr_json_path):
            with open(thr_json_path, 'r') as rf:
                data = _json.load(rf)
            thr_map_global = data.get('best_thresholds_global') or data.get('best_thresholds') or {}
            thr_map_per_symbol = data.get('best_thresholds_per_symbol') or {}
    except Exception:
        thr_map_global, thr_map_per_symbol = {}, {}

    for sym_idx, sym in enumerate(SYMS, 1):
        print(f"\n{'='*100}")
        print(f"[{sym_idx}/{len(SYMS)}] WALKFWD: {sym}")
        print(f"{'='*100}\n")
        
        df = fetch_prices(engine, sym, limit=1200)
        if df is None or df.empty or len(df) < 500:
            print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
            continue

        # Compute forward returns for evaluation
        y_true: Dict[int, np.ndarray] = {h: compute_returns(df, h) for h in HORIZONS}

        # Anchor point T0 at 120 days from end
        T0 = max(0, len(df) - 120)

        # Test Phase 2 adaptive learning models (already trained)
        print(f"📊 Testing Phase 2 models (test on last {len(df) - T0} days)")
        ml = EnhancedMLSystem()
        
        # Test on out-of-sample period
        for h in HORIZONS:
            print(f"  Testing {h}d...")
            preds = np.full(len(df), np.nan, dtype=float)
            
            # Predict for each day in test period
            # predict_enhanced will automatically load models if available
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
            
            # Determine evaluation threshold: per-symbol > global > env default
            hk = f'{h}d'
            thr_sym = float(thr_map_per_symbol.get(sym, {}).get(hk, thr_map_global.get(hk, thr_eval)))
            y = y_true[h]
            dh = dirhit(y[T0:], preds[T0:], thr_sym)
            r2_val = r2(y[T0:], preds[T0:])
            mape_val = smape(y[T0:], preds[T0:])
            nrmse_val = nrmse(y[T0:], preds[T0:])
            
            print(f"    {h}d: DirHit={dh:.2f}%, R²={r2_val:.3f}, MAPE={mape_val:.4f}, nRMSE={nrmse_val:.3f}")
            
            rows.append({
                'symbol': sym,
                'mode': 'phase2_adaptive',
                'horizon': f'{h}d',
                'dir_hit_pct': dh,
                'r2': r2_val,
                'mape': mape_val,
                'nrmse': nrmse_val,
                'n': int(np.isfinite(y[T0:]).sum())
            })
            # Diagnostics summary per symbol/horizon
            try:
                yt = np.where(np.abs(y[T0:]) < thr_sym, 0, np.sign(y[T0:]))
                yp = np.where(np.abs(preds[T0:]) < thr_sym, 0, np.sign(preds[T0:]))
                m = np.isfinite(yt) & np.isfinite(yp)
                tp = int(((yt[m] == 1) & (yp[m] == 1)).sum())
                tn = int(((yt[m] == -1) & (yp[m] == -1)).sum())
                fp = int(((yt[m] == -1) & (yp[m] == 1)).sum())
                fn = int(((yt[m] == 1) & (yp[m] == -1)).sum())
                pos_ratio = float((yp[m] == 1).mean()) if m.any() else float('nan')
                rows.append({
                    'symbol': sym,
                    'mode': 'diagnostics',
                    'horizon': f'{h}d',
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'pos_ratio': pos_ratio,
                    'thr': thr_sym
                })
            except Exception:
                pass
        
        print(f"✅ {sym} completed")

    # Save results
    os.makedirs('/opt/bist-pattern/results', exist_ok=True)
    out_csv = f"/opt/bist-pattern/results/walkforward_bist30_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n{'='*100}")
    print(f"✅ Results saved to: {out_csv}")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    run()
