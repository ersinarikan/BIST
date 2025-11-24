#!/usr/bin/env python3
"""
Threshold Grid Search for Direction Evaluation (per horizon)

Runs a lightweight walk-forward over selected symbols and horizons,
tests multiple evaluation thresholds, and outputs the best thresholds.

Usage:
  python scripts/threshold_grid_search.py --symbols GARAN,AKBNK,EREGL --horizons 1,3,7 --lookback-days 500

Outputs JSON with best thresholds and export lines to set ENV.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


BIST30 = [
    'AKBNK', 'ARCLK', 'ASELS', 'BIMAS', 'EKGYO', 'ENJSA', 'EREGL', 'FROTO', 'GARAN', 'HEKTS',
    'ISCTR', 'KCHOL', 'KOZAL', 'KOZAA', 'KRDMD', 'PETKM', 'PGSUS', 'SAHOL', 'SASA', 'SISE',
    'TAVHL', 'TCELL', 'THYAO', 'TOASO', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK', 'ODAS', 'SMRTG'
]


def fetch_prices(engine, symbol: str, limit: int = 1200) -> pd.DataFrame:
    # Cache-backed reads to avoid exhausting DB connections
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
                result = df[['open', 'high', 'low', 'close', 'volume']]
                if isinstance(result, pd.DataFrame):
                    return result
                return pd.DataFrame(result)
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
    result = df[['open', 'high', 'low', 'close', 'volume']]
    if isinstance(result, pd.DataFrame):
        return result
    return pd.DataFrame(result)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='GARAN,AKBNK,EREGL')
    ap.add_argument('--horizons', default='1,3,7')
    ap.add_argument('--lookback-days', type=int, default=500)
    ap.add_argument('--thr-grids', default='1:0.003,0.005,0.007;3:0.004,0.006,0.008;7:0.005,0.007,0.010')
    ap.add_argument('--out', default='', help='If set, writes JSON result to this path. Progress prints to stderr.')
    args = ap.parse_args()

    os.environ.setdefault('DATABASE_URL', 'postgresql://bist_user:5ex5chan5GE5*@localhost:5432/bist_pattern_db')
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(',') if h.strip()]

    # Parse thr grids
    thr_map: Dict[int, List[float]] = {}
    for part in args.thr_grids.split(';'):
        part = part.strip()
        if not part:
            continue
        try:
            h, vals = part.split(':', 1)
            h_int = int(h)
            thr_map[h_int] = [float(x) for x in vals.split(',') if x]
        except Exception:
            continue

    # Global aggregation across symbols (for backward compatibility)
    results: Dict[int, Dict[float, List[float]]] = {h: {} for h in horizons}
    # Per-symbol best thresholds per horizon
    best_per_symbol: Dict[str, Dict[int, float]] = {}

    for sym in symbols:
        try:
            print(f"[thr-grid] start symbol={sym}", file=sys.stderr, flush=True)
        except Exception:
            pass
        df = fetch_prices(engine, sym, limit=max(args.lookback_days, 500))
        if df.empty or len(df) < 200:
            try:
                print(f"[thr-grid] skip symbol={sym} insufficient_data n={0 if df is None else len(df)}", file=sys.stderr, flush=True)
            except Exception:
                pass
            continue
        y_true = {h: compute_returns(df, h) for h in horizons}
        T0 = max(0, len(df) - min(args.lookback_days, 120))
        # Lazy import to avoid module-level import ordering issues
        import sys as _sys
        _sys.path.insert(0, '/opt/bist-pattern')
        from enhanced_ml_system import EnhancedMLSystem  # type: ignore
        ml = EnhancedMLSystem()
        if not ml.load_trained_models(sym):
            try:
                print(f"[thr-grid] skip symbol={sym} no_models", file=sys.stderr, flush=True)
            except Exception:
                pass
            continue
        for h in horizons:
            try:
                print(f"[thr-grid] symbol={sym} horizon={h}d start", file=sys.stderr, flush=True)
            except Exception:
                pass
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
                    if isinstance(pred_price, (int, float)):
                        last_close = float(cur['close'].iloc[-1])
                        if last_close > 0:
                            preds[t] = float(pred_price) / last_close - 1.0
            # Evaluate thresholds
            y = y_true[h]
            _best_thr = None
            _best_score = -1.0
            for thr in thr_map.get(h, [0.005]):
                dh = dirhit(y[T0:], preds[T0:], thr)
                results[h].setdefault(thr, []).append(dh)
                # Track per-symbol best
                if dh == dh and float(dh) > _best_score:
                    _best_score = float(dh)
                    _best_thr = float(thr)
                try:
                    print(f"[thr-grid] symbol={sym} h={h}d thr={thr:.4f} dirhit={dh:.2f}", file=sys.stderr, flush=True)
                except Exception:
                    pass
            if _best_thr is not None:
                best_per_symbol.setdefault(sym, {})[h] = float(_best_thr)

    # Aggregate
    # Global best thresholds (across all symbols)
    best_global: Dict[str, float] = {}
    for h, thr_dict in results.items():
        best_thr = None
        best_score = -1.0
        for thr, scores in thr_dict.items():
            if not scores:
                continue
            avg = float(np.nanmean(scores))
            if avg > best_score:
                best_score = avg
                best_thr = thr
        if best_thr is not None:
            best_global[f'{h}d'] = float(best_thr)

    # Per-symbol best thresholds formatted
    best_per_symbol_formatted: Dict[str, Dict[str, float]] = {}
    for sym, hmap in best_per_symbol.items():
        out = {}
        for h, thr in hmap.items():
            out[f'{h}d'] = float(thr)
        if out:
            best_per_symbol_formatted[sym] = out

    payload = {
        'generated_at': datetime.now().isoformat(),
        'symbols': symbols,
        'horizons': horizons,
        # Backward-compat: keep 'best_thresholds' as global
        'best_thresholds': best_global,
        'best_thresholds_global': best_global,
        'best_thresholds_per_symbol': best_per_symbol_formatted,
        'export': {
            'ML_DIR_EVAL_THRESH_1D': best_global.get('1d'),
            'ML_DIR_EVAL_THRESH_3D': best_global.get('3d'),
            'ML_DIR_EVAL_THRESH_7D': best_global.get('7d'),
            'ML_DIR_EVAL_THRESH_14D': best_global.get('14d'),
            'ML_DIR_EVAL_THRESH_30D': best_global.get('30d'),
        }
    }
    out_json = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, 'w') as wf:
            wf.write(out_json)
        try:
            print(f"[thr-grid] wrote result to {args.out}", file=sys.stderr, flush=True)
        except Exception:
            pass
    else:
        print(out_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
