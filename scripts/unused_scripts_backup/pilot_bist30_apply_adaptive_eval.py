#!/usr/bin/env python3
"""
Apply best adaptive K (std-based) + best evaluation thresholds per symbol×horizon
for short horizons (1/3/7d), then rerun BIST30 and write CSV metrics.

Env:
  - DATABASE_URL (required)
  - ADAPTIVE_GRID_CSV: path to bist30_adaptive_deadband_grid_*.csv (optional; auto-detect latest)
  - EVAL_THRESH_GRID_CSV: path to bist30_eval_thresh_grid_*.csv (optional; auto-detect latest)
  - BIST_LOG_PATH (optional; default /opt/bist-pattern/logs)

Output CSV columns:
  symbol,horizon,r2,dir_hit_pct,mape,nrmse,adaptive_k,eval_thresh
"""

import os
import sys
import glob
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, '/opt/bist-pattern')
from enhanced_ml_system import EnhancedMLSystem  # noqa: E402


BIST30: List[str] = [
    'AKBNK', 'ASELS', 'BIMAS', 'EKGYO', 'EREGL', 'FROTO',
    'GARAN', 'GUBRF', 'HEKTS', 'ISCTR', 'KCHOL', 'KOZAL',
    'KOZAA', 'KRDMD', 'PETKM', 'PGSUS', 'SAHOL', 'SASA',
    'SISE', 'TAVHL', 'TCELL', 'THYAO', 'TKFEN', 'TOASO',
    'TTKOM', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK', 'ODAS'
]


def fetch_prices(engine, symbol: str, limit: int = 1000) -> pd.DataFrame:
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
    return df[['open', 'high', 'low', 'close', 'volume']]


def find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_adaptive_k(path: str) -> Dict[str, Dict[str, float]]:
    ks: Dict[str, Dict[str, float]] = {}
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        sym = str(r['symbol']).strip()
        hz = str(r['horizon']).strip()
        try:
            val = float(r['best_k'])
        except Exception:
            continue
        ks.setdefault(sym, {})[hz] = val
    return ks


def load_eval_thresh(path: str) -> Dict[str, Dict[str, float]]:
    th: Dict[str, Dict[str, float]] = {}
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        sym = str(r['symbol']).strip()
        hz = str(r['horizon']).strip()
        try:
            val = float(r['best_eval_thresh'])
        except Exception:
            continue
        th.setdefault(sym, {})[hz] = val
    return th


def run_apply() -> None:
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url)

    adaptive_csv = os.getenv('ADAPTIVE_GRID_CSV') or find_latest('/opt/bist-pattern/logs/bist30_adaptive_deadband_grid_*.csv')
    eval_csv = os.getenv('EVAL_THRESH_GRID_CSV') or find_latest('/opt/bist-pattern/logs/bist30_eval_thresh_grid_*.csv')
    if not adaptive_csv or not os.path.exists(adaptive_csv):
        print('❌ Adaptive grid CSV not found')
        sys.exit(1)
    if not eval_csv or not os.path.exists(eval_csv):
        print('❌ Eval-threshold grid CSV not found')
        sys.exit(1)

    print(f"Using adaptive CSV: {adaptive_csv}")
    print(f"Using eval-threshold CSV: {eval_csv}")
    k_map = load_adaptive_k(adaptive_csv)
    t_map = load_eval_thresh(eval_csv)

    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')
    os.environ['ML_USE_DIRECTIONAL_LOSS'] = os.getenv('ML_USE_DIRECTIONAL_LOSS', '0')
    os.environ['ML_USE_STACKED_SHORT'] = '0'
    os.environ['ML_HORIZONS'] = '1,3,7'
    os.environ['ML_ADAPTIVE_DEADBAND_MODE'] = os.getenv('ML_ADAPTIVE_DEADBAND_MODE', 'std')

    rows: List[Dict[str, object]] = []

    for sym in BIST30:
        print(f"\n{'='*100}\nAPPLY COMBINED: {sym}\n{'='*100}\n")
        df = fetch_prices(engine, sym, limit=1000)
        if df is None or df.empty or len(df) < int(os.getenv('ML_MIN_DATA_DAYS', '200')):
            print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
            continue

        k1 = k_map.get(sym, {}).get('1d')
        k3 = k_map.get(sym, {}).get('3d')
        k7 = k_map.get(sym, {}).get('7d')
        os.environ['ML_ADAPTIVE_K_1D'] = '' if k1 is None else str(float(k1))
        os.environ['ML_ADAPTIVE_K_3D'] = '' if k3 is None else str(float(k3))
        os.environ['ML_ADAPTIVE_K_7D'] = '' if k7 is None else str(float(k7))

        e1 = t_map.get(sym, {}).get('1d')
        e3 = t_map.get(sym, {}).get('3d')
        e7 = t_map.get(sym, {}).get('7d')
        os.environ['ML_DIR_EVAL_THRESH_1D'] = '' if e1 is None else str(float(e1))
        os.environ['ML_DIR_EVAL_THRESH_3D'] = '' if e3 is None else str(float(e3))
        os.environ['ML_DIR_EVAL_THRESH_7D'] = '' if e7 is None else str(float(e7))

        ml = EnhancedMLSystem()
        try:
            ml.enable_external_features = False
            ml.enable_fingpt_features = False
            ml.enable_yolo_features = False
            ml._add_macro_features = lambda _df: None  # type: ignore
        except Exception:
            pass

        ok = ml.train_enhanced_models(sym, df)
        if not ok:
            print(f"❌ Training failed: {sym}")
            continue

        perf = ml.model_performance.get(sym, {})
        metrics = perf.get('metrics', {}) if isinstance(perf, dict) else {}
        for hz, kval, eth in [('1d', k1, e1), ('3d', k3, e3), ('7d', k7, e7)]:
            m = metrics.get(hz)
            if not m:
                continue
            rows.append({
                'symbol': sym,
                'horizon': hz,
                'r2': m.get('r2'),
                'dir_hit_pct': m.get('dir_hit_pct'),
                'mape': m.get('mape'),
                'nrmse': m.get('nrmse'),
                'adaptive_k': (float(kval) if kval is not None else None),
                'eval_thresh': (float(eth) if eth is not None else None),
            })

    if not rows:
        print('⚠️ No metrics collected.')
        return

    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bist30_short_apply_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Wrote short-apply-combined summary: {out_path}")


if __name__ == '__main__':
    run_apply()
