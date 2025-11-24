#!/usr/bin/env python3
"""
Apply best short-horizon deadbands (1/3/7d) per symbol from the grid CSV,
then rerun training for BIST30 symbols and write a CSV summary.

Environment:
  - DATABASE_URL: required (PostgreSQL URL)
  - GRID_CSV_PATH: optional explicit path to grid CSV. If not set, the latest
                   /opt/bist-pattern/logs/bist30_short_grid_*.csv is used.
  - BIST_LOG_PATH: optional output directory (default: /opt/bist-pattern/logs)

Outputs CSV with columns:
  symbol,horizon,r2,dir_hit_pct,mape,nrmse,deadband
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
    return df.sort_values('date').reset_index(drop=True)[['open', 'high', 'low', 'close', 'volume']]


def find_latest_grid_csv() -> Optional[str]:
    path = os.getenv('GRID_CSV_PATH')
    if path and os.path.exists(path):
        return path
    candidates = glob.glob('/opt/bist-pattern/logs/bist30_short_grid_*.csv')
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_deadbands(grid_csv_path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(grid_csv_path)
    # Expect columns: symbol,horizon,best_deadband, ...
    deadbands: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        sym = str(row['symbol']).strip()
        hz = str(row['horizon']).strip()  # like '1d','3d','7d'
        try:
            band = float(row['best_deadband'])
        except Exception:
            continue
        if sym not in deadbands:
            deadbands[sym] = {}
        deadbands[sym][hz] = band
    return deadbands


def run_apply() -> None:
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url)

    grid_csv = find_latest_grid_csv()
    if not grid_csv:
        print('❌ Grid CSV not found (set GRID_CSV_PATH or run grid first).')
        sys.exit(1)

    print(f"Using grid CSV: {grid_csv}")
    deadband_map = load_deadbands(grid_csv)

    # Base config consistent with grid run
    os.environ.setdefault('ML_USE_DIRECTIONAL_LOSS', '0')
    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')
    os.environ['ML_USE_STACKED_SHORT'] = '0'  # stay on regressor path for comparability
    os.environ['ML_HORIZONS'] = '1,3,7'

    rows: List[Dict[str, object]] = []

    for sym in BIST30:
        print(f"\n{'='*100}\nAPPLY: {sym}\n{'='*100}\n")
        df = fetch_prices(engine, sym, limit=1000)
        if df is None or df.empty or len(df) < int(os.getenv('ML_MIN_DATA_DAYS', '200')):
            print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
            continue

        # Pull per-horizon deadbands for this symbol with sensible defaults
        db1 = float(deadband_map.get(sym, {}).get('1d', 0.006))
        db3 = float(deadband_map.get(sym, {}).get('3d', 0.006))
        db7 = float(deadband_map.get(sym, {}).get('7d', 0.005))

        os.environ['ML_DEADBAND_1D'] = str(db1)
        os.environ['ML_DEADBAND_3D'] = str(db3)
        os.environ['ML_DEADBAND_7D'] = str(db7)

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
        for hz, band in [('1d', db1), ('3d', db3), ('7d', db7)]:
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
                'deadband': float(band),
            })

    if not rows:
        print('⚠️ No metrics collected.')
        return

    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bist30_short_apply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Wrote short-apply summary: {out_path}")


if __name__ == '__main__':
    run_apply()