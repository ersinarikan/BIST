#!/usr/bin/env python3
"""
Cap percentile grid (DB-backed) for short horizons 1/3/7d.

Applies adopted adaptive K + best pattern-weight scales and sweeps cap percentiles
per horizon to find best DirHit (tie-break nRMSE).

Env:
  - DATABASE_URL (required)
  - ADAPTIVE_GRID_CSV: path to bist30_adaptive_deadband_grid_*.csv (optional; autodetect latest)
  - PATT_WEIGHT_GRID_CSV: path to bist30_pattern_weight_grid_*.csv (optional; autodetect latest)
  - BIST_LOG_PATH: output dir

Outputs CSV: symbol,horizon,best_cap_pctl,r2,dir_hit_pct,mape,nrmse
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple, Optional

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


def find_latest(pattern: str) -> Optional[str]:
    import glob
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_adaptive_k(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    ks: Dict[str, Dict[str, float]] = {}
    if not path or not os.path.exists(path):
        return ks
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        ks.setdefault(str(r['symbol']).strip(), {})[str(r['horizon']).strip()] = float(r['best_k'])
    return ks


def load_patt_scale(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    sc: Dict[str, Dict[str, float]] = {}
    if not path or not os.path.exists(path):
        return sc
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        sc.setdefault(str(r['symbol']).strip(), {})[str(r['horizon']).strip()] = float(r['best_scale'])
    return sc


def score_tuple(m: Dict[str, float]) -> Tuple[float, float]:
    return (
        float(m.get('dir_hit_pct') or 0.0),
        -float(m.get('nrmse') or 1e9),
    )


def run_grid() -> None:
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url)

    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')
    os.environ['ML_USE_DIRECTIONAL_LOSS'] = os.getenv('ML_USE_DIRECTIONAL_LOSS', '0')
    os.environ['ML_USE_STACKED_SHORT'] = '0'
    os.environ['ML_HORIZONS'] = '1,3,7'
    os.environ['ML_ADAPTIVE_DEADBAND_MODE'] = os.getenv('ML_ADAPTIVE_DEADBAND_MODE', 'std')

    k_csv = os.getenv('ADAPTIVE_GRID_CSV') or find_latest('/opt/bist-pattern/logs/bist30_adaptive_deadband_grid_*.csv')
    s_csv = os.getenv('PATT_WEIGHT_GRID_CSV') or find_latest('/opt/bist-pattern/logs/bist30_pattern_weight_grid_*.csv')
    k_map = load_adaptive_k(k_csv)
    s_map = load_patt_scale(s_csv)

    p_grids = {
        '1d': [90, 95, 97.5],
        '3d': [90, 95, 97.5],
        '7d': [90, 95, 97.5],
    }

    rows: List[Dict[str, object]] = []

    for sym in BIST30:
        print(f"\n{'='*100}\nCAP GRID: {sym}\n{'='*100}\n")
        df = fetch_prices(engine, sym, limit=1000)
        if df is None or df.empty or len(df) < int(os.getenv('ML_MIN_DATA_DAYS', '200')):
            print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
            continue

        best: Dict[str, Dict[str, object]] = {}
        for hz, plist in p_grids.items():
            hnum = int(hz.replace('d', ''))
            best_metrics: Dict[str, object] = {}
            best_score: Tuple[float, float] = (-1.0, -1e9)
            best_p: Optional[float] = None

            for p in plist:
                # Set adaptive K and pattern weight scales for stability
                os.environ['ML_ADAPTIVE_K_1D'] = str(k_map.get(sym, {}).get('1d', ''))
                os.environ['ML_ADAPTIVE_K_3D'] = str(k_map.get(sym, {}).get('3d', ''))
                os.environ['ML_ADAPTIVE_K_7D'] = str(k_map.get(sym, {}).get('7d', ''))
                os.environ['ML_PATTERN_WEIGHT_SCALE_1D'] = str(s_map.get(sym, {}).get('1d', '1.0'))
                os.environ['ML_PATTERN_WEIGHT_SCALE_3D'] = str(s_map.get(sym, {}).get('3d', '1.0'))
                os.environ['ML_PATTERN_WEIGHT_SCALE_7D'] = str(s_map.get(sym, {}).get('7d', '1.0'))

                # Set cap only for target horizon
                os.environ['ML_CAP_PCTL_1D'] = str(p if hnum == 1 else '')
                os.environ['ML_CAP_PCTL_3D'] = str(p if hnum == 3 else '')
                os.environ['ML_CAP_PCTL_7D'] = str(p if hnum == 7 else '')

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
                    continue
                perf = ml.model_performance.get(sym, {})
                metrics = perf.get('metrics', {}) if isinstance(perf, dict) else {}
                m = metrics.get(hz)
                if not m:
                    continue
                sc = score_tuple(m)
                if sc > best_score:
                    best_score = sc
                    best_p = p
                    best_metrics = {
                        'r2': m.get('r2'),
                        'dir_hit_pct': m.get('dir_hit_pct'),
                        'mape': m.get('mape'),
                        'nrmse': m.get('nrmse'),
                    }

            if best_p is not None:
                best[hz] = {
                    'best_cap_pctl': best_p,
                    **best_metrics,
                }

        for hz in ['1d', '3d', '7d']:
            b = best.get(hz)
            if not b:
                continue
            rows.append({
                'symbol': sym,
                'horizon': hz,
                'best_cap_pctl': b['best_cap_pctl'],
                'r2': b['r2'],
                'dir_hit_pct': b['dir_hit_pct'],
                'mape': b['mape'],
                'nrmse': b['nrmse'],
            })

    if not rows:
        print('⚠️ No cap grid results.')
        return

    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bist30_cap_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Wrote cap grid summary: {out_path}")


if __name__ == '__main__':
    run_grid()
