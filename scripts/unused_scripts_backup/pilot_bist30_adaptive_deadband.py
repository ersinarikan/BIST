#!/usr/bin/env python3
"""
Adaptive deadband k-grid for short horizons (DB-backed, no Flask import).

For horizons 1/3/7d, sweeps ML_ADAPTIVE_K_{H}D under ML_ADAPTIVE_DEADBAND_MODE=std
and records best metrics per symbol/horizon optimizing DirHit first, then nRMSE.

Outputs CSV: symbol,horizon,best_k,r2,dir_hit_pct,mape,nrmse
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple

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

    os.environ.setdefault('ML_USE_DIRECTIONAL_LOSS', '0')
    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')
    os.environ['ML_USE_STACKED_SHORT'] = '0'  # use regressor path
    os.environ['ML_HORIZONS'] = '1,3,7'
    os.environ['ML_ADAPTIVE_DEADBAND_MODE'] = os.getenv('ML_ADAPTIVE_DEADBAND_MODE', 'std')

    k_grids = {
        '1d': [0.6, 0.8, 1.0, 1.2],
        '3d': [0.6, 0.8, 1.0, 1.2],
        '7d': [0.6, 0.8, 1.0, 1.2],
    }

    rows: List[Dict[str, object]] = []

    for sym in BIST30:
        print(f"\n{'='*100}\nADAPTIVE GRID: {sym}\n{'='*100}\n")
        df = fetch_prices(engine, sym, limit=1000)
        if df is None or df.empty or len(df) < int(os.getenv('ML_MIN_DATA_DAYS', '200')):
            print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
            continue

        best: Dict[str, Dict[str, object]] = {}
        for hz, band_list in k_grids.items():
            hnum = int(hz.replace('d', ''))
            best_metrics: Dict[str, object] = {}
            best_score: Tuple[float, float] = (-1.0, -1e9)
            best_k = None

            for k in band_list:
                # set only the target horizon's adaptive K; others unset to use fixed deadband
                os.environ['ML_ADAPTIVE_K_1D'] = str(k if hnum == 1 else '')
                os.environ['ML_ADAPTIVE_K_3D'] = str(k if hnum == 3 else '')
                os.environ['ML_ADAPTIVE_K_7D'] = str(k if hnum == 7 else '')

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
                    best_k = k
                    best_metrics = {
                        'r2': m.get('r2'),
                        'dir_hit_pct': m.get('dir_hit_pct'),
                        'mape': m.get('mape'),
                        'nrmse': m.get('nrmse'),
                    }

            if best_k is not None:
                best[hz] = {
                    'best_k': best_k,
                    **best_metrics,
                }

        # write rows for this symbol
        for hz in ['1d', '3d', '7d']:
            b = best.get(hz)
            if not b:
                continue
            rows.append({
                'symbol': sym,
                'horizon': hz,
                'best_k': b['best_k'],
                'r2': b['r2'],
                'dir_hit_pct': b['dir_hit_pct'],
                'mape': b['mape'],
                'nrmse': b['nrmse'],
            })

    if not rows:
        print('⚠️ No adaptive grid results.')
        return

    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bist30_adaptive_deadband_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Wrote adaptive grid summary: {out_path}")


if __name__ == '__main__':
    run_grid()
