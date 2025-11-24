#!/usr/bin/env python3
"""
BIST30 Multi-Horizon Pilot (DB-backed, no Flask app import)

Outputs CSV summary with columns: symbol,horizon,r2,dir_hit_pct,mape,nrmse

Usage:
  source /opt/bist-pattern/.venv/bin/activate
  export DATABASE_URL=postgresql://user:pass@host:5432/db  # or rely on .secrets fallback in caller
  ML_HORIZONS=1,3,7,14,30 ML_MIN_DATA_DAYS=200 python3 scripts/pilot_bist30_multihorizon.py
"""

import os
import sys
from datetime import datetime
from typing import List, Dict

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

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


def run_bist30_pilot() -> None:
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('❌ DATABASE_URL not set')
        sys.exit(1)
    engine = create_engine(db_url, pool_pre_ping=True, poolclass=NullPool, connect_args={"connect_timeout": 5})

    os.environ.setdefault('ML_HORIZONS', '1,3,7,14,30')
    os.environ.setdefault('ML_USE_DIRECTIONAL_LOSS', '0')
    os.environ.setdefault('ENABLE_TALIB_PATTERNS', '1')

    print('=' * 100)
    print('🎯 BIST30 MULTI-HORIZON PILOT (DB-backed)')
    print('=' * 100)
    print()
    print(f"Config: ML_HORIZONS={os.getenv('ML_HORIZONS')} ML_USE_DIRECTIONAL_LOSS={os.getenv('ML_USE_DIRECTIONAL_LOSS')}")
    print()

    ml = EnhancedMLSystem()
    # Disable features that require Flask app context or external backfills for stability
    try:
        ml.enable_external_features = False
        ml.enable_fingpt_features = False
        ml.enable_yolo_features = False
        ml._add_macro_features = lambda _df: None  # type: ignore
    except Exception:
        pass

    rows: List[Dict[str, object]] = []
    horizons = ['1d', '3d', '7d', '14d', '30d']

    for sym in BIST30:
        print(f"\n{'='*100}\nTraining: {sym}\n{'='*100}\n")
        df = fetch_prices(engine, sym, limit=1000)
        if df is None or df.empty or len(df) < int(os.getenv('ML_MIN_DATA_DAYS', '200')):
            print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
            continue
        ok = ml.train_enhanced_models(sym, df)
        if not ok:
            print(f"❌ Training failed: {sym}")
            continue
        perf = ml.model_performance.get(sym, {})
        metrics = perf.get('metrics', {}) if isinstance(perf, dict) else {}
        for h in horizons:
            m = metrics.get(h)
            if not m:
                continue
            rows.append({
                'symbol': sym,
                'horizon': h,
                'r2': m.get('r2'),
                'dir_hit_pct': m.get('dir_hit_pct'),
                'mape': m.get('mape'),
                'nrmse': m.get('nrmse'),
            })

    if not rows:
        print('⚠️ No metrics collected.')
        return

    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"bist30_multihorizon_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Wrote summary: {out_path}")


if __name__ == '__main__':
    run_bist30_pilot()
