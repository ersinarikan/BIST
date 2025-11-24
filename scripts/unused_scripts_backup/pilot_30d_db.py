#!/usr/bin/env python3
"""
Pilot: VT (PostgreSQL) üzerinden 30g eğitim (Flask app import ETMEDEN)

Kullanım:
  source /opt/bist-pattern/.venv/bin/activate
  export DATABASE_URL=postgresql://user:pass@host:5432/db
  ML_HORIZONS=30 ML_USE_DIRECTIONAL_LOSS=0 python3 scripts/pilot_30d_db.py
"""

import os
import sys
import json
from typing import List

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, '/opt/bist-pattern')
from enhanced_ml_system import EnhancedMLSystem  # noqa: E402


SYMBOLS: List[str] = ['HEKTS', 'KRDMD', 'AKBNK']


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


def run_pilot() -> None:
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        # Fallback: try Settings default
        try:
            from bist_pattern.settings import Settings  # type: ignore
            db_url = Settings.SQLALCHEMY_DATABASE_URI
        except Exception:
            print('❌ DATABASE_URL not set')
            sys.exit(1)
    engine = create_engine(db_url)

    # Varsayılanlar
    os.environ.setdefault('ML_HORIZONS', '30')
    os.environ.setdefault('ML_USE_DIRECTIONAL_LOSS', '0')
    # Avoid requiring Redis in app import path
    os.environ.setdefault('SOCKETIO_MESSAGE_QUEUE', '')

    print('=' * 100)
    print('🎯 30G PILOT (DB-backed, no Flask app import)')
    print('=' * 100)
    print()
    print(f"Config: ML_HORIZONS={os.getenv('ML_HORIZONS')} ML_USE_DIRECTIONAL_LOSS={os.getenv('ML_USE_DIRECTIONAL_LOSS')}")
    print()

    ml = EnhancedMLSystem()
    # Disable features requiring Flask app context / external files
    try:
        ml.enable_external_features = False
        ml.enable_fingpt_features = False
        ml.enable_yolo_features = False
        # No-op macro features to avoid db.session within app context
        ml._add_macro_features = lambda _df: None  # type: ignore
    except Exception:
        pass
    results: dict[str, dict] = {}
    try:
        # Import app and run within app context to allow macro features (db.session)
        from app import app  # type: ignore
        ctx = app.app_context()
        ctx.push()
    except Exception:
        ctx = None

    try:
        for sym in SYMBOLS:
            print(f"\n{'='*100}")
            print(f"Training: {sym}")
            print(f"{'='*100}\n")
            df = fetch_prices(engine, sym, limit=1000)
            if df is None or df.empty or len(df) < 200:
                print(f"❌ {sym} insufficient data ({0 if df is None else len(df)})")
                continue
            ok = ml.train_enhanced_models(sym, df)
            if not ok:
                print(f"❌ Training failed: {sym}")
                continue
            # Take full performance dict for symbol (contains 'metrics')
            results[sym] = ml.model_performance.get(sym, {})
    finally:
        if ctx is not None:
            ctx.pop()

    print('\nRESULTS:')
    for sym, perf in results.items():
        try:
            metrics = perf.get('metrics', {}) if isinstance(perf, dict) else {}
            if metrics:
                # Compact JSON: only horizons we have
                print(sym, json.dumps(metrics))
            else:
                print(sym, json.dumps(perf))
        except Exception:
            print(sym, str(perf))


if __name__ == '__main__':
    run_pilot()
