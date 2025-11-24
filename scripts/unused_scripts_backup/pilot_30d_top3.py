#!/usr/bin/env python3
"""
Pilot: Train 30d models for 3 symbols (HEKTS, KRDMD, AKBNK) and print metrics

Usage:
  ML_HORIZONS=30 ML_USE_DIRECTIONAL_LOSS=0 python3 scripts/pilot_30d_top3.py
"""

import os
import sys
import json
import pandas as pd

# Ensure project root on path
sys.path.insert(0, '/opt/bist-pattern')

from app import app  # noqa: E402
from models import db, Stock  # noqa: E402
from enhanced_ml_system import EnhancedMLSystem  # noqa: E402


SYMBOLS = ['HEKTS', 'KRDMD', 'AKBNK']


def run_pilot() -> None:
    # Defaults for pilot if not set
    os.environ.setdefault('ML_HORIZONS', '30')
    os.environ.setdefault('ML_USE_DIRECTIONAL_LOSS', '0')

    print("=" * 100)
    print("🎯 30D PILOT TRAINING (HEKTS, KRDMD, AKBNK)")
    print("=" * 100)
    print()
    print(f"Config: ML_HORIZONS={os.getenv('ML_HORIZONS')} ML_USE_DIRECTIONAL_LOSS={os.getenv('ML_USE_DIRECTIONAL_LOSS')}")
    print()

    ml = EnhancedMLSystem()
    results: dict[str, dict] = {}

    with app.app_context():
        for sym in SYMBOLS:
            print(f"\n{'='*100}")
            print(f"Training: {sym}")
            print(f"{'='*100}\n")
            stock = Stock.query.filter_by(symbol=sym).first()
            if not stock:
                print(f"❌ {sym} not found in database")
                continue
            query = (
                "SELECT date, open, high, low, close, volume "
                "FROM stock_prices WHERE stock_id = :sid ORDER BY date DESC LIMIT 1000"
            )
            rows = db.session.execute(db.text(query), {"sid": stock.id}).fetchall()
            if not rows or len(rows) < 120:
                print(f"❌ {sym} insufficient data ({len(rows) if rows else 0} days)")
                continue
            df = pd.DataFrame([
                {
                    'date': r[0],
                    'open': float(r[1]),
                    'high': float(r[2]),
                    'low': float(r[3]),
                    'close': float(r[4]),
                    'volume': float(r[5]) if r[5] else 0.0,
                }
                for r in rows
            ]).sort_values('date').reset_index(drop=True)

            ok = ml.train_enhanced_models(sym, df)
            if not ok:
                print(f"❌ Training failed for {sym}")
                continue
            perf = ml.model_performance.get(f"{sym}_30d", {})
            results[sym] = perf

    print("\nRESULTS:")
    for sym, perf in results.items():
        print(sym, json.dumps(perf))


if __name__ == '__main__':
    run_pilot()
