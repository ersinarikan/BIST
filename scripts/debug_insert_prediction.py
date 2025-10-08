#!/usr/bin/env python3
"""
Debug helper: insert a single PredictionsLog row to validate DB writes.

Usage:
  venv/bin/python3 scripts/debug_insert_prediction.py [SYMBOL]

Requires DATABASE_URL in environment before importing app/config.
"""
from __future__ import annotations

import sys
from datetime import datetime

from app import app
from models import db, PredictionsLog


def main() -> int:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else 'AKBNK').upper()
    with app.app_context():
        try:
            log = PredictionsLog(
                stock_id=None,
                symbol=symbol,
                horizon='1d',
                ts_pred=datetime.utcnow(),
                price_now=100.0,
                pred_price=100.5,
                delta_pred=(100.5 - 100.0) / 100.0,
                model='basic',
                unified_best='basic',
                confidence=0.65,
            )
            db.session.add(log)
            db.session.commit()
            print(f"inserted_id {log.id}")
            return 0
        except Exception as e:
            try:
                db.session.rollback()
            except Exception:
                pass
            print(f"insert_error {type(e).__name__} {e}")
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
