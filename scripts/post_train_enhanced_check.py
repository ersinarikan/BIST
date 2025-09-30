#!/usr/bin/env python3
# flake8: noqa
"""
Post-train enhanced prediction health check.

Purpose:
- After nightly training, verify that enhanced models can load and produce
  predictions for a small sample of symbols (default 5).
- If predictions are produced, merge them into logs/ml_bulk_predictions.json
  so the UI can display ENHANCED immediately.

Usage:
  venv/bin/python scripts/post_train_enhanced_check.py --limit 5
  venv/bin/python scripts/post_train_enhanced_check.py --symbols THYAO,GARAN,ASELS
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import List, Dict, Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--symbols", type=str, default="")
    p.add_argument("--all", action="store_true", help="Process all active symbols that meet min days")
    return p.parse_args()


def merge_predictions(path: str, symbol: str, enhanced_obj: Dict[str, Any]) -> bool:
    try:
        obj: Dict[str, Any] = {"predictions": {}}
        if os.path.exists(path):
            with open(path, "r") as rf:
                obj = json.load(rf) or {"predictions": {}}
        preds = obj.setdefault("predictions", {})
        ent = preds.setdefault(symbol, {})
        ent["enhanced"] = enhanced_obj
        with open(path, "w") as wf:
            json.dump(obj, wf)
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()

    try:
        from app import create_app
    except Exception as e:
        print("ERROR: app import failed:", e)
        return 1

    app = create_app("default")
    with app.app_context():
        try:
            from models import Stock, StockPrice
            import pandas as pd
            from enhanced_ml_system import get_enhanced_ml_system
        except Exception as e:
            print("ERROR: dependency import failed:", e)
            return 1

        # Select symbols
        symbols: List[str]
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()][: args.limit]
        elif args.all:
            q = Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc())
            symbols = [s.symbol for s in q.all()]
        else:
            q = Stock.query.filter_by(is_active=True).order_by(Stock.symbol.asc()).limit(max(5, args.limit))
            symbols = [s.symbol for s in q.all()]

        enh = get_enhanced_ml_system()
        ok = 0
        fail = 0
        checked = []

        for sym in symbols:
            try:
                stock = Stock.query.filter_by(symbol=sym).first()
                if not stock:
                    fail += 1
                    continue
                rows = (
                    StockPrice.query.filter_by(stock_id=stock.id)
                    .order_by(StockPrice.date.asc())
                    .all()
                )
                try:
                    min_days = int(os.getenv('ML_MIN_DATA_DAYS', os.getenv('ML_MIN_DAYS', '180')))
                except Exception:
                    min_days = 180
                if not rows or len(rows) < min_days:
                    fail += 1
                    continue
                import pandas as pd  # local alias
                df = pd.DataFrame(
                    [
                        {
                            "date": r.date,
                            "open": float(r.open_price),
                            "high": float(r.high_price),
                            "low": float(r.low_price),
                            "close": float(r.close_price),
                            "volume": int(r.volume),
                        }
                        for r in rows
                    ]
                )
                df["date"] = pd.to_datetime(df["date"])  # type: ignore
                df.set_index("date", inplace=True)
                if len(df) > 730:
                    df = df.tail(730)

                try:
                    enh.load_trained_models(sym)
                except Exception:
                    pass

                out = enh.predict_enhanced(sym, df) or {}
                if not out:
                    # Attempt refresh train for this symbol only
                    try:
                        res = enh.train_enhanced_models(sym, df)
                        if res:
                            enh.load_trained_models(sym)
                            out = enh.predict_enhanced(sym, df) or {}
                    except Exception:
                        out = {}

                if out:
                    log_dir = os.getenv("BIST_LOG_PATH", "/opt/bist-pattern/logs")
                    os.makedirs(log_dir, exist_ok=True)
                    p = os.path.join(log_dir, "ml_bulk_predictions.json")
                    merge_predictions(p, sym, out)
                    ok += 1
                else:
                    fail += 1
                checked.append(sym)
            except Exception:
                fail += 1
                checked.append(sym)
                continue

        # Broadcast brief summary if app has socketio helper
        try:
            if hasattr(app, "broadcast_log"):
                app.broadcast_log(
                    "INFO",
                    f"Post-train check: ok={ok} fail={fail} symbols={','.join(checked)}",
                    "ml",
                )
        except Exception:
            pass

        print(json.dumps({"ok": ok, "fail": fail, "checked": checked}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


