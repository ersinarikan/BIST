#!/usr/bin/env python3
"""
Populate outcomes_log by evaluating matured predictions in predictions_log.

Policy:
  - For horizon 'Xd', evaluate at ts_pred + X days (market day granülarite: fiyat olarak DB'deki son tarih<=eval_ts kullanılır).
  - direction hit = sign(delta_pred) == sign(delta_real).
  - abs_err = |pred_price - price_eval|, mape = |pred - eval|/price_eval.

Idempotent: skips if an outcomes_log row already exists for prediction_id.

Timezone: Converts UTC timestamps to Istanbul market time for accurate date comparisons.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
import os
from typing import Optional

from app import app
from models import db, PredictionsLog, OutcomesLog, Stock, StockPrice

logger = logging.getLogger(__name__)


def _horizon_to_days(h: str) -> Optional[int]:
    try:
        h = (h or '').lower().strip()
        if h.endswith('d'):
            return int(h[:-1])
        return int(h)
    except Exception:
        return None


def _get_price_at_or_before(stock_id: int, ts: datetime) -> Optional[float]:
    # Find latest price row with date <= ts.date()
    try:
        tz_off = int(os.getenv('MARKET_TZ_OFFSET_HOURS', '0'))
    except Exception:
        tz_off = 0
    d = (ts + timedelta(hours=tz_off)).date()
    row = (
        StockPrice.query.filter_by(stock_id=stock_id)
        .filter(StockPrice.date <= d)
        .order_by(StockPrice.date.desc())
        .first()
    )
    if not row:
        return None
    try:
        return float(row.close_price)
    except Exception:
        return None


def run(limit: int = 1000) -> int:
    with app.app_context():
        # Ensure tables exist
        try:
            db.create_all()
        except Exception:
            pass
        # Candidates: predictions with no outcome yet and matured by horizon
        q = (
            db.session.query(PredictionsLog)
            .outerjoin(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
            .filter(OutcomesLog.id.is_(None))
            .order_by(PredictionsLog.ts_pred.asc())
        )
        
        candidates = q.limit(limit).all()
        logger.info(f"📊 Found {len(candidates)} prediction candidates without outcomes")
        
        processed = 0
        skipped_no_days = 0
        skipped_not_matured = 0
        skipped_no_stock = 0
        skipped_no_price = 0
        skipped_missing_prices = 0
        for p in candidates:
            days = _horizon_to_days(p.horizon)
            if not days:
                skipped_no_days += 1
                continue
            eval_ts = p.ts_pred + timedelta(days=days)
            if datetime.utcnow() < eval_ts:
                # Not matured yet
                skipped_not_matured += 1
                continue
            # Resolve stock_id if missing
            sid = p.stock_id
            if not sid:
                st = Stock.query.filter_by(symbol=p.symbol).first()
                sid = getattr(st, 'id', None)
            if not sid:
                skipped_no_stock += 1
                continue
            price_eval = _get_price_at_or_before(sid, eval_ts)
            if price_eval is None:
                skipped_no_price += 1
                continue
            # Compute realized delta
            price_now = float(p.price_now) if p.price_now is not None else None
            pred_price = float(p.pred_price) if p.pred_price is not None else None
            if price_now is None or pred_price is None:
                # Try to reconstruct price_now from close at ts_pred date
                pn = _get_price_at_or_before(sid, p.ts_pred)
                if pn is not None and pred_price is not None:
                    price_now = pn
            if price_now is None or pred_price is None:
                skipped_missing_prices += 1
                continue
            delta_real = (price_eval - price_now) / price_now if price_now else 0.0
            delta_pred = float(p.delta_pred) if p.delta_pred is not None else (pred_price - price_now) / price_now
            dir_hit = (delta_real >= 0 and delta_pred >= 0) or (delta_real < 0 and delta_pred < 0)
            abs_err = abs(pred_price - price_eval)
            mape = abs_err / price_eval if price_eval else 0.0

            o = OutcomesLog(
                prediction_id=p.id,
                ts_eval=eval_ts,
                price_eval=price_eval,
                delta_real=delta_real,
                dir_hit=dir_hit,
                abs_err=abs_err,
                mape=mape,
            )
            db.session.add(o)
            processed += 1
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
        print(
            f"populate_outcomes: processed={processed} "
            f"skips={{no_days:{skipped_no_days},not_matured:{skipped_not_matured},no_stock:{skipped_no_stock},no_price:{skipped_no_price},missing_prices:{skipped_missing_prices}}}"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=1000)
    args = ap.parse_args()
    return run(args.limit)


if __name__ == '__main__':
    raise SystemExit(main())
