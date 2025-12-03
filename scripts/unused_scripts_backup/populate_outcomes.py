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


def _add_business_days(start_date: datetime, business_days: int) -> datetime:
    """
    Add N business days to a date, skipping weekends (Saturday=5, Sunday=6).
    BIST trading days: Monday-Friday (0-4 in Python's weekday()).
    """
    current = start_date
    days_added = 0
    while days_added < business_days:
        current += timedelta(days=1)
        # weekday(): Monday=0, Sunday=6
        if current.weekday() < 5:  # Monday-Friday
            days_added += 1
    return current


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
        
        # Strategy: Daily snapshot approach - End of Day (EOD) predictions
        # For each (symbol, horizon, DATE(ts_pred)), select the LATEST prediction made AFTER market close
        # Market close: ~18:00 Istanbul time (15:00 UTC + 3 hours = 18:00 local)
        # This ensures we use end-of-day predictions with complete market data, not overnight training predictions
        
        # Subquery: Find latest ts_pred per (symbol, horizon, date) made after 18:00 local time
        from sqlalchemy import func, and_, text
        
        try:
            tz_off = int(os.getenv('MARKET_TZ_OFFSET_HOURS', '0'))
        except Exception:
            tz_off = 0
        
        # Convert UTC timestamp to local date for grouping
        # PostgreSQL: date(ts_pred + interval '3 hours')
        if tz_off != 0:
            pred_date = func.date(PredictionsLog.ts_pred + text(f"INTERVAL '{tz_off} hours'"))
        else:
            pred_date = func.date(PredictionsLog.ts_pred)
        
        # Subquery: latest ts_pred per (symbol, horizon, pred_date) made after 18:00 local time
        # Also clean symbol names (strip BOM and whitespace)
        # Filter: Only predictions after market close (18:00 local time)
        eod_cutoff_hour_local = 18  # Market closes around 18:00-18:30 Istanbul time
        
        # Apply timezone offset to get local hour - define once for reuse
        if tz_off != 0:
            local_hour_expr = func.extract('hour', PredictionsLog.ts_pred + text(f"INTERVAL '{tz_off} hours'"))
        else:
            local_hour_expr = func.extract('hour', PredictionsLog.ts_pred)
        
        subq = (
            db.session.query(
                func.upper(func.trim(PredictionsLog.symbol)).label('clean_symbol'),
                PredictionsLog.horizon,
                pred_date.label('pred_date'),
                func.max(PredictionsLog.ts_pred).label('latest_ts')
            )
            .outerjoin(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
            .filter(OutcomesLog.id.is_(None))  # No outcome yet
            .filter(local_hour_expr >= eod_cutoff_hour_local)  # After market close (local time)
            .filter(PredictionsLog.horizon.in_(['1d', '3d', '7d', '14d', '30d']))  # ✅ FIX: Process ALL horizons (7d, 14d, 30d were missing!)
            .group_by(func.upper(func.trim(PredictionsLog.symbol)), PredictionsLog.horizon, pred_date)
            .subquery()
        )
        
        # Main query: join with subquery to get only latest predictions
        # Order by pred_date ASC to process oldest mature predictions first
        # IMPORTANT: Also filter main query for EOD predictions to ensure we only get 18:00+ predictions
        # Recompute local_hour_expr for main query (SQLAlchemy needs fresh expression)
        if tz_off != 0:
            main_local_hour = func.extract('hour', PredictionsLog.ts_pred + text(f"INTERVAL '{tz_off} hours'"))
        else:
            main_local_hour = func.extract('hour', PredictionsLog.ts_pred)
        
        q = (
            db.session.query(PredictionsLog)
            .join(
                subq,
                and_(
                    func.upper(func.trim(PredictionsLog.symbol)) == subq.c.clean_symbol,
                    PredictionsLog.horizon == subq.c.horizon,
                    PredictionsLog.ts_pred == subq.c.latest_ts
                )
            )
            .filter(main_local_hour >= eod_cutoff_hour_local)  # Ensure main query also filters for EOD
            .order_by(subq.c.pred_date.asc(), func.upper(func.trim(PredictionsLog.symbol)).asc())
        )
        
        candidates = q.limit(limit).all()
        
        # Log sample of selected predictions for debugging
        if candidates:
            sample_pred = candidates[0]
            local_ts = sample_pred.ts_pred + timedelta(hours=tz_off)
            logger.info(
                f"📊 Found {len(candidates)} daily snapshot predictions (EOD>=18:00) | "
                f"Sample: {sample_pred.symbol} @ {local_ts.strftime('%Y-%m-%d %H:%M')} local"
            )
        else:
            logger.info(f"📊 Found {len(candidates)} daily snapshot predictions without outcomes (latest per symbol/horizon/day)")
        
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
            # Use business days instead of calendar days for eval timestamp
            eval_ts = _add_business_days(p.ts_pred, days)
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
            
            # Direction hit with threshold-based logic (fixes asymmetric zero bias)
            # Threshold: 0.5% - aligned with typical model prediction scale
            # Adjusted from 1.0% to 0.5% to match conservative model predictions
            threshold = float(os.getenv('DIRECTION_HIT_THRESHOLD', '0.005'))
            real_dir = 0 if abs(delta_real) < threshold else (1 if delta_real > 0 else -1)
            pred_dir = 0 if abs(delta_pred) < threshold else (1 if delta_pred > 0 else -1)
            
            # Logic:
            # - Both zero (no significant change) → HIT
            # - One zero, one non-zero → MISS (prediction failed to capture movement or predicted false movement)
            # - Same direction → HIT
            # - Opposite direction → MISS
            if real_dir == 0 and pred_dir == 0:
                dir_hit = True  # Both predicted and realized no significant change
            elif real_dir == 0 or pred_dir == 0:
                dir_hit = False  # One changed significantly, other didn't
            else:
                dir_hit = (real_dir == pred_dir)  # Same direction = HIT
            abs_err = abs(pred_price - price_eval)
            mape = abs_err / price_eval if price_eval else 0.0

            # ⚡ NEW: Magnitude-based hit calculation (keeps dir_hit unchanged for backward compatibility)
            # Calculate additional metrics for better evaluation:
            # 1. magnitude_hit: Direction is correct AND magnitude is within tolerance
            # 2. partial_hit: Partial score based on both direction and magnitude accuracy
            # Note: These variables are calculated but not stored in DB (can be computed on-the-fly when needed)
            _magnitude_hit = None  # noqa: F841
            _partial_hit = None  # noqa: F841
            try:
                # Magnitude tolerance: consider it a "magnitude hit" if abs_err is within 5% of actual price
                # or if both predicted and actual movements are small (within threshold)
                magnitude_tolerance = float(os.getenv('MAGNITUDE_HIT_TOLERANCE', '0.05'))  # 5% default
                
                # Direction hit (already calculated)
                dir_correct = dir_hit
                
                # Magnitude hit: direction correct AND price error within tolerance
                if dir_correct:
                    # If both movements are small (within threshold), consider it a magnitude hit
                    if abs(delta_real) < threshold and abs(delta_pred) < threshold:
                        _magnitude_hit = True  # noqa: F841  # Both predicted and realized small/no movement
                    else:
                        # Check if price error is within tolerance
                        if price_eval and abs_err / price_eval <= magnitude_tolerance:
                            _magnitude_hit = True  # noqa: F841
                        else:
                            _magnitude_hit = False  # noqa: F841
                else:
                    _magnitude_hit = False  # noqa: F841  # Direction wrong, cannot be magnitude hit
                
                # Partial hit: weighted score combining direction and magnitude
                # Score: 0.0 (completely wrong) to 1.0 (perfect prediction)
                # Note: These variables are calculated but not stored in DB (can be computed on-the-fly when needed)
                if dir_correct:
                    # Base score from direction correctness: 0.6 (60%)
                    base_score = 0.6
                    
                    # Magnitude accuracy score: 0.4 (40%)
                    if price_eval and price_eval > 0:
                        # Normalize error: smaller error = higher score
                        normalized_error = min(1.0, abs_err / (price_eval * magnitude_tolerance))
                        magnitude_score = 0.4 * (1.0 - normalized_error)
                    else:
                        magnitude_score = 0.0
                    
                    _partial_hit = base_score + magnitude_score  # noqa: F841
                else:
                    # Direction wrong: partial hit based on how close the magnitude was
                    if price_eval and price_eval > 0:
                        normalized_error = min(1.0, abs_err / (price_eval * magnitude_tolerance))
                        # Even if direction is wrong, give some credit for magnitude accuracy (max 0.3)
                        _partial_hit = 0.3 * (1.0 - normalized_error)  # noqa: F841
                    else:
                        _partial_hit = 0.0  # noqa: F841
            except Exception:
                # If calculation fails, leave as None (backward compatible)
                # Note: These variables are calculated but not stored in DB (see comment below)
                _magnitude_hit = None  # noqa: F841
                _partial_hit = None  # noqa: F841

            o = OutcomesLog(
                prediction_id=p.id,
                ts_eval=eval_ts,
                price_eval=price_eval,
                delta_real=delta_real,
                dir_hit=dir_hit,
                abs_err=abs_err,
                mape=mape,
            )
            # Note: magnitude_hit and partial_hit are calculated but not stored in DB
            # They can be computed on-the-fly when needed (backward compatible)
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
    try:
        return run(args.limit)
    finally:
        # ✅ FIX: Safely shutdown logging to prevent cleanup errors
        # This prevents "TypeError: 'NoneType' object is not callable" during cleanup
        try:
            logging.shutdown()
        except Exception:
            # Ignore any errors during logging shutdown (non-critical)
            pass


if __name__ == '__main__':
    raise SystemExit(main())
