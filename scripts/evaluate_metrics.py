#!/usr/bin/env python3
"""
Aggregate daily metrics from predictions_log and outcomes_log.

Usage (cron-safe):
  venv/bin/python3 scripts/evaluate_metrics.py --date 2025-10-07

Writes/updates rows in metrics_daily for each symbol×horizon seen that day.
Best-effort; never raises to cron on partial errors.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, List
import json
import math

from models import db, PredictionsLog, OutcomesLog, MetricsDaily
from app import app


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _load_param_store() -> Dict:
    """Load thresholds from param_store.json if present."""
    try:
        base = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
        path = os.path.join(base, 'param_store.json')
        if os.path.exists(path):
            with open(path, 'r') as rf:
                data = json.load(rf) or {}
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _get_thresholds(store: Dict, horizon: str) -> Tuple[float, float]:
    try:
        h = (store.get('horizons', {}) or {}).get(horizon) or {}
        th = h.get('thresholds') or {}
        d = float(th.get('delta_thr', 0.03))
        c = float(th.get('conf_thr', 0.65))
        return d, c
    except Exception:
        return 0.03, 0.65


def _compute_sharpe(returns: List[float]) -> float | None:
    vals = [r for r in returns if r is not None]
    if len(vals) < 2:
        return None
    mean_r = sum(vals) / len(vals)
    var = sum((r - mean_r) * (r - mean_r) for r in vals) / (len(vals) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return None
    return mean_r / std * math.sqrt(len(vals))


def _compute_max_drawdown(returns: List[float]) -> float | None:
    if not returns:
        return None
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    for r in returns:
        cum += r
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
    return mdd


def compute_metrics(day: date) -> None:
    with app.app_context():
        # Ensure tables exist
        try:
            db.create_all()
        except Exception:
            pass
        param_store = _load_param_store()
        # Join predictions and outcomes for the market day
        # MARKET_TZ_OFFSET_HOURS allows aligning market local day to UTC window
        try:
            tz_off = int(os.getenv('MARKET_TZ_OFFSET_HOURS', '0'))
        except Exception:
            tz_off = 0
        day_start_local = datetime(day.year, day.month, day.day)
        # Convert local day window to UTC by subtracting offset
        day_start_utc = day_start_local - timedelta(hours=tz_off)
        day_end_utc = day_start_utc + timedelta(days=1)
        rows = (
            db.session.query(PredictionsLog, OutcomesLog)
            .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
            .filter(OutcomesLog.ts_eval >= day_start_utc)
            .filter(OutcomesLog.ts_eval < day_end_utc)
            .all()
        )

        # Group by (symbol, horizon)
        buckets: Dict[Tuple[str, str], list] = {}
        for p, o in rows:
            key = (p.symbol, p.horizon)
            buckets.setdefault(key, []).append((p, o))

        for (symbol, horizon), items in buckets.items():
            n = len(items)
            if n == 0:
                continue

            # Directional metrics
            hits = sum(1 for _p, o in items if bool(o.dir_hit))
            acc = hits / n if n else 0.0

            # Error metrics
            mae_vals = [abs(_safe_float(o.abs_err) or 0.0) for _p, o in items]
            mape_vals = [abs(_safe_float(o.mape) or 0.0) for _p, o in items]
            mae = sum(mae_vals) / n if n else None
            mape = sum(mape_vals) / n if n else None

            # Brier score (if prediction confidence present)
            briers = []
            for p, o in items:
                c = _safe_float(p.confidence)
                if c is None:
                    continue
                y = 1.0 if bool(o.dir_hit) else 0.0
                briers.append((c - y) * (c - y))
            brier = (sum(briers) / len(briers)) if briers else None

            # Thresholded classification (precision/recall) and PnL metrics
            delta_thr, conf_thr = _get_thresholds(param_store, horizon)
            tp = fp = fn = 0
            trade_returns: List[float] = []
            # Sort by evaluation time for drawdown
            items_sorted = sorted(items, key=lambda x: getattr(x[1], 'ts_eval', None) or datetime.min)
            for p, o in items_sorted:
                d_pred = _safe_float(p.delta_pred)
                if d_pred is None:
                    # try reconstruct
                    price_now = _safe_float(p.price_now)
                    pred_price = _safe_float(p.pred_price)
                    if price_now is not None and pred_price is not None and price_now != 0:
                        d_pred = (pred_price - price_now) / price_now
                d_pred = d_pred if d_pred is not None else 0.0
                conf = _safe_float(p.confidence) or 0.0
                predicted = (abs(d_pred) >= delta_thr) and (conf >= conf_thr)
                hit = bool(o.dir_hit)
                if predicted and hit:
                    tp += 1
                elif predicted and (not hit):
                    fp += 1
                elif (not predicted) and hit:
                    fn += 1
                # PnL: take trades only when predicted; long if d_pred>=0 else short
                if predicted:
                    d_real = _safe_float(o.delta_real) or 0.0
                    direction = 1.0 if d_pred >= 0 else -1.0
                    trade_returns.append(direction * d_real)
            precision = (tp / (tp + fp)) if (tp + fp) else None
            recall = (tp / (tp + fn)) if (tp + fn) else None
            pnl_pct = (sum(trade_returns) if trade_returns else None)
            sharpe = _compute_sharpe(trade_returns) if trade_returns else None
            max_dd = _compute_max_drawdown(trade_returns) if trade_returns else None

            # Basic write/update
            rec = (
                db.session.query(MetricsDaily)
                .filter_by(date=day, symbol=symbol, horizon=horizon)
                .first()
            )
            if not rec:
                rec = MetricsDaily(date=day, symbol=symbol, horizon=horizon)
                db.session.add(rec)
            rec.acc = acc
            rec.precision = precision
            rec.recall = recall
            rec.mae = mae
            rec.mape = mape
            rec.brier = brier
            # Store PnL in percentage points (e.g. 2.5 => +2.5%)
            rec.pnl = (pnl_pct * 100.0) if pnl_pct is not None else None
            rec.sharpe = sharpe
            rec.max_dd = (max_dd * 100.0) if max_dd is not None else None
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None, help='YYYY-MM-DD (default: today)')
    args = parser.parse_args()

    if args.date:
        day = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        day = datetime.utcnow().date()

    # Ensure log path exists for cron friendliness
    os.makedirs(os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs'), exist_ok=True)

    compute_metrics(day)
    print(f"metrics_daily updated for {day}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
