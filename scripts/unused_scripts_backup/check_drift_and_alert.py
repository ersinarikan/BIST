#!/usr/bin/env python3
"""
Drift and health check with optional alerting and auto-rollback.

Compares recent window vs baseline in metrics_daily and triggers alerts
if accuracy drops or Brier worsens beyond thresholds. Optionally performs
auto rollback to previous param version.

Env/Args:
  - LOOKBACK_DAYS (default: 7)
  - BASELINE_DAYS (default: 21)
  - ACC_DROP_PCT (default: 0.10)  # 10% relative drop
  - BRIER_RISE_PCT (default: 0.20)  # 20% relative rise
  - MIN_ROWS (default: 100)  # minimum rows per window
  - SLACK_WEBHOOK_URL (optional)
  - AUTO_ROLLBACK (default: false)
"""
from __future__ import annotations

import os
import json
from datetime import date, timedelta
from typing import Optional, Tuple
from urllib import request

from app import app
from models import db, MetricsDaily
from sqlalchemy import func  # noqa: F401  # kept for potential aggregation extensions


def _mean_safe(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _window_stats(day: date, days: int) -> Tuple[Optional[float], Optional[float], int]:
    start = day - timedelta(days=days - 1)
    rows = (
        db.session.query(MetricsDaily)
        .filter(MetricsDaily.date >= start)
        .filter(MetricsDaily.date <= day)
        .all()
    )
    accs = [float(r.acc) for r in rows if r.acc is not None]
    briers = [float(r.brier) for r in rows if r.brier is not None]
    return _mean_safe(accs), _mean_safe(briers), len(rows)


def _post_slack(webhook: str, text: str) -> None:
    try:
        data = json.dumps({"text": text}).encode("utf-8")
        req = request.Request(webhook, data=data, headers={"Content-Type": "application/json"})
        request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def _auto_rollback_if_enabled(active_path: str) -> Optional[str]:
    if os.getenv("AUTO_ROLLBACK", "false").lower() != "true":
        return None
    versions_dir = os.path.join(os.path.dirname(active_path), "versions")
    if not os.path.isdir(versions_dir):
        return None
    # Pick previous version by mtime (excluding the active target)
    try:
        active_target = os.path.realpath(active_path) if os.path.islink(active_path) else ""
        candidates = []
        for name in os.listdir(versions_dir):
            path = os.path.join(versions_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.realpath(path) == active_target:
                continue
            candidates.append((os.path.getmtime(path), path))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        chosen = candidates[0][1]
        # Switch active symlink
        os.system(f"/opt/bist-pattern/scripts/rollback_params.sh '{chosen}' >/dev/null 2>&1")
        return chosen
    except Exception:
        return None


def main() -> int:
    lookback_days = int(os.getenv("LOOKBACK_DAYS", "7"))
    baseline_days = int(os.getenv("BASELINE_DAYS", "21"))
    acc_drop_pct = float(os.getenv("ACC_DROP_PCT", "0.10"))
    brier_rise_pct = float(os.getenv("BRIER_RISE_PCT", "0.20"))
    min_rows = int(os.getenv("MIN_ROWS", "100"))
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")

    today = date.today()

    with app.app_context():
        try:
            db.create_all()
        except Exception:
            pass

        recent_acc, recent_brier, recent_n = _window_stats(today, lookback_days)
        base_end = today - timedelta(days=lookback_days)
        baseline_acc, baseline_brier, baseline_n = _window_stats(base_end, baseline_days)

        status = {
            "recent_days": lookback_days,
            "baseline_days": baseline_days,
            "recent_rows": recent_n,
            "baseline_rows": baseline_n,
            "recent_acc": recent_acc,
            "baseline_acc": baseline_acc,
            "recent_brier": recent_brier,
            "baseline_brier": baseline_brier,
            "alert": False,
            "reasons": [],
        }

        if recent_n < min_rows or baseline_n < min_rows:
            # Not enough data for decision; just log
            print(json.dumps({"status": "insufficient_data", **status}))
            return 0

        # Compute relative changes
        if baseline_acc and recent_acc is not None:
            drop = (baseline_acc - recent_acc) / baseline_acc
            if drop >= acc_drop_pct:
                status["alert"] = True
                status["reasons"].append({"metric": "acc", "drop_pct": drop})
        if baseline_brier and recent_brier is not None:
            rise = (recent_brier - baseline_brier) / baseline_brier
            if rise >= brier_rise_pct:
                status["alert"] = True
                status["reasons"].append({"metric": "brier", "rise_pct": rise})

        # Decide and act
        if status["alert"]:
            text = (
                f"BIST-Pattern drift alert: recent({lookback_days}d) vs baseline({baseline_days}d). "
                f"acc_recent={recent_acc:.3f} acc_base={baseline_acc:.3f} "
                f"brier_recent={recent_brier:.3f} brier_base={baseline_brier:.3f}; reasons={status['reasons']}"
            )
            print(text)
            if webhook:
                _post_slack(webhook, text)
            # Optional rollback
            active_link = "/opt/bist-pattern/logs/params/active.json"
            rolled = _auto_rollback_if_enabled(active_link)
            if rolled:
                print(json.dumps({"rollback": rolled}))
        else:
            print(json.dumps({"status": "ok", **status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
