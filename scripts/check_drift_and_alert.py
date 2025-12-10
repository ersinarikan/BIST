#!/usr/bin/env python3
"""
Check for model drift and send alerts if detected.

This script monitors prediction performance over time and alerts if
significant drift is detected. Currently a placeholder - implement drift
detection logic as needed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict

from app import app
from models import db, PredictionsLog, OutcomesLog


def check_drift(window_days: int = 30, threshold: float = 0.10) -> Dict:
    """Check for model drift by comparing recent performance to baseline.

    Args:
        window_days: Number of days to look back for recent performance
        threshold: Minimum performance drop to consider as drift
            (e.g., 0.10 = 10%)

    Returns:
        Dict with drift status and details
    """
    cutoff = datetime.utcnow() - timedelta(days=window_days)

    from contextlib import AbstractContextManager
    from typing import cast
    app_ctx = cast(AbstractContextManager[None], app.app_context())
    with app_ctx:
        # Get recent performance
        recent_rows = (
            db.session.query(PredictionsLog, OutcomesLog)
            .join(OutcomesLog, OutcomesLog.prediction_id == PredictionsLog.id)
            .filter(PredictionsLog.ts_pred >= cutoff)
            .all()
        )

        if not recent_rows:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough data to check for drift'
            }

        # Calculate recent accuracy
        recent_hits = sum(1 for _, o in recent_rows if bool(o.dir_hit))
        recent_total = len(recent_rows)
        recent_acc = recent_hits / recent_total if recent_total > 0 else 0.0

        # TODO: Compare with baseline (e.g., previous period or expected
        # performance)
        # For now, just return current status
        return {
            'status': 'ok',
            'recent_accuracy': recent_acc,
            'recent_samples': recent_total,
            'window_days': window_days,
            'checked_at': datetime.utcnow().isoformat()
        }


def main() -> int:
    """Main entry point for drift check script."""
    out_dir = os.getenv('BIST_LOG_PATH', '/opt/bist-pattern/logs')
    os.makedirs(out_dir, exist_ok=True)

    result = check_drift()

    # Log result
    log_path = os.path.join(out_dir, 'drift_check.log')
    with open(log_path, 'a') as f:
        f.write(f"[{datetime.utcnow().isoformat()}] {json.dumps(result)}\n")

    print(json.dumps({'status': 'ok', 'result': result}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
