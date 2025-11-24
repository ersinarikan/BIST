"""
Threshold Calibration Script

Calibrates decision thresholds (in percent) for 1/3/7/14/30 day horizons by
maximizing directional accuracy on walk-forward backtests while reporting
coverage. Uses the same selection policy as the UI (enhanced-first best-of).

Usage:
  python scripts/calibrate_thresholds.py --limit 50 --horizons 1,3,7,14,30 \
      --eval-points 60 --lookback-days 365 --min-coverage 0.2

Outputs JSON to stdout with recommended thresholds per horizon (percent).
"""

from __future__ import annotations

import os
import sys
import json
import math
from typing import Dict, List, Tuple, Any
from datetime import datetime
import argparse

import numpy as np
import pandas as pd  # noqa: F401  # imported for consistency with helpers

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Reuse helpers from selection backtest
from scripts.backtest_selection_policy import (  # type: ignore  # noqa: E402
    pick_symbols,
    get_df,
    tanh_calibrate,
    predict_basic,
    predict_enh,
    load_predictors,
)


def compute_pairs(symbols: List[str], horizons: List[int], eval_points: int, lookback_days: int) -> Dict[int, List[Tuple[float, float]]]:
    """Return {horizon_days: [(pred_delta, gt_delta), ...]} using enhanced-first policy."""
    basic_ml, enhanced = load_predictors()
    pairs: Dict[int, List[Tuple[float, float]]] = {h: [] for h in horizons}

    # Reliability defaults
    try:
        basic_rel = float(os.getenv('BASIC_RELIABILITY', '0.6'))
    except Exception:
        basic_rel = 0.6
    base_w_basic = 0.6
    base_w_enh = 0.7

    for sym in symbols:
        df = get_df(sym, lookback_days=lookback_days)
        if df.empty or len(df) < 60:
            continue
        max_h = max(horizons)
        candidates_idx = list(range(30, max(31, len(df) - max_h)))
        eval_idx = candidates_idx[-eval_points:]
        for idx in eval_idx:
            df_cut = df.iloc[: idx + 1]
            p0 = float(df_cut['close'].iloc[-1])
            truths: Dict[int, float] = {}
            for h in horizons:
                if idx + h < len(df):
                    p1 = float(df['close'].iloc[idx + h])
                    truths[h] = (p1 - p0) / p0
            if not truths:
                continue
            # Predictions
            basic_map = predict_basic(basic_ml, sym, df_cut) if basic_ml else {}
            enh_map = predict_enh(enhanced, sym, df_cut) if enhanced else {}
            for h, gt in truths.items():
                hkey = f"{h}d"
                bp = basic_map.get(hkey)
                ep_obj = enh_map.get(hkey)
                ep = None if not isinstance(ep_obj, dict) else ep_obj.get('price')
                er = None if not isinstance(ep_obj, dict) else ep_obj.get('confidence')

                if not isinstance(bp, (int, float)) and not isinstance(ep, (int, float)):
                    continue
                # Convert to deltas
                bd = None if not isinstance(bp, (int, float)) else (float(bp) - p0) / p0
                ed = None if not isinstance(ep, (int, float)) else (float(ep) - p0) / p0

                # Selection policy (enhanced-first best-of)
                pick_delta: float
                if isinstance(bd, (int, float)) and isinstance(ed, (int, float)):
                    c_bd = tanh_calibrate(bd)
                    c_ed = tanh_calibrate(ed)
                    rel_e = er if isinstance(er, (int, float)) else 0.65
                    score_b = abs(c_bd) * basic_rel * base_w_basic
                    score_e = abs(c_ed) * float(max(0.0, min(1.0, rel_e))) * base_w_enh
                    pick_delta = ed if score_e >= score_b else bd
                elif isinstance(ed, (int, float)):
                    pick_delta = ed
                else:
                    pick_delta = bd  # type: ignore[assignment]

                if isinstance(pick_delta, (int, float)):
                    pairs[h].append((float(pick_delta), float(gt)))
    return pairs


def calibrate_threshold(pairs: List[Tuple[float, float]], min_coverage: float = 0.2) -> Tuple[float, Dict[str, Any]]:
    """Find threshold (percent) maximizing directional accuracy on |pred|>=t subset.

    Returns (best_threshold_percent, metrics)
    """
    if not pairs:
        return 1.0, {'accuracy': None, 'coverage': 0.0, 'samples': 0}

    preds = np.array([p for p, _ in pairs], dtype=float)
    gts = np.array([g for _, g in pairs], dtype=float)
    total = len(pairs)
    best_t = 1.0
    best_acc = -1.0
    best_cov = 0.0

    # Grid from 0.1% to 3.0% (inclusive)
    grid = np.round(np.arange(0.001, 0.030 + 1e-9, 0.001), 3)
    for t in grid:
        mask = np.abs(preds) >= t
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(((np.sign(preds[mask]) == np.sign(gts[mask])).sum()) / max(1, n))
        cov = n / float(total)
        # require minimal coverage; otherwise skip unless no candidate yet
        if cov < min_coverage and best_acc >= 0:
            continue
        # choose higher accuracy; tie-break by higher coverage then lower threshold
        if acc > best_acc or (math.isclose(acc, best_acc, rel_tol=1e-4) and (cov > best_cov or (math.isclose(cov, best_cov, rel_tol=1e-4) and t < best_t))):
            best_t, best_acc, best_cov = t, acc, cov

    # Convert to percent for UI (e.g., 0.012 -> 1.2)
    return best_t * 100.0, {
        'accuracy': round(best_acc, 4) if best_acc >= 0 else None,
        'coverage': round(best_cov, 4),
        'samples': total,
    }


def main():
    parser = argparse.ArgumentParser(description='Calibrate UI thresholds by walk-forward backtest')
    parser.add_argument('--limit', type=int, default=50, help='number of symbols')
    parser.add_argument('--horizons', type=str, default='1,3,7,14,30', help='comma-separated horizons (days)')
    parser.add_argument('--eval-points', type=int, default=60, help='evaluation points per symbol')
    parser.add_argument('--lookback-days', type=int, default=365, help='lookback days to fetch')
    parser.add_argument('--min-days', type=int, default=int(os.getenv('ML_MIN_DATA_DAYS', '200')), help='min rows per symbol')
    parser.add_argument('--min-coverage', type=float, default=0.2, help='minimum prediction coverage to accept a threshold')
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(',') if x.strip().isdigit()]
    if not horizons:
        horizons = [1, 3, 7, 14, 30]
    syms = pick_symbols(limit=args.limit, min_days=args.min_days)
    if not syms:
        print(json.dumps({'status': 'error', 'message': 'no_symbols'}))
        return

    pairs_map = compute_pairs(syms, horizons, eval_points=args.eval_points, lookback_days=args.lookback_days)
    out: Dict[str, Any] = {
        'status': 'success',
        'generated_at': datetime.now().isoformat(),
        'meta': {
            'limit': args.limit,
            'horizons': horizons,
            'eval_points': args.eval_points,
            'lookback_days': args.lookback_days,
            'min_coverage': args.min_coverage,
            'symbols': syms,
        },
        'thresholds_percent': {},
        'metrics': {},
    }

    for h in horizons:
        best_t, metrics = calibrate_threshold(pairs_map.get(h, []), min_coverage=args.min_coverage)
        out['thresholds_percent'][f'{h}d'] = round(float(best_t), 2)
        out['metrics'][f'{h}d'] = metrics

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
