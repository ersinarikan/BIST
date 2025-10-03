"""
Walk-Forward Comparison Report

Compares selection policies over multiple horizons using walk-forward:
  - always_basic
  - always_enhanced
  - best_of_v1 (legacy tanh-calibrated, fixed weights)
  - best_of_v2 (regime-aware + horizon-scaled, matches server policy)

Usage:
  python scripts/walkforward_compare.py --limit 50 --horizons 1,3,7,14,30 \
      --eval-points 60 --lookback-days 365
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
import argparse
import logging

# numpy not required here
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Reduce noisy logs
logging.getLogger('enhanced_ml_system').setLevel(logging.WARNING)

from scripts.backtest_selection_policy import (  # type: ignore  # noqa: E402
    pick_symbols,
    get_df,
    tanh_calibrate,
    predict_basic,
    predict_enh,
    load_predictors,
)


def regime_score(df: pd.DataFrame) -> float:
    try:
        ret20 = df['close'].pct_change().tail(20)
        ret60 = df['close'].pct_change().tail(60)
        vol20 = float(ret20.std()) if len(ret20) > 5 else 0.0
        vol60 = float(ret60.std()) if len(ret60) > 5 else 0.0
        if vol60 > 0:
            r = vol20 / vol60
        else:
            r = vol20 / 0.05
        return float(min(1.0, max(0.0, r)))
    except Exception:
        return 0.5


def run_compare(symbols: List[str], horizons: List[int], eval_points: int = 60, lookback_days: int = 365) -> Dict[str, Any]:
    basic_ml, enhanced = load_predictors()
    stats: Dict[str, Dict[str, float]] = {
        'always_basic': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
        'always_enhanced': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
        'best_of_v1': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
        'best_of_v2': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
    }

    # Constants
    try:
        basic_rel = float(os.getenv('BASIC_RELIABILITY', '0.6'))
    except Exception:
        basic_rel = 0.6
    base_w_basic = 0.6
    base_w_enh = 0.7

    total = len(symbols)
    for idx_sym, sym in enumerate(symbols, start=1):
        print(f"[progress] {idx_sym}/{total} symbol={sym}", flush=True)
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

            reg = regime_score(df_cut)

            for h in truths.keys():
                gt = truths[h]
                hkey = f"{h}d"
                bp = basic_map.get(hkey)
                ep_obj = enh_map.get(hkey)
                ep = None if not isinstance(ep_obj, dict) else ep_obj.get('price')
                er = None if not isinstance(ep_obj, dict) else ep_obj.get('confidence')

                if not isinstance(bp, (int, float)) and not isinstance(ep, (int, float)):
                    continue

                bd = None if not isinstance(bp, (int, float)) else (float(bp) - p0) / p0
                ed = None if not isinstance(ep, (int, float)) else (float(ep) - p0) / p0

                # always_basic
                if isinstance(bd, (int, float)):
                    stats['always_basic']['n'] += 1
                    stats['always_basic']['dir_acc'] += int((bd >= 0 and gt >= 0) or (bd < 0 and gt < 0))
                    stats['always_basic']['mae'] += abs(bd - gt)

                # always_enhanced
                if isinstance(ed, (int, float)):
                    stats['always_enhanced']['n'] += 1
                    stats['always_enhanced']['dir_acc'] += int((ed >= 0 and gt >= 0) or (ed < 0 and gt < 0))
                    stats['always_enhanced']['mae'] += abs(ed - gt)

                # best_of_v1 (legacy)
                pick = None
                if isinstance(bd, (int, float)) and isinstance(ed, (int, float)):
                    c_bd = tanh_calibrate(bd)
                    c_ed = tanh_calibrate(ed)
                    rel_e = er if isinstance(er, (int, float)) else 0.65
                    score_b = abs(c_bd) * basic_rel * base_w_basic
                    score_e = abs(c_ed) * float(max(0.0, min(1.0, rel_e))) * base_w_enh
                    pick = ed if score_e >= score_b else bd
                elif isinstance(ed, (int, float)):
                    pick = ed
                else:
                    pick = bd
                if isinstance(pick, (int, float)):
                    stats['best_of_v1']['n'] += 1
                    stats['best_of_v1']['dir_acc'] += int((pick >= 0 and gt >= 0) or (pick < 0 and gt < 0))
                    stats['best_of_v1']['mae'] += abs(pick - gt)

                # best_of_v2 (regime-aware + horizon-scaled)
                pick2 = None
                if isinstance(bd, (int, float)) and isinstance(ed, (int, float)):
                    c_bd = tanh_calibrate(bd)
                    c_ed = tanh_calibrate(ed)
                    # keep variable for readability without linter warning
                    _ = max(1, h)
                    base_w_e = 0.6 + 0.2 * reg
                    base_w_b = 0.65 - 0.15 * reg
                    rel_e = er if isinstance(er, (int, float)) else 0.65
                    score_b2 = abs(c_bd) * basic_rel * base_w_b
                    score_e2 = abs(c_ed) * float(max(0.0, min(1.0, rel_e))) * base_w_e
                    # slight bonus in high-vol regime
                    if reg >= 0.6:
                        score_e2 *= 1.15
                    pick2 = ed if score_e2 >= score_b2 else bd
                elif isinstance(ed, (int, float)):
                    pick2 = ed
                else:
                    pick2 = bd
                if isinstance(pick2, (int, float)):
                    stats['best_of_v2']['n'] += 1
                    stats['best_of_v2']['dir_acc'] += int((pick2 >= 0 and gt >= 0) or (pick2 < 0 and gt < 0))
                    stats['best_of_v2']['mae'] += abs(pick2 - gt)
        # periodic checkpoint
        if idx_sym % 5 == 0:
            try:
                chk = {k: {'n': int(v['n']), 'acc': round((v['dir_acc'] / max(1, int(v['n']))), 4)} for k, v in stats.items()}
                print(f"[checkpoint] after {idx_sym}/{total}: {json.dumps(chk)}", flush=True)
            except Exception:
                pass

    # Finalize
    out: Dict[str, Any] = {}
    for k, v in stats.items():
        n = max(1, int(v['n']))
        out[k] = {
            'n': int(v['n']),
            'directional_accuracy': round(float(v['dir_acc']) / n, 4),
            'mae_delta': round(float(v['mae']) / n, 5),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description='Walk-forward report for selection policies')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--horizons', type=str, default='1,3,7,14,30')
    parser.add_argument('--eval-points', type=int, default=60)
    parser.add_argument('--lookback-days', type=int, default=365)
    parser.add_argument('--min-days', type=int, default=int(os.getenv('ML_MIN_DATA_DAYS', '200')))
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(',') if x.strip().isdigit()]
    if not horizons:
        horizons = [1, 3, 7, 14, 30]
    syms = pick_symbols(limit=args.limit, min_days=args.min_days)
    if not syms:
        print(json.dumps({'status': 'error', 'message': 'no_symbols'}))
        return

    res = run_compare(syms, horizons, eval_points=args.eval_points, lookback_days=args.lookback_days)
    print(json.dumps({
        'status': 'success',
        'generated_at': datetime.now().isoformat(),
        'meta': {
            'limit': args.limit,
            'symbols': syms,
            'horizons': horizons,
            'eval_points': args.eval_points,
            'lookback_days': args.lookback_days,
        },
        'results': res
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
