"""
Selection Policy Backtest

Compares three policies on recent history:
  - best_of (calibrated |delta| × reliability, enhanced-biased)
  - always_enhanced
  - always_basic

Horizon set: 1d, 3d, 7d
Universe: top N symbols with sufficient DB data
Window: last M trading days

Usage (inside project venv):
  python scripts/backtest_selection_policy.py
"""

from __future__ import annotations

import os
import sys
import math
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def tanh_calibrate(delta: float, tau_default: float = 0.08) -> float:
    try:
        tau = float(os.getenv('DELTA_CAL_TAU', str(tau_default)))
    except Exception:
        tau = tau_default
    try:
        return float(math.tanh(delta / max(1e-9, tau)) * tau)
    except Exception:
        return float(delta)


def load_predictors():
    """Lazy load ML systems with error insulation."""
    basic_ml = None
    enhanced = None
    try:
        from ml_prediction_system import get_ml_prediction_system  # type: ignore
        basic_ml = get_ml_prediction_system()
    except Exception as e:
        print(f"WARN basic_ml load: {e}")
    try:
        from enhanced_ml_system import get_enhanced_ml_system  # type: ignore
        enhanced = get_enhanced_ml_system()
    except Exception as e:
        print(f"WARN enhanced load: {e}")
    return basic_ml, enhanced


def pick_symbols(limit: int = 12, min_days: int = 200) -> List[str]:
    """Pick symbols with most recent rows >= min_days."""
    out: List[Tuple[str, int]] = []
    from app import app as _app
    from models import db as _db, Stock as _Stock, StockPrice as _StockPrice
    with _app.app_context():
        q = _db.session.query(_Stock.id, _Stock.symbol).all()
        for sid, sym in q:
            cnt = _db.session.query(_StockPrice).filter(_StockPrice.stock_id == sid).count()
            if cnt >= min_days:
                out.append((sym, cnt))
    out.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in out[:limit]]


def get_df(symbol: str, lookback_days: int = 280) -> pd.DataFrame:
    from app import app as _app
    from models import Stock as _Stock, StockPrice as _StockPrice
    with _app.app_context():
        stock = _Stock.query.filter_by(symbol=symbol.upper()).first()
        if not stock:
            return pd.DataFrame()
        rows = (
            _StockPrice.query.filter_by(stock_id=stock.id)
            .order_by(_StockPrice.date.desc())
            .limit(lookback_days)
            .all()
        )
    if not rows:
        return pd.DataFrame()
    data = [
        {
            'date': r.date,
            'open': float(r.open_price),
            'high': float(r.high_price),
            'low': float(r.low_price),
            'close': float(r.close_price),
            'volume': int(r.volume),
        }
        for r in reversed(rows)
    ]
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def predict_basic(basic_ml, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
    try:
        preds = basic_ml.predict_prices(symbol, df, None) or {}
        out: Dict[str, float] = {}
        for k, v in preds.items():
            if isinstance(v, (int, float)):
                out[str(k).lower()] = float(v)
            elif isinstance(v, dict):
                for cand in ('price', 'prediction', 'target', 'value', 'y'):
                    if isinstance(v.get(cand), (int, float)):
                        out[str(k).lower()] = float(v[cand])
                        break
        return out
    except Exception:
        return {}


def predict_enh(enhanced, symbol: str, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    try:
        # Ensure models are available (load pre-trained if exists)
        try:
            if enhanced.has_trained_models(symbol):  # type: ignore[attr-defined]
                enhanced.load_trained_models(symbol)  # type: ignore[attr-defined]
        except Exception:
            pass
        res = enhanced.predict_enhanced(symbol, df) or {}
        # Normalize
        out: Dict[str, Dict[str, Any]] = {}
        for h, obj in res.items():
            if isinstance(obj, dict):
                price = obj.get('ensemble_prediction')
                conf = obj.get('confidence')
                if isinstance(price, (int, float)):
                    out[str(h).lower()] = {
                        'price': float(price),
                        'confidence': float(conf) if isinstance(conf, (int, float)) else None,
                    }
        return out
    except Exception:
        return {}


def run_backtest(symbols: List[str], horizons: List[int], eval_points: int = 60, lookback_days: int = 280) -> Dict[str, Any]:
    basic_ml, enhanced = load_predictors()
    if enhanced is None and basic_ml is None:
        return {'error': 'predictors_unavailable'}

    try:
        basic_rel = float(os.getenv('BASIC_RELIABILITY', '0.6'))
    except Exception:
        basic_rel = 0.6
    base_w_basic = 0.6
    base_w_enh = 0.7

    stats = {
        'always_basic': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
        'always_enhanced': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
        'best_of': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
        'samples': 0,
    }

    for sym in symbols:
        df = get_df(sym, lookback_days=lookback_days)
        if df.empty or len(df) < 60:
            continue
        # Choose evenly spaced evaluation points in last 180 bars excluding last max(horizons)
        max_h = max(horizons)
        candidates_idx = list(range(30, max(31, len(df) - max_h)))
        # sample last eval_points
        eval_idx = candidates_idx[-eval_points:]
        for idx in eval_idx:
            df_cut = df.iloc[: idx + 1]
            p0 = float(df_cut['close'].iloc[-1])
            # Ground truth for each horizon
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

            for h in truths.keys():
                gt = truths[h]
                hkey = f"{h}d"

                # Basic pred
                bp = basic_map.get(hkey)
                # Enhanced pred
                ep_obj = enh_map.get(hkey)
                ep = None if not isinstance(ep_obj, dict) else ep_obj.get('price')
                er = None if not isinstance(ep_obj, dict) else ep_obj.get('confidence')

                # Skip if neither available
                if not isinstance(bp, (int, float)) and not isinstance(ep, (int, float)):
                    continue

                # Convert to deltas
                bd = None if not isinstance(bp, (int, float)) else (float(bp) - p0) / p0
                ed = None if not isinstance(ep, (int, float)) else (float(ep) - p0) / p0

                # Policies
                # always_basic
                if isinstance(bd, (int, float)):
                    stats['always_basic']['n'] += 1
                    stats['always_basic']['dir_acc'] += int((bd >= 0 and gt >= 0) or (bd < 0 and gt < 0))
                    stats['always_basic']['mae'] += abs((bd) - gt)

                # always_enhanced
                if isinstance(ed, (int, float)):
                    stats['always_enhanced']['n'] += 1
                    stats['always_enhanced']['dir_acc'] += int((ed >= 0 and gt >= 0) or (ed < 0 and gt < 0))
                    stats['always_enhanced']['mae'] += abs((ed) - gt)

                # best_of
                pick_delta = None
                if isinstance(bd, (int, float)) and isinstance(ed, (int, float)):
                    c_bd = tanh_calibrate(bd)
                    c_ed = tanh_calibrate(ed)
                    rel_e = er
                    if not isinstance(rel_e, (int, float)):
                        rel_e = 0.65
                    score_b = abs(c_bd) * basic_rel * base_w_basic
                    score_e = abs(c_ed) * float(max(0.0, min(1.0, rel_e))) * base_w_enh
                    pick_delta = ed if score_e >= score_b else bd
                elif isinstance(ed, (int, float)):
                    pick_delta = ed
                else:
                    pick_delta = bd

                if isinstance(pick_delta, (int, float)):
                    stats['best_of']['n'] += 1
                    stats['best_of']['dir_acc'] += int((pick_delta >= 0 and gt >= 0) or (pick_delta < 0 and gt < 0))
                    stats['best_of']['mae'] += abs((pick_delta) - gt)

            stats['samples'] += 1

    # Finalize
    out = {}
    for k, v in stats.items():
        if k == 'samples':
            continue
        n = max(1, v['n'])
        out[k] = {
            'n': v['n'],
            'directional_accuracy': round(v['dir_acc'] / n, 4),
            'mae_delta': round(v['mae'] / n, 5),
        }
    out['meta'] = {
        'symbols': symbols,
        'horizons': horizons,
        'generated_at': datetime.now().isoformat(),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description='Selection policy backtest')
    parser.add_argument('--limit', type=int, default=12, help='number of symbols')
    parser.add_argument('--horizons', type=str, default='1,3,7', help='comma-separated horizons (days)')
    parser.add_argument('--eval-points', type=int, default=60, help='evaluation points per symbol')
    parser.add_argument('--lookback-days', type=int, default=280, help='lookback days to fetch')
    parser.add_argument('--min-days', type=int, default=int(os.getenv('ML_MIN_DATA_DAYS', '180')), help='min rows per symbol')
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(',') if x.strip().isdigit()]
    if not horizons:
        horizons = [1, 3, 7]
    syms = pick_symbols(limit=args.limit, min_days=args.min_days)
    if not syms:
        print("No symbols available for backtest")
        return
    res = run_backtest(syms, horizons, eval_points=args.eval_points, lookback_days=args.lookback_days)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
