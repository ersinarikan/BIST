"""
Walk-Forward Meta-Stacking Test (1D/3D/7D)

Meta-features combine base/enh deltas, reliability, regime and simple micro-features
to predict delta via ridge regression and direction via logistic regression.

Usage:
  python scripts/walkforward_meta_stacking.py --limit 40 --horizons 1,3,7 --eval-points 60 --lookback-days 365
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.backtest_selection_policy import (  # type: ignore  # noqa: E402
    pick_symbols,
    get_df,
    tanh_calibrate,
    predict_basic,
    predict_enh,
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


def micro_features(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        close = df['close']
        high = df['high']
        low = df['low']
        # RSI(3)
        delta = close.diff()
        up = delta.clip(lower=0).rolling(3).mean()
        down = (-delta.clip(upper=0)).rolling(3).mean()
        rs = up / (down + 1e-9)
        rsi3 = 100.0 - (100.0 / (1.0 + rs))
        out['rsi3'] = float(rsi3.iloc[-1]) if not np.isnan(rsi3.iloc[-1]) else 50.0
        # Boll position (20,2)
        ma20 = close.rolling(20).mean()
        sd20 = close.rolling(20).std()
        boll = (close - ma20) / (sd20 + 1e-9)
        out['boll_pos'] = float(boll.iloc[-1]) if not np.isnan(boll.iloc[-1]) else 0.0
        # Tail asymmetry today
        tail_up = ((high - close) / (close + 1e-9)) * 100.0
        tail_dn = ((close - low) / (close + 1e-9)) * 100.0
        out['tail_asym'] = float((tail_up.iloc[-1] - tail_dn.iloc[-1]))
        # Recent momentum
        out['ret1'] = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0.0
        out['mom3'] = float((close.iloc[-1] / (close.rolling(3).mean().iloc[-1] + 1e-9)) - 1.0) if len(close) >= 3 else 0.0
        # TA-Lib candlestick patterns (subset) → last-3 days bull/bear counts
        try:
            import talib  # type: ignore
            op = df['open'] if 'open' in df else close.shift(1).fillna(method='bfill')
            hi = high
            lo = low
            cl = close
            # Use a small subset for speed/robustness
            pat_series = []
            pat_series.append(talib.CDLENGULFING(op, hi, lo, cl))
            pat_series.append(talib.CDLHAMMER(op, hi, lo, cl))
            pat_series.append(talib.CDLSHOOTINGSTAR(op, hi, lo, cl))
            pat_series.append(talib.CDLHARAMI(op, hi, lo, cl))
            pat_series.append(talib.CDLDOJI(op, hi, lo, cl))
            pats = sum(pat_series)
            last3 = pats.tail(3)
            bull3 = int((last3 > 0).sum())
            bear3 = int((last3 < 0).sum())
            out['pat_bull3'] = float(bull3)
            out['pat_bear3'] = float(bear3)
            out['pat_net3'] = float(bull3 - bear3)
            # Strength today (normalized)
            today_raw = float(pats.iloc[-1]) if len(pats) else 0.0
            out['pat_today'] = max(-1.0, min(1.0, today_raw / 100.0))
        except Exception:
            out.setdefault('pat_bull3', 0.0)
            out.setdefault('pat_bear3', 0.0)
            out.setdefault('pat_net3', 0.0)
            out.setdefault('pat_today', 0.0)
    except Exception:
        # Defaults
        out.setdefault('rsi3', 50.0)
        out.setdefault('boll_pos', 0.0)
        out.setdefault('tail_asym', 0.0)
        out.setdefault('ret1', 0.0)
        out.setdefault('mom3', 0.0)
        out.setdefault('pat_bull3', 0.0)
        out.setdefault('pat_bear3', 0.0)
        out.setdefault('pat_net3', 0.0)
        out.setdefault('pat_today', 0.0)
    return out


def compute_booster_proba(df: pd.DataFrame) -> float | None:
    try:
        from scripts.walkforward_boost_compare import _features_1d  # type: ignore
    except Exception:
        return None
    feats = _features_1d(df)
    close = df['close']
    y = np.sign(close.shift(-1) - close)
    y = y.map({-1.0: 0, 0.0: 0, 1.0: 1}).astype('float')
    mask = feats.notna().all(axis=1) & y.notna()
    feats = feats.loc[mask]
    y = y.loc[mask]
    if len(feats) < 140:
        return None
    X_train = feats.iloc[:-1]
    y_train = y.iloc[:-1]
    X_last = feats.iloc[[-1]]
    if len(np.unique(y_train)) < 2:
        return None
    clf = LogisticRegression(max_iter=300, solver='liblinear')
    clf.fit(X_train.values, y_train.values)
    p_up = float(clf.predict_proba(X_last.values)[0, 1])
    return p_up


def build_feature_vector(df_cut: pd.DataFrame, bd: float | None, ed: float | None, er: float | None) -> Tuple[np.ndarray, List[str]]:
    reg = regime_score(df_cut)
    mic = micro_features(df_cut)
    bprob = compute_booster_proba(df_cut)
    c_bd = tanh_calibrate(bd) if isinstance(bd, (int, float)) else 0.0
    c_ed = tanh_calibrate(ed) if isinstance(ed, (int, float)) else 0.0
    rel_e = float(er) if isinstance(er, (int, float)) else 0.65
    x_list: List[float] = [
        c_bd, c_ed, float(bd) if isinstance(bd, (int, float)) else 0.0, float(ed) if isinstance(ed, (int, float)) else 0.0,
        rel_e, reg,
        mic.get('rsi3', 50.0), mic.get('boll_pos', 0.0), mic.get('tail_asym', 0.0), mic.get('ret1', 0.0), mic.get('mom3', 0.0),
        mic.get('pat_bull3', 0.0), mic.get('pat_bear3', 0.0), mic.get('pat_net3', 0.0), mic.get('pat_today', 0.0),
        (float(bprob) if isinstance(bprob, float) else 0.5),
    ]
    names = ['c_bd', 'c_ed', 'bd', 'ed', 'rel_e', 'reg', 'rsi3', 'boll_pos', 'tail_asym', 'ret1', 'mom3', 'pat_bull3', 'pat_bear3', 'pat_net3', 'pat_today', 'boost_p']
    return np.array(x_list, dtype=float), names


def run_meta_stacking(symbols: List[str], horizons: List[int], eval_points: int = 60, lookback_days: int = 365) -> Dict[str, Any]:
    # Per-horizon models
    models_reg: Dict[int, Ridge] = {h: Ridge(alpha=0.0, random_state=42) for h in horizons}
    models_cls: Dict[int, LogisticRegression] = {h: LogisticRegression(max_iter=400, solver='liblinear') for h in horizons}
    buffers_X: Dict[int, List[np.ndarray]] = {h: [] for h in horizons}
    buffers_y: Dict[int, List[float]] = {h: [] for h in horizons}
    buffers_s: Dict[int, List[int]] = {h: [] for h in horizons}  # sign labels

    stats_base: Dict[int, Dict[str, float]] = {h: {'n': 0, 'acc': 0, 'mae': 0.0} for h in horizons}
    stats_meta: Dict[int, Dict[str, float]] = {h: {'n': 0, 'acc': 0, 'mae': 0.0} for h in horizons}

    try:
        from scripts.backtest_selection_policy import load_predictors  # type: ignore
        basic_ml, enhanced = load_predictors()
    except Exception:
        basic_ml, enhanced = (None, None)

    for sym in symbols:
        df = get_df(sym, lookback_days=lookback_days)
        if df.empty or len(df) < 200:
            continue
        max_h = max(horizons)
        candidates_idx = list(range(60, max(61, len(df) - max_h)))
        eval_idx = candidates_idx[-eval_points:]
        for idx in eval_idx:
            df_cut = df.iloc[: idx + 1]
            p0 = float(df_cut['close'].iloc[-1])
            # Ground truth per horizon
            truths: Dict[int, float] = {}
            for h in horizons:
                if idx + h < len(df):
                    p1 = float(df['close'].iloc[idx + h])
                    truths[h] = (p1 - p0) / p0
            if not truths:
                continue

            basic_map = predict_basic(basic_ml, sym, df_cut) if basic_ml else {}
            enh_map = predict_enh(enhanced, sym, df_cut) if enhanced else {}

            for h in horizons:
                gt = truths.get(h)
                if gt is None:
                    continue
                hkey = f"{h}d"
                bp = basic_map.get(hkey)
                ep_obj = enh_map.get(hkey)
                ep = None if not isinstance(ep_obj, dict) else ep_obj.get('price')
                er = None if not isinstance(ep_obj, dict) else ep_obj.get('confidence')

                if not isinstance(bp, (int, float)) and not isinstance(ep, (int, float)):
                    continue
                bd = None if not isinstance(bp, (int, float)) else (float(bp) - p0) / p0
                ed = None if not isinstance(ep, (int, float)) else (float(ep) - p0) / p0

                # Baseline best_of_v2
                c_bd = tanh_calibrate(bd) if isinstance(bd, (int, float)) else 0.0
                c_ed = tanh_calibrate(ed) if isinstance(ed, (int, float)) else 0.0
                reg = regime_score(df_cut)
                base_w_e = 0.6 + 0.2 * reg
                base_w_b = 0.65 - 0.15 * reg
                rel_e = er if isinstance(er, (int, float)) else 0.65
                try:
                    basic_rel = float(os.getenv('BASIC_RELIABILITY', '0.6'))
                except Exception:
                    basic_rel = 0.6
                score_b2 = abs(c_bd) * basic_rel * base_w_b
                score_e2 = abs(c_ed) * float(max(0.0, min(1.0, rel_e))) * base_w_e
                if reg >= 0.6:
                    score_e2 *= 1.15
                pick2 = ed if (isinstance(bd, (int, float)) and isinstance(ed, (int, float)) and score_e2 >= score_b2) else (ed if isinstance(ed, (int, float)) else bd)

                if isinstance(pick2, (int, float)):
                    stats_base[h]['n'] += 1
                    stats_base[h]['acc'] += int((pick2 >= 0 and gt >= 0) or (pick2 < 0 and gt < 0))
                    stats_base[h]['mae'] += abs(pick2 - gt)

                # Meta features and labels
                X_vec, _ = build_feature_vector(df_cut, bd, ed, er)
                buffers_X[h].append(X_vec)
                buffers_y[h].append(float(gt))
                buffers_s[h].append(int(1 if gt >= 0 else 0))

                # Train when we have enough history and predict next
                if len(buffers_X[h]) >= 80:
                    X_train = np.vstack(buffers_X[h][:-1])
                    y_train = np.array(buffers_y[h][:-1], dtype=float)
                    s_train = np.array(buffers_s[h][:-1], dtype=int)
                    X_last = buffers_X[h][-1].reshape(1, -1)

                    # Fit models (simple, fast)
                    try:
                        models_reg[h].fit(X_train, y_train)
                    except Exception:
                        pass
                    try:
                        # ensure both classes present
                        if len(np.unique(s_train)) >= 2:
                            models_cls[h].fit(X_train, s_train)
                    except Exception:
                        pass

                    # Predict
                    try:
                        y_pred = float(models_reg[h].predict(X_last)[0])
                    except Exception:
                        y_pred = pick2 if isinstance(pick2, (int, float)) else 0.0
                    try:
                        p_dir = float(models_cls[h].predict_proba(X_last)[0, 1]) if len(np.unique(s_train)) >= 2 else 0.5
                    except Exception:
                        p_dir = 0.5
                    # Compose: magnitude from reg, sign from classifier above 0.5
                    sign = 1.0 if p_dir >= 0.5 else -1.0
                    pred_meta = abs(y_pred) * sign

                    stats_meta[h]['n'] += 1
                    stats_meta[h]['acc'] += int((pred_meta >= 0 and gt >= 0) or (pred_meta < 0 and gt < 0))
                    stats_meta[h]['mae'] += abs(pred_meta - gt)

    # Aggregate results
    out = {
        'base': {h: {
            'n': int(stats_base[h]['n']),
            'directional_accuracy': round(stats_base[h]['acc'] / max(1, stats_base[h]['n']), 4),
            'mae_delta': round(stats_base[h]['mae'] / max(1, stats_base[h]['n']), 5),
        } for h in horizons},
        'meta': {h: {
            'n': int(stats_meta[h]['n']),
            'directional_accuracy': round(stats_meta[h]['acc'] / max(1, stats_meta[h]['n']), 4),
            'mae_delta': round(stats_meta[h]['mae'] / max(1, stats_meta[h]['n']), 5),
        } for h in horizons},
    }
    return out


def main():
    parser = argparse.ArgumentParser(description='Walk-forward meta-stacking test')
    parser.add_argument('--limit', type=int, default=40)
    parser.add_argument('--horizons', type=str, default='1,3,7')
    parser.add_argument('--eval-points', type=int, default=60)
    parser.add_argument('--lookback-days', type=int, default=365)
    parser.add_argument('--min-days', type=int, default=int(os.getenv('ML_MIN_DATA_DAYS', '200')))
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(',') if x.strip().isdigit()]
    horizons = horizons or [1, 3, 7]
    syms = pick_symbols(limit=args.limit, min_days=args.min_days)
    if not syms:
        print(json.dumps({'status': 'error', 'message': 'no_symbols'}))
        return 1

    res = run_meta_stacking(syms, horizons, eval_points=args.eval_points, lookback_days=args.lookback_days)
    payload = {
        'status': 'success',
        'generated_at': datetime.now().isoformat(),
        'meta': {
            'limit': args.limit,
            'symbols': syms,
            'horizons': horizons,
            'eval_points': args.eval_points,
            'lookback_days': args.lookback_days,
        },
        'results': res,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


