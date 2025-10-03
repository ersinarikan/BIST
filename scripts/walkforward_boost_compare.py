"""
Walk-Forward Booster Impact Test (1D)

Measures the effect of the 1D directional booster tilt on MAE and directional
accuracy using walk-forward evaluation. It compares baseline best_of_v2 (regime-
aware) vs best_of_v2 with booster tilt applied to the selected delta.

Usage:
  python scripts/walkforward_boost_compare.py --limit 40 --eval-points 40 --lookback-days 365
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


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


def _features_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume'].astype(float)

    if 'open' in df:
        overnight = (df['open'] / close.shift(1) - 1.0) * 100.0
    else:
        overnight = (close / close.shift(1) - 1.0) * 100.0
    out['overnight'] = overnight
    out['ret1'] = (close.pct_change(1) * 100.0)
    out['mom3'] = ((close / close.rolling(3).mean()) - 1.0) * 100.0
    out['rv5'] = np.log(close).diff().rolling(5).std() * np.sqrt(252) * 100.0

    delta = close.diff()
    up = delta.clip(lower=0).rolling(3).mean()
    down = (-delta.clip(upper=0)).rolling(3).mean()
    rs = up / (down + 1e-9)
    out['rsi3'] = 100.0 - (100.0 / (1.0 + rs))

    out['gap'] = ((df['open'] - close.shift(1)) / (close.shift(1) + 1e-9)) * 100.0
    out['tail_up'] = ((high - close) / (close + 1e-9)) * 100.0
    out['tail_dn'] = ((close - low) / (close + 1e-9)) * 100.0

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr5 = tr.rolling(5).mean()
    out['atr5_n'] = (atr5 / (close + 1e-9)) * 100.0

    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out['boll_pos'] = (close - ma20) / (sd20 + 1e-9)

    ll14 = low.rolling(14).min()
    hh14 = high.rolling(14).max()
    out['stoch_k'] = ((close - ll14) / ((hh14 - ll14) + 1e-9)) * 100.0

    ema10 = close.ewm(span=10, adjust=False).mean()
    out['ema10_slope'] = ema10.pct_change(1) * 100.0

    vol_ma = volume.rolling(20).mean()
    vol_sd = volume.rolling(20).std()
    out['vol_z'] = (volume - vol_ma) / (vol_sd + 1e-9)

    dow = out.index.dayofweek
    for d in range(5):
        out[f'dow_{d}'] = (dow == d).astype(int)

    return out.replace([np.inf, -np.inf], np.nan)


def compute_booster_proba(df: pd.DataFrame) -> float | None:
    feats = _features_1d(df)
    close = df['close']
    y = np.sign(close.shift(-1) - close)
    y = y.map({-1.0: 0, 0.0: 0, 1.0: 1}).astype('float')
    mask = feats.notna().all(axis=1) & y.notna()
    feats = feats.loc[mask]
    y = y.loc[mask]
    if len(feats) < 120:
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


def run_compare(symbols: List[str], eval_points: int = 40, lookback_days: int = 365) -> Dict[str, Any]:
    # Load predictors
    try:
        from scripts.backtest_selection_policy import load_predictors  # type: ignore
        basic_ml, enhanced = load_predictors()
    except Exception:
        basic_ml, enhanced = (None, None)
    # weights used inline in selection
    stats = {
        'best_of_v2': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
        'best_of_v2_tilt': {'n': 0, 'dir_acc': 0, 'mae': 0.0},
    }

    total = len(symbols)
    for idx_sym, sym in enumerate(symbols, start=1):
        print(f"[progress] {idx_sym}/{total} symbol={sym}", flush=True)
        df = get_df(sym, lookback_days=lookback_days)
        if df.empty or len(df) < 180:
            continue
        max_h = 1  # 1D only
        candidates_idx = list(range(60, max(61, len(df) - max_h)))
        eval_idx = candidates_idx[-eval_points:]

        for idx in eval_idx:
            df_cut = df.iloc[: idx + 1]
            p0 = float(df_cut['close'].iloc[-1])
            if (idx + 1) >= len(df):
                continue
            p1 = float(df['close'].iloc[idx + 1])
            gt = (p1 - p0) / p0

            basic_map = predict_basic(basic_ml, sym, df_cut) if basic_ml else {}
            enh_map = predict_enh(enhanced, sym, df_cut) if enhanced else {}

            bp = basic_map.get('1d')
            ep_obj = enh_map.get('1d')
            ep = None if not isinstance(ep_obj, dict) else ep_obj.get('price')
            er = None if not isinstance(ep_obj, dict) else ep_obj.get('confidence')

            if not isinstance(bp, (int, float)) and not isinstance(ep, (int, float)):
                continue

            bd = None if not isinstance(bp, (int, float)) else (float(bp) - p0) / p0
            ed = None if not isinstance(ep, (int, float)) else (float(ep) - p0) / p0

            reg = regime_score(df_cut)
            # best_of_v2 selection on deltas
            pick2 = None
            if isinstance(bd, (int, float)) and isinstance(ed, (int, float)):
                c_bd = tanh_calibrate(bd)
                c_ed = tanh_calibrate(ed)
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
                pick2 = ed if score_e2 >= score_b2 else bd
            elif isinstance(ed, (int, float)):
                pick2 = ed
            else:
                pick2 = bd
            if not isinstance(pick2, (int, float)):
                continue

            # baseline stats
            stats['best_of_v2']['n'] += 1
            stats['best_of_v2']['dir_acc'] += int((pick2 >= 0 and gt >= 0) or (pick2 < 0 and gt < 0))
            stats['best_of_v2']['mae'] += abs(pick2 - gt)

            # booster tilt
            try:
                p_up = compute_booster_proba(df_cut)
            except Exception:
                p_up = None
            pick_tilt = pick2
            if isinstance(p_up, float):
                base_thr = 0.008
                alpha = 0.25
                bmag = abs(p_up - 0.5) * 2.0
                bagree = (pick2 >= 0 and p_up >= 0.5) or (pick2 < 0 and p_up < 0.5)
                tilt_boost = (0.5 * alpha) * base_thr * bmag * (1.0 if bagree else -1.0)
                pick_tilt = pick2 + tilt_boost
                if pick_tilt > 0.5:
                    pick_tilt = 0.5
                if pick_tilt < -0.5:
                    pick_tilt = -0.5

            stats['best_of_v2_tilt']['n'] += 1
            stats['best_of_v2_tilt']['dir_acc'] += int((pick_tilt >= 0 and gt >= 0) or (pick_tilt < 0 and gt < 0))
            stats['best_of_v2_tilt']['mae'] += abs(pick_tilt - gt)

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
    parser = argparse.ArgumentParser(description='Walk-forward 1D booster impact')
    parser.add_argument('--limit', type=int, default=40)
    parser.add_argument('--eval-points', type=int, default=40)
    parser.add_argument('--lookback-days', type=int, default=365)
    parser.add_argument('--min-days', type=int, default=int(os.getenv('ML_MIN_DATA_DAYS', '200')))
    args = parser.parse_args()

    syms = pick_symbols(limit=args.limit, min_days=args.min_days)
    if not syms:
        print(json.dumps({'status': 'error', 'message': 'no_symbols'}))
        return 1

    res = run_compare(syms, eval_points=args.eval_points, lookback_days=args.lookback_days)
    payload = {
        'status': 'success',
        'generated_at': datetime.now().isoformat(),
        'meta': {
            'limit': args.limit,
            'symbols': syms,
            'eval_points': args.eval_points,
            'lookback_days': args.lookback_days,
        },
        'results': res,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
