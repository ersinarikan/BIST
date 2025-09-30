"""
BIST30 Anchor-Date Accuracy Report

Given an anchor date (default: 2025-09-01), compute 1d/3d/14d predictions
for BIST30 symbols using only data up to the anchor, then compare to realized
returns and report directional accuracy and MAE per horizon.

Usage:
  FLASK_ENV=production DATABASE_URL=... \
  python scripts/report_bist30_anchor.py --anchor 2025-09-01 --horizons 1,3,14
"""

from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timedelta
import math
from typing import List, Dict, Any

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app import app  # noqa: E402
from models import db, Stock, StockPrice  # noqa: E402
from bist_pattern.core.ml_coordinator import get_ml_coordinator  # noqa: E402
from bist_pattern.utils.symbols import to_yf_symbol as _to_yf  # noqa: E402


def get_bist30() -> List[str]:
    # Minimal, static list; adjust if you have a dynamic source
    return [
        'AKBNK', 'ARCLK', 'ASELS', 'BIMAS', 'BRSAN', 'DOAS', 'EKGYO', 'EREGL', 'FROTO', 'GARAN',
        'HEKTS', 'ISCTR', 'KCHOL', 'KOZAA', 'KOZAL', 'KRDMD', 'PETKM', 'PGSUS', 'SAHOL', 'SASA',
        'SISE', 'TAVHL', 'TCELL', 'THYAO', 'TKNB', 'TOASO', 'TTKOM', 'TTRAK', 'TUPRS', 'YKBNK'
    ]


def _load_df_yf(symbol: str, until: datetime, lookback_days: int) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
        yf_symbol = _to_yf(symbol)
        if not yf_symbol:
            return pd.DataFrame()
        start = (until.date() - timedelta(days=int(lookback_days))).isoformat()
        end = (until.date() + timedelta(days=1)).isoformat()
        df = yf.download(yf_symbol, start=start, end=end, interval='1d', auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalize columns
        rename_map = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        for k, v in rename_map.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        # Ensure index is date-only (no time)
        try:
            df.index = pd.to_datetime(df.index).normalize()
        except Exception:
            pass
        cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        return df[cols].copy() if cols else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_df(symbol: str, until: datetime, lookback_days: int = 400) -> pd.DataFrame:
    # Try DB first
    try:
        with app.app_context():
            s = Stock.query.filter_by(symbol=symbol.upper()).first()
            if s:
                rows = (
                    db.session.query(StockPrice)
                    .filter(StockPrice.stock_id == s.id)
                    .filter(StockPrice.date <= until.date())
                    .order_by(StockPrice.date.desc())
                    .limit(lookback_days)
                    .all()
                )
            else:
                rows = []
    except Exception:
        rows = []

    if rows:
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
        # If very few rows, fallback enrich from YF
        if len(df) >= 30:
            return df

    # Fallback to YF if DB missing/insufficient
    return _load_df_yf(symbol, until, lookback_days)


def realized_return(df_all: pd.DataFrame, at: datetime, horizon: int) -> float | None:
    try:
        # Ensure sorted index
        try:
            df_all = df_all.sort_index()
        except Exception:
            pass

        anchor_ts = pd.Timestamp(at).normalize()
        if anchor_ts not in df_all.index:
            # pick last trading day <= anchor (pad)
            idx = df_all.index.get_loc(anchor_ts, method='pad')
            base = df_all.index[idx]
        else:
            base = anchor_ts
        base_px = float(df_all.loc[base, 'close'])
        # forward day index
        idx0 = df_all.index.get_loc(base)
        idx1 = idx0 + horizon
        if idx1 >= len(df_all.index):
            return None
        px1 = float(df_all.iloc[idx1]['close'])
        return (px1 - base_px) / base_px
    except Exception:
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', type=str, default='2025-09-01')
    parser.add_argument('--horizons', type=str, default='1,3,14')
    args = parser.parse_args()

    anchor = datetime.fromisoformat(args.anchor)
    horizons = [int(x) for x in args.horizons.split(',') if x.strip().isdigit()]
    symbols = get_bist30()

    mlc = get_ml_coordinator()

    stats: Dict[str, Dict[str, Any]] = {
        str(h): {
            'n': 0,
            'dir_acc': 0,
            'mae': 0.0,
            'se_sum': 0.0,           # sum of squared errors for RMSE
            'abs_list': []           # collect abs errors for median and thresholds
        } for h in horizons
    }
    details: Dict[str, Dict[str, Any]] = {}

    # Extend future window to cover non-trading days (weekends/holidays)
    future_days = max(35, int(max(horizons) * 3))
    for sym in symbols:
        df_until = load_df(sym, anchor, lookback_days=400)
        df_full = load_df(sym, anchor + timedelta(days=future_days), lookback_days=460)
        if df_until.empty or df_full.empty:
            continue
        # Predict using only data up to anchor
        coord = mlc.predict_with_coordination(sym, df_until)
        enh = coord.get('enhanced') or {}
        basic = coord.get('basic') or {}
        # Prefer enhanced if available
        for h in horizons:
            hkey = f"{h}d"
            pred_px = None
            if isinstance(enh.get(hkey), dict) and 'ensemble_prediction' in enh[hkey]:
                pred_px = float(enh[hkey]['ensemble_prediction'])
            else:
                v = basic.get(hkey)
                if isinstance(v, dict):
                    for k in ('price', 'prediction', 'target', 'value', 'y'):
                        if isinstance(v.get(k), (int, float)):
                            pred_px = float(v[k])
                            break
                elif isinstance(v, (int, float)):
                    pred_px = float(v)
            # Need base price at anchor
            try:
                try:
                    df_until = df_until.sort_index()
                except Exception:
                    pass
                anchor_ts = pd.Timestamp(anchor).normalize()
                if anchor_ts in df_until.index:
                    base_date = anchor_ts
                else:
                    # find last trading day <= anchor
                    idx = df_until.index.get_indexer([anchor_ts], method='pad')
                    if idx is None or len(idx) == 0 or idx[0] == -1:
                        continue
                    base_date = df_until.index[int(idx[0])]
                base_px = float(df_until.loc[base_date, 'close'])
            except Exception:
                continue
            if not isinstance(pred_px, (int, float)) or base_px <= 0:
                continue
            pred_ret = (pred_px - base_px) / base_px
            real_ret = realized_return(df_full, base_date, h)
            if real_ret is None:
                continue
            s = stats[str(h)]
            err = pred_ret - real_ret
            aerr = abs(err)
            s['n'] += 1
            s['dir_acc'] += int((pred_ret >= 0 and real_ret >= 0) or (pred_ret < 0 and real_ret < 0))
            s['mae'] += aerr
            s['se_sum'] += (err * err)
            try:
                s['abs_list'].append(float(aerr))
            except Exception:
                pass
            details.setdefault(sym, {})[hkey] = {'pred': pred_ret, 'real': real_ret}

    out = {
        'anchor': args.anchor,
        'horizons': horizons,
        'symbols': symbols,
        'metrics': {
            h: (lambda v: {
                'n': v['n'],
                'directional_accuracy': round(v['dir_acc'] / max(1, v['n']), 4),
                'mae_delta': round(v['mae'] / max(1, v['n']), 5),
                'rmse_delta': round(math.sqrt(v['se_sum'] / max(1, v['n'])), 5),
                'median_abs_error': (round(sorted(v['abs_list'])[len(v['abs_list'])//2], 5) if v['abs_list'] else 0.0),
                'within_1pct': (round(sum(1 for x in v['abs_list'] if x <= 0.01) / max(1, v['n']), 4) if v['abs_list'] else 0.0),
                'within_2pct': (round(sum(1 for x in v['abs_list'] if x <= 0.02) / max(1, v['n']), 4) if v['abs_list'] else 0.0),
                'within_5pct': (round(sum(1 for x in v['abs_list'] if x <= 0.05) / max(1, v['n']), 4) if v['abs_list'] else 0.0),
            })(v)
            for h, v in stats.items()
        },
        'per_symbol': details
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
