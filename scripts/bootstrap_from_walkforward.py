#!/usr/bin/env python3
"""
Run walkforward compare and import its predictions/outcomes into predictions_log/outcomes_log.

We reuse walkforward_compare.py to generate ground truth and model predictions, then
materialize them as if they were produced online, to kick-start calibration.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Any

from app import app
from models import db, PredictionsLog, OutcomesLog
from scripts.backtest_selection_policy import pick_symbols, get_df, predict_basic, predict_enh, load_predictors  # type: ignore


def import_walkforward(limit: int = 100, horizons=(1, 3, 7, 14, 30), eval_points: int = 60, lookback_days: int = 365) -> Dict[str, Any]:
    with app.app_context():
        try:
            db.create_all()
        except Exception:
            pass

        syms = pick_symbols(limit=limit, min_days=200)
        basic_ml, enhanced = load_predictors()
        total_pred = 0
        total_out = 0
        for sym in syms:
            df = get_df(sym, lookback_days=lookback_days)
            if df.empty or len(df) < 60:
                continue
            max_h = max(horizons)
            idxs = list(range(30, max(31, len(df) - max_h)))
            eval_idx = idxs[-eval_points:]
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
                basic_map = predict_basic(basic_ml, sym, df_cut) if basic_ml else {}
                enh_map = predict_enh(enhanced, sym, df_cut) if enhanced else {}
                # fallbacks if predictors not available
                basic_map = basic_map or {}
                enh_map = enh_map or {}
                for h in truths.keys():
                    hkey = f"{h}d"
                    bp_val = basic_map.get(hkey)
                    ep_obj = enh_map.get(hkey)
                    ep_price = None if not isinstance(ep_obj, dict) else ep_obj.get('price')
                    er_conf = None if not isinstance(ep_obj, dict) else ep_obj.get('confidence')
                    # Ensure numeric types
                    bp: float | None = float(bp_val) if isinstance(bp_val, (int, float)) else None
                    ep: float | None = float(ep_price) if isinstance(ep_price, (int, float)) else None
                    er: float | None = float(er_conf) if isinstance(er_conf, (int, float)) else None
                    if bp is None and ep is None:
                        continue
                    if ep is not None:
                        pred_px: float = ep
                    else:
                        # narrow type for mypy/pylance
                        assert bp is not None
                        pred_px = bp
                    delta_pred = (pred_px - p0) / p0
                    # Insert prediction
                    plog = PredictionsLog(
                        stock_id=None,
                        symbol=sym,
                        horizon=hkey,
                        ts_pred=df_cut.index[-1].to_pydatetime(),
                        price_now=p0,
                        pred_price=pred_px,
                        delta_pred=delta_pred,
                        model='enhanced' if ep is not None else 'basic',
                        unified_best='enhanced' if ep is not None else 'basic',
                        confidence=er,  # type: ignore[arg-type]
                    )
                    db.session.add(plog)
                    db.session.flush()
                    total_pred += 1
                    # Outcome
                    gt = truths[h]
                    price_eval = float(p0 * (1.0 + gt))
                    out = OutcomesLog(
                        prediction_id=plog.id,
                        ts_eval=df.index[idx + h].to_pydatetime(),
                        price_eval=price_eval,
                        delta_real=gt,
                        dir_hit=((gt >= 0 and delta_pred >= 0) or (gt < 0 and delta_pred < 0)),
                        abs_err=abs((pred_px - price_eval)),
                        mape=(abs((pred_px - price_eval)) / price_eval if price_eval else 0.0),
                    )
                    db.session.add(out)
                    total_out += 1
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
        return {'symbols': syms, 'predictions': total_pred, 'outcomes': total_out}


def main():
    res = import_walkforward()
    print(json.dumps({'status': 'ok', 'imported': res, 'generated_at': datetime.now().isoformat()}))


if __name__ == '__main__':
    main()
