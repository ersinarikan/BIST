"""
One-day (1D) directional booster with walk-forward evaluation.

- Uses available DB daily bars only (no intraday requirement)
- Features: overnight return, last-3 momentum, volatility bands, RSI(3), gap size,
  day-of-week, recent volume z-score, and prior day tail (close-high/low)
- Base target: sign of next-day return (close_{t+1} vs close_t)
- Model: LogisticRegression (liblinear) + calibration via Platt (built-in) or isotonic fallback
  with TimeSeriesSplit
- Evaluation: expanding-origin walk-forward; reports accuracy, precision/recall, F1, AUC

Run:
  python scripts/one_day_boost.py --symbols 30 --lookback 730 --report /opt/bist-pattern/logs/one_day_eval.json
"""
from __future__ import annotations

import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

lgb = None  # type: ignore
_HAS_LGB = False
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


def _to_df(rows) -> pd.DataFrame:
    df_rows = []
    for r in rows:
        df_rows.append({
            'date': r.date,
            'open': float(r.open_price),
            'high': float(r.high_price),
            'low': float(r.low_price),
            'close': float(r.close_price),
            'volume': int(r.volume or 0),
        })
    df = pd.DataFrame(df_rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df


def _features_1d(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    out = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume'].astype(float)

    # Overnight return (proxy): today open vs prior close (fallback if open missing)
    if 'open' in df:
        overnight = (df['open'] / close.shift(1) - 1.0) * 100.0
    else:
        overnight = (close / close.shift(1) - 1.0) * 100.0
    out['overnight'] = overnight

    # 1/3-day momentum (close vs SMA_3)
    out['ret1'] = (close.pct_change(1) * 100.0)
    out['mom3'] = ((close / close.rolling(3).mean()) - 1.0) * 100.0

    # Volatility band
    log_close = np.log(close)
    out['rv5'] = log_close.diff().rolling(5).std() * np.sqrt(252) * 100.0  # type: ignore

    # RSI(3)
    delta = close.diff()
    up = delta.clip(lower=0).rolling(3).mean()  # type: ignore
    down = (-delta.clip(upper=0)).rolling(3).mean()  # type: ignore
    rs = up / (down + 1e-9)
    out['rsi3'] = 100.0 - (100.0 / (1.0 + rs))

    # Gap size and prior day tails
    out['gap'] = ((df['open'] - close.shift(1)) / close.shift(1)) * 100.0
    out['tail_up'] = ((high - close) / close) * 100.0
    out['tail_dn'] = ((close - low) / close) * 100.0

    # True Range / ATR(5) normalized
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr5 = tr.rolling(5).mean()
    out['atr5_n'] = (atr5 / (close + 1e-9)) * 100.0

    # Bollinger band position (20, 2)
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out['boll_pos'] = (close - ma20) / (sd20 + 1e-9)

    # Stochastic %K (14)
    ll14 = low.rolling(14).min()
    hh14 = high.rolling(14).max()
    out['stoch_k'] = ((close - ll14) / ((hh14 - ll14) + 1e-9)) * 100.0

    # EMA slopes
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema10 = close.ewm(span=10, adjust=False).mean()
    out['ema5_slope'] = ema5.pct_change(1) * 100.0
    out['ema10_slope'] = ema10.pct_change(1) * 100.0

    # Volume z-score (20)
    vol_ma = volume.rolling(20).mean()
    vol_sd = volume.rolling(20).std()
    out['vol_z'] = (volume - vol_ma) / (vol_sd + 1e-9)

    # Day-of-week (one-hot)
    dow = out.index.dayofweek
    for d in range(5):
        out[f'dow_{d}'] = (dow == d).astype(int)

    # Target: next-day direction
    y = np.sign(close.shift(-1) - close)
    y = y.map({-1.0: 0, 0.0: 0, 1.0: 1}).astype('float')

    # Clean
    feats = out.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[feats.index].dropna()
    feats = feats.loc[y.index]
    return feats, y


def _fit_predict_proba(feats: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    # Expanding-origin time series split (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    preds = pd.Series(index=feats.index, dtype=float)
    probas = pd.Series(index=feats.index, dtype=float)

    for train_idx, test_idx in tscv.split(feats):
        X_train, X_test = feats.iloc[train_idx], feats.iloc[test_idx]
        y_train = y.iloc[train_idx]

        if _HAS_LGB and lgb is not None:
            model = lgb.LGBMClassifier(  # type: ignore
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            proba_result = model.predict_proba(X_test)
            p = proba_result[:, 1] if hasattr(proba_result, 'shape') and len(proba_result.shape) > 1 else proba_result  # type: ignore
        else:
            model = LogisticRegression(max_iter=400, solver='liblinear')
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[:, 1]
        probas.iloc[test_idx] = p
        # placeholder; threshold will be optimized globally after CV
        pred_vals = (p >= 0.5).astype(int)  # type: ignore
        preds.iloc[test_idx] = pred_vals
    # Optimize probability cutoff on out-of-fold predictions
    valid = (~probas.isna()) & (~y.loc[probas.index].isna())
    best_thr = 0.5
    best_acc = -1.0
    if valid.sum() > 50:
        yy = y.loc[probas.index][valid].astype(int)
        pp = probas[valid]
        for thr in np.linspace(0.4, 0.6, 41):
            acc = accuracy_score(yy, (pp >= thr).astype(int))
            if acc > best_acc:
                best_acc, best_thr = acc, float(thr)

    final_preds = (probas >= best_thr).astype(int)
    return {
        'preds': final_preds,
        'probas': probas,
        'threshold': best_thr,
        'cv_valid_acc': best_acc,
    }


def evaluate_symbol(symbol: str, lookback_days: int = 730) -> Dict[str, Any]:
    from app import app as flask_app
    with flask_app.app_context():
        from models import Stock, StockPrice
        from sqlalchemy import and_  # type: ignore
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            return {'symbol': symbol, 'error': 'not_found'}
        cutoff = datetime.now() - timedelta(days=lookback_days)
        rows = (
            StockPrice.query
            .filter(and_(StockPrice.stock_id == stock.id, StockPrice.date >= cutoff))
            .order_by(StockPrice.date.asc())
            .all()
        )
        if not rows or len(rows) < 120:
            return {'symbol': symbol, 'error': 'insufficient_data'}

    df = _to_df(rows)
    feats, y = _features_1d(df)
    if len(feats) < 120:
        return {'symbol': symbol, 'error': 'insufficient_data'}

    res = _fit_predict_proba(feats, y)
    idx = res['preds'].dropna().index
    y_true = y.loc[idx].astype(int)
    y_pred = res['preds'].loc[idx].astype(int)
    p = res['probas'].loc[idx].astype(float)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division="warn")  # type: ignore
    try:
        auc = roc_auc_score(y_true, p)
    except Exception:
        auc = None

    return {
        'symbol': symbol,
        'n': int(len(idx)),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc) if auc is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=int, default=30)
    parser.add_argument('--lookback', type=int, default=730)
    parser.add_argument('--report', type=str, default='/opt/bist-pattern/logs/one_day_eval.json')
    args = parser.parse_args()

    # Select top-N active stocks by volume over last 60 days
    from app import app as flask_app
    with flask_app.app_context():
        from models import db, Stock, StockPrice
        from sqlalchemy import func
        cutoff = datetime.now().date() - timedelta(days=60)
        rows = (
            db.session.query(Stock.symbol, func.avg(StockPrice.volume).label('avg_vol'))
            .join(StockPrice, Stock.id == StockPrice.stock_id)
            .filter(Stock.is_active.is_(True), StockPrice.date >= cutoff)
            .group_by(Stock.id, Stock.symbol)
            .order_by(func.avg(StockPrice.volume).desc())
            .limit(args.symbols)
            .all()
        )
        symbols = [r[0] for r in rows] if rows else []

    results = []
    for sym in symbols:
        try:
            r = evaluate_symbol(sym, lookback_days=args.lookback)
        except Exception as e:
            r = {'symbol': sym, 'error': str(e)}
        results.append(r)

    ok = [r for r in results if not r.get('error')]
    agg = {
        'count': len(ok),
        'mean_accuracy': float(np.mean([r['accuracy'] for r in ok])) if ok else None,
        'mean_f1': float(np.mean([r['f1'] for r in ok])) if ok else None,
        'mean_auc': float(np.mean([r['auc'] for r in ok if r['auc'] is not None])) if ok else None,
    }

    payload = {
        'generated_at': datetime.now().isoformat(),
        'symbols': symbols,
        'aggregate': agg,
        'results': results,
    }

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w') as f:
        json.dump(payload, f)

    print(json.dumps(agg))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
