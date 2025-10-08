"""
Backfill FinGPT sentiment features per symbol into CSV files for model training.

Outputs per-symbol CSV under EXTERNAL_FEATURE_DIR/fingpt/{SYMBOL}.csv with columns:
  - date (YYYY-MM-DD)
  - sentiment_score (float in [0,1], 0.5 neutral)
  - news_count (int)

If a news archive is not available, the script writes neutral placeholders
so that the training pipeline has aligned columns. Replace placeholders later
when a historical sentiment archive is integrated.
"""

from __future__ import annotations

import os
import csv
import sys
from datetime import datetime, timedelta
from typing import Iterable, List


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _get_symbols(limit: int | None = None) -> List[str]:
    from app import app as flask_app
    with flask_app.app_context():
        from models import Stock
        q = Stock.query.order_by(Stock.symbol.asc())
        if isinstance(limit, int) and limit > 0:
            q = q.limit(limit)
        rows = q.all()
        return [r.symbol for r in rows] if rows else []


def _get_trading_dates(symbol: str, lookback_days: int | None) -> List[str]:
    from app import app as flask_app
    with flask_app.app_context():
        from models import Stock, StockPrice
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            # Try pipeline fallback (Yahoo) via pattern_detector
            try:
                from pattern_detector import HybridPatternDetector  # type: ignore
                det = HybridPatternDetector()
                df = det.get_stock_data(symbol, days=0)
                if df is not None and not getattr(df, 'empty', True):
                    return [d.date().isoformat() for d in df.index.to_pydatetime()]
            except Exception:
                return []
        q = StockPrice.query.filter(StockPrice.stock_id == stock.id)
        if isinstance(lookback_days, int) and lookback_days > 0:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            q = q.filter(StockPrice.date >= cutoff)
        q = q.order_by(StockPrice.date.asc())
        rows = q.all()
        if rows:
            return [r.date.date().isoformat() for r in rows]
        # DB empty -> fallback to Yahoo via pattern_detector
        try:
            from pattern_detector import HybridPatternDetector  # type: ignore
            det = HybridPatternDetector()
            df = det.get_stock_data(symbol, days=0)
            if df is not None and not getattr(df, 'empty', True):
                return [d.date().isoformat() for d in df.index.to_pydatetime()]
        except Exception:
            pass
        return []


def _write_csv(dest: str, rows: Iterable[List[str | float | int]]) -> None:
    _ensure_dir(os.path.dirname(dest))
    with open(dest, 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf)
        w.writerow(['date', 'sentiment_score', 'news_count'])
        for row in rows:
            w.writerow(row)


def main(argv: List[str]) -> int:
    try:
        lookback_days = int(os.getenv('BACKFILL_LOOKBACK_DAYS', '0') or '0')
    except Exception:
        lookback_days = 0
    basedir = os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
    outdir = os.path.join(basedir, 'fingpt')
    _ensure_dir(outdir)

    syms: List[str]
    if len(argv) > 1:
        # Symbols provided as CLI args
        syms = [s.strip().upper() for s in argv[1:] if s.strip()]
    else:
        syms = _get_symbols()

    wrote = 0
    for sym in syms:
        try:
            dates = _get_trading_dates(sym, lookback_days)
            if not dates:
                continue
            # Placeholder neutral values (0.5 score, 0 news) until archive exists
            rows = [[d, 0.5, 0] for d in dates]
            dest = os.path.join(outdir, f'{sym}.csv')
            _write_csv(dest, rows)
            wrote += 1
        except Exception:
            continue

    print(f"✅ FinGPT backfill CSVs written: {wrote} → {outdir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
