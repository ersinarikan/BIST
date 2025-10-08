"""
Backfill YOLO visual pattern density features per symbol into CSV files for model training.

Outputs per-symbol CSV under EXTERNAL_FEATURE_DIR/yolo/{SYMBOL}.csv with columns:
  - date (YYYY-MM-DD)
  - yolo_density (float, detections per 100 bars; 0.0 if unavailable)
  - yolo_bull (float/int count proxy)
  - yolo_bear (float/int count proxy)
  - yolo_score (float alignment score in [-1,1], 0 if unknown)

This script writes placeholders (zeros) if historical detections are not yet archived.
Replace with real aggregated outputs once an offline YOLO pass is available.
"""

from __future__ import annotations

import os
import csv
import sys
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple, Any
import io
import warnings
warnings.filterwarnings('ignore')
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore
YOLO = None  # type: ignore
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


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
            return []
        q = StockPrice.query.filter(StockPrice.stock_id == stock.id)
        if isinstance(lookback_days, int) and lookback_days > 0:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            q = q.filter(StockPrice.date >= cutoff)
        q = q.order_by(StockPrice.date.asc())
        rows = q.all()
        return [r.date.date().isoformat() for r in rows] if rows else []


def _get_ohlc_df(symbol: str, lookback_days: int | None):
    if pd is None:
        return None
    from app import app as flask_app
    with flask_app.app_context():
        from models import Stock, StockPrice
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            return None
        q = StockPrice.query.filter(StockPrice.stock_id == stock.id)
        if isinstance(lookback_days, int) and lookback_days > 0:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            q = q.filter(StockPrice.date >= cutoff)
        q = q.order_by(StockPrice.date.asc())
        rows = q.all()
        if not rows:
            # Fallback to pipeline's unified collector via pattern_detector (DB empty)
            try:
                from pattern_detector import HybridPatternDetector  # type: ignore
                det = HybridPatternDetector()
                df_pd = det.get_stock_data(symbol, days=0)
                # Ensure pandas DataFrame shape
                if hasattr(df_pd, 'empty') and not df_pd.empty:
                    return df_pd
            except Exception:
                return None
        data = {
            'date': [r.date.date() for r in rows],
            'open': [float(r.open_price) for r in rows],
            'high': [float(r.high_price) for r in rows],
            'low': [float(r.low_price) for r in rows],
            'close': [float(r.close_price) for r in rows],
            'volume': [int(r.volume or 0) for r in rows],
        }
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
        return df


def _canonicalize_class_name(raw_name: str) -> str:
    try:
        n = (raw_name or "").strip().lower().replace('-', ' ').replace('_', ' ')
        
        def has(*tokens):
            return all(tok in n for tok in tokens)
        if has('inverse', 'shoulder') or has('inverse', 'head'):
            return 'INVERSE_HEAD_SHOULDERS'
        if has('head', 'shoulder'):
            return 'HEAD_SHOULDERS'
        if has('double', 'top'):
            return 'DOUBLE_TOP'
        if has('double', 'bottom'):
            return 'DOUBLE_BOTTOM'
        if has('ascending', 'triangle'):
            return 'ASCENDING_TRIANGLE'
        if has('descending', 'triangle'):
            return 'DESCENDING_TRIANGLE'
        if has('rising', 'wedge'):
            return 'RISING_WEDGE'
        if has('falling', 'wedge'):
            return 'FALLING_WEDGE'
        if ('bull' in n and 'flag' in n) or has('bullish', 'flag'):
            return 'BULLISH_FLAG'
        if ('bear' in n and 'flag' in n) or has('bearish', 'flag'):
            return 'BEARISH_FLAG'
        if has('cup', 'handle'):
            return 'CUP_AND_HANDLE'
        import re
        up = re.sub(r'[^a-z0-9]+', '_', n).strip('_').upper()
        return up or 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'


def _bull_bear_from_class(cls_name: str) -> Tuple[int, int]:
    cls = (cls_name or '').upper()
    bullish = {
        'INVERSE_HEAD_SHOULDERS', 'DOUBLE_BOTTOM', 'ASCENDING_TRIANGLE',
        'FALLING_WEDGE', 'BULLISH_FLAG', 'CUP_AND_HANDLE'
    }
    bearish = {
        'HEAD_SHOULDERS', 'DOUBLE_TOP', 'DESCENDING_TRIANGLE',
        'RISING_WEDGE', 'BEARISH_FLAG'
    }
    if cls in bullish:
        return 1, 0
    if cls in bearish:
        return 0, 1
    return 0, 0


def _render_chart_image(df_window) -> Any:
    try:
        if plt is None or Image is None or df_window is None or len(df_window) < 20:
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_window['close'], linewidth=2, color='blue')
        if 'volume' in df_window.columns:
            ax2 = ax.twinx()
            ax2.bar(range(len(df_window)), df_window['volume'], alpha=0.3, color='gray')
            ax2.set_ylabel('Volume')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    except Exception:
        return None


def _write_csv(dest: str, rows: Iterable[List[str | float | int]]) -> None:
    _ensure_dir(os.path.dirname(dest))
    with open(dest, 'w', newline='', encoding='utf-8') as wf:
        w = csv.writer(wf)
        w.writerow(['date', 'yolo_density', 'yolo_bull', 'yolo_bear', 'yolo_score'])
        for row in rows:
            w.writerow(row)


def main(argv: List[str]) -> int:
    try:
        lookback_days = int(os.getenv('BACKFILL_LOOKBACK_DAYS', '0') or '0')
    except Exception:
        lookback_days = 0
    basedir = os.getenv('EXTERNAL_FEATURE_DIR', '/opt/bist-pattern/logs/feature_backfill')
    outdir = os.path.join(basedir, 'yolo')
    _ensure_dir(outdir)
    # Load YOLO model if available
    model = None
    yolo_ready = False
    if YOLO_AVAILABLE and YOLO is not None:
        model_path = os.getenv('YOLO_MODEL_PATH', '/opt/bist-pattern/yolo/patterns_all_v7_rectblend.pt')
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)  # type: ignore
                yolo_ready = True
            except Exception:
                yolo_ready = False

    syms: List[str]
    if len(argv) > 1:
        # Symbols provided as CLI args
        syms = [s.strip().upper() for s in argv[1:] if s.strip()]
    else:
        syms = _get_symbols()

    wrote = 0
    for sym in syms:
        try:
            df = _get_ohlc_df(sym, lookback_days)
            if df is None or (hasattr(df, 'empty') and df.empty):
                continue
            try:
                window = int(os.getenv('YOLO_BACKFILL_WINDOW', '100') or '100')
            except Exception:
                window = 100
            rows: List[List[Any]] = []
            for i in range(len(df)):
                date_str = df.index[i].date().isoformat()
                if i + 1 < max(30, window) or not yolo_ready:
                    rows.append([date_str, 0.0, 0, 0, 0.0])
                    continue
                df_win = df.iloc[i + 1 - window: i + 1]
                img = _render_chart_image(df_win)
                if img is None:
                    rows.append([date_str, 0.0, 0, 0, 0.0])
                    continue
                det_count = 0
                bull_sum = 0
                bear_sum = 0
                try:
                    min_conf = float(os.getenv('YOLO_MIN_CONF', '0.12') or '0.12')
                except Exception:
                    min_conf = 0.12
                try:
                    if model is None:
                        continue
                    results = model(img, conf=min_conf, verbose=False)  # type: ignore
                    if results and len(results) > 0:
                        result = results[0]
                        try:
                            names = getattr(result, 'names', None) or getattr(model, 'names', None) or {}
                            if isinstance(names, list):
                                names = {idx: n for idx, n in enumerate(names)}
                        except Exception:
                            names = {}
                        boxes = getattr(result, 'boxes', None)
                        if boxes is not None and hasattr(boxes, 'cls') and hasattr(boxes, 'conf'):
                            try:
                                num = int(len(getattr(boxes, 'cls')))
                            except Exception:
                                num = 0
                            for j in range(num):
                                try:
                                    conf = float(boxes.conf[j]) if hasattr(boxes, 'conf') else 0.0
                                    if conf < min_conf:
                                        continue
                                    cls_idx = int(boxes.cls[j]) if hasattr(boxes, 'cls') else 0
                                    raw_name = str(names.get(cls_idx, f'class_{cls_idx}')) if names else f'class_{cls_idx}'
                                    cname = _canonicalize_class_name(raw_name)
                                    b, r = _bull_bear_from_class(cname)
                                    bull_sum += b
                                    bear_sum += r
                                    det_count += 1
                                except Exception:
                                    continue
                except Exception:
                    pass
                denom = max(1, (bull_sum + bear_sum))
                score = (bull_sum - bear_sum) / float(denom)
                rows.append([date_str, float(det_count), int(bull_sum), int(bear_sum), float(score)])
            dest = os.path.join(outdir, f'{sym}.csv')
            _write_csv(dest, rows)
            wrote += 1
        except Exception:
            continue

    print(f"✅ YOLO backfill CSVs written: {wrote} → {outdir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
