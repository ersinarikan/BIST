import os
os.environ.setdefault('PYTHONWARNINGS','ignore')
from app import app, get_pattern_detector

with app.app_context():
    from models import Stock
    det = get_pattern_detector()
    ml = det.ml_predictor
    try:
        from simple_enhanced_ml import get_simple_enhanced_ml
        simple_ml = get_simple_enhanced_ml()
    except Exception:
        simple_ml = None

    symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
    ok_ml = fail_ml = ok_simple = fail_simple = skipped = 0

    for i, sym in enumerate(symbols, 1):
        try:
            df = det.get_stock_data(sym, days=730) or det.get_stock_data(sym)
            if df is None or len(df) < 200:
                skipped += 1
                continue

            try:
                if ml:
                    res = ml.train_models(sym, df)
                    ok_ml += 1 if res else 0
            except Exception:
                fail_ml += 1

            try:
                if simple_ml:
                    res2 = simple_ml.train_simple_models(sym, df)
                    ok_simple += 1 if res2 else 0
            except Exception:
                fail_simple += 1

        except Exception:
            fail_ml += 1

        if i % 25 == 0:
            print(f"[{i}/{len(symbols)}] ok_ml={ok_ml} fail_ml={fail_ml} ok_simple={ok_simple} fail_simple={fail_simple} skipped={skipped}")

    print(f"DONE: ok_ml={ok_ml} fail_ml={fail_ml} ok_simple={ok_simple} fail_simple={fail_simple} skipped={skipped} total={len(symbols)}")
