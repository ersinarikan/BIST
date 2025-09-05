import os, pathlib, random
from app import app, get_pattern_detector
with app.app_context():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from models import Stock

    ROOT = pathlib.Path("/opt/bist-pattern/datasets/patterns")
    TRAIN_IMG = ROOT/"train/images"; TRAIN_LAB = ROOT/"train/labels"
    VAL_IMG   = ROOT/"val/images";   VAL_LAB   = ROOT/"val/labels"
    for p in (TRAIN_IMG, TRAIN_LAB, VAL_IMG, VAL_LAB):
        p.mkdir(parents=True, exist_ok=True)

    window   = int(os.getenv("WINDOW", "120"))
    stride   = int(os.getenv("STRIDE", "12"))
    per_sym  = int(os.getenv("PER_SYMBOL_MAX","120"))
    train_r  = float(os.getenv("TRAIN_RATIO","0.8"))
    sym_env  = [s.strip().upper() for s in (os.getenv("SYMBOLS","").split(",") if os.getenv("SYMBOLS") else []) if s.strip()]

    rect_cls = 12  # patterns.yaml'da 'rectangle' id
    bx = (0.5, 0.5, 0.9, 0.9)  # x_center, y_center, width, height (normalize)

    def render(seg, out):
        try:
            fig = plt.figure(figsize=(6.4,6.4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(range(len(seg)), seg, color='#1f77b4', linewidth=2)
            ax.set_axis_off(); fig.tight_layout(pad=0)
            fig.savefig(out, dpi=100, bbox_inches='tight', pad_inches=0); plt.close(fig)
            return True
        except Exception:
            return False

    det = get_pattern_detector()
    symbols = sym_env if sym_env else [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
    random.shuffle(symbols)

    total = 0
    for sym in symbols:
        try:
            df = det.get_stock_data(sym, days=1000)
            if df is None or getattr(df, 'empty', False):
                df = det.get_stock_data(sym)
            if df is None or getattr(df, 'empty', False):
                continue
            if len(df) < window + 10:
                continue
            close = df['close'] if 'close' in df.columns else (df['Close'] if 'Close' in df.columns else None)
            if close is None:
                continue
            x = close.values.astype(float)
            n = len(x); written = 0
            for st in range(0, max(1, n - window - 1), stride):
                if written >= per_sym:
                    break
                seg = x[st:st+window]
                if seg.size < window:
                    continue
                # hedef dosyalar
                fname = f"{sym}_{st}.png"
                is_train = (random.random() < train_r)
                img_p = (TRAIN_IMG if is_train else VAL_IMG) / fname
                lab_p = (TRAIN_LAB if is_train else VAL_LAB) / fname.replace(".png", ".txt")
                # çiz ve etiket yaz
                if render(seg, img_p):
                    with open(lab_p, "w", encoding="utf-8") as f:
                        f.write(f"{rect_cls} {bx[0]} {bx[1]} {bx[2]} {bx[3]}\n")
                    written += 1
                    total += 1
        except Exception:
            continue

    print("rect_dataset_written:", total)
