
import os, pathlib, random, sys
sys.path.insert(0,"/opt/bist-pattern")
from app import app, get_pattern_detector
with app.app_context():
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from models import Stock

    ROOT=pathlib.Path("/opt/bist-pattern/datasets/patterns")
    TIMG=ROOT/"train/images"; TLAB=ROOT/"train/labels"
    VIMG=ROOT/"val/images";   VLAB=ROOT/"val/labels"
    for p in (TIMG, TLAB, VIMG, VLAB): p.mkdir(parents=True, exist_ok=True)

    WINDOW=int(os.getenv("WINDOW","120"))
    STRIDE=int(os.getenv("STRIDE","1"))
    PER=int(os.getenv("PER_SYMBOL_MAX","300"))
    TR=float(os.getenv("TRAIN_RATIO","0.95"))
    CLASS_CAP=int(os.getenv("CLASS_CAP","60000"))

    # sweep: pre yüzdeleri (virgülle)
    PRE_SWEEP=[float(x) for x in os.getenv("FLAG_PRE_SWEEP","0.4,0.5,0.6,0.65,0.7").split(",")]

    # çok gevşek varsayılanlar (ENV ile override edilebilir)
    TREND_PCT_MIN=float(os.getenv("FLAG_TREND_PCT_MIN","0.01"))     # ilk bölümde >= %1 yönlü hareket
    POST_RANGE_MAX=float(os.getenv("FLAG_POST_RANGE_MAX","0.35"))   # konsolidasyon bandı <= %35
    POST_SLOPE_MAX=float(os.getenv("FLAG_POST_SLOPE_MAX","0.003"))  # konsolidasyon eğimi küçük
    RATIO_MAX=float(os.getenv("FLAG_RANGE_RATIO_MAX","1.20"))       # recent/pre range oranı

    def render(seg, out):
        try:
            f=plt.figure(figsize=(6.4,6.4), dpi=100); ax=f.add_subplot(111)
            ax.plot(range(len(seg)), seg, color='#1f77b4', linewidth=2)
            ax.set_axis_off(); f.tight_layout(pad=0)
            f.savefig(out, dpi=100, bbox_inches='tight', pad_inches=0); plt.close(f); return True
        except Exception: return False

    def extract_close(df):
        for k in ('close','Close','adj_close','Adj Close','Adj_Close','price','last'):
            if k in df.columns: return df[k]
        return None

    def lin_slope_pct(y):
        if len(y) < 2: return 0.0
        x=np.arange(len(y), dtype=float); m=np.polyfit(x, y.astype(float), 1)[0]
        base=max(1e-9, abs(float(y[0]))); return float(m)/base

    def detect_flag(x):
        n=len(x)
        if n < WINDOW: return None
        # pre yüzdelerini süpür
        for pf in PRE_SWEEP:
            pre=max(12, min(n-10, int(n*pf)))
            if pre < 12 or n-pre < 10: 
                continue
            trend_pct=(float(x[pre]) - float(x[0]))/max(1e-9, abs(float(x[0])))
            pre_range=(float(np.max(x[:pre])) - float(np.min(x[:pre])))/max(1e-9, abs(float(x[0])))
            post=x[pre:]
            post_range=(float(np.max(post)) - float(np.min(post)))/max(1e-9, abs(float(x[pre])))
            post_slope=lin_slope_pct(post)
            # koşullar
            if (abs(trend_pct) >= TREND_PCT_MIN 
                and post_range <= POST_RANGE_MAX 
                and abs(post_slope) <= POST_SLOPE_MAX
                and (pre_range <= 1e-9 or post_range/pre_range <= RATIO_MAX)):
                a, b = int(n*0.55), n-1
                return a, b
        return None

    det=get_pattern_detector()
    symbols=[s.symbol for s in Stock.query.filter_by(is_active=True).all()]
    random.shuffle(symbols)

    cid=7; total=0
    for sym in symbols:
        try:
            df=det.get_stock_data(sym, days=1000) or det.get_stock_data(sym)
            if df is None or getattr(df,'empty',False) or len(df) < WINDOW+10: 
                continue
            close=extract_close(df)
            if close is None: 
                continue
            x=close.values.astype(float); n=len(x); per_sym=0
            for st in range(0, max(1, n-WINDOW-1), STRIDE):
                if per_sym >= PER or total >= CLASS_CAP: 
                    break
                seg=x[st:st+WINDOW]
                if seg.size < WINDOW: 
                    continue
                lab=detect_flag(seg)
                if not lab: 
                    continue
                a,b=lab
                xc=(a+b)/2/WINDOW; w=min(0.9, max(0.2, (b-a)/WINDOW + 0.1)); yc=0.5; h=0.6
                fname=f"{sym}_flag_{st}.png"
                is_tr=(random.random() < TR)
                img=(TIMG if is_tr else VIMG)/fname
                labp=(TLAB if is_tr else VLAB)/fname.replace(".png",".txt")
                if render(seg, img):
                    with open(labp, "w", encoding="utf-8") as f:
                        f.write(f"{cid} {xc} {yc} {w} {h}\n")
                    total += 1; per_sym += 1
        except Exception:
            continue
    print("flag_written_v4:", total)
