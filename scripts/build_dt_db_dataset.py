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
    stride   = int(os.getenv("STRIDE", "8"))
    per_sym  = int(os.getenv("PER_SYMBOL_MAX","80"))
    train_r  = float(os.getenv("TRAIN_RATIO","0.8"))
    sym_env  = [s.strip().upper() for s in (os.getenv("SYMBOLS","").split(",") if os.getenv("SYMBOLS") else []) if s.strip()]

    # names.yaml eşleşmesi
    cls_map = {"double_top":2, "double_bottom":3}

    def render(seg, out):
        try:
            fig=plt.figure(figsize=(6.4,6.4), dpi=100)
            ax=fig.add_subplot(111)
            ax.plot(range(len(seg)), seg, color='#1f77b4', linewidth=2)
            ax.set_axis_off(); fig.tight_layout(pad=0)
            fig.savefig(out, dpi=100, bbox_inches='tight', pad_inches=0); plt.close(fig)
            return True
        except Exception: return False

    def peaks(x):
        return [i for i in range(1,len(x)-1) if x[i]>x[i-1] and x[i]>x[i+1]]
    def troughs(x):
        return [i for i in range(1,len(x)-1) if x[i]<x[i-1] and x[i]<x[i+1]]

    def weak_dt_db(x):
        labs=[]
        pks=peaks(x); trs=troughs(x)
        span_min=int(os.getenv("DT_SPAN_MIN","8"))
        span_max=int(os.getenv("DT_SPAN_MAX","80"))
        tol=float(os.getenv("DT_TOL","0.03"))
        # double top
        for i in range(len(pks)-1):
            a,b=pks[i],pks[i+1]
            if b-a<span_min or b-a>span_max: continue
            if abs(x[a]-x[b])<=tol*max(x[a],x[b]):
                mid = min(range(a,b+1), key=lambda k: x[k])
                if x[mid] < min(x[a],x[b]) - tol*max(x[a],x[b]):
                    labs.append(("double_top", a, b))
                    break
        # double bottom
        for i in range(len(trs)-1):
            a,b=trs[i],trs[i+1]
            if b-a<span_min or b-a>span_max: continue
            if abs(x[a]-x[b])<=tol*max(x[a],x[b]):
                mid = max(range(a,b+1), key=lambda k: x[k])
                if x[mid] > max(x[a],x[b]) + tol*max(x[a],x[b]):
                    labs.append(("double_bottom", a, b))
                    break
        return labs

    def bbox_from_indices(a,b,n):
        xc=(a+b)/2/n; yc=0.5
        w=min(0.9, max(0.2, (b-a)/n + 0.1))
        h=0.6
        return xc,yc,w,h

    det = get_pattern_detector()
    symbols = sym_env if sym_env else [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
    random.shuffle(symbols)

    total=0
    for sym in symbols:
        try:
            df = det.get_stock_data(sym, days=1000)
            if df is None or getattr(df,'empty',False):
                df = det.get_stock_data(sym)
            if df is None or getattr(df,'empty',False) or len(df)<window+10:
                continue
            close = df['close'] if 'close' in df.columns else (df['Close'] if 'Close' in df.columns else None)
            if close is None: continue
            x = close.values.astype(float)
            n = len(x); written=0
            for st in range(0, max(1,n-window-1), stride):
                if written>=per_sym: break
                seg = x[st:st+window]
                if seg.size<window: continue
                labs = weak_dt_db(seg)
                if not labs: continue
                fname=f"{sym}_dt_db_{st}.png"
                is_train = (random.random()<train_r)
                img_p=(TRAIN_IMG if is_train else VAL_IMG)/fname
                lab_p=(TRAIN_LAB if is_train else VAL_LAB)/fname.replace(".png",".txt")
                if render(seg,img_p):
                    with open(lab_p,'w',encoding='utf-8') as f:
                        for typ,a,b in labs:
                            cid = cls_map[typ]
                            xc,yc,w,h = bbox_from_indices(a,b,window)
                            f.write(f"{cid} {xc} {yc} {w} {h}\n")
                    total+=1; written+=1
        except Exception:
            continue
    print("dt_db_dataset_written:", total)
