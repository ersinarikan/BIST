import os, random, pathlib, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONWARNINGS','ignore')

from app import app, get_pattern_detector

with app.app_context():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from models import Stock

    ROOT = pathlib.Path("/opt/bist-pattern/datasets/patterns")
    TRAIN_IMG = ROOT/"train/images"; TRAIN_LAB = ROOT/"train/labels"
    VAL_IMG = ROOT/"val/images";     VAL_LAB = ROOT/"val/labels"
    for p in (TRAIN_IMG, TRAIN_LAB, VAL_IMG, VAL_LAB):
        p.mkdir(parents=True, exist_ok=True)

    class_map = {
        "head_and_shoulders": 0, "inverse_head_and_shoulders": 1,
        "double_top": 2, "double_bottom": 3,
        "ascending_triangle": 4, "descending_triangle": 5, "symmetrical_triangle": 6,
        "flag": 7, "pennant": 8,
        "wedge_up": 9, "wedge_down": 10,
        "cup_handle": 11, "rectangle": 12,
        "channel_up": 13, "channel_down": 14,
    }

    def render_chart(seg_close: np.ndarray, out_path: pathlib.Path) -> bool:
        try:
            if seg_close.size < 20: return False
            fig = plt.figure(figsize=(6.4, 6.4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(range(len(seg_close)), seg_close, color='#1f77b4', linewidth=2)
            ax.set_axis_off(); fig.tight_layout(pad=0)
            fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            return True
        except Exception:
            return False

    def weak_detect(x: np.ndarray) -> list[str]:
        out=[]
        n=len(x)
        if n<50: return out
        peaks=[i for i in range(1,n-1) if x[i]>x[i-1] and x[i]>x[i+1]]
        troughs=[i for i in range(1,n-1) if x[i]<x[i-1] and x[i]<x[i+1]]
        # double top
        for i in range(len(peaks)-1):
            a,b=peaks[i],peaks[i+1]
            if abs(x[a]-x[b])<=0.03*max(x[a],x[b]) and (b-a)>=5 and any(a<k<b for k in troughs):
                out.append("double_top"); break
        # double bottom
        for i in range(len(troughs)-1):
            a,b=troughs[i],troughs[i+1]
            if abs(x[a]-x[b])<=0.03*max(x[a],x[b]) and (b-a)>=5 and any(a<k<b for k in peaks):
                out.append("double_bottom"); break
        # head & shoulders (kaba)
        if len(peaks)>=3:
            for i in range(len(peaks)-2):
                l,m,r=peaks[i],peaks[i+1],peaks[i+2]
                if x[m]>x[l] and x[m]>x[r] and abs(x[l]-x[r])<=0.05*max(x[l],x[r]):
                    out.append("head_and_shoulders"); break
        # inverse h&s (kaba)
        if len(troughs)>=3:
            for i in range(len(troughs)-2):
                l,m,r=troughs[i],troughs[i+1],troughs[i+2]
                if x[m]<x[l] and x[m]<x[r] and abs(x[l]-x[r])<=0.05*max(x[l],x[r]):
                    out.append("inverse_head_and_shoulders"); break
        # rectangle
        win=x[-60:] if n>=60 else x
        if (win.max()-win.min())<=0.03*max(1.0,np.median(x)):
            out.append("rectangle")
        # channel slope
        try:
            t=np.arange(n); A=np.vstack([t,np.ones(n)]).T
            m,_=np.linalg.lstsq(A,x,rcond=None)[0]
            if m>0 and (x.max()-x.min())>0.05*np.median(x): out.append("channel_up")
            if m<0 and (x.max()-x.min())>0.05*np.median(x): out.append("channel_down")
        except Exception: pass
        # triangle (kaba)
        if len(peaks)>=3 and len(troughs)>=3:
            tops=np.array([x[i] for i in peaks[-3:]])
            bots=np.array([x[i] for i in troughs[-3:]])
            if tops.std()<=0.02*max(1.0,tops.mean()) and np.all(np.diff(bots)>0): out.append("ascending_triangle")
            if bots.std()<=0.02*max(1.0,bots.mean()) and np.all(np.diff(tops)<0): out.append("descending_triangle")
        # flag (momentum+daralma)
        ret=np.diff(x)/np.clip(x[:-1],1e-9,None)
        if ret.size>30 and abs(ret[-30:].mean())>0.01 and np.std(ret[-15:])<np.std(ret[-60:-45])+1e-9:
            out.append("flag")
        uniq=[]
        for c in out:
            if c in class_map and c not in uniq: uniq.append(c)
        return uniq

    det=get_pattern_detector()
    min_len=250; window=180; stride=30; train_ratio=0.8
    symbols=[s.symbol for s in Stock.query.filter_by(is_active=True).all()]
    random.shuffle(symbols)

    total=0
    for sym in symbols:
        try:
            df=det.get_stock_data(sym, days=1000) or det.get_stock_data(sym)
            if df is None or len(df)<min_len: continue
            close=df['close'] if 'close' in df.columns else df.get('Close')
            if close is None: continue
            x=close.values.astype(float); n=len(x)
            for start in range(0, max(1,n-window-1), stride):
                seg=x[start:start+window]
                if len(seg)<window: continue
                labels=weak_detect(seg)
                if not labels: continue
                is_train = (random.random()<train_ratio)
                img_dir=TRAIN_IMG if is_train else VAL_IMG
                lab_dir=TRAIN_LAB if is_train else VAL_LAB
                name=f"{sym}_{start}_{start+window}.png"
                img_path=img_dir/name
                if not render_chart(seg, img_path):
                    try: img_path.unlink(missing_ok=True)
                    except Exception: pass
                    continue
                xc,yc,w,h=0.5,0.5,0.8,0.7
                lab_lines=[f"{class_map[c]} {xc} {yc} {w} {h}" for c in labels]
                if not lab_lines:
                    try: img_path.unlink(missing_ok=True)
                    except Exception: pass
                    continue
                lab_path=lab_dir/(img_path.stem+".txt")
                with open(lab_path,"w") as f: f.write("\n".join(lab_lines))
                total+=1
        except Exception:
            continue
    print(f"Dataset build completed. Images+labels written: {total}")
