
import os, pathlib, random, sys
sys.path.insert(0, "/opt/bist-pattern")
from app import app, get_pattern_detector
with app.app_context():
    import numpy as np
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from models import Stock

    ROOT = pathlib.Path("/opt/bist-pattern/datasets/patterns")
    TRAIN_IMG = ROOT/"train/images"; TRAIN_LAB = ROOT/"train/labels"
    VAL_IMG   = ROOT/"val/images";   VAL_LAB   = ROOT/"val/labels"
    for p in (TRAIN_IMG, TRAIN_LAB, VAL_IMG, VAL_LAB): p.mkdir(parents=True, exist_ok=True)

    WINDOW=int(os.getenv("WINDOW","120")); STRIDE=int(os.getenv("STRIDE","8"))
    PER=int(os.getenv("PER_SYMBOL_MAX","80")); TR=float(os.getenv("TRAIN_RATIO","0.95"))

    # Gevşek eşikler (ENV ile override edilebilir)
    WEDGE_EPS=float(os.getenv("WEDGE_EPS","0.02")); WEDGE_MIN_SPAN=int(os.getenv("WEDGE_MIN_SPAN","10"))
    CHANNEL_EPS=float(os.getenv("CHANNEL_EPS","0.003")); CHANNEL_MIN_SLOPE=float(os.getenv("CHANNEL_MIN_SLOPE","0.0002"))
    FLAG_MIN_TREND=float(os.getenv("FLAG_MIN_TREND","0.01")); FLAG_MAX_RANGE=float(os.getenv("FLAG_MAX_RANGE","0.20"))
    PENNANT_MIN_SLOPE_SUM=float(os.getenv("PENNANT_MIN_SLOPE_SUM","0.001"))
    CUP_MIN_DEPTH=float(os.getenv("CUP_MIN_DEPTH","0.02")); HANDLE_MAX_NEG_SLOPE=float(os.getenv("HANDLE_MAX_NEG_SLOPE","-0.01"))

    cls_map={"wedge_up":9,"wedge_down":10,"flag":7,"pennant":8,"cup_handle":11,"channel_up":13,"channel_down":14}

    def render(seg,out):
        try:
            f=plt.figure(figsize=(6.4,6.4),dpi=100); ax=f.add_subplot(111)
            ax.plot(range(len(seg)),seg,color='#1f77b4',linewidth=2)
            ax.set_axis_off(); f.tight_layout(pad=0)
            f.savefig(out,dpi=100,bbox_inches='tight',pad_inches=0); plt.close(f); return True
        except Exception: return False

    def peaks(x):   return [i for i in range(1,len(x)-1) if x[i]>x[i-1] and x[i]>x[i+1]]
    def troughs(x): return [i for i in range(1,len(x)-1) if x[i]<x[i-1] and x[i]<x[i+1]]
    def fit_line(idxs,vals):
        if len(idxs)<3: return 0.0,0.0
        c=np.polyfit(np.array(idxs,float),np.array(vals,float),1); return c[0],c[1]
    def bbox(a,b,n): xc=(a+b)/2/n; w=min(0.9,max(0.2,(b-a)/n+0.1)); return xc,0.5,w,0.6

    def detect_wedge(x):
        p=peaks(x); t=troughs(x)
        if len(p)<3 or len(t)<3: return None
        K=min(6,len(p),len(t)); p_idx,t_idx=p[-K:],t[-K:]
        m_top,_=fit_line(p_idx,[x[i] for i in p_idx]); m_bot,_=fit_line(t_idx,[x[i] for i in t_idx])
        a,b=min(t_idx[0],p_idx[0]),max(t_idx[-1],p_idx[-1])
        if (m_top>0 and m_bot>0 or m_top<0 and m_bot<0) and abs(m_top-m_bot)<=WEDGE_EPS and (b-a)>=WEDGE_MIN_SPAN:
            return ("wedge_up" if m_top>0 else "wedge_down", a, b)
        return None

    def detect_channel(x):
        p=peaks(x); t=troughs(x)
        if len(p)<3 or len(t)<3: return None
        K=min(6,len(p),len(t)); p_idx,t_idx=p[-K:],t[-K:]
        m_top,_=fit_line(p_idx,[x[i] for i in p_idx]); m_bot,_=fit_line(t_idx,[x[i] for i in t_idx])
        if abs(m_top-m_bot)<=CHANNEL_EPS and max(abs(m_top),abs(m_bot))>=CHANNEL_MIN_SLOPE:
            a,b=min(t_idx[0],p_idx[0]),max(t_idx[-1],p_idx[-1])
            return ("channel_up" if m_top>0 else "channel_down", a, b)
        return None

    def detect_flag_pennant(x):
        n=len(x); pre=n//2
        if n<WINDOW or pre<10: return None
        trend=(x[pre-1]-x[0])/max(1e-9,abs(x[0]))
        recent=x[pre:]; rng=(max(recent)-min(recent))/max(1e-9,abs(x[pre]))
        p=peaks(x); t=troughs(x); tri=False
        if len(p)>=3 and len(t)>=3:
            K=min(5,len(p),len(t)); p_idx,t_idx=p[-K:],t[-K:]
            m_top,_=fit_line(p_idx,[x[i] for i in p_idx]); m_bot,_=fit_line(t_idx,[x[i] for i in t_idx])
            tri=(m_top<0 and m_bot>0 and (abs(m_top)+abs(m_bot))>=PENNANT_MIN_SLOPE_SUM)
        a,b=int(n*0.55),n-1
        if abs(trend)>=FLAG_MIN_TREND:
            if rng<=FLAG_MAX_RANGE: return ("flag",a,b)
            if tri: return ("pennant",a,b)
        return None

    def detect_cup_handle(x):
        n=len(x); mid=int(np.argmin(x))
        if not (int(n*0.20)<mid<int(n*0.75)): return None
        left,right=x[0],x[-1]; depth=min(left,right)-x[mid]
        if depth < CUP_MIN_DEPTH*max(left,right): return None
        h_len=max(10,int(n*0.18)); handle=x[-h_len:]; slope,_=fit_line(list(range(h_len)),handle)
        if HANDLE_MAX_NEG_SLOPE <= slope <= 0: return ("cup_handle", int(n*0.15), n-1)
        return None

    det=get_pattern_detector()
    symbols=[s.symbol for s in Stock.query.filter_by(is_active=True).all()]
    random.shuffle(symbols)

    total=0
    for sym in symbols:
        try:
            df=det.get_stock_data(sym,days=1000) or det.get_stock_data(sym)
            if df is None or getattr(df,'empty',False) or len(df)<WINDOW+10: continue
            close=df['close'] if 'close' in df.columns else (df['Close'] if 'Close' in df.columns else None)
            if close is None: continue
            x=close.values.astype(float); n=len(x); written=0
            for st in range(0,max(1,n-WINDOW-1),STRIDE):
                if written>=PER: break
                seg=x[st:st+WINDOW]
                if seg.size<WINDOW: continue
                lab = detect_wedge(seg) or detect_channel(seg) or detect_flag_pennant(seg) or detect_cup_handle(seg)
                if not lab: continue
                typ,a,b=lab; cid=cls_map[typ]
                fname=f"{sym}_more_{typ}_{st}.png"
                is_tr=(random.random()<TR)
                img=(TRAIN_IMG if is_tr else VAL_IMG)/fname
                labp=(TRAIN_LAB if is_tr else VAL_LAB)/fname.replace(".png",".txt")
                if render(seg,img):
                    xc,yc,w,h=bbox(a,b,WINDOW)
                    with open(labp,"w",encoding="utf-8") as f: f.write(f"{cid} {xc} {yc} {w} {h}\n")
                    total+=1; written+=1
        except Exception:
            continue
    print("more_patterns_written:", total)
