#!/usr/bin/env python3
import os, pathlib, random, sys, subprocess

DATASET_ROOT = os.getenv('DATASET_ROOT', '/opt/bist-pattern/datasets/patterns')
WINDOW  = int(os.getenv('WINDOW','120'))
STRIDE  = int(os.getenv('STRIDE','8'))
TRAIN_RATIO = float(os.getenv('TRAIN_RATIO','0.8'))
PER_SYMBOL_MAX = int(os.getenv('PER_SYMBOL_MAX','120'))

DT_SPAN_MIN = int(os.getenv('DT_SPAN_MIN','8'))
DT_SPAN_MAX = int(os.getenv('DT_SPAN_MAX','100'))
DT_TOL      = float(os.getenv('DT_TOL','0.03'))
HS_TOL      = float(os.getenv('HS_TOL','0.05'))
HS_SPAN_MIN = int(os.getenv('HS_SPAN_MIN','12'))
HS_SPAN_MAX = int(os.getenv('HS_SPAN_MAX','90'))
HS_HEAD_MARGIN = float(os.getenv('HS_HEAD_MARGIN','0.04'))
TRI_SLOPE_FLAT = float(os.getenv('TRI_SLOPE_FLAT','5e-4'))
TRI_SLOPE_MIN  = float(os.getenv('TRI_SLOPE_MIN','1e-3'))

ROOT = pathlib.Path(DATASET_ROOT)
TRAIN_IMG = ROOT/'train/images'; TRAIN_LAB = ROOT/'train/labels'
VAL_IMG   = ROOT/'val/images';   VAL_LAB   = ROOT/'val/labels'
for p in (TRAIN_IMG, TRAIN_LAB, VAL_IMG, VAL_LAB):
    p.mkdir(parents=True, exist_ok=True)

CLASS_TO_ID = {
    "head_and_shoulders": 0,
    "inverse_head_and_shoulders": 1,
    "double_top": 2,
    "double_bottom": 3,
    "ascending_triangle": 4,
    "descending_triangle": 5,
    "symmetrical_triangle": 6,
    "flag": 7,
    "pennant": 8,
    "wedge_up": 9,
    "wedge_down": 10,
    "cup_handle": 11,
    "rectangle": 12,
    "channel_up": 13,
    "channel_down": 14,
}

def ensure_yaml():
    yml = ROOT/'patterns.yaml'
    if not yml.exists():
        yml.write_text("""path: /opt/bist-pattern/datasets/patterns
train: train/images
val: val/images
names:
  0: head_and_shoulders
  1: inverse_head_and_shoulders
  2: double_top
  3: double_bottom
  4: ascending_triangle
  5: descending_triangle
  6: symmetrical_triangle
  7: flag
  8: pennant
  9: wedge_up
  10: wedge_down
  11: cup_handle
  12: rectangle
  13: channel_up
  14: channel_down
""", encoding='utf-8')

def bbox_from_indices(a, b, n):
    xc=(a+b)/2/n; yc=0.5
    w=min(0.9, max(0.2, (b-a)/n + 0.1))
    h=0.6
    return xc,yc,w,h

def render_chart(plt, np, seg, out_path):
    try:
        fig = plt.figure(figsize=(6.4,6.4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(range(len(seg)), seg, color='#1f77b4', linewidth=2)
        ax.set_axis_off(); fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True
    except Exception:
        return False

def build_all_datasets():
    from app import app, get_pattern_detector
    with app.app_context():
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from models import Stock

        def peaks(x):
            return [i for i in range(1,len(x)-1) if x[i]>x[i-1] and x[i]>x[i+1]]
        def troughs(x):
            return [i for i in range(1,len(x)-1) if x[i]<x[i-1] and x[i]<x[i+1]]

        def weak_dt_db(x):
            labs=[]
            pks=peaks(x); trs=troughs(x)
            for i in range(len(pks)-1):
                a,b=pks[i],pks[i+1]
                if b-a<DT_SPAN_MIN or b-a>DT_SPAN_MAX: continue
                if abs(x[a]-x[b])<=DT_TOL*max(x[a],x[b]):
                    mid = min(range(a,b+1), key=lambda k: x[k])
                    if x[mid] < min(x[a],x[b]) - DT_TOL*max(x[a],x[b]):
                        labs.append(("double_top", a, b)); break
            for i in range(len(trs)-1):
                a,b=trs[i],trs[i+1]
                if b-a<DT_SPAN_MIN or b-a>DT_SPAN_MAX: continue
                if abs(x[a]-x[b])<=DT_TOL*max(x[a],x[b]):
                    mid = max(range(a,b+1), key=lambda k: x[k])
                    if x[mid] > max(x[a],x[b]) + DT_TOL*max(x[a],x[b]):
                        labs.append(("double_bottom", a, b)); break
            return labs

        def weak_hs(x):
            pks=peaks(x)
            for i in range(1, len(pks)-1):
                l, h, r = pks[i-1], pks[i], pks[i+1]
                if r-l < HS_SPAN_MIN or r-l > HS_SPAN_MAX: continue
                hl, hr = x[l], x[r]
                if abs(hl-hr) <= HS_TOL*max(hl,hr) and x[h] > max(hl,hr)*(1+HS_HEAD_MARGIN):
                    return ("head_and_shoulders", l, r)
            trs=troughs(x)
            for i in range(1, len(trs)-1):
                l, h, r = trs[i-1], trs[i], trs[i+1]
                if r-l < HS_SPAN_MIN or r-l > HS_SPAN_MAX: continue
                hl, hr = x[l], x[r]
                if abs(hl-hr) <= HS_TOL*max(hl,hr) and x[h] < min(hl,hr)*(1-HS_HEAD_MARGIN):
                    return ("inverse_head_and_shoulders", l, r)
            return None

        def fit_line(idxs, vals, np):
            if len(idxs)<3: return 0.0, 0.0
            c = np.polyfit(np.array(idxs, dtype=float), np.array(vals, dtype=float), 1)
            return c[0], c[1]

        def weak_triangles(x, np):
            p=peaks(x); t=troughs(x)
            if len(p)<3 or len(t)<3: return None
            K= min(6, len(p), len(t))
            p_idx, t_idx = p[-K:], t[-K:]
            m_top,_ = fit_line(p_idx, [x[i] for i in p_idx], np)
            m_bot,_ = fit_line(t_idx, [x[i] for i in t_idx], np)
            if abs(m_top) <= TRI_SLOPE_FLAT and m_bot >= TRI_SLOPE_MIN:
                return ("ascending_triangle", min(t_idx[0],p_idx[0]), max(t_idx[-1],p_idx[-1]))
            if abs(m_bot) <= TRI_SLOPE_FLAT and m_top <= -TRI_SLOPE_MIN:
                return ("descending_triangle", min(t_idx[0],p_idx[0]), max(t_idx[-1],p_idx[-1]))
            if m_top < -TRI_SLOPE_MIN and m_bot > TRI_SLOPE_MIN:
                return ("symmetrical_triangle", min(t_idx[0],p_idx[0]), max(t_idx[-1],pidx[-1]))
            return None

        def write_example(seg, labels, np, plt, sym, st):
            if not labels: return 0
            fname=f"{sym}_{st}.png"
            is_train = (random.random() < TRAIN_RATIO)
            img_p=(TRAIN_IMG if is_train else VAL_IMG)/fname
            lab_p=(TRAIN_LAB if is_train else VAL_LAB)/fname.replace(".png",".txt")
            if render_chart(plt, np, seg, img_p):
                with open(lab_p,'w',encoding='utf-8') as f:
                    for typ,a,b in labels:
                        cid = CLASS_TO_ID[typ]
                        xc,yc,w,h = bbox_from_indices(a,b,WINDOW)
                        f.write(f"{cid} {xc} {yc} {w} {h}\n")
                return 1
            return 0

        det = get_pattern_detector()
        symbols = [s.symbol for s in Stock.query.filter_by(is_active=True).all()]
        random.shuffle(symbols)

        total=0
        for sym in symbols:
            try:
                df = det.get_stock_data(sym, days=1000) or det.get_stock_data(sym)
                if df is None or getattr(df,'empty',False) or len(df)<WINDOW+10: continue
                close = df['close'] if 'close' in df.columns else (df['Close'] if 'Close' in df.columns else None)
                if close is None: continue
                x = close.values.astype(float)
                n = len(x); written=0
                for st in range(0, max(1,n-WINDOW-1), STRIDE):
                    if written>=PER_SYMBOL_MAX: break
                    seg = x[st:st+WINDOW]
                    if seg.size<WINDOW: continue
                    labels=[]
                    hs = weak_hs(seg)
                    if hs: labels.append(hs)
                    labels += weak_dt_db(seg)
                    tri = weak_triangles(seg, np)
                    if tri: labels.append(tri)
                    if not labels:
                        labels.append(("rectangle", int(WINDOW*0.1), int(WINDOW*0.9)))
                    total += write_example(seg, labels, np, plt, sym, f"all_{st}")
                    written += 1
            except Exception:
                continue
        print("all_patterns_dataset_written:", total)

def run(cmd):
    print("+", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    ensure_yaml()
    print("Building datasets...")
    build_all_datasets()
    runs_dir = "/opt/bist-pattern/runs"
    data_yaml = str(ROOT/'patterns.yaml')
    start_model = os.getenv('START_MODEL', '/opt/bist-pattern/yolo/yolov8n_v82.pt')
    name = os.getenv('RUN_NAME', 'all_patterns_overnight')
    epochs = int(os.getenv('EPOCHS','25'))
    batch  = int(os.getenv('BATCH','16'))

    cmd = [
        '/opt/bist-pattern/venv/bin/yolo', 'detect', 'train',
        f'data={data_yaml}', f'model={start_model}',
        'imgsz=640', f'batch={batch}', f'epochs={epochs}', 'patience=5',
        f'name={name}', f'project={runs_dir}'
    ]
    rc = run(cmd)
    if rc!=0: sys.exit(rc)
    best = f"/opt/bist-pattern/runs/{name}/weights/best.pt"
    target = "/opt/bist-pattern/yolo/patterns_all_v1.pt"
    if os.path.exists(best):
        os.system(f"cp {best} {target}")
        print("saved_best_to:", target)
    else:
        print("best_not_found:", best)

if __name__ == "__main__":
    main()
