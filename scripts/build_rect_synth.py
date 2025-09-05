
import os, pathlib, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/opt/bist-pattern/datasets/patterns")
TIMG = ROOT/"train/images"; TLAB = ROOT/"train/labels"
VIMG = ROOT/"val/images";   VLAB = ROOT/"val/labels"
for p in (TIMG, TLAB, VIMG, VLAB): p.mkdir(parents=True, exist_ok=True)

WINDOW = int(os.getenv("WINDOW","120"))
TOTAL  = int(os.getenv("SYN_RECT_NUM","500"))
TR     = float(os.getenv("TRAIN_RATIO","0.9"))
CID    = 12  # rectangle

def render(seg, out):
    try:
        f=plt.figure(figsize=(6.4,6.4), dpi=100); ax=f.add_subplot(111)
        ax.plot(range(len(seg)), seg, color='#1f77b4', linewidth=2)
        ax.set_axis_off(); f.tight_layout(pad=0)
        f.savefig(out, dpi=100, bbox_inches='tight', pad_inches=0); plt.close(f); return True
    except Exception:
        return False

def synth_rect_series():
    # yatay/dar bant hareketi
    base = random.uniform(80.0, 150.0)
    amp  = random.uniform(0.003, 0.02)  # bant genliği (yüzdesel)
    drift = random.uniform(-0.0005, 0.0005)  # çok hafif sürüklenme
    noise = random.uniform(0.0005, 0.004)
    y = np.zeros(WINDOW, dtype=float); y[0] = base
    for i in range(1, WINDOW):
        cyc = amp * np.sin(2*np.pi * (i / random.uniform(12, 30)))
        y[i] = y[i-1] * (1.0 + drift + cyc + np.random.normal(0, noise))
    return y

def write_one(idx, seg):
    # rectangle kutusunu pencerenin ortasına sabitle
    a = int(WINDOW * 0.15); b = int(WINDOW * 0.85)
    xc = (a + b) / 2 / WINDOW
    w  = min(0.9, max(0.2, (b - a) / WINDOW + 0.05))
    yc, h = 0.5, 0.6

    is_tr = (random.random() < TR)
    fname = f"synth_rect_{idx}.png"
    imgp  = (TIMG if is_tr else VIMG)/fname
    labp  = (TLAB if is_tr else VLAB)/fname.replace(".png",".txt")
    if render(seg, imgp):
        with open(labp, "w", encoding="utf-8") as f:
            f.write(f"{CID} {xc} {yc} {w} {h}\n")
        return True
    return False

written = 0
for i in range(1, TOTAL+1):
    series = synth_rect_series()
    if write_one(written+1, series):
        written += 1

print("rect_synth_written:", written)
