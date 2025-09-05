
import os, pathlib, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/opt/bist-pattern/datasets/patterns")
TIMG = ROOT/"train/images"; TLAB = ROOT/"train/labels"
VIMG = ROOT/"val/images";   VLAB = ROOT/"val/labels"
for p in (TIMG, TLAB, VIMG, VLAB): p.mkdir(parents=True, exist_ok=True)

WINDOW = int(os.getenv("WINDOW","120"))
TOTAL  = int(os.getenv("SYN_FLAG_NUM","3000"))
TR     = float(os.getenv("TRAIN_RATIO","0.9"))
CID    = 7  # flag

def render(seg, out):
    try:
        f=plt.figure(figsize=(6.4,6.4), dpi=100); ax=f.add_subplot(111)
        ax.plot(range(len(seg)), seg, color='#1f77b4', linewidth=2)
        ax.set_axis_off(); f.tight_layout(pad=0)
        f.savefig(out, dpi=100, bbox_inches='tight', pad_inches=0); plt.close(f); return True
    except Exception:
        return False

def lin_slope_pct(y):
    if len(y) < 2: return 0.0
    x = np.arange(len(y), dtype=float)
    m = np.polyfit(x, y.astype(float), 1)[0]
    base = max(1e-9, abs(float(y[0])))
    return float(m)/base

def synth_flag_series():
    # trend + dar kanal
    base = random.uniform(80.0, 150.0)
    pre_frac = random.uniform(0.45, 0.7)
    pre = max(15, min(WINDOW-15, int(WINDOW*pre_frac)))
    direction = random.choice((-1, 1))
    m_pre  = direction * random.uniform(0.0015, 0.01)
    m_post = random.uniform(-0.0008, 0.0008)
    n_pre_noise  = random.uniform(0.001, 0.004)
    n_post_noise = random.uniform(0.0005, 0.003)

    y = np.zeros(WINDOW, dtype=float)
    y[0] = base
    for i in range(1, pre):
        y[i] = y[i-1] * (1.0 + m_pre + np.random.normal(0, n_pre_noise))
    for i in range(pre, WINDOW):
        y[i] = y[i-1] * (1.0 + m_post + np.random.normal(0, n_post_noise))

    trend_pct = (y[pre] - y[0]) / max(1e-9, abs(y[0]))
    post = y[pre:]
    post_range = (float(np.max(post)) - float(np.min(post))) / max(1e-9, abs(y[pre]))
    post_slope = lin_slope_pct(post)

    ok = (abs(trend_pct) >= 0.02) and (post_range <= 0.22) and (abs(post_slope) <= 0.002)
    return y, pre, ok

def write_one(idx, seg, pre):
    a = max(int(WINDOW*0.55), pre)
    b = WINDOW - 1
    xc = (a + b) / 2 / WINDOW
    w  = min(0.9, max(0.2, (b - a) / WINDOW + 0.1))
    yc, h = 0.5, 0.6
    is_tr = (random.random() < TR)
    fname = f"synth_flag_{idx}.png"
    imgp  = (TIMG if is_tr else VIMG)/fname
    labp  = (TLAB if is_tr else VLAB)/fname.replace(".png",".txt")
    if render(seg, imgp):
        with open(labp, "w", encoding="utf-8") as f:
            f.write(f"{CID} {xc} {yc} {w} {h}\n")
        return True
    return False

written = 0
attempts = 0
max_attempts = TOTAL * 8
while written < TOTAL and attempts < max_attempts:
    attempts += 1
    series, pre, ok = synth_flag_series()
    if not ok:
        continue
    if write_one(written+1, series, pre):
        written += 1

print("flag_synth_written:", written)
