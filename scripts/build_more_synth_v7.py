
import os, pathlib, random
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/opt/bist-pattern/datasets/patterns")
TIMG = ROOT/"train/images"; TLAB = ROOT/"train/labels"
VIMG = ROOT/"val/images";   VLAB = ROOT/"val/labels"
for p in (TIMG, TLAB, VIMG, VLAB): p.mkdir(parents=True, exist_ok=True)

WINDOW = int(os.getenv("WINDOW","120"))
TR     = float(os.getenv("TRAIN_RATIO","0.9"))

# Sayılar (ENV ile override edilebilir)
N_FLAG   = int(os.getenv("SYN_FLAG_NUM","0"))
N_PENN   = int(os.getenv("SYN_PENNANT_NUM","300"))
N_WUP    = int(os.getenv("SYN_WEDGE_UP_NUM","400"))
N_WDN    = int(os.getenv("SYN_WEDGE_DOWN_NUM","400"))
N_CUP    = int(os.getenv("SYN_CUP_NUM","300"))
N_CHUP   = int(os.getenv("SYN_CHANNEL_UP_NUM","400"))
N_CHDN   = int(os.getenv("SYN_CHANNEL_DOWN_NUM","400"))

# class ids (patterns.yaml)
CID = {
  "flag":7, "pennant":8, "wedge_up":9, "wedge_down":10,
  "cup_handle":11, "channel_up":13, "channel_down":14
}

def render(seg, out):
    try:
        f=plt.figure(figsize=(6.4,6.4), dpi=100); ax=f.add_subplot(111)
        ax.plot(range(len(seg)), seg, color='#1f77b4', linewidth=2)
        ax.set_axis_off(); f.tight_layout(pad=0)
        f.savefig(out, dpi=100, bbox_inches='tight', pad_inches=0); plt.close(f); return True
    except Exception:
        return False

def write_lbl(labp, cid, a, b):
    a = max(0, min(WINDOW-2, int(a)))
    b = max(a+1, min(WINDOW-1, int(b)))
    xc = (a + b) / 2 / WINDOW
    w  = min(0.9, max(0.2, (b - a) / WINDOW + 0.08))
    yc, h = 0.5, 0.6
    with open(labp, "w", encoding="utf-8") as f:
        f.write(f"{cid} {xc} {yc} {w} {h}\n")

def write_one(name, cid, seg, a, b, idx):
    is_tr = (random.random() < TR)
    fname = f"synth_{name}_{idx}.png"
    imgp  = (TIMG if is_tr else VIMG)/fname
    labp  = (TLAB if is_tr else VLAB)/fname.replace(".png",".txt")
    if render(seg, imgp):
        write_lbl(labp, cid, a, b)
        return True
    return False

def gen_series(base, slope, amp0, amp_decay, noise, phase=None):
    y = np.zeros(WINDOW, dtype=float); y[0]=base
    ph = random.uniform(0, 2*np.pi) if phase is None else phase
    for i in range(1, WINDOW):
        amp = amp0 * (1.0 - amp_decay * (i/WINDOW))
        cyc = amp * np.sin(2*np.pi*(i/random.uniform(12,30)) + ph)
        r = slope + cyc + np.random.normal(0, noise)
        y[i] = y[i-1] * (1.0 + r)
    return y

def synth_flag():
    base = random.uniform(80,150)
    pre  = max(18, min(WINDOW-18, int(WINDOW*random.uniform(0.5,0.65))))
    y = np.zeros(WINDOW); y[0]=base
    m_pre  = random.choice((-1,1)) * random.uniform(0.0015, 0.01)
    m_post = random.uniform(-0.0008, 0.0008)
    n_pre  = random.uniform(0.001, 0.004)
    n_post = random.uniform(0.0005, 0.003)
    for i in range(1, pre):
        y[i] = y[i-1]*(1.0 + m_pre + np.random.normal(0, n_pre))
    for i in range(pre, WINDOW):
        cyc = random.uniform(0.003,0.02)*np.sin(2*np.pi*(i/random.uniform(14,28)))
        y[i] = y[i-1]*(1.0 + m_post + cyc + np.random.normal(0, n_post))
    a = int(WINDOW*0.55); b = WINDOW-1
    return y, a, b

def synth_pennant():
    base = random.uniform(80,150)
    pre  = max(18, min(WINDOW-18, int(WINDOW*random.uniform(0.5,0.7))))
    y = np.zeros(WINDOW); y[0]=base
    m_pre  = random.choice((-1,1)) * random.uniform(0.002, 0.012)
    for i in range(1, pre):
        y[i] = y[i-1]*(1.0 + m_pre + np.random.normal(0, random.uniform(0.001,0.004)))
    # üçgenleşen konsolidasyon
    amp0 = random.uniform(0.02, 0.06)
    for i in range(pre, WINDOW):
        decay = amp0 * (1 - (i-pre)/(WINDOW-pre+1e-9))
        cyc = decay * np.sin(2*np.pi*(i/random.uniform(10,24)))
        y[i] = y[i-1]*(1.0 + np.random.normal(0, random.uniform(0.0006,0.002)) + cyc)
    a = int(WINDOW*0.55); b = WINDOW-1
    return y, a, b

def synth_wedge(up=True):
    base = random.uniform(80,150)
    slope = (0.0015 if up else -0.0015) * random.uniform(0.6,1.6)
    amp0  = random.uniform(0.02,0.05)
    y = gen_series(base, slope, amp0, amp_decay=1.0, noise=random.uniform(0.0006,0.002))
    a = int(WINDOW*0.5); b = WINDOW-1
    return y, a, b

def synth_channel(up=True):
    base = random.uniform(80,150)
    slope = (0.0015 if up else -0.0015) * random.uniform(0.6,1.6)
    amp0  = random.uniform(0.01,0.025)
    y = gen_series(base, slope, amp0, amp_decay=0.0, noise=random.uniform(0.0006,0.002))
    a = int(WINDOW*0.2); b = WINDOW-1
    return y, a, b

def synth_cup_handle():
    base = random.uniform(80,150)
    t = np.linspace(0,1,WINDOW)
    depth = random.uniform(0.05, 0.18)  # %5-%18 derinlik
    cup = 1.0 - depth*4*(t-0.5)**2
    cup *= base
    noise = np.random.normal(0, base*random.uniform(0.0005,0.002), size=WINDOW)
    y = cup + noise
    # handle: son %18 dilimde hafif negatif eğim
    hlen = max(10, int(WINDOW*0.18))
    for i in range(WINDOW-hlen, WINDOW):
        y[i] = y[i-1]*(1.0 + np.random.normal(-0.0006, 0.001))
    a = int(WINDOW*0.15); b = WINDOW-1
    return y, a, b

def run_one(name, n, fn, cid):
    w = 0
    for i in range(1, n+1):
        seg, a, b = fn()
        if write_one(name, cid, seg, a, b, i): w += 1
    print(f"{name}_synth_written:", w)

run_one("flag",           N_FLAG,   synth_flag,      CID["flag"])
run_one("pennant",        N_PENN,   synth_pennant,   CID["pennant"])
run_one("wedge_up",       N_WUP,    lambda: synth_wedge(True),  CID["wedge_up"])
run_one("wedge_down",     N_WDN,    lambda: synth_wedge(False), CID["wedge_down"])
run_one("cup_handle",     N_CUP,    synth_cup_handle, CID["cup_handle"])
run_one("channel_up",     N_CHUP,   lambda: synth_channel(True),  CID["channel_up"])
run_one("channel_down",   N_CHDN,   lambda: synth_channel(False), CID["channel_down"])
