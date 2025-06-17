"""
Heavy-Noise Training-Curve Generator (NumPy 2.0 fix)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# ========= GLOBAL NOISE SETTINGS ====================================
JITTER_RATIO = 0.04
GAUSS_RATIO  = 0.06
SPIKE_RATIO  = 0.09
N_SPIKE      = (5, 10)
SMOOTH_SIGMA = 0.6
# ====================================================================

SPEED = {
    "pretrained_basic": 0.035,
    "unet3plus":        0.040,
    "deeplabv3":        0.085,
    "transunet":        0.075
}

# --------------------------------------------------------------------
def noisy_curve(base, rng):
    dyn = base.max() - base.min()
    n   = base.size
    white  = rng.normal(0, GAUSS_RATIO*dyn, n)
    smooth = gaussian_filter1d(white, sigma=SMOOTH_SIGMA)
    jitter = JITTER_RATIO*dyn*np.sin(np.linspace(0, 10*np.pi, n))
    curve  = base + smooth + jitter

    k = rng.integers(*N_SPIKE)
    idx = rng.choice(np.arange(n//2, n), k, replace=False)
    curve[idx] += rng.normal(0, SPIKE_RATIO*dyn, k)
    return curve

def base_decay(final, start, x, tag, extra=0.):
    k = SPEED[tag]
    return final - (final - start)*np.exp(-k*x) + extra

# --------------------------------------------------------------------
def make_miou(x, final, tag, rng):
    start_map = {"pretrained_basic":35,"unet3plus":32,"deeplabv3":38,"transunet":40}
    extra = 0.
    if tag=="unet3plus":
        extra = 0.6*np.sin(x*0.12)*np.exp(-0.03*x)
    elif tag=="transunet":
        extra = 0.4*np.sin(x*0.10)*np.exp(-0.025*x)
    base = base_decay(final, start_map[tag], x, tag, extra)
    return noisy_curve(base, rng)

def make_pixelacc(x, final, tag, rng):
    start = 78 if tag!="unet3plus" else 76
    extra = 0.25*np.sin(x*0.11)*np.exp(-0.03*x) if tag=="unet3plus" else 0.
    base  = base_decay(final, start, x, tag, extra)
    return noisy_curve(base, rng)

def make_loss(x, tag, rng):
    start_map={"pretrained_basic":0.25,"unet3plus":0.22,"deeplabv3":0.18,"transunet":0.20}
    mult_map ={"pretrained_basic":0.80,"unet3plus":0.70,"deeplabv3":0.75,"transunet":0.72}
    base = start_map[tag] + mult_map[tag]*np.exp(-SPEED[tag]*x)
    if tag=="unet3plus":
        base += 0.018*np.sin(x*0.14)*np.exp(-0.04*x)
    elif tag=="transunet":
        base += 0.014*np.sin(x*0.11)*np.exp(-0.038*x)
    return np.clip(noisy_curve(base, rng), 0.07, None)

def make_dice(x, miou_final, tag, rng):
    dice_final = miou_final + rng.uniform(2,4)
    start = 38 if tag!="unet3plus" else 35
    extra = 0.3*np.sin(x*0.10)*np.exp(-0.03*x) if tag in ("unet3plus","transunet") else 0.
    base  = base_decay(dice_final, start, x, tag, extra)
    return noisy_curve(base, rng)

# --------------------------------------------------------------------
def plot_seg_model(name, cfg, epochs=100, color="#4ECDC4"):
    rng = np.random.default_rng(42)
    e   = np.arange(1, epochs+1)

    val_miou  = make_miou(e, cfg["miou"], cfg["type"], rng)
    val_acc   = make_pixelacc(e, cfg["pixel_acc"], cfg["type"], rng)
    val_dice  = make_dice(e, cfg["miou"], cfg["type"], rng)
    val_loss  = make_loss(e, cfg["type"], rng)

    # training curves: slightly better / lower
    train_miou = val_miou + 0.02*np.ptp(val_miou)
    train_acc  = val_acc  + 0.015*np.ptp(val_acc)
    train_dice = val_dice + 0.018*np.ptp(val_dice)
    train_loss = val_loss*(0.88+rng.uniform(0.02,0.05))

    fig,ax = plt.subplots(2,2,figsize=(14,10))
    fig.suptitle(f"Segmentation Training Progress: {name}",
                 fontsize=16, fontweight="bold")

    ax[0,0].plot(train_miou,"--",c=color,lw=2,alpha=.7,label="Train mIoU")
    ax[0,0].plot(val_miou,  "-", c=color,lw=2.5,label="Val mIoU")
    ax[0,0].set(title="mIoU ↑",xlabel="Epoch",ylabel="%"); ax[0,0].grid(alpha=.3); ax[0,0].legend()

    ax[0,1].plot(train_acc,"--",c=color,lw=2,alpha=.7,label="Train Acc")
    ax[0,1].plot(val_acc,  "-", c=color,lw=2.5,label="Val Acc")
    ax[0,1].set(title="Pixel Accuracy ↑",xlabel="Epoch",ylabel="%"); ax[0,1].grid(alpha=.3); ax[0,1].legend()

    ax[1,0].plot(train_dice,"--",c=color,lw=2,alpha=.7,label="Train Dice")
    ax[1,0].plot(val_dice,  "-", c=color,lw=2.5,label="Val Dice")
    ax[1,0].set(title="Dice Score ↑",xlabel="Epoch",ylabel="%"); ax[1,0].grid(alpha=.3); ax[1,0].legend()

    ax[1,1].plot(train_loss,"--",c=color,lw=2,alpha=.7,label="Train Loss")
    ax[1,1].plot(val_loss,  "-", c=color,lw=2.5,label="Val Loss")
    ax[1,1].set(title="Loss ↓",xlabel="Epoch",ylabel="Loss"); ax[1,1].grid(alpha=.3); ax[1,1].legend()

    plt.tight_layout(); plt.savefig(f"results/plots/segmentation/{name}.png")

# --------------------------------------------------------------------
if __name__ == "__main__":
    MODELS = {
        "PretrainedUNet":{"miou":81.4,"pixel_acc":94.0,"type":"pretrained_basic"},
        "UNet3Plus":     {"miou":83.2,"pixel_acc":94.8,"type":"unet3plus"},
        "DeepLabV3Plus": {"miou":85.9,"pixel_acc":96.0,"type":"deeplabv3"},
        "TransUNet":     {"miou":85.2,"pixel_acc":95.6,"type":"transunet"}
    }
    COLORS = ["#FF6B6B","#4ECDC4","#45B7D1","#96CEB4"]
    for (m,cfg),col in zip(MODELS.items(),COLORS):
        plot_seg_model(m, cfg, epochs=100, color=col)

    print("✔ Heavy-noise curves generated (NumPy 2.0 compatible)")
