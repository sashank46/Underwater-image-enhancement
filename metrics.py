"""
metrics.py - Image quality measures.

No-reference (work without ground truth):
    entropy, UIQM (simplified), UCIQE
Full-reference (need a reference image, e.g. UIEB "reference-890"):
    PSNR, SSIM
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


# ---------------------------- UIQM (Panetta et al.) -----------------------
def _trimmed_mean(x, t=0.1):
    x = np.sort(x.ravel())
    n = len(x)
    return x[int(t * n):max(int((1 - t) * n), int(t * n) + 1)].mean()


def _blocks(x, bs=8):
    h, w = x.shape
    h -= h % bs
    w -= w % bs
    return x[:h, :w].reshape(h // bs, bs, w // bs, bs).swapaxes(1, 2).reshape(-1, bs * bs)


def _eme(x, bs=8):
    b = _blocks(x, bs)
    return float(2 * np.mean(np.log((b.max(1) + 1) / (b.min(1) + 1))))


def uiqm(img):
    """UIQM = 0.0282*UICM + 0.2953*UISM + 3.5753*UIConM  (simplified UIConM)."""
    b, g, r = [c.astype(np.float64) for c in cv2.split(img)]

    # Colourfulness
    rg, yb = r - g, (r + g) / 2 - b
    mu_rg, mu_yb = _trimmed_mean(rg), _trimmed_mean(yb)
    var = rg.var() + yb.var()
    uicm = -0.0268 * np.hypot(mu_rg, mu_yb) + 0.1586 * np.sqrt(var)

    # Sharpness (Sobel edges * channel, EME per block)
    uism = 0.0
    for wgt, ch in zip((0.114, 0.587, 0.299), (b, g, r)):   # B, G, R weights
        sx = cv2.Sobel(ch, cv2.CV_64F, 1, 0)
        sy = cv2.Sobel(ch, cv2.CV_64F, 0, 1)
        uism += wgt * _eme(np.hypot(sx, sy) * ch / 255.0)

    # Contrast (log-AME on intensity, simplified)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    blk = _blocks(gray)
    ratio = (blk.max(1) - blk.min(1)) / (blk.max(1) + blk.min(1) + 1e-6)
    uiconm = float(-np.mean(ratio * np.log(ratio + 1e-6)))

    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


# ---------------------------- UCIQE (Yang & Sowmya) -----------------------
def uciqe(img):
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0] / 100.0, lab[..., 1] / 100.0, lab[..., 2] / 100.0
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()
    con_l = np.percentile(L, 99) - np.percentile(L, 1)
    sat = chroma / (np.sqrt(chroma ** 2 + L ** 2) + 1e-6)
    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * sat.mean())


# ---------------------------- Full-reference ------------------------------
def psnr_ssim(img, ref):
    if ref.shape != img.shape:
        ref = cv2.resize(ref, (img.shape[1], img.shape[0]))
    p = peak_signal_noise_ratio(ref, img, data_range=255)
    s = structural_similarity(ref, img, channel_axis=2, data_range=255)
    return float(p), float(s)


def all_metrics(img, ref=None):
    m = {"Entropy": entropy(img), "UIQM": uiqm(img), "UCIQE": uciqe(img)}
    if ref is not None:
        m["PSNR"], m["SSIM"] = psnr_ssim(img, ref)
    return m
