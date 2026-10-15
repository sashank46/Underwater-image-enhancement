"""
methods.py - Underwater image enhancement algorithms (pure NumPy + OpenCV).

All functions take and return 8-bit BGR images (OpenCV convention).

Implemented methods
-------------------
1. gray_world   : Gray-World white balance (colour-cast removal)
2. he           : Histogram Equalisation on the V channel (HSV)
3. clahe        : CLAHE on the L channel (LAB)
4. udcp         : Red compensation + Underwater Dark Channel Prior dehazing
                  (transmission refined with a guided filter)
5. fusion       : PROPOSED - Red compensation + white balance, then
                  multi-scale (Laplacian pyramid) fusion of two derived inputs
                  (Ancuti et al., "Enhancing Underwater Images and Videos by Fusion")
"""
import cv2
import numpy as np


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Basic building blocks
# --------------------------------------------------------------------------
def gray_world(img):
    """Scale each channel so that all channel means become equal."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / (means + 1e-6)
    return _u8(f * gain)


def red_compensation(img, alpha=1.0):
    """Compensate the strongly attenuated red channel using the green channel.

    Ir' = Ir + alpha * (mean(Ig) - mean(Ir)) * (1 - Ir) * Ig
    """
    f = img.astype(np.float32) / 255.0
    b, g, r = cv2.split(f)
    diff = max(g.mean() - r.mean(), 0.0)
    r = r + alpha * diff * (1.0 - r) * g
    return _u8(cv2.merge([b, g, r]) * 255.0)


def contrast_stretch(img, low=1, high=99):
    """Per-channel percentile contrast stretching."""
    out = np.empty_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], (low, high))
        out[..., c] = _u8((img[..., c].astype(np.float32) - lo) / max(hi - lo, 1) * 255)
    return out


def unsharp(img, sigma=3.0, amount=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


# --------------------------------------------------------------------------
# 1-3. Classical baselines
# --------------------------------------------------------------------------
def he_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def clahe_lab(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# 4. Dark-channel-prior dehazing (underwater variant)
# --------------------------------------------------------------------------
def _guided_filter(I, p, r=30, eps=1e-3):
    """Grey-scale guided filter (He et al.) implemented with box filters."""
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = mean(I), mean(p)
    cov = mean(I * p) - mI * mp
    var = mean(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return mean(a) * I + mean(b)


def udcp(img, patch=15, omega=0.9, t0=0.2):
    """Underwater DCP: dark channel computed from the G and B channels only,
    because the red channel is already almost zero underwater."""
    img = red_compensation(img)
    f = img.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))

    dark = cv2.erode(np.minimum(f[..., 0], f[..., 1]), kernel)

    # Atmospheric (background) light: brightest pixel among top 0.1% dark-channel pixels
    n = max(1, int(dark.size * 0.001))
    idx = np.argsort(dark.ravel())[-n:]
    cand = f.reshape(-1, 3)[idx]
    A = cand[cand.sum(axis=1).argmax()]
    A = np.clip(A, 0.2, 0.95)

    # Transmission map + guided-filter refinement
    norm = f / A
    t = 1.0 - omega * cv2.erode(np.minimum(norm[..., 0], norm[..., 1]), kernel)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = np.clip(_guided_filter(gray, t.astype(np.float32)), t0, 1.0)

    J = (f - A) / t[..., None] + A
    return contrast_stretch(_u8(J * 255.0))


# --------------------------------------------------------------------------
# 5. Fusion-based enhancement (proposed)
# --------------------------------------------------------------------------
def _gauss_pyr(x, n):
    p = [x]
    for _ in range(n - 1):
        p.append(cv2.pyrDown(p[-1]))
    return p


def _lap_pyr(x, n):
    g = _gauss_pyr(x, n)
    lp = []
    for i in range(n - 1):
        h, w = g[i].shape[:2]
        lp.append(g[i] - cv2.pyrUp(g[i + 1], dstsize=(w, h)))
    lp.append(g[-1])
    return lp


def _norm01(x):
    return x / (x.max() + 1e-6)


def _weight_map(img_u8):
    """Combine 4 quality measures: Laplacian contrast, saliency,
    saturation and exposedness."""
    f = img_u8.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lab = cv2.cvtColor(cv2.GaussianBlur(img_u8, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    saliency = np.linalg.norm(lab - lab.reshape(-1, 3).mean(axis=0), axis=2)
    saturation = f.std(axis=2)
    exposed = np.exp(-((gray - 0.5) ** 2) / (2 * 0.25 ** 2))

    return _norm01(contrast) + _norm01(saliency) + _norm01(saturation) + exposed


def fusion(img, levels=5):
    """Proposed method: colour correction + multi-scale fusion."""
    wb = gray_world(red_compensation(img))

    # Input 1 : sharpened version (recovers edges / fine detail)
    in1 = unsharp(wb, sigma=3.0, amount=0.8)
    # Input 2 : CLAHE version (global contrast / visibility)
    in2 = clahe_lab(wb, clip=3.0)
    inputs = [in1, in2]

    # Normalised weights
    w = [_weight_map(i) for i in inputs]
    total = sum(w) + 1e-6
    w = [x / total for x in w]

    # Number of pyramid levels limited by image size
    h, wd = img.shape[:2]
    levels = max(2, min(levels, int(np.log2(min(h, wd))) - 3))

    # Blend each Laplacian level using Gaussian pyramid of the weights
    fused = None
    for inp, wt in zip(inputs, w):
        lp = _lap_pyr(inp.astype(np.float32), levels)
        gp = _gauss_pyr(wt.astype(np.float32), levels)
        cur = [l * g[..., None] for l, g in zip(lp, gp)]
        fused = cur if fused is None else [a + b for a, b in zip(fused, cur)]

    out = fused[-1]
    for lvl in reversed(fused[:-1]):
        hh, ww = lvl.shape[:2]
        out = cv2.pyrUp(out, dstsize=(ww, hh)) + lvl
    return _u8(out)


METHODS = {
    "gray_world": gray_world,
    "he": he_hsv,
    "clahe": clahe_lab,
    "udcp": udcp,
    "fusion": fusion,
}
