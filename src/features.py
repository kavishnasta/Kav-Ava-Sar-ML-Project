"""Hand-made features: HOG + HSV colour histogram + Hu moments. Everything is controlled by a cfg dict."""
import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog

import config

DEFAULT_CFG = dict(median=True, clahe=False, size=32, hog_orient=9, hog_cells=(8,),
                   hsv_bins=16, hsv_grid=1, hsv_ch="hsv", hu=True)


def make_cfg(**kw):
    cfg = dict(DEFAULT_CFG)
    cfg.update(config.FEATURE_CFG)
    cfg.update(kw)
    return cfg


def _preprocess(bgr, cfg):
    if cfg["median"]:
        bgr = cv2.medianBlur(bgr, 3)
    if cfg["size"] != bgr.shape[0]:
        bgr = cv2.resize(bgr, (cfg["size"], cfg["size"]), interpolation=cv2.INTER_CUBIC)
    if cfg["clahe"]:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[..., 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def _hog(gray, cfg):
    return np.concatenate([hog(gray, orientations=cfg["hog_orient"], pixels_per_cell=(c, c),
                               cells_per_block=(2, 2), block_norm="L2-Hys") for c in cfg["hog_cells"]])


def _hsv(bgr, cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ranges = {"h": (0, 0, 180), "s": (1, 0, 256), "v": (2, 0, 256)}
    g = cfg["hsv_grid"]
    step = hsv.shape[0] // g
    parts = []
    for i in range(g):
        for j in range(g):
            cell = np.ascontiguousarray(hsv[i * step:(i + 1) * step, j * step:(j + 1) * step])
            for ch in cfg["hsv_ch"]:
                idx, lo, hi = ranges[ch]
                h = cv2.calcHist([cell], [idx], None, [cfg["hsv_bins"]], [lo, hi])
                parts.append(cv2.normalize(h, h).ravel())
    return np.concatenate(parts)


def _hu(gray):
    hu = cv2.HuMoments(cv2.moments(gray)).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def extract_groups(bgr, cfg):
    bgr = _preprocess(bgr, cfg)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = {"HOG": _hog(gray, cfg), "HSV": _hsv(bgr, cfg)}
    if cfg["hu"]:
        out["Hu"] = _hu(gray)
    return out


def extract(bgr, cfg):
    return np.concatenate(list(extract_groups(bgr, cfg).values())).astype(np.float32)


def group_slices(cfg):
    """{'HOG': slice, 'HSV': slice, 'Hu': slice} column ranges of the feature matrix."""
    sizes = {k: len(v) for k, v in extract_groups(np.zeros((32, 32, 3), np.uint8), cfg).items()}
    out, start = {}, 0
    for k, n in sizes.items():
        out[k] = slice(start, start + n)
        start += n
    return out


def _chunk(images, cfg):
    cv2.setNumThreads(0)
    return np.stack([extract(im, cfg) for im in images])


def extract_batch(images, cfg=None, n_jobs=None, desc=None):
    cfg = cfg or make_cfg()
    n_jobs = n_jobs or config.N_JOBS
    if len(images) < 600 or n_jobs == 1:
        return _chunk(images, cfg)
    chunks = np.array_split(np.arange(len(images)), min(len(images) // 200, n_jobs * 4))
    res = Parallel(n_jobs=n_jobs)(delayed(_chunk)(images[c], cfg) for c in chunks)
    return np.concatenate(res)
