"""
feature_extraction.py

Reusable feature extraction: for a single BGR uint8 image (32x32x3), builds
one concatenated feature vector from:
    1. HOG on grayscale (skimage.feature.hog)
    2. Color histogram in HSV (cv2.calcHist, 16 bins/channel, flattened)
    3. Hu Moments (cv2.HuMoments(cv2.moments(gray)), 7-dim, log-scaled)

Total dimensionality (for 32x32 images, 8x8 cells, 2x2 block, 9 orientations):
    HOG: cells = (32/8)x(32/8) = 4x4; blocks = (4-2+1)x(4-2+1)=3x3;
         dims = 3*3*2*2*9 = 324
    HSV hist: 16*3 = 48
    Hu moments: 7
    TOTAL = 324 + 48 + 7 = 379  (well under the 600-dim budget)

Exposes:
    extract_features(img_bgr) -> np.ndarray[379]
    extract_features_batch(images_bgr) -> np.ndarray[N, 379]
"""
import os
import sys

import numpy as np
import cv2
from skimage.feature import hog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def _hog_features(gray_img: np.ndarray) -> np.ndarray:
    feats = hog(
        gray_img,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feats.astype(np.float32)


def _hsv_hist_features(bgr_img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    bins = config.HSV_BINS_PER_CHANNEL
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    # NOTE: per spec this is "16 bins per channel, flattened" -- we compute a
    # joint 3D histogram (bins^3) which is large; to respect the "flattened,
    # concatenated per-channel" reading and the <=600 dim budget, we instead
    # compute three independent 1D histograms below (16*3=48 dims). This
    # helper is kept for reference but is unused; see _hsv_hist_features_1d.
    return hist.flatten().astype(np.float32)


def _hsv_hist_features_1d(bgr_img: np.ndarray) -> np.ndarray:
    """Three independent per-channel histograms (H, S, V), 16 bins each -> 48-dim."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    bins = config.HSV_BINS_PER_CHANNEL
    ranges = [(0, 180), (0, 256), (0, 256)]
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], list(ranges[ch]))
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


def _hu_moments_features(gray_img: np.ndarray) -> np.ndarray:
    moments = cv2.moments(gray_img)
    hu = cv2.HuMoments(moments).flatten()
    # log-scale for numerical stability (Hu moments span many orders of magnitude)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log.astype(np.float32)


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """img_bgr: uint8 array [H, W, 3] in BGR order (OpenCV convention)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    hog_feats = _hog_features(gray)
    hsv_feats = _hsv_hist_features_1d(img_bgr)
    hu_feats = _hu_moments_features(gray)

    return np.concatenate([hog_feats, hsv_feats, hu_feats]).astype(np.float32)


def extract_features_batch(images_bgr: np.ndarray, desc: str = None) -> np.ndarray:
    """images_bgr: uint8 array [N, H, W, 3]. Returns [N, D] float32 array."""
    from tqdm import tqdm

    feats = []
    iterator = images_bgr
    if desc is not None:
        iterator = tqdm(images_bgr, desc=desc)
    for img in iterator:
        feats.append(extract_features(img))
    return np.stack(feats)


def feature_dim() -> int:
    dummy = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
    return extract_features(dummy).shape[0]


if __name__ == "__main__":
    d = feature_dim()
    print(f"Feature vector dimensionality: {d}")
    assert d <= 600, "Feature vector exceeds 600-dim budget!"
