"""
corrupt_and_cache.py

Applies all 14 corruption types x 5 severities from the `imagecorruptions`
package to the TEST SET ONLY, then extracts features for each corrupted
version and caches everything to disk so it is never regenerated.

Cache layout (data/cache/):
    corrupted/{corruption}_{severity}.npy   -- corrupted images [N,32,32,3] uint8
    features/test_clean.npy                 -- clean test features [N,D]
    features/train_clean.npy                -- clean train features [N,D]
    features/{corruption}_{severity}.npy    -- corrupted test features [N,D]

imagecorruptions expects RGB uint8 images with a minimum side of 32 (we use
32x32, which is the package's tested minimum) and returns RGB uint8. Our
pipeline stores images as BGR (OpenCV convention) internally, so we convert
BGR<->RGB around the corrupt() call.

Run standalone:
    python src/corrupt_and_cache.py
"""
import os
import sys

import numpy as np
import cv2
from tqdm import tqdm

# Compat shim: imagecorruptions==1.1.2 uses np.float_, removed in NumPy>=2.0.
# Must be set before importing imagecorruptions (it references np.float_ at
# call time inside corruptions.py's plasma_fractal(), used by fog()).
if not hasattr(np, "float_"):
    np.float_ = np.float64

from imagecorruptions import corrupt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loading import load_or_build_splits
from feature_extraction import extract_features_batch


def _corrupted_cache_path(corruption: str, severity: int) -> str:
    return os.path.join(config.CORRUPTED_CACHE_DIR, f"{corruption}_s{severity}.npy")


def _feature_cache_path(name: str) -> str:
    return os.path.join(config.FEATURE_CACHE_DIR, f"{name}.npy")


def generate_corrupted_images(X_test: np.ndarray, corruption: str, severity: int) -> np.ndarray:
    """Returns corrupted BGR uint8 images [N,32,32,3], cached to disk."""
    cache_path = _corrupted_cache_path(corruption, severity)
    if os.path.exists(cache_path):
        return np.load(cache_path)

    out = np.empty_like(X_test)
    for i, img_bgr in enumerate(X_test):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        corrupted_rgb = corrupt(img_rgb, corruption_name=corruption, severity=severity)
        corrupted_bgr = cv2.cvtColor(corrupted_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out[i] = corrupted_bgr

    np.save(cache_path, out)
    return out


def build_all_caches():
    X_train, y_train, X_test, y_test = load_or_build_splits()

    # --- Clean features (train + test) ---
    train_feat_path = _feature_cache_path("train_clean")
    if not os.path.exists(train_feat_path):
        print("[corrupt_and_cache] Extracting clean TRAIN features...")
        feats = extract_features_batch(X_train, desc="train_clean features")
        np.save(train_feat_path, feats)
    else:
        print("[corrupt_and_cache] Clean TRAIN features already cached.")

    test_feat_path = _feature_cache_path("test_clean")
    if not os.path.exists(test_feat_path):
        print("[corrupt_and_cache] Extracting clean TEST features...")
        feats = extract_features_batch(X_test, desc="test_clean features")
        np.save(test_feat_path, feats)
    else:
        print("[corrupt_and_cache] Clean TEST features already cached.")

    # --- Corrupted test images + features, all 70 conditions ---
    conditions = [
        (c, s) for c in config.CORRUPTION_TYPES for s in config.SEVERITIES
    ]
    for corruption, severity in tqdm(conditions, desc="corruption x severity grid"):
        feat_name = f"{corruption}_s{severity}"
        feat_path = _feature_cache_path(feat_name)
        if os.path.exists(feat_path):
            continue

        corrupted_imgs = generate_corrupted_images(X_test, corruption, severity)
        feats = extract_features_batch(corrupted_imgs)
        np.save(feat_path, feats)

    print("[corrupt_and_cache] All caches built.")


if __name__ == "__main__":
    build_all_caches()
