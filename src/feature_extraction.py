import os
import sys

import numpy as np
import cv2
from skimage.feature import hog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def hog_features(gray):
    return hog(
        gray,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PIXELS_PER_CELL,
        cells_per_block=config.HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def hsv_features(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ranges = [(0, 180), (0, 256), (0, 256)]
    parts = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [config.HSV_BINS], list(ranges[ch]))
        hist = cv2.normalize(hist, hist).flatten()
        parts.append(hist)
    return np.concatenate(parts).astype(np.float32)


def hu_features(gray):
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu.astype(np.float32)


def extract(bgr):
    bgr = cv2.medianBlur(bgr, 3)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.concatenate([hog_features(gray), hsv_features(bgr), hu_features(gray)]).astype(np.float32)


def extract_batch(images, desc=None):
    from tqdm import tqdm
    iterator = tqdm(images, desc=desc) if desc else images
    return np.stack([extract(img) for img in iterator])


if __name__ == "__main__":
    dummy = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
    print(extract(dummy).shape[0])
