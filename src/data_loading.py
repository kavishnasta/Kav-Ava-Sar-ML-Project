"""
data_loading.py

Downloads the official GTSRB dataset (INI Rub benchmark), extracts it,
loads images + labels for the official train/test split, resizes to
32x32, and optionally subsamples the training set per class.

Produces two pickled manifests under data/cache/:
    - train_manifest.pkl : list of (image_path_or_array, label)
    - test_manifest.pkl

To keep memory bounded we load images into numpy arrays once here and
cache them as .npy files (images) + .npy (labels), which downstream
scripts (feature_extraction.py) consume directly.

Run standalone:
    python src/data_loading.py
"""
import os
import sys
import csv
import zipfile
import random

import numpy as np
import cv2
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def _download(url: str, dest_path: str):
    if os.path.exists(dest_path):
        print(f"[data_loading] Already downloaded: {dest_path}")
        return
    print(f"[data_loading] Downloading {url} -> {dest_path}")
    tmp_path = dest_path + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    os.replace(tmp_path, dest_path)


def _extract(zip_path: str, extract_dir: str):
    marker = extract_dir + ".extracted"
    if os.path.exists(marker):
        print(f"[data_loading] Already extracted: {extract_dir}")
        return
    print(f"[data_loading] Extracting {zip_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    with open(marker, "w") as f:
        f.write("done")


def download_gtsrb():
    """Download + extract train images, test images, and test ground truth."""
    train_zip = os.path.join(config.RAW_DIR, "GTSRB-Training_fixed.zip")
    test_img_zip = os.path.join(config.RAW_DIR, "GTSRB_Final_Test_Images.zip")
    test_gt_zip = os.path.join(config.RAW_DIR, "GTSRB_Final_Test_GT.zip")

    _download(config.GTSRB_TRAIN_URL, train_zip)
    _download(config.GTSRB_TEST_IMAGES_URL, test_img_zip)
    _download(config.GTSRB_TEST_GT_URL, test_gt_zip)

    _extract(train_zip, os.path.join(config.RAW_DIR, "train_extracted"))
    _extract(test_img_zip, os.path.join(config.RAW_DIR, "test_extracted"))
    _extract(test_gt_zip, os.path.join(config.RAW_DIR, "test_gt_extracted"))


def _find_train_root():
    """Locate the 'Training' directory (contains 00000..00042 class subdirs)."""
    base = os.path.join(config.RAW_DIR, "train_extracted")
    for root, dirs, _files in os.walk(base):
        if os.path.basename(root) == "Images" or any(
            d.isdigit() and len(d) == 5 for d in dirs
        ):
            # root itself may directly contain 00000/, 00001/, ...
            if any(d.isdigit() and len(d) == 5 for d in dirs):
                return root
    raise FileNotFoundError("Could not locate GTSRB training class directories.")


def _load_train_split():
    """
    Loads GTSRB training images from per-class subdirs (00000..00042), each with
    a GT-<classid>.csv annotation file (Filename;Width;Height;Roi.X1;...;ClassId).
    Subsamples to config.MAX_TRAIN_PER_CLASS images per class (random, seeded).
    Returns (images: uint8 array [N,32,32,3] BGR, labels: int array [N]).
    """
    train_root = _find_train_root()
    class_dirs = sorted(
        d for d in os.listdir(train_root)
        if d.isdigit() and len(d) == 5 and os.path.isdir(os.path.join(train_root, d))
    )
    assert len(class_dirs) == config.NUM_CLASSES, (
        f"Expected {config.NUM_CLASSES} classes, found {len(class_dirs)}"
    )

    rng = random.Random(config.RANDOM_SEED)
    images, labels = [], []

    for class_dir in tqdm(class_dirs, desc="Loading train classes"):
        class_id = int(class_dir)
        class_path = os.path.join(train_root, class_dir)
        gt_csv = os.path.join(class_path, f"GT-{class_dir}.csv")

        rows = []
        with open(gt_csv, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                rows.append(row["Filename"])

        if config.MAX_TRAIN_PER_CLASS is not None and len(rows) > config.MAX_TRAIN_PER_CLASS:
            rows = rng.sample(rows, config.MAX_TRAIN_PER_CLASS)

        for fname in rows:
            img_path = os.path.join(class_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(class_id)

    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64)


def _load_test_split():
    """
    Loads the official GTSRB test set (single flat directory of images) with
    labels from GT-final_test.csv. Uses the FULL official test set (no subsampling,
    since corruption benchmarking should reflect the whole benchmark).
    Returns (images: uint8 array [N,32,32,3] BGR, labels: int array [N]).
    """
    test_img_dir = os.path.join(config.RAW_DIR, "test_extracted")
    # locate directory containing the .ppm test images
    img_root = None
    for root, _dirs, files in os.walk(test_img_dir):
        if any(f.lower().endswith(".ppm") for f in files):
            img_root = root
            break
    if img_root is None:
        raise FileNotFoundError("Could not locate GTSRB test images.")

    gt_dir = os.path.join(config.RAW_DIR, "test_gt_extracted")
    gt_csv = None
    for root, _dirs, files in os.walk(gt_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                gt_csv = os.path.join(root, f)
                break
        if gt_csv:
            break
    if gt_csv is None:
        raise FileNotFoundError("Could not locate GTSRB test ground-truth CSV.")

    images, labels = [], []
    with open(gt_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in tqdm(list(reader), desc="Loading test set"):
            img_path = os.path.join(img_root, row["Filename"])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
            images.append(img)
            labels.append(int(row["ClassId"]))

    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64)


def load_or_build_splits():
    """
    Returns (X_train, y_train, X_test, y_test) as numpy arrays, caching the
    result to data/cache/*.npy so repeated runs skip re-decoding images.
    Images are uint8 BGR (OpenCV convention), shape [N, 32, 32, 3].
    """
    paths = {
        "X_train": os.path.join(config.CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(config.CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(config.CACHE_DIR, "X_test.npy"),
        "y_test": os.path.join(config.CACHE_DIR, "y_test.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("[data_loading] Loading cached splits from disk.")
        X_train = np.load(paths["X_train"])
        y_train = np.load(paths["y_train"])
        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"])
        return X_train, y_train, X_test, y_test

    download_gtsrb()

    print("[data_loading] Building train split...")
    X_train, y_train = _load_train_split()
    print(f"[data_loading] Train: {X_train.shape}, classes={len(np.unique(y_train))}")

    print("[data_loading] Building test split...")
    X_test, y_test = _load_test_split()
    print(f"[data_loading] Test: {X_test.shape}, classes={len(np.unique(y_test))}")

    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_test"], X_test)
    np.save(paths["y_test"], y_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_or_build_splits()
    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("Train class distribution (min/max per class):",
          np.bincount(y_train).min(), np.bincount(y_train).max())
