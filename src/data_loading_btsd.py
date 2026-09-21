"""
data_loading_btsd.py

Downloads and loads the Belgian Traffic Sign Dataset (BTSD / BelgiumTSC),
the classification variant used as a companion/auxiliary dataset alongside
GTSRB. This module ONLY loads raw data (download, extract, resize, cache to
numpy) — it is not wired into the GTSRB feature/corruption/training pipeline,
since BTSD's class taxonomy (62 classes) differs from GTSRB's (43 classes).

Official source: https://btsd.ethz.ch/shareddata/BelgiumTSC/
    - BelgiumTSC_Training.zip
    - BelgiumTSC_Testing.zip

Directory layout inside each zip (same convention as GTSRB): per-class
subfolders named as 5-digit zero-padded class IDs (e.g. 00000, 00001, ...),
each containing .ppm images and a GT-<classid>.csv annotation file with
columns: Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId

Produces cached numpy arrays under data/cache_btsd/:
    X_train.npy, y_train.npy, X_test.npy, y_test.npy
(uint8 BGR images, resized to config.IMG_SIZE x config.IMG_SIZE)

Run standalone:
    python src/data_loading_btsd.py
"""
import os
import sys
import csv
import zipfile

import numpy as np
import cv2
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

BTSD_BASE_URL = "https://btsd.ethz.ch/shareddata/BelgiumTSC"
BTSD_TRAIN_URL = f"{BTSD_BASE_URL}/BelgiumTSC_Training.zip"
BTSD_TEST_URL = f"{BTSD_BASE_URL}/BelgiumTSC_Testing.zip"

BTSD_RAW_DIR = os.path.join(config.DATA_DIR, "raw_btsd")
BTSD_CACHE_DIR = os.path.join(config.DATA_DIR, "cache_btsd")
os.makedirs(BTSD_RAW_DIR, exist_ok=True)
os.makedirs(BTSD_CACHE_DIR, exist_ok=True)


def _download(url: str, dest_path: str):
    """Downloads with a content-length check so a truncated/interrupted
    download is detected and retried rather than silently accepted."""
    if os.path.exists(dest_path):
        expected = _remote_size(url)
        actual = os.path.getsize(dest_path)
        if expected is None or actual == expected:
            print(f"[data_loading_btsd] Already downloaded: {dest_path}")
            return
        print(f"[data_loading_btsd] Existing file size mismatch "
              f"({actual} != {expected}), re-downloading: {dest_path}")

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

    actual = os.path.getsize(tmp_path)
    if total and actual != total:
        os.remove(tmp_path)
        raise IOError(f"Download truncated: got {actual} bytes, expected {total} for {url}")

    os.replace(tmp_path, dest_path)


def _remote_size(url: str):
    try:
        r = requests.head(url, timeout=15, allow_redirects=True)
        r.raise_for_status()
        cl = r.headers.get("content-length")
        return int(cl) if cl is not None else None
    except requests.RequestException:
        return None


def _extract(zip_path: str, extract_dir: str):
    marker = extract_dir + ".extracted"
    if os.path.exists(marker):
        print(f"[data_loading_btsd] Already extracted: {extract_dir}")
        return
    print(f"[data_loading_btsd] Extracting {zip_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    with open(marker, "w") as f:
        f.write("done")


def download_btsd():
    train_zip = os.path.join(BTSD_RAW_DIR, "BelgiumTSC_Training.zip")
    test_zip = os.path.join(BTSD_RAW_DIR, "BelgiumTSC_Testing.zip")

    _download(BTSD_TRAIN_URL, train_zip)
    _download(BTSD_TEST_URL, test_zip)

    _extract(train_zip, os.path.join(BTSD_RAW_DIR, "train_extracted"))
    _extract(test_zip, os.path.join(BTSD_RAW_DIR, "test_extracted"))


def _find_class_root(extracted_dir: str) -> str:
    """Locate the directory directly containing 5-digit class subfolders."""
    for root, dirs, _files in os.walk(extracted_dir):
        if any(d.isdigit() and len(d) == 5 for d in dirs):
            return root
    raise FileNotFoundError(f"Could not locate BTSD class directories under {extracted_dir}")


def _load_split(extracted_dir: str, desc: str):
    class_root = _find_class_root(extracted_dir)
    class_dirs = sorted(
        d for d in os.listdir(class_root)
        if d.isdigit() and len(d) == 5 and os.path.isdir(os.path.join(class_root, d))
    )

    images, labels = [], []
    for class_dir in tqdm(class_dirs, desc=desc):
        class_id = int(class_dir)
        class_path = os.path.join(class_root, class_dir)
        gt_csv = os.path.join(class_path, f"GT-{class_dir}.csv")
        if not os.path.exists(gt_csv):
            # Some BTSD releases omit per-class CSVs for the test split;
            # fall back to reading every image file directly in that case.
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith((".ppm", ".png", ".jpg", ".jpeg")):
                    continue
                img = cv2.imread(os.path.join(class_path, fname), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(class_id)
            continue

        with open(gt_csv, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                img_path = os.path.join(class_path, row["Filename"])
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(class_id)

    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64)


def load_or_build_btsd_splits():
    """
    Returns (X_train, y_train, X_test, y_test) as numpy arrays for BTSD,
    caching to data/cache_btsd/*.npy. Images are uint8 BGR, resized to
    config.IMG_SIZE x config.IMG_SIZE. Class IDs follow BTSD's own 62-class
    taxonomy (0-61), NOT GTSRB's 43-class taxonomy.
    """
    paths = {
        "X_train": os.path.join(BTSD_CACHE_DIR, "X_train.npy"),
        "y_train": os.path.join(BTSD_CACHE_DIR, "y_train.npy"),
        "X_test": os.path.join(BTSD_CACHE_DIR, "X_test.npy"),
        "y_test": os.path.join(BTSD_CACHE_DIR, "y_test.npy"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("[data_loading_btsd] Loading cached BTSD splits from disk.")
        return (np.load(paths["X_train"]), np.load(paths["y_train"]),
                np.load(paths["X_test"]), np.load(paths["y_test"]))

    download_btsd()

    print("[data_loading_btsd] Building BTSD train split...")
    X_train, y_train = _load_split(os.path.join(BTSD_RAW_DIR, "train_extracted"), "BTSD train classes")
    print(f"[data_loading_btsd] BTSD train: {X_train.shape}, classes={len(np.unique(y_train))}")

    print("[data_loading_btsd] Building BTSD test split...")
    X_test, y_test = _load_split(os.path.join(BTSD_RAW_DIR, "test_extracted"), "BTSD test classes")
    print(f"[data_loading_btsd] BTSD test: {X_test.shape}, classes={len(np.unique(y_test))}")

    np.save(paths["X_train"], X_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["X_test"], X_test)
    np.save(paths["y_test"], y_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_or_build_btsd_splits()
    print("BTSD Train:", X_train.shape, y_train.shape)
    print("BTSD Test:", X_test.shape, y_test.shape)
    print("BTSD classes:", sorted(np.unique(np.concatenate([y_train, y_test]))))
