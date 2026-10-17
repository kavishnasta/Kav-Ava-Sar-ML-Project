import os
import csv
import zipfile

import numpy as np
import cv2
import requests
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw_btsd")
CACHE_DIR = os.path.join(ROOT, "data", "cache_btsd")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_URL = "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip"
TEST_URL = "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip"
IMG_SIZE = 32


def download(url, path):
    if os.path.exists(path):
        return
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(path)) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    os.replace(tmp, path)


def extract(zip_path, out_dir):
    marker = out_dir + ".done"
    if os.path.exists(marker):
        return
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    open(marker, "w").close()


def find_class_root(base):
    for root, dirs, _ in os.walk(base):
        if any(d.isdigit() and len(d) == 5 for d in dirs):
            return root
    raise FileNotFoundError("class dirs not found")


def load_split(base, desc):
    root = find_class_root(base)
    classes = sorted(d for d in os.listdir(root) if d.isdigit() and len(d) == 5)
    images, labels = [], []

    for c in tqdm(classes, desc=desc):
        class_id = int(c)
        class_dir = os.path.join(root, c)
        gt_file = os.path.join(class_dir, f"GT-{c}.csv")
        if os.path.exists(gt_file):
            with open(gt_file, newline="") as f:
                names = [row["Filename"] for row in csv.DictReader(f, delimiter=";")]
        else:
            names = [f for f in sorted(os.listdir(class_dir)) if f.lower().endswith((".ppm", ".png", ".jpg"))]
        for name in names:
            img = cv2.imread(os.path.join(class_dir, name), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_id)

    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64)


def load_or_build():
    paths = {k: os.path.join(CACHE_DIR, f"{k}.npy") for k in ["X_train", "y_train", "X_test", "y_test"]}
    if all(os.path.exists(p) for p in paths.values()):
        return tuple(np.load(paths[k]) for k in ["X_train", "y_train", "X_test", "y_test"])

    train_zip = os.path.join(RAW_DIR, "train.zip")
    test_zip = os.path.join(RAW_DIR, "test.zip")
    download(TRAIN_URL, train_zip)
    download(TEST_URL, test_zip)
    extract(train_zip, os.path.join(RAW_DIR, "train"))
    extract(test_zip, os.path.join(RAW_DIR, "test"))

    X_train, y_train = load_split(os.path.join(RAW_DIR, "train"), "train")
    X_test, y_test = load_split(os.path.join(RAW_DIR, "test"), "test")

    for k, arr in zip(["X_train", "y_train", "X_test", "y_test"], [X_train, y_train, X_test, y_test]):
        np.save(paths[k], arr)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_or_build()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
