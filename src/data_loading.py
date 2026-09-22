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


def get_raw_data():
    train_zip = os.path.join(config.RAW_DIR, "train.zip")
    test_zip = os.path.join(config.RAW_DIR, "test.zip")
    gt_zip = os.path.join(config.RAW_DIR, "test_gt.zip")

    download(config.TRAIN_URL, train_zip)
    download(config.TEST_IMG_URL, test_zip)
    download(config.TEST_GT_URL, gt_zip)

    extract(train_zip, os.path.join(config.RAW_DIR, "train"))
    extract(test_zip, os.path.join(config.RAW_DIR, "test"))
    extract(gt_zip, os.path.join(config.RAW_DIR, "test_gt"))


def find_train_root():
    base = os.path.join(config.RAW_DIR, "train")
    for root, dirs, _ in os.walk(base):
        if any(d.isdigit() and len(d) == 5 for d in dirs):
            return root
    raise FileNotFoundError("train classes not found")


def load_train():
    root = find_train_root()
    classes = sorted(d for d in os.listdir(root) if d.isdigit() and len(d) == 5)
    rng = random.Random(config.SEED)
    images, labels = [], []

    for c in tqdm(classes, desc="train"):
        class_id = int(c)
        class_dir = os.path.join(root, c)
        gt_file = os.path.join(class_dir, f"GT-{c}.csv")
        with open(gt_file, newline="") as f:
            rows = [row["Filename"] for row in csv.DictReader(f, delimiter=";")]
        if len(rows) > config.MAX_TRAIN_PER_CLASS:
            rows = rng.sample(rows, config.MAX_TRAIN_PER_CLASS)
        for name in rows:
            img = cv2.imread(os.path.join(class_dir, name), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            images.append(img)
            labels.append(class_id)

    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64)


def load_test():
    img_dir = None
    for root, _, files in os.walk(os.path.join(config.RAW_DIR, "test")):
        if any(f.lower().endswith(".ppm") for f in files):
            img_dir = root
            break

    gt_file = None
    for root, _, files in os.walk(os.path.join(config.RAW_DIR, "test_gt")):
        for f in files:
            if f.lower().endswith(".csv"):
                gt_file = os.path.join(root, f)
        if gt_file:
            break

    images, labels = [], []
    with open(gt_file, newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    for row in tqdm(rows, desc="test"):
        img = cv2.imread(os.path.join(img_dir, row["Filename"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        images.append(img)
        labels.append(int(row["ClassId"]))

    return np.stack(images).astype(np.uint8), np.array(labels, dtype=np.int64)


def load_or_build():
    paths = {k: os.path.join(config.CACHE_DIR, f"{k}.npy") for k in ["X_train", "y_train", "X_test", "y_test"]}
    if all(os.path.exists(p) for p in paths.values()):
        return tuple(np.load(paths[k]) for k in ["X_train", "y_train", "X_test", "y_test"])

    get_raw_data()
    X_train, y_train = load_train()
    X_test, y_test = load_test()

    for k, arr in zip(["X_train", "y_train", "X_test", "y_test"], [X_train, y_train, X_test, y_test]):
        np.save(paths[k], arr)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_or_build()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
