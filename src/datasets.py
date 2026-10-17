"""Unified loader for GTSRB, BTSD (BelgiumTSC) and CTSD (Chinese Traffic Signs, Kaggle version).

load(name) -> X_train, y_train, groups_train, X_test, y_test, n_classes
Images are uint8 BGR, IMG_SIZE x IMG_SIZE, cached under data/cache/<dataset>/.
"""
import argparse
import csv
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

import config

RAW = {"gtsrb": os.path.join(config.DATA_DIR, "raw"),
       "btsd": os.path.join(config.DATA_DIR, "raw_btsd"),
       "ctsd": os.path.join(config.DATA_DIR, "raw_ctsd")}
N_CLASSES = {"gtsrb": 43, "btsd": 62, "ctsd": 58}
CTSD_HELP = """CTSD data not found. Manual download:
  1. Download the Kaggle dataset "Chinese Traffic Signs" (5,998 images + annotations.csv with
     columns file_name,width,height,x1,y1,x2,y2,category), e.g. from
     https://www.kaggle.com/datasets/dmitryyemelyanov/chinese-traffic-signs
  2. Unzip so that data/raw_ctsd/annotations.csv and data/raw_ctsd/images/*.png exist.
  3. Re-run this script. CTSD is skipped until then."""


def _resize(img, s):
    return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA if max(img.shape[:2]) > s else cv2.INTER_LINEAR)


def _read(args):
    path, roi = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    s = config.IMG_SIZE
    full = _resize(img, s)
    x1, y1, x2, y2 = roi
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    crop = _resize(img[y1:y2 + 1, x1:x2 + 1], s)
    return crop, full


def _read_all(records, desc):
    """records: list of (path, (x1,y1,x2,y2)) -> X_roi, X_full uint8 arrays."""
    with ThreadPoolExecutor(config.N_JOBS) as ex:
        out = list(tqdm(ex.map(_read, records), total=len(records), desc=desc))
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def _factorize(keys):
    ids = {}
    return np.array([ids.setdefault(k, len(ids)) for k in keys], dtype=np.int64)


def group_split(y, groups, frac, seed):
    """Boolean mask of samples held out. Per class ~frac of the groups (>=1 if the class has >=2 groups)
    are held out as whole groups; classes with a single group are never held out."""
    rng = np.random.RandomState(seed)
    held = np.zeros(len(y), bool)
    for c in np.unique(y):
        gs = np.unique(groups[y == c])
        if len(gs) < 2:
            continue
        k = max(1, int(round(frac * len(gs))))
        chosen = rng.choice(gs, k, replace=False)
        held |= (y == c) & np.isin(groups, chosen)
    return held


def val_split(y_train, groups_train):
    """(train_idx, val_idx) on the training set for tuning. Whole tracks are held out per class."""
    held = group_split(y_train, groups_train, 0.2, config.SEED)
    return np.where(~held)[0], np.where(held)[0]


# ---------------------------------------------------------------- per-dataset record builders
# record = (path, class, (x1,y1,x2,y2), group_key or None)
def _gt_rows(csv_path, folder):
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            yield (os.path.join(folder, r["Filename"]), int(r["ClassId"]),
                   (int(r["Roi.X1"]), int(r["Roi.Y1"]), int(r["Roi.X2"]), int(r["Roi.Y2"])), r["Filename"])


def _build_gtsrb():
    base = RAW["gtsrb"]
    tr = []
    for gt in sorted(glob.glob(os.path.join(base, "train_extracted", "**", "GT-*.csv"), recursive=True)):
        for p, c, roi, fn in _gt_rows(gt, os.path.dirname(gt)):
            tr.append((p, c, roi, (c, fn.split("_")[0])))      # group = (class, track id)
    gt_test = os.path.join(base, "test_gt_extracted", "GT-final_test.csv")
    img_dir = os.path.dirname(glob.glob(os.path.join(base, "test_extracted", "**", "00000.ppm"), recursive=True)[0])
    te = [(p, c, roi, None) for p, c, roi, fn in _gt_rows(gt_test, img_dir)]
    return tr, te


def _build_btsd():
    recs = []
    for split in ("train_extracted", "test_extracted"):
        for gt in sorted(glob.glob(os.path.join(RAW["btsd"], split, "**", "GT-*.csv"), recursive=True)):
            for p, c, roi, fn in _gt_rows(gt, os.path.dirname(gt)):
                recs.append((p, c, roi, (c, fn.split("_")[0])))   # track ids repeat across classes -> key on (class, track)
    y = np.array([r[1] for r in recs])
    g = _factorize([r[3] for r in recs])
    held = group_split(y, g, 0.2, config.SEED)
    lines = ["BTSD (BelgiumTSC): official train+test pooled, split 80/20 by track (group = class + track id) within each class, seed 42.",
             f"pooled images: {len(y)}, tracks: {len(np.unique(g))}, test images: {int(held.sum())}"]
    single = [int(c) for c in np.unique(y) if len(np.unique(g[y == c])) < 2]
    lines.append(f"classes with a single track (train only, excluded from macro-F1): {single if single else 'none'}")
    absent = [int(c) for c in np.unique(y) if not held[y == c].any()]
    lines.append(f"classes absent from the test split: {absent if absent else 'none'}")
    lines.append(f"classes present in test: {len(np.unique(y[held]))} of {len(np.unique(y))}")
    with open(os.path.join(config.RESULTS_DIR, "dataset_notes_btsd.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return [r for r, h in zip(recs, held) if not h], [r for r, h in zip(recs, held) if h]


def _build_ctsd():
    import pandas as pd
    base = RAW["ctsd"]
    ann = pd.read_csv(os.path.join(base, "annotations.csv"))
    # some file names occur twice in the CSV (two different source images, only one on disk): keep the row whose
    # width/height match the image that exists, drop the other
    from PIL import Image
    sizes = {f: Image.open(os.path.join(base, "images", f)).size for f in ann.file_name.unique()}
    ok = np.array([sizes[r.file_name] == (r.width, r.height) for r in ann.itertuples()])
    n_dropped = int((~ok).sum())
    ann = ann[ok]
    recs = []
    for r in ann.itertuples():
        # "<cls>_<n>_j.png" is a jittered re-crop of "<cls>_<n>.png": same source -> same group (avoids leakage)
        key = (int(r.category), r.file_name.replace("_j.png", ".png"))
        recs.append((os.path.join(base, "images", r.file_name), int(r.category), (r.x1, r.y1, r.x2, r.y2), key))
    y = np.array([r[1] for r in recs])
    g = _factorize([r[3] for r in recs])
    held = group_split(y, g, 0.2, config.SEED)
    single = [int(c) for c in np.unique(y) if len(np.unique(g[y == c])) < 2]
    lines = ["CTSD (Kaggle 'Chinese Traffic Signs'): cropped to box, split 80/20 by source image within each class (seed 42).",
             "Files '*_j.png' are jittered variants of '*.png'; they share a group so a sign never appears in both train and test.",
             f"CSV rows dropped because the box belongs to a different image than the one on disk (size mismatch): {n_dropped}",
             f"rows: {len(y)}, images: {ann.file_name.nunique()}, groups: {len(np.unique(g))}, test rows: {int(held.sum())}",
             f"classes with a single group (train only, excluded from macro-F1): {single or 'none'}",
             f"classes present in test: {len(np.unique(y[held]))} of {len(np.unique(y))}"]
    with open(os.path.join(config.RESULTS_DIR, "dataset_notes_ctsd.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return [r for r, h in zip(recs, held) if not h], [r for r, h in zip(recs, held) if h]


def _available(name):
    if name == "ctsd":
        return os.path.exists(os.path.join(RAW["ctsd"], "annotations.csv"))
    return os.path.isdir(RAW[name])


def _cache_dir(name):
    d = os.path.join(config.CACHE_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def build(name):
    tr, te = {"gtsrb": _build_gtsrb, "btsd": _build_btsd, "ctsd": _build_ctsd}[name]()
    d = _cache_dir(name)
    for split, recs in (("train", tr), ("test", te)):
        roi, full = _read_all([(r[0], r[2]) for r in recs], f"{name} {split}")
        np.save(os.path.join(d, f"X_{split}_roi.npy"), roi)
        np.save(os.path.join(d, f"X_{split}_full.npy"), full)
        np.save(os.path.join(d, f"y_{split}.npy"), np.array([r[1] for r in recs], dtype=np.int64))
        if split == "train":
            np.save(os.path.join(d, "g_train.npy"), _factorize([r[3] for r in recs]))


def load(name, crop=None):
    crop = crop or config.CROP
    d = _cache_dir(name)
    need = [f"X_train_{crop}.npy", f"X_test_{crop}.npy", "y_train.npy", "y_test.npy", "g_train.npy"]
    if not all(os.path.exists(os.path.join(d, n)) for n in need):
        if not _available(name):
            raise FileNotFoundError(CTSD_HELP if name == "ctsd" else f"raw data for {name} not found in {RAW[name]}")
        build(name)
    a = lambda n: np.load(os.path.join(d, n))
    y_train = a("y_train.npy")
    assert y_train.max() < N_CLASSES[name]
    return a(f"X_train_{crop}.npy"), y_train, a("g_train.npy"), a(f"X_test_{crop}.npy"), a("y_test.npy"), N_CLASSES[name]


def resolve(arg):
    """--dataset value -> list of available dataset names (prints help and skips a dataset whose data is missing)."""
    names = config.DATASETS if arg == "all" else [arg]
    out = []
    for n in names:
        try:
            load(n)
            out.append(n)
        except FileNotFoundError as e:
            print(f"[skip {n}] {e}")
    return out


def add_dataset_arg(p):
    p.add_argument("--dataset", default="all", choices=config.DATASETS + ["all"])
    return p


if __name__ == "__main__":
    args = add_dataset_arg(argparse.ArgumentParser()).parse_args()
    for n in resolve(args.dataset):
        Xtr, ytr, gtr, Xte, yte, k = load(n)
        tr_i, va_i = val_split(ytr, gtr)
        print(f"{n}: train {Xtr.shape} test {Xte.shape} classes {k} (train {len(np.unique(ytr))}, test {len(np.unique(yte))}) "
              f"groups {len(np.unique(gtr))} val split {len(tr_i)}/{len(va_i)}")
