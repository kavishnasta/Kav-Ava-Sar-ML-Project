"""Phase 4: clean baseline + robustness grid (14 corruptions x 5 severities) for the SVM.

For every (corruption, severity) the test images are corrupted once and features are extracted once.
Severity 0 = clean (always the full test set). Outputs in results/:
  baseline_<ds>.csv            train/test accuracy + macro-F1 (full sets; plus test_subset in fast mode)
  grid_<ds>.csv                dataset,model,corruption,family,severity,accuracy,macro_f1,n_samples,mode  (resumable)
  ablation_features_<ds>.csv   SVM on feature groups, clean + severities 3 and 5
  ablation_preproc_<ds>.csv    GTSRB only: SVM with median blur / CLAHE flipped, clean + severities 1, 3, 5
  preds_<ds>.npz               y_true / y_pred for clean and every corruption at severity 5
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import config
import corruptions
import datasets
import features
import improve
from models import SVMModel, metrics, predict_parallel
from timing import timed

COLS = ["dataset", "model", "corruption", "family", "severity", "accuracy", "macro_f1", "n_samples", "mode"]


def R(name, ds, ext="csv"):
    return os.path.join(config.RESULTS_DIR, f"{name}_{ds}.{ext}")


def stratified_subset(y, n, seed=config.SEED):
    if len(y) <= n:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, min(max(1, round(n * len(ci) / len(y))), len(ci)), replace=False))
    return np.sort(np.array(idx))


def cached(path, sig, build):
    """Reuse a trained model from disk if it was trained with the same settings."""
    if os.path.exists(path):
        obj, s = joblib.load(path)
        if s == sig:
            return obj
    obj = build()
    joblib.dump((obj, sig), path)
    return obj


def fit_svm(F, y, cols=None):
    return SVMModel(config.SVM_PARAMS, cols).fit(F, y)


def eval_loop(ds, mode, X_test, y_test, idx, entries, fcfgs, conditions, store=None, desc=""):
    """entries: list of dict(name, model, key (feature cfg), sev (set or None), path (csv))."""
    seen = set()
    for p in {e["path"] for e in entries}:
        if os.path.exists(p):                       # resume: keep only rows of this mode
            df = pd.read_csv(p)
            df = df[df["mode"] == mode]
            df.to_csv(p, index=False)
            seen |= {(p, r.model, r.corruption, r.severity) for r in df.itertuples()}
    bar = tqdm(conditions, desc=desc, mininterval=20)
    for corr, sev in bar:
        todo = [e for e in entries if (e["sev"] is None or sev in e["sev"]) and (e["path"], e["name"], corr, sev) not in seen]
        if not todo:
            continue
        bar.set_postfix_str(f"{corr} s{sev}")
        if sev == 0:
            X, y = X_test, y_test
        else:
            X, y = corruptions.corrupt_batch(X_test[idx], corr, sev, ds, n_jobs=config.N_JOBS), y_test[idx]
        feats = {}
        for e in todo:
            if e["key"] not in feats:
                feats[e["key"]] = features.extract_batch(X, fcfgs[e["key"]])
            pred = predict_parallel(e["model"], feats[e["key"]])
            acc, f1 = metrics(y, pred)
            row = dict(dataset=ds, model=e["name"], corruption=corr, family=corruptions.FAMILY_OF.get(corr, "clean"),
                       severity=sev, accuracy=acc, macro_f1=f1, n_samples=len(y), mode=mode)
            pd.DataFrame([row], columns=COLS).to_csv(e["path"], mode="a", header=not os.path.exists(e["path"]), index=False)
            if store is not None and e["name"] == "SVM" and sev in (0, 5):
                store[f"{corr}__{sev}"] = pred.astype(np.int16)
        if store is not None and sev in (0, 5):
            np.savez_compressed(R("preds", ds, "npz"), **store)


def run_dataset(ds, mode, retrain, subset):
    print(f"\n===== {ds} ({mode} mode) =====")
    Xtr, ytr, gtr, Xte, yte, _ = datasets.load(ds)
    fcfg = features.make_cfg()
    cap = improve.cap_per_class(ytr, config.MAX_TRAIN_PER_CLASS)
    sig = repr((ds, config.CROP, config.MAX_TRAIN_PER_CLASS, fcfg, config.SVM_PARAMS))
    mp = lambda n: os.path.join(config.MODELS_DIR, f"{n}_{ds}.joblib")
    if retrain:
        for n in ("svm", "svm_featgroups"):
            if os.path.exists(mp(n)):
                os.remove(mp(n))

    with timed(f"grid_{ds}", "train_features"):
        Ftr = features.extract_batch(Xtr, fcfg)
    sl = features.group_slices(fcfg)
    groups = {"HOG": sl["HOG"], "HSV": sl["HSV"], "HOG+HSV": slice(sl["HOG"].start, sl["HSV"].stop)}
    if "Hu" in sl:
        groups["Hu"] = sl["Hu"]
    print(f"train set {len(Xtr)} (fit uses {len(cap)}), feature dim {Ftr.shape[1]}")

    with timed(f"grid_{ds}", "train_svm"):
        svm = cached(mp("svm"), sig, lambda: fit_svm(Ftr[cap], ytr[cap]))
        abl = cached(mp("svm_featgroups"), sig, lambda: Parallel(n_jobs=min(len(groups), config.N_JOBS))(
            delayed(fit_svm)(Ftr[cap], ytr[cap], np.arange(g.start, g.stop)) for g in groups.values()))

    idx = np.arange(len(yte)) if mode == "full" else stratified_subset(yte, subset)
    np.save(R("grid_subset", ds, "npy"), idx)
    with timed(f"grid_{ds}", "baseline"):
        Fte = features.extract_batch(Xte, fcfg)
        base = []
        for split, F, y in (("train", Ftr, ytr), ("test", Fte, yte), ("test_subset", Fte[idx], yte[idx])):
            if split == "test_subset" and len(idx) == len(yte):
                continue
            acc, f1 = metrics(y, predict_parallel(svm, F))
            base.append(dict(dataset=ds, model="SVM", split=split, accuracy=acc, macro_f1=f1, n_samples=len(y)))
        pd.DataFrame(base).to_csv(R("baseline", ds), index=False)
        print(pd.DataFrame(base).to_string(index=False))

    entries = [dict(name="SVM", model=svm, key="main", sev=None, path=R("grid", ds))]
    entries += [dict(name=f"SVM[{g}]", model=m, key="main", sev={0, 3, 5}, path=R("ablation_features", ds)) for g, m in zip(groups, abl)]
    conditions = [("clean", 0)] + [(c, s) for c in corruptions.NAMES for s in range(1, 6)]
    store = dict(np.load(R("preds", ds, "npz"))) if os.path.exists(R("preds", ds, "npz")) else {}
    store.update(y_true_clean=yte.astype(np.int16), y_true_sub=yte[idx].astype(np.int16))
    with timed(f"grid_{ds}", "grid_eval"):
        eval_loop(ds, mode, Xte, yte, idx, entries, {"main": fcfg}, conditions, store, desc=f"{ds} grid")

    if ds == "gtsrb":
        preproc_ablation(ds, mode, Xtr, ytr, Xte, yte, idx, cap, fcfg)


def preproc_ablation(ds, mode, Xtr, ytr, Xte, yte, idx, cap, fcfg):
    """SVM trained with the median blur / CLAHE setting flipped (everything else identical)."""
    tag = lambda c: f"SVM[median={'on' if c['median'] else 'off'},clahe={'on' if c['clahe'] else 'off'}]"
    variants = [fcfg, dict(fcfg, median=not fcfg["median"]), dict(fcfg, clahe=not fcfg["clahe"])]
    with timed(f"grid_{ds}", "preproc_ablation_train"):
        fits = Parallel(n_jobs=len(variants))(
            delayed(lambda c: fit_svm(features.extract_batch(Xtr[cap], c, n_jobs=1), ytr[cap]))(c) for c in variants)
    entries = [dict(name=tag(c), model=m, key=tag(c), sev={0, 1, 3, 5}, path=R("ablation_preproc", ds)) for c, m in zip(variants, fits)]
    conditions = [("clean", 0)] + [(c, s) for c in corruptions.NAMES for s in (1, 3, 5)]
    with timed(f"grid_{ds}", "preproc_ablation_eval"):
        eval_loop(ds, mode, Xte, yte, idx, entries, {tag(c): c for c in variants}, conditions, desc="preproc ablation")


def main():
    ap = datasets.add_dataset_arg(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--subset", type=int, default=2000, help="fast mode: max test images used for the corruption grid")
    args = ap.parse_args()
    for ds in datasets.resolve(args.dataset):
        run_dataset(ds, args.mode, args.retrain, args.subset)


if __name__ == "__main__":
    main()
