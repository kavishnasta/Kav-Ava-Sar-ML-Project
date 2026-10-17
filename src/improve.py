"""Phase 2: improve the clean baseline using the VALIDATION split only (held-out tracks of the training set).

Greedy and cumulative: a change is kept only if it adds >= 0.3 pt validation accuracy
(removals: Hu moments are dropped when val accuracy falls by <= 0.3 pt).
Every evaluated configuration is appended to results/improvement_log.csv. The test set is never loaded here.
Tuning is done on GTSRB; the final config is then sanity-checked on the validation split of BTSD / CTSD.

Timings in the log are measured with several worker processes running concurrently: compare them only relatively.
"""
import copy
import json
import os
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import datasets
import features

LOG = os.path.join(config.RESULTS_DIR, "improvement_log.csv")
GAIN = 0.3          # points of validation accuracy
MAX_DIM = 1500


def cap_per_class(y, n, seed=config.SEED):
    if n is None:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    keep, cnt = [], {}
    for i in idx:
        c = y[i]
        if cnt.get(c, 0) < n:
            cnt[c] = cnt.get(c, 0) + 1
            keep.append(i)
    return np.sort(np.array(keep))


def fit_eval(spec):
    """spec: dataset, crop, per_class, fcfg, params, sub (optional train subsample). Validation only."""
    import cv2
    cv2.setNumThreads(0)
    Xtr, ytr, gtr, _, _, _ = datasets.load(spec["dataset"], spec["crop"])
    tr_i, va_i = datasets.val_split(ytr, gtr)
    tr_i = tr_i[cap_per_class(ytr[tr_i], spec["per_class"])]
    if spec.get("sub"):
        tr_i = tr_i[cap_per_class(ytr[tr_i], spec["sub"] // len(np.unique(ytr[tr_i])), seed=1)]
    Ftr = features.extract_batch(Xtr[tr_i], spec["fcfg"], n_jobs=1)
    Fva = features.extract_batch(Xtr[va_i], spec["fcfg"], n_jobs=1)
    ytr_, yva = ytr[tr_i], ytr[va_i]
    p = dict(spec.get("params", {}))
    sc = StandardScaler().fit(Ftr)
    Fs = sc.transform(Ftr)
    if "gamma_mult" in p:
        p["gamma"] = p.pop("gamma_mult") / (Fs.shape[1] * Fs.var())
    clf = SVC(kernel="rbf", random_state=config.SEED, **p)
    t = time.time()
    clf.fit(Fs, ytr_)
    pre = sc.transform
    fit_s = time.time() - t
    t = time.time()
    pred = clf.predict(pre(Fva))
    pred_s = (time.time() - t) / len(Fva) * 1000
    return dict(dataset=spec["dataset"], val_accuracy=accuracy_score(yva, pred),
                val_macro_f1=f1_score(yva, pred, average="macro", labels=np.unique(yva)),
                feature_dim=Ftr.shape[1], n_train=len(tr_i), n_val=len(va_i), fit_s=fit_s, predict_s_per_1000=pred_s)


def log_rows(step, entries):
    rows = []
    for label, spec, res, adopted, note in entries:
        rows.append(dict(step=step, config=label, **{k: res[k] for k in ("dataset", "val_accuracy", "val_macro_f1", "feature_dim",
                         "n_train", "fit_s", "predict_s_per_1000")}, adopted=adopted, note=note,
                         spec=json.dumps({k: spec[k] for k in ("crop", "per_class", "fcfg", "params", "sub") if k in spec}, default=list)))
    df = pd.DataFrame(rows)
    df.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)


def run_many(specs):
    return Parallel(n_jobs=min(len(specs), config.N_JOBS))(delayed(fit_eval)(s) for s in specs)


def short(res):
    return f"val_acc {res['val_accuracy']*100:.2f} f1 {res['val_macro_f1']*100:.2f} dim {res['feature_dim']} fit {res['fit_s']:.0f}s"


def run_step(step, state, base_res, cands, prefer_smaller=False):
    """cands: list of (label, {state_key: value} or {'fcfg': {...}}). Returns new state, new result."""
    specs = []
    for label, mod in cands:
        s = copy.deepcopy(state)
        for k, v in mod.items():
            if k == "fcfg":
                s["fcfg"].update(v)
            else:
                s[k] = v
        specs.append(s)
    results = run_many(specs)
    entries = [("BASE " + step, state, base_res, "", "reference")]
    scored = []
    for (label, mod), spec, res in zip(cands, specs, results):
        gain = (res["val_accuracy"] - base_res["val_accuracy"]) * 100
        ok = gain >= GAIN or (prefer_smaller and gain >= -GAIN and res["feature_dim"] < base_res["feature_dim"])
        if res["feature_dim"] > MAX_DIM and gain < 1.0:
            ok = False
        note = f"gain {gain:+.2f} pt"
        print(f"  [{step}] {label}: {short(res)} ({note}) -> {'candidate' if ok else 'rejected'}")
        scored.append((gain, label, mod, spec, res, ok, note))
    passed = sorted([s for s in scored if s[5]], key=lambda s: -s[0])
    new_state, new_res, adopted_labels = state, base_res, []
    if passed:
        # combine passing changes (best gain first; a later change on an already-modified key is skipped)
        comb, used = copy.deepcopy(state), set()
        for gain, label, mod, spec, res, ok, note in passed:
            keys = set()
            for k, v in mod.items():
                keys |= {("fcfg", kk) for kk in v} if k == "fcfg" else {(k,)}
            if keys & used:
                continue
            used |= keys
            for k, v in mod.items():
                comb["fcfg"].update(v) if k == "fcfg" else comb.__setitem__(k, v)
            adopted_labels.append(label)
        best = passed[0]
        if len(adopted_labels) > 1:
            cres = run_many([comb])[0]
            print(f"  [{step}] combination {adopted_labels}: {short(cres)}")
            entries.append(("COMBINED " + "+".join(adopted_labels), comb, cres, "", "combination of passing changes"))
            # a combination must itself beat the best single change by the same margin, otherwise it only adds cost
            if (cres["val_accuracy"] - best[4]["val_accuracy"]) * 100 >= GAIN:
                new_state, new_res = comb, cres
            else:
                new_state, new_res, adopted_labels = best[3], best[4], [best[1]]
        else:
            new_state, new_res = best[3], best[4]
    for gain, label, mod, spec, res, ok, note in scored:
        entries.append((label, spec, res, "yes" if label in adopted_labels else "no", note))
    log_rows(step, entries)
    print(f"[{step}] -> {'adopted ' + str(adopted_labels) if adopted_labels else 'no change'}; val_acc {new_res['val_accuracy']*100:.2f}")
    return new_state, new_res


def resume_state(letter):
    """Keep the log rows of the steps before `letter`, and rebuild the state from the last adopted configuration."""
    df = pd.read_csv(LOG)
    df = df[df.step.str[0] < letter]
    df.to_csv(LOG, index=False)
    spec = json.loads(df[df.adopted == "yes"].iloc[-1].spec)
    spec["fcfg"]["hog_cells"] = tuple(spec["fcfg"]["hog_cells"])
    spec.pop("sub", None)
    return dict(dataset="gtsrb", **spec)


def main(resume=None):
    t0 = time.time()
    if resume:
        state = resume_state(resume)
        print(f"resuming before step {resume} from", {k: state[k] for k in ("crop", "per_class")}, state["fcfg"])
        base = run_many([state])[0]
        print("  state re-evaluated:", short(base))
        log_rows(f"{resume} (resume)", [("re-evaluated state at resume", state, base, "yes", "run restarted here after fixing the combination rule")])
        run_from(resume, state, base, t0)
        return
    if os.path.exists(LOG):
        os.remove(LOG)
    state = dict(dataset="gtsrb", crop="roi", per_class=400, params=dict(C=10, gamma="scale"),
                 fcfg=features.make_cfg(median=True, clahe=False, size=32, hog_orient=9, hog_cells=(8,),
                                        hsv_bins=16, hsv_grid=1, hsv_ch="hsv", hu=True))
    print("start: baseline config, GTSRB validation")
    base = run_many([state])[0]
    print("  baseline:", short(base))
    log_rows("0 baseline", [("baseline (roi, 400/class, median, HOG8/9, HSV16, Hu, SVC C=10)", state, base, "yes", "start")])

    run_from("a", state, base, t0)


def run_from(first, state, base, t0):
    go = lambda letter: letter >= first
    if go("a"):
        state, base = run_step("a ROI crop", state, base, [("full frame", dict(crop="full"))])
    if go("b"):
        state, base = run_step("b train size", state, base, [("1000/class", dict(per_class=1000)), ("all", dict(per_class=None))])
    if go("c"):
        state, base = run_step("c CLAHE", state, base, [("CLAHE on L", dict(fcfg=dict(clahe=True)))])
    if go("d"):
        state, base = run_step("d median blur", state, base, [("no median blur", dict(fcfg=dict(median=False)))])
    if go("e"):
        state, base = run_step("e HOG", state, base, [
            ("cell 4x4", dict(fcfg=dict(hog_cells=(4,)))), ("orient 12", dict(fcfg=dict(hog_orient=12))),
            ("resolution 48", dict(fcfg=dict(size=48))), ("multi-scale 8+16", dict(fcfg=dict(hog_cells=(8, 16))))])
    if go("f"):
        state, base = run_step("f HSV", state, base, [
            ("8 bins", dict(fcfg=dict(hsv_bins=8))), ("2x2 spatial grid", dict(fcfg=dict(hsv_grid=2))),
            ("drop S", dict(fcfg=dict(hsv_ch="hv"))), ("drop V", dict(fcfg=dict(hsv_ch="hs")))])
    if go("g"):
        state, base = run_step("g Hu moments", state, base, [("drop Hu", dict(fcfg=dict(hu=False)))], prefer_smaller=True)
    print("features frozen:", state["crop"], state["per_class"], state["fcfg"])

    # ---- h. SVM C x gamma on a subsample, confirm the best three on the full training set
    grid = [(C, m) for C in (1, 3, 10, 30) for m in (0.3, 1, 3)]
    specs = [dict(state, params=dict(C=C, gamma_mult=m), sub=8000) for C, m in grid]
    res = run_many(specs)
    log_rows("h SVM grid (8000-sample subsample)", [(f"C={C} gamma={m}*scale", s, r, "", "subsample") for (C, m), s, r in zip(grid, specs, res)])
    order = np.argsort([-r["val_accuracy"] for r in res])[:3]
    for i in order:
        print(f"  [h] subsample C={grid[i][0]} gamma={grid[i][1]}x: {short(res[i])}")
    specs = [dict(state, params=dict(C=grid[i][0], gamma_mult=grid[i][1])) for i in order]
    fres = run_many(specs)
    best_i = int(np.argmax([r["val_accuracy"] for r in fres]))
    log_rows("h SVM grid (confirm on full train)", [(f"C={grid[i][0]} gamma={grid[i][1]}*scale", s, r, "yes" if j == best_i else "no", "full train")
                                                    for j, (i, s, r) in enumerate(zip(order, specs, fres))])
    bestC, bestM = grid[order[best_i]]
    svm_res = fres[best_i]
    print(f"[h] SVM best: C={bestC} gamma={bestM}*scale  {short(svm_res)}")
    state["params"] = dict(C=bestC, gamma_mult=bestM)

    # ---- cross-dataset validation sanity check (baseline config vs final config)
    entries = []
    base_cfg = dict(dataset="gtsrb", crop="roi", per_class=400, params=dict(C=10, gamma="scale"), fcfg=features.make_cfg(
        median=True, clahe=False, size=32, hog_orient=9, hog_cells=(8,), hsv_bins=16, hsv_grid=1, hsv_ch="hsv", hu=True))
    for ds in ("btsd", "ctsd"):
        try:
            datasets.load(ds)
        except FileNotFoundError:
            continue
        for label, sp in (("baseline config", base_cfg), ("final config", state)):
            s = dict(sp, dataset=ds)
            r = fit_eval(s)
            print(f"  [check {ds}] {label}: {short(r)}")
            entries.append((f"{label} on {ds}", s, r, "", "validation sanity check of the GTSRB-tuned config"))
    log_rows("j cross-dataset validation check", entries)

    write_config(state)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def write_config(state):
    p = dict(state["params"])
    fcfg = {k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in state["fcfg"].items() if k in features.DEFAULT_CFG}
    block = (f"# BEGIN AUTO (written by src/improve.py from the validation-only search)\n"
             f"CROP = {state['crop']!r}\n"
             f"MAX_TRAIN_PER_CLASS = {state['per_class']!r}    # None = all\n"
             f"FEATURE_CFG = {fcfg!r}\n"
             f"SVM_PARAMS = dict(kernel=\"rbf\", C={p['C']!r}, gamma_mult={p['gamma_mult']!r}, random_state=SEED)   # gamma = gamma_mult / (n_features * var)\n"
             f"# END AUTO")
    path = os.path.join(config.ROOT, "src", "config.py")
    s = open(path).read()
    s = re.sub(r"# BEGIN AUTO.*?# END AUTO", lambda m: block, s, flags=re.S) if "# BEGIN AUTO" in s else re.sub(
        r"# --- model/feature config.*?\n(?=\n# one colour)", lambda m: block + "\n", s, flags=re.S)
    open(path, "w").write(s)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-from", default=None, help="step letter (e.g. e): keep the log of earlier steps and continue from there")
    main(ap.parse_args().resume_from)
