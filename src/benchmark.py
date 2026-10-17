"""Phase 0: time feature extraction, SVC fit and SVC predict. Writes results/timing.csv."""
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import datasets
import features
from timing import log_time


def main():
    print("cpu_count:", os.cpu_count(), "workers:", config.N_JOBS)
    X, y, g, Xte, yte, k = datasets.load("gtsrb")
    cfg = features.make_cfg()
    features.extract_batch(X[:1200], cfg)              # warm up the worker pool
    t = time.time(); F = features.extract_batch(X[:6000], cfg); t_par = (time.time() - t) / 6
    t = time.time(); features.extract_batch(X[:1000], cfg, n_jobs=1); t_ser = time.time() - t
    log_time("benchmark", "feature_extraction_per_1000_parallel", t_par)
    log_time("benchmark", "feature_extraction_per_1000_single_core", t_ser)
    print(f"feature extraction /1000 imgs: {t_par:.3f}s parallel, {t_ser:.3f}s single core, dim {F.shape[1]}")

    Ftr = features.extract_batch(X, cfg)
    sc = StandardScaler().fit(Ftr)
    Ftr = sc.transform(Ftr)
    for n in (6000, len(Ftr)):
        idx = np.random.RandomState(0).permutation(len(Ftr))[:n]
        svm = SVC(**config.SVM_PARAMS)
        t = time.time(); svm.fit(Ftr[idx], y[idx]); tf = time.time() - t
        Fte = sc.transform(features.extract_batch(Xte[:2000], cfg))
        t = time.time(); svm.predict(Fte); tp = (time.time() - t) / 2
        log_time("benchmark", f"svc_fit_n{n}", tf)
        log_time("benchmark", f"svc_predict_per_1000_n{n}", tp)
        print(f"SVC n_train={n}: fit {tf:.1f}s, predict {tp:.2f}s /1000, support vectors {svm.n_support_.sum()}")


if __name__ == "__main__":
    main()
    print(pd.read_csv(os.path.join(config.RESULTS_DIR, "timing.csv")).to_string())
