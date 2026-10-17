"""SVM (StandardScaler + RBF SVC) with parallel prediction and metrics."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


class SVMModel:
    """cols = subset of feature columns to use (None = all); used for the feature-group ablation."""

    def __init__(self, params=None, cols=None):
        self.params, self.cols = dict(params or config.SVM_PARAMS), cols

    def _sel(self, F):
        return F if self.cols is None else F[:, self.cols]

    def fit(self, F, y):
        F = self._sel(F)
        self.scaler = StandardScaler().fit(F)
        Fs = self.scaler.transform(F)
        p = dict(self.params)
        if "gamma_mult" in p:                       # gamma = mult * 'scale' = mult / (n_features * var)
            p["gamma"] = p.pop("gamma_mult") / (Fs.shape[1] * Fs.var())
        self.svc = SVC(**p).fit(Fs, y)
        return self

    def predict(self, F):
        return self.svc.predict(self.scaler.transform(self._sel(F)))


def predict_parallel(model, F, n_jobs=None):
    """SVC predict is single-threaded: split the rows over worker processes."""
    n_jobs = n_jobs or config.N_JOBS
    if len(F) < 400 or n_jobs == 1:
        return model.predict(F)
    parts = np.array_split(np.arange(len(F)), n_jobs * 2)
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(model.predict)(F[p]) for p in parts))


def metrics(y, pred):
    """accuracy and macro-F1 over the classes present in y_true (BTSD/CTSD tests may lack some classes)."""
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)
