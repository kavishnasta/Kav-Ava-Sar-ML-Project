import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


# Scaler plus RBF SVM, optionally using only some feature columns
class SVMModel:
    def __init__(self, params=None, cols=None):
        self.params, self.cols = dict(params or config.SVM_PARAMS), cols

    # Keep only the chosen feature columns
    def _sel(self, F):
        return F if self.cols is None else F[:, self.cols]

    # Scale the features and train the SVM, gamma is gamma_mult times the scikit learn scale value
    def fit(self, F, y):
        F = self._sel(F)
        self.scaler = StandardScaler().fit(F)
        Fs = self.scaler.transform(F)
        p = dict(self.params)
        if "gamma_mult" in p:
            p["gamma"] = p.pop("gamma_mult") / (Fs.shape[1] * Fs.var())
        self.svc = SVC(**p).fit(Fs, y)
        return self

    # Predict class labels for new feature rows
    def predict(self, F):
        return self.svc.predict(self.scaler.transform(self._sel(F)))


# Split the rows over several processes because SVC predict uses one core
def predict_parallel(model, F, n_jobs=None):
    n_jobs = n_jobs or config.N_JOBS
    if len(F) < 400 or n_jobs == 1:
        return model.predict(F)
    parts = np.array_split(np.arange(len(F)), n_jobs * 2)
    return np.concatenate(Parallel(n_jobs=n_jobs)(delayed(model.predict)(F[p]) for p in parts))


# Accuracy and macro F1 over the classes that appear in the true labels
def metrics(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)
