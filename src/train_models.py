import os
import sys

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loading import load_or_build


def train():
    _, y_train, _, _ = load_or_build()
    X_train = np.load(os.path.join(config.FEATURE_DIR, "train.npy"))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(config.MODELS_DIR, "scaler.joblib"))

    svm = SVC(**config.SVM_PARAMS)
    svm.fit(X_scaled, y_train)
    joblib.dump(svm, os.path.join(config.MODELS_DIR, "svm.joblib"))

    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(config.MODELS_DIR, "rf.joblib"))

    return svm, rf, scaler


if __name__ == "__main__":
    train()
