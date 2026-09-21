"""
train_models.py

Trains SVM (RBF kernel) and Random Forest classifiers once on the clean
training features, then saves them (joblib) for reuse across the entire
70-condition evaluation grid (no retraining per corruption).

Also fits a StandardScaler on train features (helps the SVM; RF is
scale-invariant but we apply the same scaled features to both for a fair,
single feature pipeline) and saves it alongside the models.

Run standalone:
    python src/train_models.py
"""
import os
import sys
import time

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loading import load_or_build_splits
from corrupt_and_cache import _feature_cache_path


def load_clean_features():
    _X_train_img, y_train, _X_test_img, y_test = load_or_build_splits()
    X_train = np.load(_feature_cache_path("train_clean"))
    X_test = np.load(_feature_cache_path("test_clean"))
    return X_train, y_train, X_test, y_test


def train_and_save():
    X_train, y_train, X_test, y_test = load_clean_features()

    scaler_path = os.path.join(config.MODELS_DIR, "scaler.joblib")
    svm_path = os.path.join(config.MODELS_DIR, "svm.joblib")
    rf_path = os.path.join(config.MODELS_DIR, "rf.joblib")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, scaler_path)

    print(f"[train_models] Training SVM on {X_train_scaled.shape} features...")
    t0 = time.time()
    svm = SVC(**config.SVM_PARAMS)
    svm.fit(X_train_scaled, y_train)
    print(f"[train_models] SVM trained in {time.time() - t0:.1f}s")
    joblib.dump(svm, svm_path)

    print(f"[train_models] Training RandomForest on {X_train.shape} features...")
    t0 = time.time()
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    print(f"[train_models] RF trained in {time.time() - t0:.1f}s")
    joblib.dump(rf, rf_path)

    print(f"[train_models] Saved models to {config.MODELS_DIR}")
    return svm, rf, scaler


if __name__ == "__main__":
    train_and_save()
