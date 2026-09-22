import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loading import load_or_build
from feature_extraction import extract_batch


def build():
    X_train, y_train, X_test, y_test = load_or_build()

    train_path = os.path.join(config.FEATURE_DIR, "train.npy")
    test_path = os.path.join(config.FEATURE_DIR, "test.npy")

    if not os.path.exists(train_path):
        feats = extract_batch(X_train, desc="train features")
        np.save(train_path, feats)

    if not os.path.exists(test_path):
        feats = extract_batch(X_test, desc="test features")
        np.save(test_path, feats)


if __name__ == "__main__":
    build()
