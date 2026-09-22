import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loading import load_or_build


def run():
    _, y_train, _, y_test = load_or_build()

    scaler = joblib.load(os.path.join(config.MODELS_DIR, "scaler.joblib"))
    svm = joblib.load(os.path.join(config.MODELS_DIR, "svm.joblib"))
    rf = joblib.load(os.path.join(config.MODELS_DIR, "rf.joblib"))

    X_train = np.load(os.path.join(config.FEATURE_DIR, "train.npy"))
    X_test = np.load(os.path.join(config.FEATURE_DIR, "test.npy"))
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rows = []
    preds = {}

    for name, model, X, Xs, y, split in [
        ("svm", svm, X_train, X_train_scaled, y_train, "train"),
        ("rf", rf, X_train, X_train_scaled, y_train, "train"),
        ("svm", svm, X_test, X_test_scaled, y_test, "test"),
        ("rf", rf, X_test, X_test_scaled, y_test, "test"),
    ]:
        pred = model.predict(Xs if name == "svm" else X)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro")
        rows.append(dict(model=name, split=split, accuracy=acc, macro_f1=f1, n_samples=len(y)))
        if split == "test":
            preds[name] = (y, pred)
        print(name, split, "acc", round(acc, 4), "f1", round(f1, 4))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "results.csv"), index=False)
    joblib.dump(preds, os.path.join(config.RESULTS_DIR, "preds.joblib"))
    return df


if __name__ == "__main__":
    run()
