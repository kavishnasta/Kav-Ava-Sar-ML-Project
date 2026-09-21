"""
evaluate.py

Loads the trained SVM + RF models (and scaler) and runs them over:
    - baseline (clean test set)
    - all 14 corruptions x 5 severities (70 conditions)
plus reports clean TRAIN accuracy/F1 for both models (train/test gap).

Writes results/results.csv with columns:
    model, condition, corruption, severity, accuracy, macro_f1, n_samples, split

`condition` is "baseline" for the clean test set, "train_clean" for the
training-set row, or "{corruption}_s{severity}" otherwise.

Also saves predictions for two special conditions needed by plots.py's
confusion matrices: clean baseline and a chosen worst-case severity-5
condition, to results/ as .npy (y_true, y_pred per model).

Run standalone:
    python src/evaluate.py
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_loading import load_or_build_splits
from corrupt_and_cache import _feature_cache_path

# Representative "worst-case" severity-5 corruptions used for confusion-matrix plots.
WORST_CASE_CORRUPTIONS_FOR_CM = ["gaussian_noise", "fog"]


def _load_models():
    scaler = joblib.load(os.path.join(config.MODELS_DIR, "scaler.joblib"))
    svm = joblib.load(os.path.join(config.MODELS_DIR, "svm.joblib"))
    rf = joblib.load(os.path.join(config.MODELS_DIR, "rf.joblib"))
    return svm, rf, scaler


def _eval_one(model, model_name, X, y, X_scaled_for_svm=None):
    X_use = X_scaled_for_svm if model_name == "svm" else X
    y_pred = model.predict(X_use)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    return acc, f1, y_pred


def run_evaluation():
    _X_train_img, y_train, _X_test_img, y_test = load_or_build_splits()
    svm, rf, scaler = _load_models()

    X_train = np.load(_feature_cache_path("train_clean"))
    X_test_clean = np.load(_feature_cache_path("test_clean"))

    rows = []
    saved_preds = {}  # (condition, model) -> (y_true, y_pred), for confusion matrices

    models = [("svm", svm), ("rf", rf)]

    total_conditions = 2 + len(config.CORRUPTION_TYPES) * len(config.SEVERITIES)  # train_clean + baseline + grid
    done_conditions = 0

    # --- Train-set clean performance (train/test gap) ---
    X_train_scaled = scaler.transform(X_train)
    for model_name, model in models:
        acc, f1, _ = _eval_one(model, model_name, X_train, y_train, X_train_scaled)
        rows.append(dict(
            model=model_name, condition="train_clean", corruption="none",
            severity=0, accuracy=acc, macro_f1=f1, n_samples=len(y_train), split="train",
        ))
        print(f"[evaluate] TRAIN clean | {model_name}: acc={acc:.4f} f1={f1:.4f}", flush=True)
    done_conditions += 1
    print(f"[evaluate] progress: {done_conditions}/{total_conditions} conditions", flush=True)

    # --- Baseline (clean test) ---
    X_test_scaled = scaler.transform(X_test_clean)
    for model_name, model in models:
        acc, f1, y_pred = _eval_one(model, model_name, X_test_clean, y_test, X_test_scaled)
        rows.append(dict(
            model=model_name, condition="baseline", corruption="none",
            severity=0, accuracy=acc, macro_f1=f1, n_samples=len(y_test), split="test",
        ))
        saved_preds[("baseline", model_name)] = (y_test, y_pred)
        print(f"[evaluate] BASELINE clean | {model_name}: acc={acc:.4f} f1={f1:.4f}", flush=True)
    done_conditions += 1
    print(f"[evaluate] progress: {done_conditions}/{total_conditions} conditions", flush=True)

    # --- Full 70-condition grid ---
    for corruption in config.CORRUPTION_TYPES:
        for severity in config.SEVERITIES:
            condition = f"{corruption}_s{severity}"
            X_cond = np.load(_feature_cache_path(condition))
            X_cond_scaled = scaler.transform(X_cond)

            for model_name, model in models:
                acc, f1, y_pred = _eval_one(model, model_name, X_cond, y_test, X_cond_scaled)
                rows.append(dict(
                    model=model_name, condition=condition, corruption=corruption,
                    severity=severity, accuracy=acc, macro_f1=f1,
                    n_samples=len(y_test), split="test",
                ))
                if severity == 5 and corruption in WORST_CASE_CORRUPTIONS_FOR_CM:
                    saved_preds[(condition, model_name)] = (y_test, y_pred)

            done_conditions += 1
            print(f"[evaluate] {condition} | svm/rf done "
                  f"({done_conditions}/{total_conditions}, "
                  f"{100*done_conditions/total_conditions:.0f}%)", flush=True)

    df = pd.DataFrame(rows)
    results_path = os.path.join(config.RESULTS_DIR, "results.csv")
    df.to_csv(results_path, index=False)
    print(f"[evaluate] Wrote {results_path} ({len(df)} rows)")

    preds_path = os.path.join(config.RESULTS_DIR, "confusion_preds.joblib")
    joblib.dump(saved_preds, preds_path)
    print(f"[evaluate] Wrote {preds_path} ({len(saved_preds)} prediction sets)")

    return df


if __name__ == "__main__":
    run_evaluation()
