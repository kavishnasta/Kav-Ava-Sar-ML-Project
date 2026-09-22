import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

LABELS = {"svm": "SVM", "rf": "Random Forest"}


def plot_confusion(y_true, y_pred, name, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate():
    preds = joblib.load(os.path.join(config.RESULTS_DIR, "preds.joblib"))
    for name, (y_true, y_pred) in preds.items():
        path = os.path.join(config.PLOTS_DIR, f"confusion_{name}.pdf")
        plot_confusion(y_true, y_pred, LABELS[name], path)


if __name__ == "__main__":
    generate()
