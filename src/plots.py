"""
plots.py

Generates all required PDF plots from results/results.csv and
results/confusion_preds.joblib. All plots saved under plots/ as PDF.

Color choices follow the validated categorical palette (8 CVD-safe hues,
fixed order) from the dataviz skill's reference palette
(references/palette.md), reused across facets since each facet holds <=5
series (within the safe adjacent-pair count). Linestyle is used as a
secondary encoding alongside color for the line charts.

Plots produced:
    1. confusion_matrix_baseline_{svm,rf}.pdf
    2. confusion_matrix_worstcase_{corruption}_{svm,rf}.pdf  (x2 corruptions)
    3. accuracy_vs_severity_{svm,rf}.pdf  (faceted by family, one subplot each)
    4. severity5_bar_comparison.pdf  (SVM vs RF grouped bars, all 14 corruptions)

Run standalone:
    python src/plots.py
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from evaluate import WORST_CASE_CORRUPTIONS_FOR_CM

# --- Validated categorical palette (light mode), fixed order (see dataviz skill) ---
PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e0e0dd"

MODEL_LABELS = {"svm": "SVM (RBF)", "rf": "Random Forest"}
MODEL_COLORS = {"svm": PALETTE[0], "rf": PALETTE[1]}

plt.rcParams.update({
    "font.size": 10,
    "text.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _savefig(fig, path):
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved {path}")


def plot_confusion_matrices():
    preds = joblib.load(os.path.join(config.RESULTS_DIR, "confusion_preds.joblib"))

    # 1. Clean baseline, both models
    for model_name in ["svm", "rf"]:
        y_true, y_pred = preds[("baseline", model_name)]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_title(f"Confusion Matrix — Clean Baseline — {MODEL_LABELS[model_name]}")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(range(0, config.NUM_CLASSES, 5))
        ax.set_yticks(range(0, config.NUM_CLASSES, 5))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
        path = os.path.join(config.PLOTS_DIR, f"confusion_matrix_baseline_{model_name}.pdf")
        _savefig(fig, path)

    # 2. Worst-case severity-5 conditions, both models
    for corruption in WORST_CASE_CORRUPTIONS_FOR_CM:
        condition = f"{corruption}_s5"
        for model_name in ["svm", "rf"]:
            key = (condition, model_name)
            if key not in preds:
                continue
            y_true, y_pred = preds[key]
            cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
            fig, ax = plt.subplots(figsize=(9, 8))
            im = ax.imshow(cm, cmap="Oranges", aspect="equal")
            ax.set_title(f"Confusion Matrix — {corruption} (severity 5) — {MODEL_LABELS[model_name]}")
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_xticks(range(0, config.NUM_CLASSES, 5))
            ax.set_yticks(range(0, config.NUM_CLASSES, 5))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
            path = os.path.join(config.PLOTS_DIR, f"confusion_matrix_worstcase_{corruption}_{model_name}.pdf")
            _savefig(fig, path)


def plot_accuracy_vs_severity(df: pd.DataFrame):
    families = config.CORRUPTION_FAMILIES
    for model_name in ["svm", "rf"]:
        sub = df[(df["model"] == model_name) & (df["split"] == "test") & (df["condition"] != "baseline")]
        baseline_acc = df[(df["model"] == model_name) & (df["condition"] == "baseline")]["accuracy"].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
        axes = axes.flatten()

        for fam_idx, (family_name, corruptions) in enumerate(families.items()):
            ax = axes[fam_idx]
            for i, corruption in enumerate(corruptions):
                cdata = sub[sub["corruption"] == corruption].sort_values("severity")
                ax.plot(
                    cdata["severity"], cdata["accuracy"],
                    marker="o", markersize=5, linewidth=2,
                    color=PALETTE[i % len(PALETTE)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    label=corruption,
                )
            ax.axhline(baseline_acc, color=TEXT_SECONDARY, linewidth=1, linestyle=":", alpha=0.7)
            ax.set_title(family_name.capitalize())
            ax.set_xlabel("Severity")
            ax.set_ylabel("Accuracy")
            ax.set_xticks(config.SEVERITIES)
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=8, loc="lower left")

        fig.suptitle(f"Accuracy vs. Corruption Severity — {MODEL_LABELS[model_name]}\n(dotted line = clean baseline accuracy)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = os.path.join(config.PLOTS_DIR, f"accuracy_vs_severity_{model_name}.pdf")
        _savefig(fig, path)


def plot_severity5_bar_comparison(df: pd.DataFrame):
    sub = df[(df["severity"] == 5) & (df["split"] == "test")]
    corruptions = config.CORRUPTION_TYPES

    svm_acc = [sub[(sub["model"] == "svm") & (sub["corruption"] == c)]["accuracy"].iloc[0] for c in corruptions]
    rf_acc = [sub[(sub["model"] == "rf") & (sub["corruption"] == c)]["accuracy"].iloc[0] for c in corruptions]

    x = np.arange(len(corruptions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, svm_acc, width, label=MODEL_LABELS["svm"], color=MODEL_COLORS["svm"])
    ax.bar(x + width / 2, rf_acc, width, label=MODEL_LABELS["rf"], color=MODEL_COLORS["rf"])

    ax.set_xticks(x)
    ax.set_xticklabels(corruptions, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("SVM vs. Random Forest Accuracy at Severity 5 — All Corruption Types")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "severity5_bar_comparison.pdf")
    _savefig(fig, path)


def generate_all_plots():
    results_path = os.path.join(config.RESULTS_DIR, "results.csv")
    df = pd.read_csv(results_path)

    plot_confusion_matrices()
    plot_accuracy_vs_severity(df)
    plot_severity5_bar_comparison(df)

    print("[plots] All plots generated.")


if __name__ == "__main__":
    generate_all_plots()
