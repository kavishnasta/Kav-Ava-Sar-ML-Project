"""Corruption gallery: rows = 14 corruptions, columns = clean + severity 1..5, one sample sign per dataset."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import corruptions
import datasets


def make_gallery(dataset, path=None, sample=None):
    _, _, _, X, y, _ = datasets.load(dataset)
    if sample is None:                               # a sharp, reasonably large-contrast sample: highest Laplacian variance among a few
        import cv2
        cand = np.random.RandomState(0).choice(len(X), 60, replace=False)
        sample = cand[int(np.argmax([cv2.Laplacian(cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() for i in cand]))]
    img = X[sample]
    n = len(corruptions.NAMES)
    fig, axes = plt.subplots(n, 6, figsize=(7.4, n * 1.2 + 0.6))
    for r, name in enumerate(corruptions.NAMES):
        for c in range(6):
            out = corruptions.apply(img, name, c, np.random.default_rng(corruptions.seed_for(dataset, name, c)))
            ax = axes[r, c]
            ax.imshow(out[..., ::-1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title("clean" if c == 0 else f"sev {c}", fontsize=11)
            if c == 0:
                ax.set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center", fontsize=10, labelpad=6)
    fig.suptitle(f"Corruption gallery ({dataset.upper()}, 32x32 test sample)", fontsize=12, y=0.998)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.985))
    path = path or os.path.join(config.PLOTS_DIR, f"fig_corruption_gallery_{dataset}.pdf")
    fig.savefig(path, format=path.rsplit(".", 1)[-1], bbox_inches="tight", dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = datasets.add_dataset_arg(argparse.ArgumentParser())
    args = ap.parse_args()
    corruptions.write_params_csv(os.path.join(config.RESULTS_DIR, "corruption_params.csv"))
    for ds in datasets.resolve(args.dataset):
        print(make_gallery(ds))
