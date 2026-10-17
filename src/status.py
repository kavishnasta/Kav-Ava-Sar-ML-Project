"""Progress checkpoints: python src/status.py  (reads results/ only, safe to run while the pipeline runs)."""
import os

import pandas as pd

import config

NCOND = 71
print(f"{'dataset':8} {'train':>6} {'baseline':>9} {'grid':>10} {'feat-abl':>9} {'preproc-abl':>12} {'plots':>6}")
for ds in config.DATASETS:
    R = lambda n: os.path.join(config.RESULTS_DIR, f"{n}_{ds}.csv")
    n = lambda p, tot: f"{len(pd.read_csv(p))}/{tot}" if os.path.exists(p) else "-"
    train = "done" if os.path.exists(os.path.join(config.MODELS_DIR, f"svm_featgroups_{ds}.joblib")) else "-"
    plots = "done" if os.path.exists(os.path.join(config.PLOTS_DIR, f"fig_heatmap_{ds}.pdf")) else "-"
    print(f"{ds:8} {train:>6} {'done' if os.path.exists(R('baseline')) else '-':>9} {n(R('grid'), NCOND):>10} "
          f"{n(R('ablation_features'), 116):>9} "
          f"{(n(R('ablation_preproc'), 129) if ds == 'gtsrb' else 'n/a'):>12} {plots:>6}")
