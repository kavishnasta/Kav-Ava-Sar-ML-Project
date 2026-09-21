"""
run_pipeline.py

Convenience orchestrator: runs the full pipeline end-to-end in order.
    1. data_loading   -> data/cache/{X,y}_{train,test}.npy
    2. corrupt_and_cache -> data/cache/corrupted/*.npy, data/cache/features/*.npy
    3. train_models   -> models/{svm,rf,scaler}.joblib
    4. evaluate       -> results/results.csv, results/confusion_preds.joblib
    5. plots          -> plots/*.pdf

Each step is idempotent (checks its own cache/output before recomputing), so
re-running this script after a partial run resumes rather than restarts.

Run:
    python src/run_pipeline.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loading
import corrupt_and_cache
import train_models
import evaluate
import plots


def main():
    t0 = time.time()

    print("\n=== [1/5] data_loading ===")
    data_loading.load_or_build_splits()

    print("\n=== [2/5] corrupt_and_cache ===")
    corrupt_and_cache.build_all_caches()

    print("\n=== [3/5] train_models ===")
    train_models.train_and_save()

    print("\n=== [4/5] evaluate ===")
    evaluate.run_evaluation()

    print("\n=== [5/5] plots ===")
    plots.generate_all_plots()

    print(f"\nPipeline complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
