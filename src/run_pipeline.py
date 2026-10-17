"""End to end from cached data:  python src/run_pipeline.py --dataset all --mode fast

Stages: dataset cache check -> corruption galleries -> evaluation grid (SVM + ablations) -> figures/tables/summary.
The Phase 2 search (src/improve.py, ~1 h) is not re-run by default because its result is already in src/config.py;
pass --improve to repeat it.
"""
import argparse
import os
import subprocess
import sys
import time

import config
import datasets
from timing import log_time

SRC = os.path.dirname(os.path.abspath(__file__))


def stage(name, *args):
    print(f"\n######## {name}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, os.path.join(SRC, *args[:1]), *args[1:]], check=True)
    log_time("pipeline", name, time.time() - t)


def main():
    ap = datasets.add_dataset_arg(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--improve", action="store_true", help="repeat the Phase 2 validation search")
    args = ap.parse_args()
    t0 = time.time()
    names = datasets.resolve(args.dataset)          # builds caches if needed, skips datasets whose data is missing
    if not names:
        sys.exit("no dataset available")
    if args.improve:
        stage("improve", "improve.py")
    stage("gallery", "gallery.py", "--dataset", args.dataset)
    stage("evaluate_grid", "evaluate_grid.py", "--dataset", args.dataset, "--mode", args.mode)
    stage("make_outputs", "make_outputs.py", "--dataset", args.dataset)
    print(f"\npipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
