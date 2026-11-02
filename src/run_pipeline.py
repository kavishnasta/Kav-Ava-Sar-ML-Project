import argparse
import os
import subprocess
import sys
import time

import config
import datasets
from timing import log_time

SRC = os.path.dirname(os.path.abspath(__file__))


# Run one script as a separate step and log how long it took
def stage(name, *args):
    print(f"\n######## {name}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, os.path.join(SRC, *args[:1]), *args[1:]], check=True)
    log_time("pipeline", name, time.time() - t)


# Run gallery, evaluation and output steps in order
def main():
    ap = datasets.add_dataset_arg(argparse.ArgumentParser())
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--improve", action="store_true", help="repeat the Phase 2 validation search")
    args = ap.parse_args()
    t0 = time.time()
    names = datasets.resolve(args.dataset)
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
