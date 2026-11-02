import csv
import os
import time
from contextlib import contextmanager

import config

PATH = os.path.join(config.RESULTS_DIR, "timing.csv")


# Add one timing row to the csv file
def log_time(phase, item, seconds):
    new = not os.path.exists(PATH)
    with open(PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["phase", "item", "seconds", "timestamp"])
        w.writerow([phase, item, f"{seconds:.3f}", time.strftime("%Y-%m-%d %H:%M:%S")])


# Measure how long the code inside a with block takes and log it
@contextmanager
def timed(phase, item):
    t = time.time()
    yield
    log_time(phase, item, time.time() - t)
