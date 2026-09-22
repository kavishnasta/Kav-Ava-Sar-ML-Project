import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loading
import features_cache
import train_models
import evaluate
import plots


def main():
    data_loading.load_or_build()
    features_cache.build()
    train_models.train()
    evaluate.run()
    plots.generate()


if __name__ == "__main__":
    main()
