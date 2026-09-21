# GTSRB Traffic Sign Classification — Robustness Under Image Corruptions

Laptop-feasible, non-deep-learning pipeline that classifies GTSRB traffic signs
(43 classes) using classical CV features (HOG + HSV color histogram + Hu
moments) and two classical classifiers (SVM-RBF, Random Forest), then
evaluates robustness across 14 corruption types x 5 severities (70
conditions) using the `imagecorruptions` package.

## Setup

```bash
pip install -r requirements.txt
```

## Getting the data

Raw and cached data (`data/`) and trained model binaries (`models/*.joblib`)
are **not** stored in this repo — they're large and fully reproducible from
public sources. Everything downloads automatically the first time you run
the pipeline:

```bash
python src/data_loading.py       # downloads official GTSRB (~276MB train zip, ~187MB test zip)
python src/data_loading_btsd.py  # optional: downloads BTSD (Belgian) raw data (~174MB train zip, ~78MB test zip)
```

Sources used:
- **GTSRB**: official [INI Rub benchmark](https://benchmark.ini.rub.de/gtsrb_dataset.html) host (`sid.erda.dk`)
- **BTSD**: official [ETH Zurich BelgiumTSC](https://btsd.ethz.ch/shareddata/BelgiumTSC/) host

Both scripts download, extract, resize to 32x32, and cache to `data/cache/`
(GTSRB) or `data/cache_btsd/` (BTSD) as `.npy` files. Every step is
idempotent — safe to re-run, it resumes/skips what's already done. After
that, run the rest of the pipeline below to regenerate corrupted test sets,
trained models, results, and plots locally.

## Pipeline

Each script is idempotent — it checks its own cache/output before
recomputing, so re-running after an interruption resumes rather than
restarts.

```bash
python src/data_loading.py       # 1. download + extract + load + resize GTSRB, cache to data/cache/*.npy
python src/feature_extraction.py # (sanity check only — prints feature dim)
python src/corrupt_and_cache.py  # 2. generate 70 corrupted test sets + extract all features, cache to disk
python src/train_models.py       # 3. train SVM + RF once on clean train features, save to models/
python src/evaluate.py           # 4. evaluate baseline + 70-condition grid, write results/results.csv
python src/plots.py              # 5. generate all PDF plots into plots/

# Or run all of the above in order:
python src/run_pipeline.py
```

## Structure

| File | Purpose |
|---|---|
| `src/config.py` | Shared paths, corruption list, model hyperparameters |
| `src/data_loading.py` | Downloads official GTSRB (train + test splits), resizes to 32x32, subsamples train to `MAX_TRAIN_PER_CLASS` per class |
| `src/feature_extraction.py` | `extract_features(img_bgr)` — HOG (324-d) + HSV histogram (48-d) + Hu moments (7-d) = 379-d vector |
| `src/corrupt_and_cache.py` | Applies all 14 corruptions x 5 severities to the test set only (via `imagecorruptions`), extracts + caches features for every condition |
| `src/train_models.py` | Trains `SVC(kernel='rbf', C=10, gamma='scale')` and `RandomForestClassifier(n_estimators=200)` once on clean train features; saves via `joblib` |
| `src/evaluate.py` | Runs both saved models over baseline + all 70 conditions (no retraining), writes `results/results.csv` (accuracy + macro F1 per model x condition) and caches predictions needed for confusion matrices |
| `src/plots.py` | All PDF plots (confusion matrices, accuracy-vs-severity faceted by corruption family, severity-5 SVM-vs-RF bar chart) |
| `src/run_pipeline.py` | Runs all of the above in order |

## Data

- **Dataset**: official [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) train/test split, downloaded automatically from the official host on first run.
- **Train set**: subsampled to `MAX_TRAIN_PER_CLASS = 400` images/class (config.py) for speed; set to `None` for the full ~39k-image training set.
- **Test set**: full official test set (~12.6k images) used for every corruption condition.
- All images resized to 32x32 (BGR, OpenCV convention internally).

## Corruptions

14 types x 5 severities from `imagecorruptions`, applied to the **test set
only**: gaussian_noise, impulse_noise, shot_noise, snow, fog, frost,
motion_blur, zoom_blur, defocus_blur, brightness, contrast, pixelate,
elastic_transform, jpeg_compression.

## Outputs

- `results/results.csv` — accuracy + macro F1 for both models across baseline, train-clean, and all 70 conditions.
- `results/confusion_preds.joblib` — cached predictions for confusion-matrix plots.
- `models/{svm,rf,scaler}.joblib` — trained models + feature scaler.
- `plots/*.pdf` — all required plots.
- `data/cache/` — cached resized images, corrupted images, and extracted feature vectors (safe to delete to force full regeneration; downloads/extraction under `data/raw/` are kept separately).
