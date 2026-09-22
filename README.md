# GTSRB Traffic Sign Classification

Classifies GTSRB traffic signs (43 classes) using HOG + HSV histogram + Hu
moments features with SVM and Random Forest.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Not stored in this repo. Downloads automatically on first run from the
official GTSRB host (`sid.erda.dk`).

## Run

```bash
python src/data_loading.py
python src/features_cache.py
python src/train_models.py
python src/evaluate.py
python src/plots.py

# or all at once:
python src/run_pipeline.py
```

## Structure

| File | Purpose |
|---|---|
| `src/config.py` | paths, params |
| `src/data_loading.py` | download + load GTSRB |
| `src/feature_extraction.py` | HOG + HSV + Hu features |
| `src/features_cache.py` | extract + cache features for train/test |
| `src/train_models.py` | train SVM + RF |
| `src/evaluate.py` | run models, write results.csv |
| `src/plots.py` | confusion matrix plots |
| `src/run_pipeline.py` | runs everything above |

## Outputs

- `results/results.csv`
- `models/{svm,rf,scaler}.joblib`
- `plots/confusion_svm.pdf`, `plots/confusion_rf.pdf`
