# Robustness of Classical Traffic-Sign Classification under Image Corruption

This project trains a classical machine-learning classifier (hand-crafted features
and a support vector machine) on three traffic-sign datasets and measures how its
accuracy degrades when the test images are corrupted by noise, blur, weather
effects and digital artefacts. No neural network is used.

## 1. Objective

Report not only clean accuracy but also robustness: how much of the classifier's
accuracy is retained under 14 types of image corruption, each applied at 5
severity levels. The classical pipeline is deliberately transparent, so that the
contribution of each feature group and each preprocessing step can be isolated by
ablation.

## 2. Datasets

| Dataset | Classes | Test images | Split |
|---|---|---|---|
| GTSRB (German Traffic Sign Recognition Benchmark) | 43 | 12,630 | Official train / test split |
| BTSD (BelgiumTSC) | 62 | 1,425 | Official train and test pooled, then re-split 80/20 |
| CTSD (Chinese Traffic Signs, Kaggle version) | 58 | 1,204 | Re-split 80/20 from a single annotation file |

For BTSD and CTSD the split is performed on whole groups rather than single
images: a group is one physical sign track (BTSD) or one source image together
with its jittered copy (CTSD). For each class, approximately 20% of the groups are
held out as the test set. This prevents near-duplicate images from appearing in
both training and test data. The same procedure applied to the training set
provides a validation set used for tuning. Classes with a single group are used
for training only, and macro-F1 is computed over the classes present in the test
labels. Per-dataset notes are written to `results/dataset_notes_*.txt`.

## 3. Method (end-to-end pipeline)

### 3.1 Data preparation

Each image is cropped to its annotated sign bounding box (region of interest) and
resized to 32 x 32 pixels (area interpolation when shrinking, bilinear when
enlarging). The arrays are cached on disk. Cropping to the region of interest
improved validation accuracy on GTSRB by about 10.6 points compared with the full
frame.

### 3.2 Hyperparameter selection

A greedy, cumulative search is run on the validation split only; the test set is
never used. Candidate changes (crop style, training-set size, CLAHE, median
filtering, HOG cell size, resolution and orientations) are evaluated one at a
time, and a change is retained only if it raises validation accuracy by at least
0.3 points. On GTSRB, validation accuracy rose from 87.8% to 94.85%. The retained
settings are stored in `src/config.py`. The search was stopped early; the HSV,
Hu-moment and SVM `C`/`gamma` steps were not searched.

### 3.3 Preprocessing and feature extraction

Each image undergoes the following steps:

1. **Median filter (3 x 3).** Each pixel is replaced by the median of its
   neighbourhood. This removes isolated salt-and-pepper pixels while preserving
   edges.
2. **CLAHE** (contrast-limited adaptive histogram equalisation, 4 x 4 tiles, clip
   limit 2.0) on the lightness channel of the LAB colour space, which evens out
   illumination without altering colour.
3. **Histogram of oriented gradients (HOG).** Brightness gradients are computed on
   the grey image and accumulated into 9-bin orientation histograms over 4 x 4
   pixel cells. Cells are grouped into overlapping 2 x 2 blocks, normalised with
   L2-Hys. This yields 1,764 values and describes edges and symbol shape.
4. **HSV colour histogram.** 16-bin normalised histograms of hue, saturation and
   value (48 values).
5. **Hu moments.** Seven translation-, scale- and rotation-invariant shape
   moments, log-scaled (7 values).

The three groups are concatenated into a vector of 1,819 values per image.

### 3.4 Classifier

Features are standardised (zero mean, unit variance) and classified with a
support vector machine using a radial basis function kernel (`C = 10`,
`gamma` set by the scikit-learn `scale` rule). Multi-class classification uses
one-against-one voting. Training uses at most 400 images per class (13,720 images
for GTSRB). Prediction is parallelised over CPU processes.

### 3.5 Corruption suite

Fourteen corruptions, implemented from scratch with NumPy, OpenCV and SciPy, are
each applied at five severity levels with fixed, seeded parameters
(listed in `results/corruption_params.csv`):

| Family | Corruptions |
|---|---|
| Noise | Gaussian, shot (Poisson), impulse (salt-and-pepper) |
| Blur | defocus (disk kernel), motion (line kernel), zoom |
| Weather | snow, frost, fog (plasma-fractal haze) |
| Digital | brightness, contrast, pixelation, elastic transform, JPEG compression |

Each corruption is applied in pixel space and the full preprocessing and feature
extraction is then repeated on the corrupted image. The parameters were calibrated
visually on 32 x 32 images (see the galleries in `plots/`). The corruption
families follow the ImageNet-C benchmark, but the implementations are independent
and the results are not comparable with published ImageNet-C numbers. Shape,
data type and determinism are covered by the tests in `tests/`.

### 3.6 Evaluation grid

For every dataset the model is evaluated under 71 conditions: the clean test set
plus 14 corruptions at 5 severities. In the default `fast` mode the corrupted
conditions use a class-stratified subset of up to 2,000 test images; the clean
condition always uses the full test set. Reported metrics:

- **Accuracy.**
- **Macro-F1**, the unweighted mean of per-class F1 scores.
- **Retention**, accuracy under corruption divided by clean accuracy on the same
  images, which allows comparison across datasets with different clean accuracy.

### 3.7 Ablations

- **Feature-group ablation:** separate SVMs trained on HOG only, HSV only, Hu only
  and HOG + HSV, each tested under all corruptions.
- **Preprocessing ablation (GTSRB):** retraining with the median filter disabled,
  and with CLAHE disabled.

### 3.8 Outputs

Figures (PDF, in `plots/`) and a single results document (`results/report.md`)
are generated from the saved evaluation results.

## 4. Main results

SVM, severity-5 corruption, mean over the 14 corruptions:

| Dataset | Clean accuracy | Clean macro-F1 | Mean accuracy (sev. 5) | Retention (sev. 5) |
|---|---|---|---|---|
| GTSRB | 0.943 | 0.921 | 0.602 | 0.637 |
| BTSD | 0.876 | 0.655 | 0.656 | 0.749 |
| CTSD | 1.000 | 1.000 | 0.642 | 0.642 |

- Weather corruptions are the most damaging (frost: 0.160 on GTSRB, 0.251 on BTSD,
  0.134 on CTSD); digital corruptions such as contrast and brightness are the least
  damaging.
- HOG features account for almost all of the accuracy; colour and moment features
  add at most 0.017 to severity-5 accuracy.
- The median filter is the most important preprocessing step under corruption:
  removing it lowers GTSRB severity-5 accuracy from 0.602 to 0.490 while leaving
  clean accuracy unchanged (0.943).

The complete tables (per corruption and severity, per family, ablations, tuning
log, confusion pairs) are in `results/report.md`.

## 5. Limitations

- Only the SVM is evaluated by the final pipeline; no neural network or other
  classifier is compared under the same conditions.
- Corrupted-image metrics rely on at most 2,000 images per condition, with one
  random seed and no confidence intervals.
- The corruptions are synthetic approximations, not recordings of real weather or
  sensor faults.
- The clean CTSD score of 1.000 should be checked for residual train/test overlap
  before it is relied upon.
- Hyperparameter tuning was stopped before all planned steps were completed.

## 6. Setup

```bash
pip install -r requirements.txt
```

## 7. Data

Datasets are not stored in this repository.

- **GTSRB:** place the extracted training images, test images and test ground
  truth under `data/raw/` (`train_extracted`, `test_extracted`,
  `test_gt_extracted`). The archives are published by the GTSRB organisers.
- **BTSD (BelgiumTSC):** extracted under `data/raw_btsd/`.
- **CTSD:** download the Kaggle dataset "Chinese Traffic Signs" and unzip it so
  that `data/raw_ctsd/annotations.csv` and `data/raw_ctsd/images/` exist.
  `src/datasets.py` prints these instructions if the data is missing and skips any
  dataset that is not available.

## 8. Usage

```bash
# Full pipeline: galleries, evaluation grid, ablations, figures
python src/run_pipeline.py --dataset all --mode fast

# Evaluate on the full test set instead of the 2,000-image subset
python src/run_pipeline.py --dataset all --mode full

# Repeat the validation-only hyperparameter search (about one hour)
python src/run_pipeline.py --improve

# Progress of a running pipeline
python src/status.py

# Tests for the corruption suite
pytest tests
```

A single dataset can be selected with `--dataset gtsrb`, `btsd` or `ctsd`. The
evaluation resumes from previously saved results. On the machine used, a full
`--mode fast` run over all three datasets took about 111 minutes.

## 9. Repository structure

| File | Purpose |
|---|---|
| `src/config.py` | Paths, seed, and the tuned feature and SVM settings |
| `src/datasets.py` | Loading, ROI cropping, resizing, group-based splits for the three datasets |
| `src/features.py` | Preprocessing and HOG / HSV / Hu feature extraction |
| `src/models.py` | Standardisation + RBF SVM, parallel prediction, accuracy and macro-F1 |
| `src/corruptions.py` | The 14 corruptions at 5 severities |
| `src/improve.py` | Validation-only hyperparameter search |
| `src/evaluate_grid.py` | Training, 71-condition evaluation grid and ablations |
| `src/gallery.py` | Corruption example galleries |
| `src/make_outputs.py` | Figures and results report |
| `src/run_pipeline.py` | End-to-end driver |
| `src/status.py` | Progress display for a running pipeline |
| `src/benchmark.py`, `src/timing.py` | Timing measurements |
| `src/data_loading.py`, `src/run.py` | Earlier GTSRB-only loader and script, retained for reference |
| `tests/test_corruptions.py` | Unit tests for the corruption suite |

## 10. Outputs

- `plots/`: corruption galleries, accuracy heatmaps, accuracy-versus-severity
  curves, confusion matrices and ablation charts (PDF).
- `results/report.md`: all accuracy and macro-F1 metrics.
- `results/*.csv`, `results/*.npz`, `results/*.npy`, `results/*.log` and
  `models/*.joblib` are intermediate working files and are excluded from version
  control.
