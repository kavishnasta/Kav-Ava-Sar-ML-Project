import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(ROOT, "plots")
for _d in (CACHE_DIR, MODELS_DIR, RESULTS_DIR, TABLES_DIR, PLOTS_DIR):
    os.makedirs(_d, exist_ok=True)

DATASETS = ["gtsrb", "btsd", "ctsd"]
IMG_SIZE = 32
SEED = 42
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# --- model/feature config (final values written after Phase 2, see results/improvement_log.csv) ---
CROP = "roi"                 # "roi" | "full"
MAX_TRAIN_PER_CLASS = 400    # None = all
FEATURE_CFG = dict(median=True, clahe=False, size=32,
                   hog_orient=9, hog_cells=(8,), hsv_bins=16, hsv_grid=1, hsv_ch="hsv", hu=True)
SVM_PARAMS = dict(kernel="rbf", C=10, gamma="scale", random_state=SEED)
RF_PARAMS = dict(n_estimators=200, random_state=SEED)

# one colour per model, used by every figure
MODEL_COLORS = {"SVM": "#1f77b4", "RF": "#2ca02c", "CNN": "#d62728"}
