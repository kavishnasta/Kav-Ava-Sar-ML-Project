"""
Shared configuration/constants for the GTSRB robustness pipeline.
All paths are relative to the project root (one level above src/).
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
FEATURE_CACHE_DIR = os.path.join(CACHE_DIR, "features")
CORRUPTED_CACHE_DIR = os.path.join(CACHE_DIR, "corrupted")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

for d in [RAW_DIR, CACHE_DIR, FEATURE_CACHE_DIR, CORRUPTED_CACHE_DIR,
          MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# GTSRB official download URLs (INI Rub benchmark, hosted on sid.erda.dk)
GTSRB_BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
GTSRB_TRAIN_URL = f"{GTSRB_BASE_URL}/GTSRB-Training_fixed.zip"
GTSRB_TEST_IMAGES_URL = f"{GTSRB_BASE_URL}/GTSRB_Final_Test_Images.zip"
GTSRB_TEST_GT_URL = f"{GTSRB_BASE_URL}/GTSRB_Final_Test_GT.zip"

NUM_CLASSES = 43
IMG_SIZE = 32  # resize target (32x32)

# Subsample cap per class for the TRAIN set only (keeps things laptop-feasible).
# Set to None to use the full training set.
MAX_TRAIN_PER_CLASS = 400

RANDOM_SEED = 42

# --- Corruptions ---
CORRUPTION_TYPES = [
    "gaussian_noise", "impulse_noise", "shot_noise",
    "snow", "fog", "frost",
    "motion_blur", "zoom_blur", "defocus_blur",
    "brightness", "contrast", "pixelate",
    "elastic_transform", "jpeg_compression",
]
SEVERITIES = [1, 2, 3, 4, 5]

CORRUPTION_FAMILIES = {
    "noise": ["gaussian_noise", "impulse_noise", "shot_noise"],
    "blur": ["motion_blur", "zoom_blur", "defocus_blur"],
    "weather": ["snow", "fog", "frost"],
    "digital": ["brightness", "contrast", "pixelate", "elastic_transform", "jpeg_compression"],
}

# --- Feature extraction ---
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

HSV_BINS_PER_CHANNEL = 16  # 16*3 = 48 dims

# --- Models ---
SVM_PARAMS = dict(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)
RF_PARAMS = dict(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
