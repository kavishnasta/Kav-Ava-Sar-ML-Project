import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
FEATURE_DIR = os.path.join(CACHE_DIR, "features")

MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")

for d in [RAW_DIR, CACHE_DIR, FEATURE_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
TRAIN_URL = f"{BASE_URL}/GTSRB-Training_fixed.zip"
TEST_IMG_URL = f"{BASE_URL}/GTSRB_Final_Test_Images.zip"
TEST_GT_URL = f"{BASE_URL}/GTSRB_Final_Test_GT.zip"

NUM_CLASSES = 43
IMG_SIZE = 32
MAX_TRAIN_PER_CLASS = 400
SEED = 42

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HSV_BINS = 16

SVM_PARAMS = dict(kernel="rbf", C=10, gamma="scale", random_state=SEED)
RF_PARAMS = dict(n_estimators=200, random_state=SEED, n_jobs=-1)
