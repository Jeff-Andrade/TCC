import os
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join("..", "data")
POS_CSV = os.path.join(DATA_DIR, "positive_data.csv")
NEG_CSV = os.path.join(DATA_DIR, "negative_data.csv")
POS_DIR = os.path.join(DATA_DIR, "positive_data")
NEG_DIR = os.path.join(DATA_DIR, "negative_data")
FEATURES_FILE = "../cache/cached_features.npz"
MODEL_OUTPUT_DIR = os.path.join("..", "models")
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "planet_classifier_xgb_best.joblib")
TUNING_RESULTS_PATH = os.path.join(MODEL_OUTPUT_DIR, "tuning_results.csv")
SEQUENCE_LENGTH = 1000