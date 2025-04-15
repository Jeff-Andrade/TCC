from xgboost import plot_tree
import joblib
import matplotlib.pyplot as plt

from src.config import MODEL_OUTPUT_DIR

FEATURE_NAMES = [
    "std_flux", "mean_flux", "min_flux", "max_flux", "ptp_flux",
    "skew_flux", "kurt_flux",
    "dom_freq", "power_ratio", "total_energy", "spectral_entropy",
    "period", "duration", "transit_depth", "symmetry_score"
]

model = joblib.load(f"../{MODEL_OUTPUT_DIR}/XGB-Simplified/planet_classifier_xgb_best.joblib")

booster = model.get_booster()
booster.feature_names = FEATURE_NAMES

plt.figure(figsize=(30, 20))
plot_tree(booster, num_trees=0, rankdir='LR')
plt.title("XGBoost Tree Visualization - Tree 0")
plt.tight_layout()
plt.show()
