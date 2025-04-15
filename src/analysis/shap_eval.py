import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.config import MODEL_OUTPUT_DIR, FEATURES_FILE

FEATURE_NAMES = [
    'std_flux', 'mean_flux', 'min_flux', 'max_flux', 'ptp_flux',
    'skew_flux', 'kurt_flux',
    'dom_freq', 'power_ratio', 'total_energy', 'spectral_entropy',
    'period', 'duration', 'transit_depth', 'symmetry_score'
]

def run_shap_analysis():
    print("[SHAP] Loading model and data...")
    model_path = f"../{MODEL_OUTPUT_DIR}/XGB-01/planet_classifier_xgb_best.joblib"
    model = joblib.load(model_path)

    data = np.load(f"../{FEATURES_FILE}")
    X = data['X']
    y = data['y']

    print("[SHAP] Computing SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    print("[SHAP] Generating SHAP beeswarm plot...")
    plt.title("SHAP Summary Plot (Beeswarm)")
    shap.summary_plot(shap_values, features=X, feature_names=FEATURE_NAMES, plot_type="dot")

if __name__ == "__main__":
    run_shap_analysis()
