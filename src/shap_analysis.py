# shap_analysis.py

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Caminhos
MODEL_PATH = "../models/XGBoost/xgb_model.joblib"
FEATURES_PATH = "../data/processed/features.csv"

# Carregar modelo treinado
model = joblib.load(MODEL_PATH)

# Carregar dados
df = pd.read_csv(FEATURES_PATH)

# Separar features e labels
X = df.drop(columns=["tid", "label"])
y = df["label"].map({"CP": 1, "KP": 1, "FP": 0})  # Garante binarização

# Verifica se o modelo está de fato binário
if hasattr(model, "n_classes_") and model.n_classes_ > 2:
    raise ValueError("O modelo carregado não é binário.")

# Criar explicador SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Gráfico de colmeia
shap.plots.beeswarm(shap_values, max_display=45)
plt.tight_layout()
plt.show()
