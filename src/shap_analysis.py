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
y = df["label"].map({"CP": 1, "KP": 1, "FP": 0})  # Binarização

# Verificar se o modelo é binário
if hasattr(model, "n_classes_") and model.n_classes_ > 2:
    raise ValueError("O modelo carregado não é binário.")

# Criar explicador SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Gerar gráfico de colmeia
plt.figure(figsize=(8, 6))  # Aumenta o tamanho para não cortar labels
shap.plots.beeswarm(shap_values, max_display=10, show=False)

# Ajuste para não cortar labels
plt.tight_layout()

# Salvar em formato vetorial (ideal para LaTeX)
plt.savefig("shap_beeswarm.pdf", format="pdf", bbox_inches="tight")
plt.savefig("shap_beeswarm.svg", format="svg", bbox_inches="tight")
plt.savefig("shap_beeswarm.png", dpi=600, bbox_inches="tight")  # opcional, alta resolução

plt.show()
