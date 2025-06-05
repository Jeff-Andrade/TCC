# -*- coding: utf-8 -*-
import os
import sys
import joblib
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Garante que o console use UTF‑8 e evita UnicodeEncodeError
sys.stdout.reconfigure(encoding='utf-8')

# Caminhos
DATA_PATH   = "../data/processed/features.csv"
MODEL_DIR   = "../models/XGBoost"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ORDER_PATH  = os.path.join(MODEL_DIR, "feature_order.txt")
STUDY_PATH  = os.path.join(MODEL_DIR, "xgb_tuning_study.joblib")
MODEL_PATH  = os.path.join(MODEL_DIR, "xgb_model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# Carrega dados
df = pd.read_csv(DATA_PATH)
label_map = {"CP": "planet", "KP": "planet", "FP": "not planet"}
df = df[df["label"].isin(label_map)]
df["class"] = df["label"].map(label_map)

# Separa features e target
X = df.drop(columns=["tid", "label", "class"])
y = (df["class"] == "planet").astype(int)

# Salva a ordem das features
with open(ORDER_PATH, "w") as f:
    f.write("\n".join(X.columns.tolist()))

# Hold-out final
X_pool, X_hold, y_pool, y_hold = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalização com MinMaxScaler (compatível com a inferência)
scaler = MinMaxScaler().fit(X_pool)
X_pool = scaler.transform(X_pool)
X_hold = scaler.transform(X_hold)
joblib.dump(scaler, SCALER_PATH)

# Função objetivo para Optuna
def objective(trial):
    params = {
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators":       trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":          trial.suggest_int("max_depth", 3, 10),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-3, 10.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-3, 10.0),
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 1.0, 1.5),
        "random_state":       42,
        "eval_metric":        "logloss"
    }
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pool, y_pool, test_size=0.2, random_state=42, stratify=y_pool
    )
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_val_prob)

# Otimização com Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
joblib.dump(study, STUDY_PATH)

# Exibe melhores hiperparâmetros
best = study.best_params
print("\n=== Melhores hiperparâmetros (val ROC-AUC) ===")
for k, v in best.items():
    print(f"{k}: {v}")

# Treina modelo final com os melhores parâmetros
final_params = best.copy()
final_params.update({
    "random_state": 42,
    "eval_metric": "logloss"
})
final_model = XGBClassifier(**final_params)
final_model.fit(X_pool, y_pool)
joblib.dump(final_model, MODEL_PATH)

# Avaliação no hold-out final
y_hold_pred = final_model.predict(X_hold)
y_hold_prob = final_model.predict_proba(X_hold)[:, 1]
hold_auc = roc_auc_score(y_hold, y_hold_prob)
report = classification_report(
    y_hold, y_hold_pred,
    target_names=["not planet", "planet"]
)
conf_matrix = confusion_matrix(y_hold, y_hold_pred)

print(f"\n=== Avaliação no Hold-out Final (ROC-AUC = {hold_auc:.4f}) ===")
print(report)

# Matriz de confusão
plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=["not planet", "planet"],
    yticklabels=["not planet", "planet"]
)
plt.title("Matriz de Confusão (Hold-out)")
plt.xlabel("Previsto")
plt.ylabel("Verdadeiro")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.show()
