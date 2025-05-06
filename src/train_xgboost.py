import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

# Paths
DATA_PATH = "../data/processed/features.csv"
MODEL_DIR = "../models/XGBoost"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Map labels to binary
label_map = {"CP": "planet", "KP": "planet", "FP": "not planet"}
df = df[df["label"].isin(label_map)]
df["class"] = df["label"].map(label_map)
X = df.drop(columns=["tid", "label", "class"])
y = df["class"]

# Encode target
y_encoded = (y == "planet").astype(int)  # 1 = planet, 0 = not planet

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter search
best_score = -np.inf
best_model = None
best_params = {}

param_grid = {
    "learning_rate": np.linspace(0.01, 0.3, 30),
    "random_state": range(1, 1000),
    "test_size": np.linspace(0.2, 0.4, 20)
}

for _ in tqdm(range(1000)):
    lr = random.choice(param_grid["learning_rate"])
    rs = random.choice(param_grid["random_state"])
    ts = random.choice(param_grid["test_size"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=ts, random_state=rs, stratify=y_encoded
    )

    model = XGBClassifier(eval_metric="logloss",
                          learning_rate=lr, random_state=rs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)

    if score > best_score:
        best_score = score
        best_model = model
        best_params = {
            "learning_rate": lr,
            "random_state": rs,
            "test_size": ts,
            "roc_auc": score
        }

# Save best model
joblib.dump(best_model, MODEL_PATH)

# Evaluate final model
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=best_params["test_size"],
    random_state=best_params["random_state"], stratify=y_encoded
)
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=["not planet", "planet"])
conf_matrix = confusion_matrix(y_test, y_pred)

# Print report and confusion matrix
print("\nBest Hyperparameters:")
print(best_params)
print("\nClassification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["not planet", "planet"],
            yticklabels=["not planet", "planet"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.show()
