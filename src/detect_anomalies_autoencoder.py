import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Caminhos
DATA_DIR = "../data/processed"
MODEL_PATH = "../models/Autoencoder_FA/autoencoder_fa.keras"

# Carregar rótulos e filtrar
label_csv = os.path.join(DATA_DIR, "tic_labels.csv")
df = pd.read_csv(label_csv)
df = df[df["tfopwg_disp"].isin(["CP", "FP", "KP"])]

# Carregar curvas e metadados
X, tids, labels = [], [], []
n_points = 256
for _, row in df.iterrows():
    tid = int(row["tid"])
    lab = row["tfopwg_disp"]
    path = os.path.join(DATA_DIR, lab, f"TIC_{tid}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        if len(arr) == n_points:
            X.append(arr)
            tids.append(tid)
            labels.append(lab)
X = np.array(X)[..., np.newaxis]

# Padronizar com scaler
scaler = StandardScaler()
X_flat = X.reshape((X.shape[0], -1))
X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

# Carregar autoencoder e prever
ae = load_model(MODEL_PATH)
X_pred = ae.predict(X_scaled, verbose=0)

# Calcular MSE de reconstrução por amostra
de_errors = np.mean((X_scaled - X_pred) ** 2, axis=(1, 2))

# Calcular threshold dinamicamente (95º percentil dos FA)
fa_dir = os.path.join(DATA_DIR, "FA")
X_fa = []
for f in os.listdir(fa_dir):
    if f.endswith(".npy"):
        arr = np.load(os.path.join(fa_dir, f))
        if len(arr) == n_points:
            X_fa.append(arr)
X_fa = np.array(X_fa)[..., np.newaxis]
X_fa_flat = X_fa.reshape((X_fa.shape[0], -1))
X_fa_scaled = scaler.transform(X_fa_flat).reshape(X_fa.shape)
X_fa_pred = ae.predict(X_fa_scaled, verbose=0)
fa_mse = np.mean((X_fa_scaled - X_fa_pred) ** 2, axis=(1, 2))
threshold = np.percentile(fa_mse, 95)
print(f"Threshold (95% FA): {threshold:.6f}")

# Classificar is_anomaly: True se reconstruction_error >= threshold
is_anomaly = de_errors >= threshold

# DataFrame de resultados
df_res = pd.DataFrame({
    "tid": tids,
    "label": labels,
    "reconstruction_error": de_errors,
    "is_anomaly": is_anomaly
})

# Estatísticas textuais
print("\nDistribuição por label e is_anomaly:")
print(df_res.groupby(["label", "is_anomaly"]).size())
print("\nTotal por label com padrão FA (is_anomaly=False):")
print(df_res[df_res["is_anomaly"] == False]["label"].value_counts())

# Visualizações
# 1) Histograma de erros
plt.figure(figsize=(10, 6))
for lab in ["CP", "FP", "KP"]:
    errs = df_res[df_res["label"] == lab]["reconstruction_error"]
    plt.hist(errs, bins=50, alpha=0.6, label=lab)
plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.6f}")
plt.title("Erro de Reconstrução por Label")
plt.xlabel("MSE")
plt.ylabel("Frequência")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Bar plot contagens
counts = df_res.groupby(["label", "is_anomaly"]).size().unstack(fill_value=0)
counts.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Contagem por Label e is_anomaly")
plt.xlabel("Label")
plt.ylabel("Número de Amostras")
plt.legend(title="is_anomaly")
plt.tight_layout()
plt.show()


# 3) Exemplos de curvas
def plot_examples(lab, status, ax):
    # status: True->anomaly, False->FA-like
    subset = df_res[(df_res["label"] == lab) & (df_res["is_anomaly"] == status)]
    if subset.empty:
        return
    tid_sample = subset.sample(n=3, random_state=42)["tid"].tolist()
    for idx, tid in enumerate(tid_sample):
        arr = np.load(os.path.join(DATA_DIR, lab, f"TIC_{tid}.npy"))
        ax[idx].plot(np.linspace(0, 1, n_points), arr, color=("crimson" if status else "steelblue"))
        ax[idx].set_title(f"{lab} {'Anômala' if status else 'FA-like'} TIC {tid}", fontsize=8)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])


fig, axes = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle("Exemplos: Linha 1 = Anômala; Linha 2 = FA-like")
for j, lab in enumerate(["CP", "FP", "KP"]):
    plot_examples(lab, True, axes[0])
    plot_examples(lab, False, axes[1])
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
