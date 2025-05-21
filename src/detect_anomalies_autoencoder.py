# autoencoders_by_class.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model

# ── Configurações ────────────────────────────────────────────────────────────
DATA_DIR   = "../data/processed"
LABEL_CSV  = os.path.join(DATA_DIR, "tic_labels.csv")
OUT_DIR    = "../models/Autoencoders_by_class"
N_POINTS   = 256
BATCH_SIZE = 32
EPOCHS     = 50
VAL_SPLIT  = 0.1

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1) Carrega dados por classe ──────────────────────────────────────────────
df = pd.read_csv(LABEL_CSV)
classes = ["CP", "FP", "KP", "FA"]
data_by_class = {c: [] for c in classes}

for _, row in df.iterrows():
    lbl = row["tfopwg_disp"]
    if lbl in classes:
        path = os.path.join(DATA_DIR, lbl, f"TIC_{int(row['tid'])}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            if arr.shape[0] == N_POINTS:
                data_by_class[lbl].append(arr)

# ── 2) Função para criar o autoencoder ───────────────────────────────────────
def build_autoencoder(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(16, 3, activation="relu", padding="same")(x)
    code = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(16, 3, activation="relu", padding="same")(code)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(1, 3, activation="linear", padding="same")(x)
    ae = models.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    return ae

# ── 3) Treina um autoencoder por classe ─────────────────────────────────────
histories = {}
for cls in classes:
    arrs = np.stack(data_by_class[cls], axis=0)[..., np.newaxis]
    flat = arrs.reshape((arrs.shape[0], -1))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(arrs.shape)

    ae = build_autoencoder(input_shape=(N_POINTS, 1))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    h = ae.fit(
        scaled, scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[es],
        verbose=2
    )

    save_dir = os.path.join(OUT_DIR, cls)
    os.makedirs(save_dir, exist_ok=True)
    ae.save(os.path.join(save_dir, "autoencoder.keras"))
    np.save(os.path.join(save_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(save_dir, "scaler_scale.npy"), scaler.scale_)
    histories[cls] = h

# ── 4) Avalia reconstrução em cada classe ───────────────────────────────────
def compute_mse_for_class(cls):
    m = np.load(os.path.join(OUT_DIR, cls, "scaler_mean.npy"))
    s = np.load(os.path.join(OUT_DIR, cls, "scaler_scale.npy"))
    scaler = StandardScaler()
    scaler.mean_ = m
    scaler.scale_ = s

    ae = load_model(os.path.join(OUT_DIR, cls, "autoencoder.keras"))

    arrs = np.stack(data_by_class[cls], axis=0)[..., np.newaxis]
    flat = arrs.reshape((arrs.shape[0], -1))
    scaled = scaler.transform(flat).reshape(arrs.shape)

    pred = ae.predict(scaled, verbose=0)
    mse = np.mean((scaled - pred) ** 2, axis=(1, 2))
    return mse

errors = {cls: compute_mse_for_class(cls) for cls in classes}

# ── 5) Estatísticas por classe ───────────────────────────────────────────────
print("\nEstatísticas de MSE por classe:")
for cls in classes:
    errs = errors[cls]
    print(f"{cls}: média={errs.mean():.3e}, mediana={np.median(errs):.3e}, std={errs.std():.3e}")

# ── 6) Plot de histórico de treinamento ───────────────────────────────────────
plt.figure(figsize=(10, 6))
for cls, h in histories.items():
    plt.plot(h.history["val_loss"], label=f"{cls} val_loss")
plt.yscale("log")
plt.xlabel("Época")
plt.ylabel("Val Loss (MSE)")
plt.legend()
plt.title("Histórico de Val Loss por Classe")
plt.tight_layout()
plt.show()

# ── 7) Distribuição de erros por classe ──────────────────────────────────────
plt.figure(figsize=(10, 6))
for cls in classes:
    plt.hist(errors[cls], bins=50, alpha=0.5, label=cls)
plt.xlabel("Reconstruction MSE")
plt.ylabel("Frequência")
plt.title("Distribuição de Erro de Reconstrução por Classe")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
df_box = pd.DataFrame({cls: pd.Series(errors[cls]) for cls in classes})
df_box.boxplot()
plt.ylabel("MSE")
plt.title("Boxplot de Erro de Reconstrução por Classe")
plt.tight_layout()
plt.show()
