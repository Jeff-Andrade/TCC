import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Caminhos
DATA_DIR = "../data/processed"
LABEL_CSV = os.path.join(DATA_DIR, "tic_labels.csv")

# Carregar rótulos
df = pd.read_csv(LABEL_CSV)

# Mapear rótulos para nomes amigáveis
label_map = {
    "CP": "planet",
    "KP": "planet",
    "FP": "not planet",
    "FA": "false alarm"
}

# Filtrar apenas rótulos de interesse
df = df[df["tfopwg_disp"].isin(label_map.keys())]
df["class"] = df["tfopwg_disp"].map(label_map)

# Exibir contagem de amostras por rótulo original e por classe
print("Contagem por label original (tfopwg_disp):")
print(df["tfopwg_disp"].value_counts(), "\n")

print("Contagem por classe (agrupada):")
print(df["class"].value_counts(), "\n")

# Estatísticas por classe com as curvas .npy
stats = []

for label, group in df.groupby("class"):
    fluxes = []
    for tid in group["tid"]:
        sublabel = group[group["tid"] == tid]["tfopwg_disp"].values[0]
        npy_path = os.path.join(DATA_DIR, sublabel, f"TIC_{tid}.npy")
        if os.path.exists(npy_path):
            flux = np.load(npy_path)
            fluxes.append(flux)
    fluxes = np.array(fluxes)
    if len(fluxes) > 0:
        flat = fluxes.flatten()
        stats.append({
            "class": label,
            "mean": np.mean(flat),
            "std": np.std(flat),
            "min": np.min(flat),
            "max": np.max(flat),
            "median": np.median(flat)
        })

# Mostrar estatísticas
print("Resumo estatístico por classe:\n")
stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))

# Plots em uma única janela
n_samples = 5
fig, axes = plt.subplots(n_samples, len(stats_df["class"].unique()), figsize=(14, 10))
fig.suptitle("Amostras de curvas de luz por classe", fontsize=16)

for col_idx, (label, group) in enumerate(df.groupby("class")):
    sampled = group.sample(n=min(n_samples, len(group)), random_state=42)
    for row_idx, tid in enumerate(sampled["tid"]):
        sublabel = sampled[sampled["tid"] == tid]["tfopwg_disp"].values[0]
        npy_path = os.path.join(DATA_DIR, sublabel, f"TIC_{tid}.npy")
        if os.path.exists(npy_path):
            flux = np.load(npy_path)
            ax = axes[row_idx, col_idx] if n_samples > 1 else axes[col_idx]
            ax.plot(np.linspace(0, 1, len(flux)), flux, color="black")
            ax.set_title(f"{label} — TIC {tid}")
            ax.set_ylim(min(flux) - 0.1, max(flux) + 0.1)
            if row_idx == n_samples - 1:
                ax.set_xlabel("Fase normalizada")
            if col_idx == 0:
                ax.set_ylabel("Fluxo")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
