import os

import numpy as np
import pandas as pd
from lightkurve import read, LightCurve, LightCurveCollection
from tqdm import tqdm

from preprocessing import (
    remove_outliers,
    normalize,
    detrend,
    interpolate_gaps,
    resample_lightcurve
)

# Configurações de diretório e constantes
RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "tic_labels.csv")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "tic_labels.csv")
N_POINTS = 256
VALID_LABELS = {"CP", "FP", "KP", "FA"}

# Criação dos diretórios de saída
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
for label in VALID_LABELS:
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, label), exist_ok=True)

# Carregar os rótulos
df = pd.read_csv(RAW_CSV_PATH)
df = df[df["tfopwg_disp"].isin(VALID_LABELS)]

processed_samples = []

# Pré-processamento
for _, row in tqdm(df.iterrows(), total=len(df), desc="Pré-processando curvas"):
    tid = int(row["tid"])
    label = row["tfopwg_disp"]
    fits_path = os.path.join(RAW_DATA_DIR, label, f"TIC_{tid}.fits")

    if not os.path.exists(fits_path):
        tqdm.write(f"Arquivo não encontrado: {fits_path}")
        continue

    try:
        raw_lc = read(fits_path)
        if isinstance(raw_lc, LightCurveCollection):
            raw_lc = raw_lc.stitch()

        # Remover unidades problemáticas (flux.value e time.value)
        lc = LightCurve(time=raw_lc.time.value, flux=raw_lc.flux.value)

        # Pipeline de pré-processamento
        lc = remove_outliers(lc)
        lc = normalize(lc)
        lc = detrend(lc)
        lc = interpolate_gaps(lc)
        _, flux_vector = resample_lightcurve(lc, num_points=N_POINTS)

        # Salvar curva pré-processada
        save_path = os.path.join(PROCESSED_DATA_DIR, label, f"TIC_{tid}.npy")
        np.save(save_path, flux_vector)
        processed_samples.append({"tid": tid, "tfopwg_disp": label})

    except Exception as e:
        tqdm.write(f"Erro ao processar TIC {tid}: {e}")

# Salvar novo CSV com amostras válidas
pd.DataFrame(processed_samples).to_csv(PROCESSED_CSV_PATH, index=False)
print(f"\n✅ Pré-processamento concluído: {len(processed_samples)} curvas salvas.")
