# src/download.py

import os

import pandas as pd
from lightkurve import search_lightcurve
from tqdm import tqdm

# Caminho para o CSV
CSV_PATH = "../data/raw/tic_labels.csv"

# Carrega o CSV e assume que tfopwg_disp já é a sigla desejada (FP, PC, etc.)
df = pd.read_csv(CSV_PATH, dtype={"tfopwg_disp": str})
df["label"] = df["tfopwg_disp"]

# Descarta quaisquer linhas sem label (NaN)
df = df.dropna(subset=["label"])

# Garante que só vamos rodar nas 6 classes previstas
VALID_LABELS = {"CP", "FP", "KP", "FA"}
df = df[df["label"].isin(VALID_LABELS)]

# Loop por classe
for label_code in sorted(df["label"].unique()):
    print(f"\n===> Processando classe: {label_code}")

    class_df = df[df["label"] == label_code]
    tids = class_df["tid"].unique()

    # Diretório para esta classe
    output_dir = f"../data/raw/{label_code}"
    os.makedirs(output_dir, exist_ok=True)

    # TIDs já baixados
    downloaded = {
        f.split("_")[1].split(".")[0]
        for f in os.listdir(output_dir)
        if f.endswith(".fits")
    }

    for tid in tqdm(tids, desc=f"Baixando {label_code}", unit="TIC"):
        if str(tid) in downloaded:
            tqdm.write(f"Já existe: TIC {tid}")
            continue

        try:
            lc_collection = search_lightcurve(f"TIC {tid}", mission="TESS")
            if lc_collection and len(lc_collection) > 0:
                lc = lc_collection[0].download()
                if lc:
                    filename = os.path.join(output_dir, f"TIC_{tid}.fits")
                    lc.to_fits(filename, overwrite=True)
                    tqdm.write(f"✔ TIC {tid} salvo em {filename}")
                else:
                    tqdm.write(f"✖ Nada baixado para TIC {tid}")
            else:
                tqdm.write(f"✖ Nenhuma curva para TIC {tid}")
        except Exception as e:
            tqdm.write(f"⚠ Erro em TIC {tid}: {e}")
