import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import FEATURES_FILE
from utils import extract_features_wrapper
from trainer import PlanetClassifierTrainer
from data_loader import load_data


if __name__ == "__main__":
    TUNING_RUNS = 5          # <- Number of full tuning attempts
    ITERATIONS_PER_RUN = 1000

    combined_df = load_data()

    if os.path.exists(FEATURES_FILE):
        print("Carregando features em cache...")
        data = np.load(FEATURES_FILE, allow_pickle=True)
        X, y = data['X'], data['y']
    else:
        print("Extraindo features...")
        features, labels = [], []
        with Pool() as pool:
            results = list(tqdm(pool.imap(extract_features_wrapper, [row for _, row in combined_df.iterrows()]), total=len(combined_df)))
            for result in results:
                if result:
                    f, label = result
                    features.append(f)
                    labels.append(label)
        X = np.array(features)
        y = np.array(labels)
        np.savez(FEATURES_FILE, X=X, y=y)
        print("Features armazenadas em cache no disco.")

    print("Distribuição de Classes:")
    print(pd.Series(y).value_counts(normalize=True))

    for run in range(1, TUNING_RUNS + 1):
        print(f"\n🎯 Tuning Run {run}/{TUNING_RUNS}")
        trainer = PlanetClassifierTrainer(X, y)
        trainer.hyperparameter_tuning(iterations=ITERATIONS_PER_RUN)
        trainer.train_final_model()
