import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import FEATURES_FILE
from utils import extract_features_wrapper


if __name__ == "__main__":
    # Load data using data_loader functionality
    from data_loader import load_data
    combined_df = load_data()

    # Extract features
    if os.path.exists(FEATURES_FILE):
        print("Loading cached features...")
        data = np.load(FEATURES_FILE, allow_pickle=True)
        X, y = data['X'], data['y']
    else:
        print("Extracting features...")
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
        print("Features cached to disk.")

    print("Class Distribution:")
    print(pd.Series(y).value_counts(normalize=True))

    # Hyperparameter tuning and training
    from trainer import PlanetClassifierTrainer
    trainer = PlanetClassifierTrainer(X, y)
    trainer.hyperparameter_tuning(iterations=1000)
    trainer.train_final_model()