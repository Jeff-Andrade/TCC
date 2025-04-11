import os
import random

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from config import TUNING_RESULTS_PATH, MODEL_SAVE_PATH, MODEL_OUTPUT_DIR


class PlanetClassifierTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.best_params = None

    def hyperparameter_tuning(self, iterations=1000):
        learning_rates = [0.01, 0.05, 0.1, 0.2]
        test_sizes = [0.15, 0.2, 0.25, 0.3]
        random_states = random.sample(range(1, 10000), iterations)

        best_score = 0
        best_params = {}
        results = []

        for i in tqdm(range(iterations), desc="Tuning"):
            lr = random.choice(learning_rates)
            ts = random.choice(test_sizes)
            rs = random_states[i]

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=ts, random_state=rs)
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

            model = XGBClassifier(
                n_estimators=300,
                learning_rate=lr,
                max_depth=6,
                scale_pos_weight=weights[0] / weights[1],
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            f1 = report['1']['f1-score']
            results.append({'learning_rate': lr, 'test_size': ts, 'random_state': rs, 'f1_score': f1})
            if f1 > best_score:
                best_score = f1
                best_params = {'learning_rate': lr, 'test_size': ts, 'random_state': rs}

        self.best_params = best_params
        pd.DataFrame(results).to_csv(TUNING_RESULTS_PATH, index=False)
        print(f"Tuning results saved to {TUNING_RESULTS_PATH}")
        print("Best Parameters:", best_params)
        print("Best F1 Score:", best_score)
        return best_params

    def train_final_model(self):
        if not self.best_params:
            raise ValueError("No best parameters found. Please run hyperparameter_tuning() first.")
        lr = self.best_params['learning_rate']
        ts = self.best_params['test_size']
        rs = self.best_params['random_state']

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=ts, random_state=rs)
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

        final_model = XGBClassifier(
            n_estimators=300,
            learning_rate=lr,
            max_depth=6,
            scale_pos_weight=weights[0] / weights[1],
            use_label_encoder=False,
            eval_metric='logloss'
        )
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        print("\nFinal Evaluation Report:")
        print(classification_report(y_test, y_pred))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
        plt.title("Confusion Matrix (Best Hyperparameters)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        joblib.dump(final_model, MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")