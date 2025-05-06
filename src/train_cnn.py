import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Diretórios
MODEL_DIR = "../models/CNN"
os.makedirs(MODEL_DIR, exist_ok=True)


# Carregamento dos dados
def load_data(processed_dir="../data/processed", n_points=256):
    X = []
    y = []
    for label in os.listdir(processed_dir):
        if label not in ["CP", "FP", "KP"]:
            continue  # ignorar FA
        label_path = os.path.join(processed_dir, label)
        for file in os.listdir(label_path):
            if file.endswith(".npy"):
                x = np.load(os.path.join(label_path, file))
                if len(x) == n_points:
                    X.append(x)
                    y.append(1 if label in ["CP", "KP"] else 0)
    return np.array(X), np.array(y)


X, y = load_data()
X = X[..., np.newaxis]  # adicionar canal

# Divisão treino/val/teste
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# Modelo CNN 1D
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = create_cnn_model(X_train.shape[1:])

# Callbacks
checkpoint_path = os.path.join(MODEL_DIR, "best_model.keras")
history_path = os.path.join(MODEL_DIR, "training_history.npy")
history_plot_path = os.path.join(MODEL_DIR, "training_plot.png")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", mode="min"),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# Treinamento
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Salvar histórico
np.save(history_path, history.history)

# Avaliação
y_pred = (model.predict(X_test) > 0.50).astype(int).flatten()
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, target_names=["not planet", "planet"]))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))

# Plot do treinamento
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("CNN Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(history_plot_path)
plt.close()
