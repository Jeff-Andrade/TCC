import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models, optimizers

# ------------------------
# Configurações
# ------------------------
DATA_DIR = "../data/processed"
LABEL_FA = "FA"  # pasta com os Falsos Alarmes
MODEL_DIR = "../models/Autoencoder_FA"
os.makedirs(MODEL_DIR, exist_ok=True)

N_POINTS = 256
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3

# ------------------------
# Carregar apenas FA para treino
# ------------------------
fa_dir = os.path.join(DATA_DIR, LABEL_FA)
X = []
for fn in os.listdir(fa_dir):
    if fn.endswith(".npy"):
        arr = np.load(os.path.join(fa_dir, fn))
        if arr.shape[0] == N_POINTS:
            X.append(arr)
X = np.array(X)[..., np.newaxis]  # (N, 256, 1)

# Normalização simples (já vêm normalizadas em volta de 1.0)
X -= X.mean(axis=1, keepdims=True)
X /= X.std(axis=1, keepdims=True)


# ------------------------
# Construção do Autoencoder “estreito”
# ------------------------
def build_narrow_autoencoder(input_shape):
    inp = layers.Input(shape=input_shape)

    # Encoder: poucas camadas, gargalo pequeno
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling1D(2, padding='same')(x)

    x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)

    # Bottleneck muito estreito
    x = layers.Conv1D(4, 3, activation='relu', padding='same')(x)

    # Decoder simétrico
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)

    out = layers.Conv1D(1, 3, activation='linear', padding='same')(x)

    autoenc = models.Model(inputs=inp, outputs=out)
    autoenc.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='mse')
    return autoenc


autoencoder = build_narrow_autoencoder((N_POINTS, 1))
autoencoder.summary()

# ------------------------
# Treinamento sem early stopping
# ------------------------
history = autoencoder.fit(
    X, X,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.1,
    verbose=2
)

# Salvar modelo e histórico
autoencoder.save(os.path.join(MODEL_DIR, "autoencoder_fa.keras"))
np.save(os.path.join(MODEL_DIR, "history_loss.npy"), history.history['loss'])
np.save(os.path.join(MODEL_DIR, "history_val_loss.npy"), history.history['val_loss'])

# ------------------------
# Plot do Loss
# ------------------------
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label="train loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"))
plt.show()
