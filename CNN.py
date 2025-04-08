import os
import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input, \
    concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import random

positive_dataset_path = '../../../data/positive_data.csv'
negative_dataset_path = '../../../data/negative_data.csv'
data_dir_positive = '../../../data/positive_data'
data_dir_negative = '../../../data/negative_data'

sequence_length = 1000
noise_std = 0.01
shift_max = 5
augment_prob = 0.5

df_pos = pd.read_csv(positive_dataset_path)
df_pos = df_pos[df_pos['tfopwg_disp'].isin(['CP', 'KP'])]
df_pos['label'] = 1

df_neg = pd.read_csv(negative_dataset_path)
df_neg = df_neg[df_neg['tfopwg_disp'].isin(['FA', 'FP'])]
df_neg['label'] = 0

df_all = pd.concat([df_pos, df_neg], ignore_index=True)


def augment_signal(signal, noise_std=0.01, shift_max=5):
    noisy = signal + np.random.normal(0, noise_std, size=signal.shape)
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(noisy, shift)


light_curves, labels = [], []
scaler = MinMaxScaler(feature_range=(0, 1))

for _, row in df_all.iterrows():
    tid = row['tid']
    label = row['label']
    file_path = os.path.join(data_dir_positive if label == 1 else data_dir_negative, f"TIC_{tid}.fits")
    if os.path.exists(file_path):
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            time = data['TIME']
            flux = data['PDCSAP_FLUX'] if 'PDCSAP_FLUX' in data.columns.names else data['FLUX']
            valid_mask = np.isfinite(time) & np.isfinite(flux)
            if np.sum(valid_mask) > 0:
                flux_norm = scaler.fit_transform(flux[valid_mask].reshape(-1, 1)).flatten()
                light_curves.append(flux_norm)
                labels.append(label)

X = pad_sequences(light_curves, maxlen=sequence_length, padding='post', truncating='post', dtype='float32')
y = np.array(labels)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_aug, y_train_aug = [], []
for i in range(len(X_train)):
    X_train_aug.append(X_train[i])
    y_train_aug.append(y_train[i])
    if random.random() < augment_prob:
        augmented = augment_signal(X_train[i].flatten(), noise_std=noise_std, shift_max=shift_max)
        X_train_aug.append(augmented.reshape((sequence_length, 1)))
        y_train_aug.append(y_train[i])

X_train_aug = np.array(X_train_aug)
y_train_aug = np.array(y_train_aug)
class_weights = dict(
    enumerate(compute_class_weight(class_weight='balanced', classes=np.unique(y_train_aug), y=y_train_aug)))

model = Sequential([
    Conv1D(64, 5, activation='relu', padding='same', input_shape=(sequence_length, 1)),
    BatchNormalization(),
    MaxPooling1D(2),

    Conv1D(256, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),

    # Additional convolutional block for deeper feature extraction
    Conv1D(512, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ModelCheckpoint('model_improved.h5', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

model.fit(X_train_aug, y_train_aug, epochs=50, batch_size=32, validation_split=0.2, class_weight=class_weights,
          callbacks=callbacks)

predictions = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, predictions, target_names=['Not Planet (FA/FP)', 'Planet (CP/KP)']))
