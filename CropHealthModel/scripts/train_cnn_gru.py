import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models.cnn_gru_crop import build_cnn_gru

IMG_SIZE = 224
TIMESTEPS = 3  # how many timepoints per plot
DATA_DIR = "../data/patches"

# 1. Load metadata with labels
meta = pd.read_csv("../data/patch_metadata.csv")

# Filter out unlabeled
meta = meta[meta['label'].notnull()]

# Map labels to integers
label2id = {label: idx for idx, label in enumerate(sorted(meta['label'].unique()))}
meta['label_id'] = meta['label'].map(label2id)

# 2. Build sequences: you need a "plot_id" and "time" column in metadata
# For now, assume each row is an independent frame (no time series)
# and we fake sequences of length 1 by repeating:
def make_sequence_for_row(row):
    img_path = os.path.join(DATA_DIR, row['id'])
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img[..., ::-1]  # BGR to RGB
    img = img.astype('float32') / 255.0

    # replicate same frame TIMESTEPS times (if you don't have time data yet)
    seq = np.stack([img] * TIMESTEPS, axis=0)  # (TIMESTEPS, H, W, 3)
    return seq

X = np.stack([make_sequence_for_row(r) for _, r in meta.iterrows()])
y = meta['label_id'].values

# 3. Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Build model
model = build_cnn_gru(num_classes=len(label2id),
                      timesteps=TIMESTEPS,
                      img_size=IMG_SIZE)

# 5. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=4
)

# 6. Save model + label mapping
os.makedirs("../models/saved", exist_ok=True)
model.save("../models/saved/crop_health_cnn_gru.h5")
pd.Series(label2id).to_json("../models/saved/label2id.json")
