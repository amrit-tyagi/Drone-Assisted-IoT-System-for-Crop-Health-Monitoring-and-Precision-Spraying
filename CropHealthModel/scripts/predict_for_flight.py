import numpy as np
import rasterio
import cv2
import json
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from skimage.util import view_as_windows

from models.cnn_gru_crop import build_cnn_gru  # if needed for custom objects

IMG_SIZE = 224
TIMESTEPS = 3
MODEL_PATH = "../models/saved/crop_health_cnn_gru.h5"
LABEL2ID_PATH = "../models/saved/label2id.json"

def load_label_map():
    label2id = json.loads(Path(LABEL2ID_PATH).read_text())
    id2label = {int(v): k for k, v in label2id.items()}
    return id2label

def tile_image(ortho_arr, patch_size):
    patches = view_as_windows(ortho_arr, (3, patch_size, patch_size), step=patch_size)
    # patches shape: (H_tiles, W_tiles, 3, P, P)
    return patches

def main():
    id2label = load_label_map()
    model = load_model(MODEL_PATH)

    ortho = rasterio.open("../CropAnalysis/data/DrnMppr-ORT-AOI.tif").read()[0:3]  # (3, H, W)
    H, W = ortho.shape[1:]
    PATCH = 128

    patches = view_as_windows(ortho, (3, PATCH, PATCH), step=PATCH)
    H_tiles, W_tiles, _, _, _ = patches.shape

    results = []
    for i in range(H_tiles):
        for j in range(W_tiles):
            rgb_patch = np.transpose(patches[i, j], (1, 2, 0))  # (P,P,3)
            img = cv2.resize(rgb_patch, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0

            seq = np.stack([img] * TIMESTEPS, axis=0)  # (T,H,W,3)
            pred = model.predict(seq[None, ...], verbose=0)
            class_id = int(np.argmax(pred, axis=1)[0])
            label = id2label[class_id]

            results.append({
                "tile_row": i,
                "tile_col": j,
                "label": label,
                "x_min": j * PATCH,
                "y_min": i * PATCH,
            })

    df = pd.DataFrame(results)
    df.to_csv("../output/predictions_tiles.csv", index=False)

if __name__ == "__main__":
    main()
