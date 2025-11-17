import numpy as np
import rasterio
import cv2
import json
import pandas as pd
from tensorflow.keras.models import load_model
from skimage.util import view_as_windows

IMG_SIZE = 224
PATCH = 128
TIMESTEPS = 3

model = load_model("../models/saved/crop_health.h5")
label_map = json.load(open("../models/saved/label_map.json"))
id2label = {v:k for k,v in label_map.items()}

ortho = rasterio.open("../CropAnalysis/data/DrnMppr-ORT-AOI.tif").read()[0:3]
patches = view_as_windows(ortho, (3,PATCH,PATCH), step=PATCH)

rows, cols = patches.shape[:2]
results = []

for i in range(rows):
    for j in range(cols):
        patch = patches[i,j]
        patch = np.transpose(patch, (1,2,0))
        img = cv2.resize(patch, (IMG_SIZE,IMG_SIZE))[...,::-1]/255.

        seq = np.stack([img]*TIMESTEPS)
        pred = model.predict(seq[None], verbose=0)
        c = int(np.argmax(pred))

        results.append({"row":i, "col":j, "class":id2label[c]})

pd.DataFrame(results).to_csv("../output/prediction.csv", index=False)
