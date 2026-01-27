import os
import sys
import cv2
import numpy as np
from PIL import Image
import glob
import rasterio
from tqdm import tqdm
import pandas as pd
from typing import (
    List,Tuple
)

def rgb_to_mask(
    rgb: "np.typing.NDArray[np.uint8]", colors: List[Tuple[int, int, int]]
) -> "np.typing.NDArray[np.uint8]":
    assert len(colors) <= 256  # we currently return a uint8 array, so the largest value
    # we can map is 255
    h, w = rgb.shape[:2]
    mask: "np.typing.NDArray[np.uint8]" = np.zeros(shape=(h, w), dtype=np.uint8)
    for i, c in enumerate(colors):
        cmask = rgb == c
        # Only update mask if class is present in mask
        if isinstance(cmask, np.ndarray):
            mask[cmask.all(axis=-1)] = i
    return mask

colormap = [
    (255, 0, 0),
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
]

# potsdam
coordinate_file_path="../coord_files/potsdam_train_coordinate_list.txt"
file_list = pd.read_csv(coordinate_file_path, dtype=None, delimiter=' ', header=None).to_numpy()
selected_samples = file_list[:, :]

files = []
for x in selected_samples:
    image_path = os.path.join("../Potsdam/", "masks", x[2] + '.tif')
    files.append(dict(target=image_path, coord_x=int(x[3]), coord_y=int(x[4])))
num_classes = np.zeros(6)
dict_class = {
        'ImageName':[],
        'class1':[],
        'class2':[],
        'class3':[],
        'class4':[],
        'class5':[],
        'class6':[],
      }
df = pd.DataFrame(dict_class)
for file in tqdm(files):
    # with rasterio.open(file["target"]) as f:
    with Image.open(file["target"]) as img:
        array: "np.typing.NDArray[np.uint8]" = np.array(img.convert("RGB"))
        coord_x, coord_y = file['coord_x'], file['coord_y']
        mask = array[coord_x:coord_x + 256, coord_y:coord_y + 256]
        mask = rgb_to_mask(mask, colormap)
        h,w = mask.shape
        for i in range(h):
            for j in range(w):
                class_ = mask[i,j]
                num_classes[class_] += 1

        image_name = file["target"].split("/")[-1].replace("label.tif","RGBIR")
        dsm_name = file["target"].split("/")[-1].replace("top_potsdam_","dsm_potsdam_0")
        dsm_name = dsm_name.replace("_label.tif","_normalized_lastools")
        df.loc[len(df.index)] = [image_name+" "+dsm_name+" "+file["target"].split("/")[-1].replace("label.tif","label")+" "+str(file["coord_x"])+" "+str(file["coord_y"])] + list(num_classes)
    num_classes = np.zeros(6)

df.to_csv("potsdam_class_distribution.csv",index=False)
