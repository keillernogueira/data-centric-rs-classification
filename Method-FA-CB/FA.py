import rasterio
from rasterio.enums import Resampling
import torch
import torch.nn as nn
import torchvision.models as models
import resnet
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import csv

# Load a pre-trained ResNet model
resnet = resnet.__dict__['resnet18'](pretrained=True).to('cuda')

resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer

# Set the model to evaluation mode
resnet.eval()

print("resnet18")

def calculate_1_meanlogstd(features):
    # Flatten the feature tensor
    flattened_features = features.view(features.size(0), -1)
    # Calculate the mean and standard deviation of each feature
    mean = torch.mean(flattened_features, dim=1)
    std = torch.std(flattened_features, dim=1)
    # Calculate the entropy
    entropy = -(1-mean) * torch.log(std + 1e-8) 
#     print(entropy)
    return entropy

# Potsdam
coordinate_file_path="coord_files/potsdam_train_coordinate_list.txt"
file_list = pd.read_csv(coordinate_file_path, dtype=None, delimiter=' ', header=None).to_numpy()
selected_samples = file_list[:, :]

files = []
for x in selected_samples:
#     image_path = os.path.join("./Potsdam/", "images", x[0] + '.tif')
    image_path = os.path.join("Potsdam/", "patches_images", x[0] + '_' + str(x[3]) + '_' + str(x[4]) + '.tif')
    files.append(dict(image=image_path, coord_x=int(x[3]), coord_y=int(x[4])))

entropies = []
for file in tqdm(files):
    with rasterio.open(file["image"]) as f:
        array: "np.typing.NDArray[np.float_]" = f.read(
            out_shape=None, out_dtype="float32", resampling=Resampling.bilinear
        )
#         coord_x, coord_y = file['coord_x'], file['coord_y']
        image = array / 255.0 #array[:, coord_x:coord_x + 256, coord_y:coord_y + 256] / 255.0
        tensor = torch.from_numpy(image).unsqueeze(0).to('cuda')
        tensor = tensor[:,:3,:,:]
        features = resnet(tensor)

        # Calculate the entropy of the features
        entropy = calculate_1_meanlogstd(features).item()
        entropies.append([entropy])

with open('method2.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(entropies)
