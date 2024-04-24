import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

import torch
from torch.utils import data


# Image.MAX_IMAGE_PIXELS = None


class ISPRSDataLoader(data.Dataset):
    classes = [
        "Impervious surfaces",
        "Building",
        "Low vegetation",
        "Tree",
        "Car",
        "Clutter/background"
    ]

    colormap = [
        "#ffffff",
        "#0000ff",
        "#00ffff",
        "#00ff00",
        "#ffff00",
        "#ff0000"
    ]

    metadata = {
        # same images used as test set for the ISPRS challenge
        "vaihingen_test": [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38],
        "potsdam_test": ['2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15',
                         '5_13', '5_14', '5_15', '6_13', '6_14', '6_15', '7_13'],
    }

    image_root = "images"
    dsm_root = "ndsm"
    target_root = "masks"

    def __init__(self,
                 root,
                 coordinate_file_path=None,
                 split="train",
                 patch_size=256,
                 training_sample_perct=1.0,
                 transforms=None):
        super().__init__()
        assert split in ['train', 'val', 'test']

        self.root = root
        self.coordinate_file_path = coordinate_file_path
        self.split = split
        self.patch_size = patch_size
        self.training_sample_perct = training_sample_perct
        self.transforms = transforms

        self.files = self._load_files()
        print(self.split, len(self.files))

    def __getitem__(self, index):
        img_path, dsm_path, label_path = self.files[index]["image"], \
            self.files[index]["dsm"], self.files[index]["target"]

        image = self._load_image(img_path)
        dem = self._load_image(dsm_path)
        image = torch.cat(tensors=[image, dem], dim=0)

        mask = self._encode_mask(self._load_target(label_path))
        # print(mask.shape, np.unique(mask), np.bincount(mask.flatten()))

        if self.coordinate_file_path is not None and self.patch_size != -1:
            coord_x, coord_y = self.files[index]['coord_x'], self.files[index]['coord_y']
            image = image[:, coord_x:coord_x + self.patch_size, coord_y:coord_y + self.patch_size]
            mask = mask[:, coord_x:coord_x + self.patch_size, coord_y:coord_y + self.patch_size]

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        # if self.split == 'val':
        #     return len(self.files) * 50  # 50 random patches from each validation image
        # else:
        return len(self.files)

    def _encode_mask(self, msk):
        msk = np.rollaxis(msk.numpy(), 0, 3)
        new = np.zeros((msk.shape[0], msk.shape[1]), dtype=int)

        msk = msk // 255
        msk = msk * (1, 7, 49)
        msk = msk.sum(axis=2)

        new[msk == 1 + 7 + 49] = 0  # Street.
        new[msk ==         49] = 1  # Building.
        new[msk ==     7 + 49] = 2  # Grass.
        new[msk ==     7     ] = 3  # Tree.
        new[msk == 1 + 7     ] = 4  # Car.
        new[msk == 1         ] = 5  # Surfaces.

        tensor = torch.from_numpy(new[np.newaxis, :, :])

        return tensor

    def _load_files(self):
        if self.coordinate_file_path is not None:
            # read list of patches
            file_list = pd.read_csv(self.coordinate_file_path, dtype=None, delimiter=' ').to_numpy()

            if (self.coordinate_file_path == 'vaihingen_train_coordinate_list.txt' or
                    self.coordinate_file_path == 'potsdam_train_coordinate_list.txt'):
                # this is the original coord list; this is only used to train the baseline model
                # shuffle because all instances have score 1.0
                np.random.shuffle(file_list)  # shuffle
                sort_samples = file_list
            else:
                # otherwise, sort based on the score
                # this are the correct files usually named: xxx_train_coords.txt, where xxx = name of team
                sort_samples = file_list[file_list[:, 5].argsort()[::-1]]

            # selecting samples based on the threshold
            total_size = len(sort_samples)
            selected_samples = sort_samples[0:int(total_size*self.training_sample_perct), :]
            print('sanity check - select and total samples, average score: ',
                  len(selected_samples), total_size, np.mean(selected_samples[:, 5]))

            files = []
            for x in selected_samples:
                img_path = os.path.join(self.root, self.image_root, x[0] + '.tif')
                dsm_path = os.path.join(self.root, self.dsm_root, x[1] + '.jpg')
                label_path = os.path.join(self.root, self.target_root, x[2] + '.tif')
                files.append(dict(image=img_path, dsm=dsm_path, target=label_path,
                                  coord_x=int(x[3]), coord_y=int(x[4])))
        else:
            files = []
            for img in self.metadata[('vaihingen_test' if 'vaihingen' in self.root else 'potsdam_test')]:
                if 'vaihingen' in self.root:
                    image = os.path.join(self.root, self.image_root, 'top_mosaic_09cm_area' + str(img) + '.tif')
                    dsm = os.path.join(self.root, self.dsm_root, 'dsm_09cm_matching_area' + str(img) + '_normalized.jpg')
                    target = os.path.join(self.root, self.target_root, 'top_mosaic_09cm_area' + str(img) + '.tif')
                else:
                    image = os.path.join(self.root, self.image_root, 'top_potsdam_' + str(img) + '_RGBIR.tif')
                    dsm = os.path.join(self.root, self.dsm_root, 'dsm_potsdam_0' + str(img) + '_normalized_lastools.jpg')
                    target = os.path.join(self.root, self.target_root, 'top_potsdam_' + str(img) + '_label.tif')

                files.append(dict(image=image, dsm=dsm, target=target))

        return files

    def _load_image(self, path, shape=None):
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            return tensor

    def _load_target(self, path):
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                out_dtype="int32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
            return tensor
