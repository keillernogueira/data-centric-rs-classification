from pathlib import Path
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class DFC2022(Dataset):
    classes = [
        "No information",
        "Urban fabric",
        "Industrial, commercial, public, military, private and transport units",
        "Mine, dump and construction sites",
        "Artificial non-agricultural vegetated areas",
        "Arable land (annual crops)",
        "Permanent crops",
        "Pastures",
        "Complex and mixed cultivation patterns",
        "Orchards at the fringe of urban classes",
        "Forests",
        "Herbaceous vegetation associations",
        "Open spaces with little or no vegetation",
        "Wetlands",
        "Water",
        "Clouds and Shadows",
    ]

    colormap = [
        "#231F20",
        "#DB5F57",
        "#DB9757",
        "#DBD057",
        "#ADDB57",
        "#75DB57",
        "#7BC47B",
        "#58B158",
        "#D4F6D4",
        "#B0E2B0",
        "#008000",
        "#58B0A7",
        "#995D13",
        "#579BDB",
        "#0062FF",
        "#231F20",
    ]

    metadata = {
        "train": "train",
        "train_val": "train",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self, root, split, scores_file, sample_size, transforms=None, embed_fp=False
    ) -> None:
        assert split in self.metadata
        self.root = root
        self.split = split
        self.scores_file = scores_file
        self.sample_size = sample_size
        self.transforms = transforms
        self.embed_fp = embed_fp
        self.files = self._load_files()

    def _load_files(self):
        if self.scores_file is not None:
            scores = pd.read_csv(self.scores_file, index_col=0)
            patch_files = list()
            for _, row in scores.iterrows():
                id, xmin, ymin, score = (
                    row["id"],
                    int(row["xmin"]),
                    int(row["ymin"]),
                    float(row["score"]),
                )
                file_path = Path(self.root) / self.split / f"{id}_{xmin}_{ymin}.tif"
                patch_files.append((file_path, float(score)))
            patch_files = sorted(patch_files, key=lambda e: e[1], reverse=True)
            patch_files = (
                patch_files[: self.sample_size]
                if self.sample_size is not None
                else patch_files
            )
            patch_files = [file for file, _ in patch_files]
        else:
            patch_files = list(Path(self.root).glob(f"{self.split}/*.tif"))
        return patch_files

    def __getitem__(self, index):
        file_path = self.files[index]
        image, mask = self._load_data(file_path)
        sample = {"image": image, "mask": mask}
        if self.embed_fp:
            sample["fp"] = str(file_path.absolute())
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.files)

    def _load_data(self, path, shape=None):
        with rasterio.open(path) as f:
            arr = f.read()
            image_tensor = torch.from_numpy(arr[:-1, :, :].astype("float32"))
            mask_tensor = torch.from_numpy(arr[-1, :, :].astype("int32")).to(torch.long)
            return image_tensor, mask_tensor


class Vaihingen(Dataset):

    classes = [
        "Impervious surfaces",
        "Building",
        "Low vegetation",
        "Tree",
        "Car",
        "Clutter/background",
    ]

    colormap = [
        "#FFFFFF",  # Impervious surfaces (RGB: 255, 255, 255)
        "#0000FF",  # Building (RGB: 0, 0, 255)
        "#00FFFF",  # Low vegetation (RGB: 0, 255, 255)
        "#00FF00",  # Tree (RGB: 0, 255, 0)
        "#FFFF00",  # Car (RGB: 255, 255, 0)
        "#FF0000",  # Clutter/background (RGB: 255, 0, 0)
    ]

    metadata = {
        "train": "train",
        "train_val": "train",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self, root, split, scores_file, sample_size, transforms=None, embed_fp=False
    ) -> None:
        assert split in self.metadata
        self.root = root
        self.split = split
        self.scores_file = scores_file
        self.sample_size = sample_size
        self.transforms = transforms
        self.embed_fp = embed_fp
        self.files = self._load_files()

    def _load_files(self):
        if self.scores_file is not None:
            scores = pd.read_csv(self.scores_file, sep=" ", header=None)
            scores = scores.iloc[:, [0, -3, -2, -1]]
            scores.columns = ["id", "xmin", "ymin", "score"]
            scores["id"] = scores["id"].apply(lambda x: x.split("_")[3])
            patch_files = list()
            for _, row in scores.iterrows():
                id, xmin, ymin, score = (
                    row["id"],
                    int(row["xmin"]),
                    int(row["ymin"]),
                    float(row["score"]),
                )
                file_path = Path(self.root) / self.split / f"{id}_{xmin}_{ymin}.tif"
                patch_files.append((file_path, float(score)))
            patch_files = sorted(patch_files, key=lambda e: e[1], reverse=True)
            patch_files = (
                patch_files[: self.sample_size]
                if self.sample_size is not None
                else patch_files
            )
            patch_files = [file for file, _ in patch_files]
        else:
            patch_files = list(Path(self.root).glob(f"{self.split}/*.tif"))
        return patch_files

    def __getitem__(self, index):
        file_path = self.files[index]
        image, mask = self._load_data(file_path)
        sample = {"image": image, "mask": mask}
        if self.embed_fp:
            sample["fp"] = str(file_path.absolute())
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.files)

    def _load_data(self, path, shape=None):
        with rasterio.open(path) as f:
            arr = f.read()
            image_tensor = torch.from_numpy(arr[:-1, :, :].astype("float32"))
            mask_tensor = torch.from_numpy(arr[-1, :, :].astype("int32")).to(torch.long)
            return image_tensor, mask_tensor


class Potsdam(Dataset):

    classes = [
        "Impervious surfaces",
        "Building",
        "Low vegetation",
        "Tree",
        "Car",
        "Clutter/background",
    ]

    colormap = [
        "#FFFFFF",  # Impervious surfaces (RGB: 255, 255, 255)
        "#0000FF",  # Building (RGB: 0, 0, 255)
        "#00FFFF",  # Low vegetation (RGB: 0, 255, 255)
        "#00FF00",  # Tree (RGB: 0, 255, 0)
        "#FFFF00",  # Car (RGB: 255, 255, 0)
        "#FF0000",  # Clutter/background (RGB: 255, 0, 0)
    ]

    metadata = {
        "train": "train",
        "train_val": "train",
        "val": "val",
        "test": "test",
    }

    def __init__(
        self, root, split, scores_file, sample_size, transforms=None, embed_fp=False
    ) -> None:
        assert split in self.metadata
        self.root = root
        self.split = split
        self.scores_file = scores_file
        self.sample_size = sample_size
        self.transforms = transforms
        self.embed_fp = embed_fp
        self.files = self._load_files()

    def _load_files(self):
        if self.scores_file is not None:
            scores = pd.read_csv(self.scores_file, sep=" ", header=None)
            scores = scores.iloc[:, [0, -3, -2, -1]]
            scores.columns = ["id", "xmin", "ymin", "score"]
            scores["id"] = scores["id"].apply(lambda x: "_".join(x.split("_")[2:4]))
            patch_files = list()
            for _, row in scores.iterrows():
                id, xmin, ymin, score = (
                    row["id"],
                    int(row["xmin"]),
                    int(row["ymin"]),
                    float(row["score"]),
                )
                file_path = Path(self.root) / self.split / f"{id}_{xmin}_{ymin}.tif"
                patch_files.append((file_path, float(score)))
            patch_files = sorted(patch_files, key=lambda e: e[1], reverse=True)
            patch_files = (
                patch_files[: self.sample_size]
                if self.sample_size is not None
                else patch_files
            )
            patch_files = [file for file, _ in patch_files]
        else:
            patch_files = list(Path(self.root).glob(f"{self.split}/*.tif"))
        return patch_files

    def __getitem__(self, index):
        file_path = self.files[index]
        image, mask = self._load_data(file_path)
        sample = {"image": image, "mask": mask}
        if self.embed_fp:
            sample["fp"] = str(file_path.absolute())
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.files)

    def _load_data(self, path, shape=None):
        with rasterio.open(path) as f:
            arr = f.read()
            image_tensor = torch.from_numpy(arr[:-1, :, :].astype("float32"))
            mask_tensor = torch.from_numpy(arr[-1, :, :].astype("int32")).to(torch.long)
            return image_tensor, mask_tensor
