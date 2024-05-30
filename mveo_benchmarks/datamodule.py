"""Data module classes for the DFC2022 and Potsdam datasets."""

import kornia.augmentation as K
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision.transforms as T
from einops import rearrange
from matplotlib import colors
from torch.utils.data import DataLoader
from torchgeo.datasets.utils import percentile_normalization
from mveo_benchmarks.dataset import DFC2022, Potsdam, Vaihingen

# Standard vertical and horizontal flips
DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["input", "mask"],
)


class Preprocessor:
    def __init__(self, extra_vmin, extra_vmax) -> None:
        self.extra_vmin = extra_vmin
        self.extra_vmax = extra_vmax

    def __call__(self, sample):

        # RGB-... is uint8 so divide by 255
        sample["image"][:-1] /= 255.0

        # DEM is float and has a different range. Perform min-max normalization
        sample["image"][-1] = (sample["image"][-1] - self.extra_vmin) / (
            self.extra_vmax - self.extra_vmin
        )

        # If the provided min/max are not the actual min/max, clip the values
        sample["image"][-1] = torch.clip(sample["image"][-1], min=0.0, max=1.0)

        if "mask" in sample:
            # ignore the clouds and shadows class (not used in scoring)
            sample["mask"][sample["mask"] == 15] = 0
            sample["mask"] = rearrange(sample["mask"], "h w -> () h w")

        return sample


class DFC2022DataModule(pl.LightningDataModule):
    # Derived DEM stats from the training dataset
    dem_min, dem_max = -79.18, 3020.26

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        train_scores_file: str,
        train_size: int,
        augmentations=DEFAULT_AUGS,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_scores_file = train_scores_file
        self.train_size = train_size
        self.augmentations = augmentations
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.preprocess = Preprocessor(self.dem_min, self.dem_max)

    def setup(self, stage=None):
        # Define the transforms for train, val, and the test sets
        transforms = T.Compose([self.preprocess])

        # Create the training dataset
        self.train_ds = DFC2022(
            self.root,
            split="train",
            scores_file=self.train_scores_file,
            sample_size=self.train_size,
            transforms=transforms,
        )

        # Create the validation dataset
        self.val_ds = DFC2022(
            self.root,
            split="val",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

        # Create the test dataset
        self.test_ds = DFC2022(
            self.root,
            split="test",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dl_idx):
        if self.trainer.training:
            if self.augmentations is not None:
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                batch["mask"] = batch["mask"].to(torch.long)
        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(
        self,
        sample,
        show_titles=True,
        suptitle=None,
    ):
        ncols = 2

        # Prepare the RGB image
        image = sample["image"][:3]
        image = (image * 255.0).to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()

        # Prepare the DEM Image
        dem = sample["image"][-1].numpy()
        dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1))

        # Check if we have a mask and/or prediction
        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        # Define the color map for visualization
        cmap = colors.ListedColormap(DFC2022.colormap)

        # Add columns if we have mask and/or prediction
        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        # Visualize
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(
                mask, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(
                    pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
                )
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(
                pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM")
            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class PotsdamDataModule(pl.LightningDataModule):

    # Set the min and max values for the DSM
    dsm_min, dsm_max = -79.18, 3020.26

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        train_scores_file: str,
        train_size: int,
        augmentations=DEFAULT_AUGS,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_scores_file = train_scores_file
        self.train_size = train_size
        self.augmentations = augmentations
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.preprocess = Preprocessor(self.dsm_min, self.dsm_max)

    def setup(self, stage=None):
        # Define the transforms for train, val, and the test sets
        transforms = T.Compose([self.preprocess])

        # Create the training dataset
        self.train_ds = Potsdam(
            self.root,
            split="train",
            scores_file=self.train_scores_file,
            sample_size=self.train_size,
            transforms=transforms,
        )

        # Create the validation dataset
        self.val_ds = Potsdam(
            self.root,
            split="val",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

        # Create the test dataset
        self.test_ds = Potsdam(
            self.root,
            split="test",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 4,  # Because the training batches are small!
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size * 4,  # Because the training batches are small!
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dl_idx):
        if self.trainer.training:
            if self.augmentations is not None:
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                batch["mask"] = batch["mask"].to(torch.long)
        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(
        self,
        sample,
        show_titles=True,
        suptitle=None,
    ):
        ncols = 2

        # Prepare the RGB image
        image = sample["image"][:3]
        image = (image * 255.0).to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()

        # Prepare the DEM Image
        dem = sample["image"][-1].numpy()
        dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1))

        # Check if we have a mask and/or prediction
        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        # Define the color map for visualization
        cmap = colors.ListedColormap(Potsdam.colormap)

        # Add columns if we have mask and/or prediction
        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        # Visualize
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(
                mask, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(
                    pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
                )
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(
                pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM")
            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class VaihingenDataModule(pl.LightningDataModule):

    # Set the min and max values for the DSM
    dsm_min, dsm_max = -79.18, 3020.26

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        train_scores_file: str,
        train_size: int,
        augmentations=DEFAULT_AUGS,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_scores_file = train_scores_file
        self.train_size = train_size
        self.augmentations = augmentations
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.preprocess = Preprocessor(self.dsm_min, self.dsm_max)

    def setup(self, stage=None):
        # Define the transforms for train, val, and the test sets
        transforms = T.Compose([self.preprocess])

        # Create the training dataset
        self.train_ds = Vaihingen(
            self.root,
            split="train",
            scores_file=self.train_scores_file,
            sample_size=self.train_size,
            transforms=transforms,
        )

        # Create the validation dataset
        self.val_ds = Vaihingen(
            self.root,
            split="val",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

        # Create the test dataset
        self.test_ds = Vaihingen(
            self.root,
            split="test",
            scores_file=None,
            sample_size=None,
            transforms=transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 4,  # Because the training batches are small!
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size * 4,  # Because the training batches are small!
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dl_idx):
        if self.trainer.training:
            if self.augmentations is not None:
                batch["mask"] = batch["mask"].to(torch.float)
                batch["image"], batch["mask"] = self.augmentations(
                    batch["image"], batch["mask"]
                )
                batch["mask"] = batch["mask"].to(torch.long)
        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(
        self,
        sample,
        show_titles=True,
        suptitle=None,
    ):
        ncols = 2

        # Prepare the RGB image
        image = sample["image"][:3]
        image = (image * 255.0).to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()

        # Prepare the DEM Image
        dem = sample["image"][-1].numpy()
        dem = percentile_normalization(dem, lower=0, upper=100, axis=(0, 1))

        # Check if we have a mask and/or prediction
        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        # Define the color map for visualization
        cmap = colors.ListedColormap(Vaihingen.colormap)

        # Add columns if we have mask and/or prediction
        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        # Visualize
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dem)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(
                mask, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(
                    pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
                )
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(
                pred, cmap=cmap, interpolation="none", vmin=0, vmax=cmap.N - 1
            )
            axs[2].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DEM")
            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
