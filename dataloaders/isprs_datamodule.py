import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import kornia.augmentation as K
import lightning.pytorch as pl
from einops import rearrange

from matplotlib import colors
import matplotlib.pyplot as plt

from torchgeo.datasets.utils import percentile_normalization

from dataloaders.isprs_dataloader import ISPRSDataLoader, ISPRSDataLoaderPatch


DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["input", "mask"],
)


class ISPRSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        num_workers: int = 0,
        train_coordinate_file_path: str = 'train_coords.txt',
        training_sample_perct: float = 1.0,
        val_image_file_path: str = 'val_coords.txt',
        patch_size: int = 256,
        augmentations=DEFAULT_AUGS,
        patch_loader=True,
        **kwargs,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_coordinate_file_path = train_coordinate_file_path
        self.training_sample_perct = training_sample_perct
        self.val_image_file_path = val_image_file_path
        self.patch_size = patch_size
        self.augmentations = augmentations
        self.patch_loader = patch_loader
        # self.random_crop = T.RandomCrop((self.patch_size, self.patch_size))
        self.random_crop = K.AugmentationSequential(
            K.RandomCrop((self.patch_size, self.patch_size), p=1.0, keepdim=False),
            data_keys=["input", "mask"],
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def preprocess(self, sample):
        if 'postdam' in self.train_coordinate_file_path:
            # for potsdam, the images and dsms are encoded as uint8, varying from 0 to 255
            sample["image"] /= 255.0
        else:
            # Stats computed in labeled train set
            dsm_min, dsm_max = 240.7338, 359.9981

            # RGB is uint8 so divide by 255
            sample["image"][:-1] /= 255.0

            # min-max normalization for the DSM
            sample["image"][-1] = (sample["image"][-1] - dsm_min) / (dsm_max - dsm_min)
            sample["image"][-1] = torch.clip(sample["image"][-1], min=0.0, max=1.0)
        return sample

    # this is only used to simplify the validation process
    # the validation process is performed using crops of the validation images
    def crop(self, sample):
        sample["image"] = rearrange(sample["image"], "c h w -> () c h w")
        sample["mask"] = rearrange(sample["mask"], "c h w -> () c h w")
        sample["mask"] = sample["mask"].to(torch.float)

        sample["image"], sample["mask"] = self.random_crop(
            sample["image"], sample["mask"]
        )

        sample["mask"] = sample["mask"].to(torch.long)
        sample["image"] = rearrange(sample["image"], "() c h w -> c h w")
        sample["mask"] = rearrange(sample["mask"], "() c h w -> c h w")
        return sample

    def setup(self, stage=None):
        train_transforms = T.Compose([self.preprocess])
        val_transforms = T.Compose([self.preprocess, self.crop])
        test_transforms = T.Compose([self.preprocess])

        if self.patch_loader is False:
            # create patches on the fly
            self.train_dataset = ISPRSDataLoader(self.root_dir, self.train_coordinate_file_path, "train",
                                                 self.patch_size, training_sample_perct=self.training_sample_perct,
                                                 transforms=train_transforms)
        else:
            # otherwise, load patches from disk
            self.train_dataset = ISPRSDataLoaderPatch(self.root_dir, self.train_coordinate_file_path, "train",
                                                      self.patch_size, training_sample_perct=self.training_sample_perct,
                                                      transforms=train_transforms)

        # for validation, there is no patch size for the dataloader since patches are generated using the transforms
        self.val_dataset = ISPRSDataLoader(self.root_dir, self.val_image_file_path, "val",
                                           patch_size=-1, transforms=val_transforms)

        # for test, there is no patch size since images are processed entirely
        self.test_dataset = ISPRSDataLoader(self.root_dir, None, "test",
                                            patch_size=-1, transforms=test_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
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
        image = sample["image"][:3]
        image = image.to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()

        dsm = sample["image"][-1].numpy()
        dsm = percentile_normalization(dsm, lower=0, upper=100, axis=(0, 1))

        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample

        cmap = colors.ListedColormap(ISPRSDataLoader.colormap)

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 10))

        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(dsm)
        axs[1].axis("off")
        if showing_mask:
            axs[2].imshow(mask, cmap=cmap, interpolation="none")
            axs[2].axis("off")
            if showing_prediction:
                axs[3].imshow(pred, cmap=cmap, interpolation="none")
                axs[3].axis("off")
        elif showing_prediction:
            axs[2].imshow(pred, cmap=cmap, interpolation="none")
            axs[2].axis("off")

        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("DSM")

            if showing_mask:
                axs[2].set_title("Ground Truth")
                if showing_prediction:
                    axs[3].set_title("Predictions")
            elif showing_prediction:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
