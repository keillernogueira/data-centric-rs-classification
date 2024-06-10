import argparse
import warnings
import rasterio

# import pytorch_lightning as pl
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from torchgeo.trainers import SemanticSegmentationTask
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from dataloaders.dfc2022_datamodule import DFC2022DataModule
from dataloaders.isprs_datamodule import ISPRSDataModule

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SegmentationTaskCustomScheduler(SemanticSegmentationTask):
    def __init__(
        self, model,
        backbone,
        weights=None,
        in_channels=3,
        num_classes=1000,
        num_filters=3,
        loss="ce",
        class_weights=None,
        ignore_index=None,
        lr=0.001,
        patience=5,
        freeze_backbone=False,
        freeze_decoder=False,
        steps=10
    ) -> None:
        super().__init__(model, backbone, weights, in_channels, num_classes,
                         num_filters, loss, class_weights, ignore_index, lr, patience, freeze_backbone, freeze_decoder)
        self.steps = steps

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }


def main(_config):
    pl.seed_everything(0)

    task = SemanticSegmentationTask(**_config.learning)  # model = unet (with imagenet), loss = ce (no class weights)
    # task = SegmentationTaskCustomScheduler(**_config.learning)

    if _config.datamodule.dataset == 'DFC2022':
        datamodule = DFC2022DataModule(**_config.datamodule)
    elif _config.datamodule.dataset == 'potsdam' or _config.datamodule.dataset == 'vaihingen':
        datamodule = ISPRSDataModule(**_config.datamodule)
    else:
        raise NotImplementedError('Dataset not implemented. Options are DFC2022 and ISPRS.')

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback],
                         **_config.trainer)
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to config.yaml file")
    args = parser.parse_args()

    _config = OmegaConf.load(args.config_file)

    main(_config)
