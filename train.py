import argparse

# import pytorch_lightning as pl
import lightning.pytorch as pl
from omegaconf import OmegaConf

from torchgeo.trainers import SemanticSegmentationTask

from dataloaders.dfc2022_datamodule import DFC2022DataModule
from dataloaders.isprs_datamodule import ISPRSDataModule


def main(_config):
    pl.seed_everything(0)

    task = SemanticSegmentationTask(**_config.learning)  # model = unet (with imagenet), loss = ce (no class weights)

    if _config.datamodule.dataset == 'DFC2022':
        datamodule = DFC2022DataModule(**_config.datamodule)
    elif _config.datamodule.dataset == 'ISPRS':
        datamodule = ISPRSDataModule(**_config.datamodule)
    else:
        raise NotImplementedError('Dataset not implemented. Options are DFC2022 and ISPRS.')

    trainer = pl.Trainer(**_config.trainer)
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to config.yaml file")
    args = parser.parse_args()

    _config = OmegaConf.load(args.config_file)

    main(_config)
