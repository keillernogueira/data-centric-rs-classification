"""Evaluates a scores file by training 5 UNet models on the top 5 highest scored sample folds and averages the resulting Jaccard scores.

Usage (Example):

    $ python scripts/train.py \
        --dataset "potsdam" \
        --method_name "PotsdamDiversity" \
        --scores_file_path /home/akramzaytar/ssdshared/akramz_datasets/wtl/submissions/potsdam/complexity.txt \
        --gpu 7
"""

import argparse
from pathlib import Path
from omegaconf import OmegaConf

import pandas as pd

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mveo_benchmarks.tasks import SemanticSegmentationTask
from mveo_benchmarks.datamodule import (
    DFC2022DataModule,
    PotsdamDataModule,
    VaihingenDataModule,
)


def main(config):

    # Get the name of the dataset
    dataset = config["datamodule"]["dataset"]

    # Depending on the name, create the data module object
    if dataset == "dfc22":
        dm = DFC2022DataModule
    elif dataset == "potsdam":
        dm = PotsdamDataModule
    elif dataset == "vaihingen":
        dm = VaihingenDataModule
    else:
        raise ValueError("Unknown dataset")

    # Initialize a dictionary to hold scores for each size
    scores = {}

    # Iterate over the top fraction of scored samples
    for sample_size in config["evaluation"]["sizes"]:
        # Initialize list to store scores for this sample size
        scores[sample_size] = []

        # Perform multiple runs for each sample size
        for run in range(config["evaluation"]["runs"]):

            # Train!
            datamodule = dm(
                **config.datamodule,
                train_size=sample_size,
                train_scores_file=config["evaluation"]["scores_file"],
            )
            task = SemanticSegmentationTask(**config.learning)
            ckpt_dir_path = f"checkpoints/{config['evaluation']['method_name'].lower().replace(' ', '_')}_{sample_size}_run{run}"

            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=0.00, patience=100, mode="min"
            )
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=ckpt_dir_path,
                filename="best_model",
                save_top_k=1,
                mode="min",
            )
            logger = pl.loggers.TensorBoardLogger(
                "logs/",
                name=f"{config['evaluation']['method_name'].lower().replace(' ', '_')}_{sample_size}_run{run}",
            )
            trainer = pl.Trainer(
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=logger,
                **config.trainer,
            )
            trainer.fit(model=task, datamodule=datamodule)

            # Get the test metrics
            test_metrics = trainer.test(
                model=task,
                datamodule=datamodule,
                ckpt_path=checkpoint_callback.best_model_path,
            )

            # Get the metrics
            jaccard_scores = test_metrics[0]
            jaccard_scores.pop("test_loss")
            keys = list(jaccard_scores.keys())
            scores[sample_size].append([jaccard_scores[key] for key in keys])

            # Delete objects and free memory
            torch.cuda.empty_cache()
            del task, datamodule, trainer

    # Save the scores to a text file
    scores_file_path = (
        Path("./results")
        / f"{config['evaluation']['method_name'].replace(' ', '_')}.txt"
    )
    with open(scores_file_path, "w") as file:
        for size, metrics in scores.items():
            metrics_str = ", ".join(map(str, metrics))
            file.write(f"{size}, {metrics_str}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["vaihingen", "potsdam", "dfc22"],
        type=str,
        help="The name of the scoring method.",
    )
    parser.add_argument(
        "--method_name", required=True, type=str, help="The name of the scoring method."
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--scores_file_path",
        required=True,
        type=str,
        help="Path to scores file (similar to `train_ids.txt`)",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file",
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config_file_path)
    config["evaluation"]["method_name"] = args.method_name
    config["evaluation"]["scores_file"] = Path(args.scores_file_path)
    config["trainer"]["devices"] = [args.gpu] if args.gpu >= 0 else -1
    config["datamodule"]["root"] = Path(config["datamodule"]["root"]) / args.dataset
    config["datamodule"]["dataset"] = args.dataset

    # Set the number of channels by dataset
    if args.dataset == "vaihingen":
        config["learning"]["in_channels"] = 4
        config["learning"]["ignore_index"] = -1
        config["learning"]["num_classes"] = 6
    elif args.dataset == "potsdam":
        config["learning"]["in_channels"] = 5
        config["learning"]["ignore_index"] = -1
        config["learning"]["num_classes"] = 6
    elif args.dataset == "dfc22":
        config["learning"]["in_channels"] = 4
        config["learning"]["ignore_index"] = 0
        config["learning"]["num_classes"] = 16
    else:
        raise ValueError("Unknown dataset")

    # Set the sizes according to how many rows are in the dataset
    evaluation_percentages = config["evaluation"]["sizes"]
    scores = pd.read_csv(args.scores_file_path, sep=" ", header=None)
    n_samples = len(scores)

    # Set the sizes according to how many rows are in the dataset
    config["evaluation"]["sizes"] = [
        int(n_samples * percentage) for percentage in evaluation_percentages
    ]

    main(config)
