"""
Script to extract indidivual patches from the Potsdam imagery/masks.
===========================================================

Usage
-----

# Train
python potsdam.py \
    --indices_file ../../data/indices/potsdam_train_coordinate_list.txt \
    --rgb_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/potsdam/train

# Validation
python potsdam.py \
    --indices_file ../../data/indices/potsdam_val_image_list.txt \
    --rgb_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/potsdam/val

# Testing
python potsdam.py \
    --indices_file ../../data/indices/potsdam_test_image_list.txt \
    --rgb_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/potsdam/test

Arguments
---------
--indices_file : Path to a text file containing the indices of the patches to be extracted.
--rgb_dir : Path to the directory containing the RGB orthophotos.
--masks_dir : Path to the directory containing the masks.
--output_dir : Path to the directory where the extracted patches will be saved.
--patch_size : Size of each square patch.
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import rasterio
from itertools import product
from rasterio.windows import Window
import numpy as np
import pandas as pd


def encode_mask(mask):
    mask = np.rollaxis(mask, 0, 3)
    new = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)
    mask = mask // 255
    mask = mask * (1, 7, 49)
    mask = mask.sum(axis=2)
    new[mask == 1 + 7 + 49] = 0  # Street.
    new[mask == 49] = 1  # Building.
    new[mask == 7 + 49] = 2  # Grass.
    new[mask == 7] = 3  # Tree.
    new[mask == 1 + 7] = 4  # Car.
    new[mask == 1] = 5  # Clutter.
    return new[np.newaxis, :, :]


def extract_patches(
    indices_file, rgb_dir, dsm_dir, masks_dir, output_dir, patch_size, test
):

    # Convert the input directories to Path objects
    rgb_dir = Path(rgb_dir)
    dsm_dir = Path(dsm_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read the indices file
    indices = pd.read_csv(
        indices_file,
        sep=" ",
        header=None,
        names=["img", "dsm", "mask", "xmin", "ymin", "score"],
    )

    # If in test mode, only process the first 100 indices
    if test:
        indices = indices.head(100)
    for _, row in tqdm(indices.iterrows(), total=len(indices)):

        # Get the image and masks paths
        img_fp = rgb_dir / f"{row['img']}.tif"
        dsm_fp = dsm_dir / f"{row['dsm']}.jpg"
        mask_fp = masks_dir / f"{row['mask']}.tif"

        # Check if the files exist
        assert img_fp.exists(), f"File does not exist: {img_fp}"
        assert dsm_fp.exists(), f"File does not exist: {dsm_fp}"
        assert mask_fp.exists(), f"File does not exist: {mask_fp}"

        # Get the image identifiers
        img_els = img_fp.stem.split("_")
        id0, id1 = img_els[2], img_els[3]
        img_id = f"{id0}_{id1}"

        # Get the indices
        xmin, ymin = int(row["xmin"]), int(row["ymin"])

        # Read the image and mask
        with rasterio.open(img_fp) as img_src, rasterio.open(
            dsm_fp
        ) as dsm_src, rasterio.open(mask_fp) as mask_src:

            # Read the DSM
            dsm_resampled = dsm_src.read(
                out_shape=(img_src.height, img_src.width),
                resampling=rasterio.enums.Resampling.bilinear,
            )[0]

            # .. when processing the validation or the test indices
            if xmin == -1 and ymin == -1:
                stride = 200
                for i, j in product(
                    range(0, img_src.width - stride, stride),
                    range(0, img_src.height - stride, stride),
                ):
                    window = Window.from_slices(
                        (j, j + patch_size), (i, i + patch_size)
                    )
                    rgb = img_src.read(window=window)
                    dsm_patch = dsm_resampled[j : j + patch_size, i : i + patch_size]
                    mask = encode_mask(mask_src.read(window=window))
                    patch = np.vstack([rgb, dsm_patch[np.newaxis, :, :], mask])
                    out_file_path = output_dir / f"{img_id}_{i}_{j}.tif"
                    transform = img_src.window_transform(window)
                    meta = img_src.meta.copy()
                    meta.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "count": patch.shape[0],
                            "dtype": patch.dtype,
                            "transform": transform,
                        }
                    )
                    with rasterio.open(out_file_path, "w", **meta) as dst:
                        dst.write(patch)
            else:
                window = Window.from_slices(
                    (xmin, xmin + patch_size), (ymin, ymin + patch_size)
                )
                rgb = img_src.read(window=window)
                dsm_patch = dsm_resampled[
                    xmin : xmin + patch_size, ymin : ymin + patch_size
                ]
                mask = encode_mask(mask_src.read(window=window))
                patch = np.vstack([rgb, dsm_patch[np.newaxis, :, :], mask])
                out_file_path = output_dir / f"{img_id}_{xmin}_{ymin}.tif"
                transform = img_src.window_transform(window)
                meta = img_src.meta.copy()
                meta.update(
                    {
                        "height": patch_size,
                        "width": patch_size,
                        "count": patch.shape[0],
                        "dtype": patch.dtype,
                        "transform": transform,
                    }
                )
                with rasterio.open(out_file_path, "w", **meta) as dst:
                    dst.write(patch)


def main():
    parser = argparse.ArgumentParser(description="Extract geospatial patches.")
    parser.add_argument(
        "--indices_file", type=str, required=True, help="File containing patch indices."
    )
    parser.add_argument(
        "--rgb_dir",
        type=str,
        help="Directory containing the RGB TIF files to be cropped.",
    )
    parser.add_argument(
        "--dsm_dir",
        type=str,
        help="Directory containing the DSM TIF files to be cropped.",
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        help="Directory containing the masks.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where patches will be saved.",
    )
    parser.add_argument(
        "--patch_size", type=int, default=256, help="Size of each square patch."
    )
    parser.add_argument(
        "--test", action="store_true", help="Run the script in test mode."
    )

    args = parser.parse_args()
    extract_patches(
        args.indices_file,
        args.rgb_dir,
        args.dsm_dir,
        args.masks_dir,
        args.output_dir,
        args.patch_size,
        args.test,
    )


if __name__ == "__main__":
    main()
