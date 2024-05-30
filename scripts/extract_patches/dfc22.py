"""
Script to Extract Geospatial Patches from Satellite Imagery
===========================================================

This script is designed to extract patches of a specified size from larger geospatial datasets.
It reads from three different types of raster files for each patch:

1. RGB orthophotos (BDORTHO)
2. Digital Elevation Model (DEM) (RGEALTI)
3. Urban Atlas landcover data (UrbanAtlas)

Each patch contains these three types of data stacked together.
The extracted patches are saved in GeoTIFF format with updated geospatial metadata.

Usage
-----
You can run this script from the command line:

# Patchify!
python dfc22.py \
    --indices_file ../../data/indices/dfc2022_train_val_test.csv \
    --source_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/raw/dfc22 \
    --output_dir /home/akramzaytar/ssdprivate/akramz_datasets/wtl/dfc22

Arguments
---------
--indices_file : str The file containing patch indices information.
--source_dir : str The root directory containing the source raster files organized by city.
--output_dir : str The directory where the output patches will be saved.
--patch_size : int The size of each square patch.
"""

from pathlib import Path
import argparse
from tqdm import tqdm
import os
import rasterio
from itertools import product
from rasterio.windows import Window
import numpy as np
import pandas as pd


def extract_patches(indices_file, source_dir, output_dir, patch_size, test):

    # Turn the directories into paths
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # read the indices file
    indices = pd.read_csv(indices_file)

    # if test mode, we sample 100 files
    if test:
        indices = indices.sample(100)

    # Init a missing list
    missing = list()

    for _, row in tqdm(indices.iterrows()):

        # Get the row values
        city, img_name, split, xmin, ymin = (
            row["city"],
            row["tile"],
            row["split"],
            int(row["xmin"]),
            int(row["ymin"]),
        )

        # Get the paths to the 3 raster files (RGB, DEM, and Mask)
        bdortho_path = (
            source_dir
            / split.replace("val", "train")
            / city
            / "BDORTHO"
            / f"{img_name}.tif"
        )
        rgealti_path = (
            source_dir
            / split.replace("val", "train")
            / city
            / "RGEALTI"
            / f"{img_name}_RGEALTI.tif"
        )
        urbanatlas_path = (
            source_dir
            / split.replace("val", "train")
            / city
            / "UrbanAtlas"
            / f"{img_name}_UA2012.tif"
        )

        # # Check if the files are missing one by one
        # assert bdortho_path.exists(), f"Missing {bdortho_path}"
        # assert rgealti_path.exists(), f"Missing {rgealti_path}"
        # assert urbanatlas_path.exists(), f"Missing {urbanatlas_path}"

        # if any of the files is missing, save it in `missing` then skip this image
        if (
            not bdortho_path.exists()
            or not rgealti_path.exists()
            or not urbanatlas_path.exists()
        ):
            missing.append((split, city, img_name))
            continue

        with rasterio.open(bdortho_path) as bdortho_src, rasterio.open(
            rgealti_path
        ) as rgealti_src, rasterio.open(urbanatlas_path) as urbanatlas_src:
            dem_resampled = rgealti_src.read(
                out_shape=(bdortho_src.height, bdortho_src.width),
                resampling=rasterio.enums.Resampling.bilinear,
            )[0]

            if (
                xmin == -1 and ymin == -1
            ):  # .. when processing the validation or the test indices
                stride = 200
                for i, j in product(
                    range(0, bdortho_src.width - stride, stride),
                    range(0, bdortho_src.height - stride, stride),
                ):
                    window = Window.from_slices(
                        (j, j + patch_size), (i, i + patch_size)
                    )
                    rgb = bdortho_src.read(window=window)
                    dem_patch = dem_resampled[j : j + patch_size, i : i + patch_size]
                    mask = urbanatlas_src.read(window=window)
                    patch = np.vstack([rgb, dem_patch[np.newaxis, :, :], mask])
                    out_file_path = (
                        output_dir / split / f"{city}_{img_name}_{i}_{j}.tif"
                    )
                    transform = bdortho_src.window_transform(window)
                    meta = bdortho_src.meta.copy()
                    meta.update(
                        {
                            "height": patch_size,
                            "width": patch_size,
                            "count": patch.shape[0],
                            "dtype": patch.dtype,
                            "transform": transform,
                        }
                    )
                    out_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with rasterio.open(out_file_path, "w", **meta) as dst:
                        dst.write(patch)
            else:
                window = Window.from_slices(
                    (xmin, xmin + patch_size), (ymin, ymin + patch_size)
                )
                rgb = bdortho_src.read(window=window)
                dem_patch = dem_resampled[
                    xmin : xmin + patch_size, ymin : ymin + patch_size
                ]
                mask = urbanatlas_src.read(window=window)
                patch = np.vstack([rgb, dem_patch[np.newaxis, :, :], mask])
                out_file_path = (
                    output_dir / split / f"{city}_{img_name}_{xmin}_{ymin}.tif"
                )
                transform = bdortho_src.window_transform(window)
                meta = bdortho_src.meta.copy()
                meta.update(
                    {
                        "height": patch_size,
                        "width": patch_size,
                        "count": patch.shape[0],
                        "dtype": patch.dtype,
                        "transform": transform,
                    }
                )
                out_file_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_file_path, "w", **meta) as dst:
                    dst.write(patch)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Extract geospatial patches.")
    parser.add_argument(
        "--indices_file", type=str, required=True, help="File containing patch indices."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing the original TIF files to be cropped.",
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
    missing = extract_patches(
        args.indices_file, args.source_dir, args.output_dir, args.patch_size, args.test
    )

    # Save the missing list as a text file in the output directory
    with open(os.path.join(args.output_dir, "missing.txt"), "w") as f:
        for split, city, img_name in missing:
            f.write(f"{split}, {city}, {img_name}\n")


if __name__ == "__main__":
    main()
