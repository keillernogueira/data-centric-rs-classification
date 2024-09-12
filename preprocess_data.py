import os
import argparse
import numpy as np
import pandas as pd

import glob

import cv2
import imageio
import rasterio
from rasterio.enums import Resampling


def normalize_float2uint(patch):
    info = np.iinfo(patch.dtype)
    patch = patch / info.max
    patch = 255 * patch
    return patch.astype(np.uint8)


def _load_target(path, shape=None):
    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.int_]" = f.read(
            out_shape=shape, out_dtype="int32", resampling=Resampling.bilinear
        )
        return array


def _load_image(path, shape=None):
    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.float_]" = f.read(
            out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
        )
        return array


def _encode_mask(msk):
    msk = np.rollaxis(msk, 0, 3)
    new = np.zeros((msk.shape[0], msk.shape[1]), dtype=int)

    msk = msk // 255
    msk = msk * (1, 7, 49)
    msk = msk.sum(axis=2)

    new[msk == 1 + 7 + 49] = 0  # Street.
    new[msk == 49] = 1  # Building.
    new[msk == 7 + 49] = 2  # Grass.
    new[msk == 7] = 3  # Tree.
    new[msk == 1 + 7] = 4  # Car.
    new[msk == 1] = 5  # Surfaces.

    return new


def _decode_mask(msk):
    # sanity check
    new = np.zeros((msk.shape[0], msk.shape[1], 3), dtype=int)

    new[msk == 0] = (255, 255, 255)  # Street.
    new[msk == 1] = (255, 0, 0)  # Building.
    new[msk == 2] = (255, 255, 0) # Grass.
    new[msk == 3] = (0, 255, 0)  # Tree.
    new[msk == 4] = (0, 255, 255)  # Car.
    new[msk == 5] = (0, 0, 255)  # Surfaces.

    return new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=str, required=True,
                        choices=['encode_labels', 'create_patches', 'calculate_max_min'], help="Operation to perform")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to data")
    parser.add_argument("--coord_file", type=str, required=False,
                        help="Path to coordinates file. Used to create the patches.")
    parser.add_argument("--patches_all", type=str, required=False,
                        choices=['all', 'top10', 'bottom10'], help="Create all patches or just for top10 or bottom10")
    parser.add_argument("--patch_size", type=int, required=False, help="Size of the patch.")
    args = parser.parse_args()

    if args.operation == 'encode_labels':
        # python preprocess_data.py --operation encode_labels --root_dir /home/kno/datasets/postdam/Potsdam/masks/
        for file in glob.glob(os.path.join(args.root_dir, "*.tif")):
            mask = _load_target(file)
            mask[mask > 128] = 255
            mask[mask <= 128] = 0
            enc_mask = _encode_mask(mask)
            print(file, mask.shape, enc_mask.shape, np.bincount(enc_mask.flatten()))
            cv2.imwrite(os.path.join(args.root_dir, file[:-4] + '_encoded.png'), enc_mask)
            # sanity check
            # cv2.imwrite(os.path.join(args.root_dir, file[:-4] + '_decoded.png'), _decode_mask(enc_mask))
    elif args.operation == 'create_patches':
        file_list = pd.read_csv(args.coord_file, dtype=None, delimiter=' ', header=None).to_numpy()
        if args.patches_all == 'top10':
            file_list = file_list[:20]
        elif args.patches_all == 'bottom10':
            file_list = file_list[-10:]
        for i, x in enumerate(file_list):
            print('processing...', i)
            # open image
            image = _load_image(os.path.join(args.root_dir, 'images', x[0] + '.tif'))
            dsm = _load_image(os.path.join(args.root_dir, 'dsm', x[1] +
                                           ('.tif' if 'vaihingen' in args.coord_file else '.jpg')),
                              shape=(1, image.shape[1], image.shape[2]))
            mask = _load_target(os.path.join(args.root_dir, 'masks', x[2] + '.tif'))  # '_encoded.png'))
            # print('1', image.shape, dsm.shape, mask.shape)

            # extract patch
            coord_x, coord_y = x[3], x[4]
            image_patch = image[:, coord_x:coord_x + args.patch_size, coord_y:coord_y + args.patch_size]
            dsm_patch = dsm[:, coord_x:coord_x + args.patch_size, coord_y:coord_y + args.patch_size]
            mask_patch = mask[:, coord_x:coord_x + args.patch_size, coord_y:coord_y + args.patch_size]
            # print('2', image_patch.shape, dsm_patch.shape, mask_patch.shape)

            # save
            if args.patches_all == 'all':
                # save all patches
                with rasterio.open(os.path.join(args.root_dir, 'patches', 'images',
                                                x[0] + '_' + str(coord_x) + '_' + str(coord_y) + '.tif'), 'w',
                                   width=args.patch_size, height=args.patch_size,
                                   count=(3 if 'vaihingen' in args.coord_file else 4), dtype='float32') as dst:
                    dst.write(image_patch)
                with rasterio.open(os.path.join(args.root_dir, 'patches', 'dsm',
                                                x[1] + '_' + str(coord_x) + '_' + str(coord_y) + '.tif'), 'w',
                                   width=args.patch_size, height=args.patch_size, count=1, dtype='float32') as dst:
                    dst.write(dsm_patch)
                with rasterio.open(os.path.join(args.root_dir, 'patches', 'masks',
                                                x[2] + '_' + str(coord_x) + '_' + str(coord_y) + '_encoded.png'), 'w',
                                   width=args.patch_size, height=args.patch_size, count=1, dtype='uint8') as dst:
                    dst.write(mask_patch)
            else:
                dsm_patch = cv2.normalize(src=dsm_patch, dst=None, alpha=0, beta=255,
                                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # print(np.min(image_patch), np.max(image_patch), np.min(dsm_patch),
                #       np.max(dsm_patch), np.min(mask_patch), np.max(mask_patch))

                # save few patches just for visualization
                imageio.imwrite(os.path.join(args.root_dir, 'patches', 'images',
                                             x[0] + '_' + str(coord_x) + '_' + str(coord_y) + '_image.png'),
                                np.rollaxis(image_patch, 0, 3).astype(np.uint8))
                imageio.imwrite(os.path.join(args.root_dir, 'patches', 'dsm',
                                             x[0] + '_' + str(coord_x) + '_' + str(coord_y) + '_dsm.png'),
                                np.repeat(np.rollaxis(dsm_patch, 0, 3), 3, 2).astype(np.uint8))
                imageio.imwrite(os.path.join(args.root_dir, 'patches', 'masks',
                                             x[0] + '_' + str(coord_x) + '_' + str(coord_y) + '_label.png'),
                                np.rollaxis(mask_patch, 0, 3).astype(np.uint8))


            # sanity check
            # s_i = _load_image(os.path.join(args.root_dir, 'patches', 'images',
            #                                x[0] + '_' + str(coord_x) + '_' + str(coord_y) + '.tif'))
            # s_d = _load_image(os.path.join(args.root_dir, 'patches', 'dsm',
            #                                x[1] + '_' + str(coord_x) + '_' + str(coord_y) + '.tif'))
            # s_m = _load_target(os.path.join(args.root_dir, 'patches', 'masks',
            #                                 x[2] + '_' + str(coord_x) + '_' + str(coord_y) + '_encoded.png'))
            # print('3', s_i.shape, s_d.shape, s_m.shape)
            # print('f', (image_patch == s_i).all(), (dsm_patch == s_d).all(), (mask_patch == s_m).all())
    elif args.operation == 'calculate_max_min':
        # file_list = glob.glob(os.path.join(args.root_dir, "*_normalized_lastools.jpg"), recursive=True)
        file_list = glob.glob(os.path.join(args.root_dir, "*.tif"), recursive=True)
        print(file_list)
        min = []
        max = []
        for i, x in enumerate(file_list):
            dsm = _load_image(x, shape=(1, 6000, 6000))
            min.append(np.min(dsm))
            max.append(np.max(dsm))
        print(np.min(min), min)
        print(np.max(max), max)
    else:
        raise NotImplementedError
