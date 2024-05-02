import os
import argparse
import random

import numpy as np
import glob
import rasterio
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from rasterio.enums import Resampling

random.seed(10)


def _load_image(path, shape=None):
    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.float_]" = f.read(
            out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
        )
        return array


def _load_files(dataset, root, split='labeled_train'):
    if dataset == 'DFC2022':
        files = glob.glob(os.path.join(root, split, "**", 'UrbanAtlas', "*.tif"), recursive=True)
    elif dataset == 'vaihingen':
        images = sorted(glob.glob(os.path.join(root, 'images', "*.tif")), key=lambda x: (int(x[:-4].split('_')[-1][4:]), x))
        ndsm = sorted(glob.glob(os.path.join(root, 'ndsm', "*.jpg")), key=lambda x: (int(x.split('_')[-2][4:]), x))
        masks = sorted(glob.glob(os.path.join(root, 'masks', "*.tif")), key=lambda x: (int(x[:-4].split('_')[-1][4:]), x))
        files = np.vstack((np.asarray(images), np.asarray(ndsm), np.asarray(masks))).T  # shape = x, 3
    else:  # potsdam
        images = sorted(glob.glob(os.path.join(root, 'images', "*.tif")),
                        key=lambda x: (int(x.split('/')[-1][12:-10]), x))
        ndsm = sorted(glob.glob(os.path.join(root, 'ndsm', "*_normalized_lastools.jpg")),
                      key=lambda x: (int(x.split('/')[-1][12:-24].replace("07", "7").replace("08", "8").replace("09", "9")), x))
        masks = sorted(glob.glob(os.path.join(root, 'masks', "*.tif")),
                       key=lambda x: (int(x.split('/')[-1][12:-10]), x))
        files = np.vstack((np.asarray(images), np.asarray(ndsm), np.asarray(masks))).T  # shape = x, 3

    return files


def plot_example(img_path, crop_size, stride_size):
    img_path = img_path.replace('UrbanAtlas', 'BDORTHO')
    img_path = img_path.replace('_UA2012', '')

    colors = ["red", "blue", "yellow"]
    color_counter = 0

    source_img = Image.open(img_path)
    draw = ImageDraw.Draw(source_img)

    img = _load_image(img_path)
    _, h, w = img.shape
    for i in range(0, h, stride_size):
        for j in range(0, w, stride_size):
            cur_x = i
            cur_y = j
            patch = img[0, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

            if len(patch) != crop_size and len(patch[0]) != crop_size:
                cur_x = cur_x - (crop_size - len(patch))
                cur_y = cur_y - (crop_size - len(patch[0]))
            elif len(patch) != crop_size:
                cur_x = cur_x - (crop_size - len(patch))
            elif len(patch[0]) != crop_size:
                cur_y = cur_y - (crop_size - len(patch[0]))
            patch = img[0, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

            assert patch.shape == (crop_size, crop_size), \
                "Error create_distrib: Current patch size is " + str(len(patch[0])) + "x" + str(len(patch[0][0]))

            draw.rectangle(((int(cur_x), int(cur_y)), (int(cur_x + crop_size), int(cur_y + crop_size))),
                           outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=3)
            color_counter += 1

    source_img.save("example.png", "PNG")


def create_coord_distribution(files, dataset, crop_size, stride_size):
    instances = []

    for x, data in enumerate(files):
        if dataset == 'DFC2022':
            img_path = data
        else:
            img_path = data[0]
            ndsm_path = data[1]
            mask_path = data[2]

        img = _load_image(img_path)
        # print('shape', img_path, img.shape)
        _, h, w = img.shape
        for i in range(0, h, stride_size):
            for j in range(0, w, stride_size):
                if dataset == 'DFC2022':
                    cur_map = img_path.split('/')[6]
                    cur_img = img_path.split('/')[-1].replace('_UA2012.tif', '')
                else:
                    cur_map = data[0][:-4].split('/')[-1]
                    ndsm_path = data[1][:-4].split('/')[-1]
                    mask_path = data[2][:-4].split('/')[-1]

                cur_x = i
                cur_y = j
                patch = img[0, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch) != crop_size and len(patch[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch))
                    cur_y = cur_y - (crop_size - len(patch[0]))
                elif len(patch) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch))
                elif len(patch[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch[0]))
                patch = img[0, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                assert patch.shape == (crop_size, crop_size), \
                    "Error create_distrib: Current patch size is " + str(len(patch[0])) + "x" + str(len(patch[0][0]))

                if dataset == 'DFC2022':
                    instances.append((cur_map, cur_img, cur_x, cur_y, 1.0))
                else:
                    instances.append((cur_map, ndsm_path, mask_path, cur_x, cur_y, 1.0))

    return np.asarray(instances)


def prepare_val(files, dataset):
    instances = []
    for x, data in enumerate(files):
        if dataset == 'DFC2022':
            cur_map = data.split('/')[6]
            cur_img = data.split('/')[-1].replace('_UA2012.tif', '')
            instances.append((cur_map, cur_img, -1, -1, 1.0))
        else:
            cur_map = data[0][:-4].split('/')[-1]
            ndsm_path = data[1][:-4].split('/')[-1]
            mask_path = data[2][:-4].split('/')[-1]
            instances.append((cur_map, ndsm_path, mask_path, -1, -1, 1.0))

    return np.asarray(instances)


def remove_duplicates(file_list):
    uni_set = set()
    all = []
    for x in file_list:
        uni_set.add('#'.join(x))
        all.append('#'.join(x))
    print(len(file_list), len(uni_set))  # should be equal

    from collections import Counter
    duplicates = [(k, v) for k, v in Counter(all).items() if v > 1]
    print(duplicates, len(duplicates))

    instances = []
    for x, img_join in enumerate(uni_set):
        instances.append((img_join.split('#')))

    return instances


# python generate_coord.py --root /datasets/df2022/ --patch_size 256 --stride_size 200 --val_split_pcr 0.1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset",
                        choices=['DFC2022', 'vaihingen', 'potsdam'])
    parser.add_argument("--root", type=str, required=True, help="Path to the data root")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size")
    parser.add_argument("--stride_size", type=int, required=True, help="Stride size")
    parser.add_argument("--val_split_pcr", type=float, required=False, help="Validation percentage")
    args = parser.parse_args()
    print(args)

    vaihingen_valid_images = ['11', '15', '28', '30', '34']
    potsdam_valid_images = ['2_12', '3_12', '4_12', '5_12', '6_12', '7_12']

    all_files = _load_files(args.dataset, args.root, split='labeled_train')  # get all training file path

    if args.dataset == 'DFC2022':
        assert args.val_split_pcr is not None
        # divide the path into train (90%)...
        subset_train_files = random.sample(all_files, int(len(all_files) * (1 - args.val_split_pcr)))
        # ... and validation (10%)
        subset_val_files = list(set(all_files) - set(subset_train_files))
    else:
        subset_val_files = []
        remove_indexes = []
        for val_img in (vaihingen_valid_images if args.dataset == 'vaihingen' else potsdam_valid_images):
            for i, img in enumerate(all_files):
                if val_img in img[0]:  # or val_img in img[1] or val_img in img[2]:
                    subset_val_files.append(img)
                    remove_indexes.append(i)
                    break
        subset_train_files = np.delete(all_files, remove_indexes, 0)

    print(len(all_files), len(subset_train_files), len(subset_val_files))
    # print('subset_train_files', len(subset_train_files))
    print('subset_val_files', subset_val_files, len(subset_val_files))

    # create a list of images and coordinates
    train_instances = create_coord_distribution(subset_train_files, args.dataset, args.patch_size, args.stride_size)
    train_instances = remove_duplicates(train_instances)
    print(len(train_instances))
    np.savetxt(os.path.join('coord_files', args.dataset + '_train_coordinate_list.txt'),
               train_instances, fmt='%s', delimiter=' ')

    val_instances = prepare_val(subset_val_files, args.dataset)
    np.savetxt(os.path.join('coord_files', args.dataset + '_val_image_list.txt'),
               val_instances, fmt='%s', delimiter=' ')

    # plot example
    # plot_example(subset_train_files[2], args.patch_size, args.stride_size)
