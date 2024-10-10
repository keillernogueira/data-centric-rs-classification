import os
import glob
import warnings
from collections import defaultdict

import cv2
import rasterio
import imageio
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from rasterio.enums import Resampling


# ignore NotGeoreferencedWarning
warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)


def _load_image(path, shape=None):
    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.float_]" = f.read(
            out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
        )
        return array


def _load_target(path):
    with rasterio.open(path) as f:
        array: "np.typing.NDArray[np.int_]" = f.read(
            out_dtype="int32", resampling=Resampling.bilinear
        )
        return array


def get_vaihingen_kvs(files=None, metric='probs'):
    key_vals = defaultdict(list)
    for fn in os.listdir("vaihingen/"):
        if files is None or (files is not None and fn in files):
            df = pd.read_csv(f"vaihingen/{fn}", delimiter=" ", header=None)
            df = df.sort_values(5, ascending=False)
            df["key"] = df.apply(lambda row: f"{row[0]}_{row[3]}_{row[4]}", axis=1)
            keys = df["key"].values
            if metric == 'probs':
                data = df[5].values
                data = data / data.max()
            else:
                data = df.index.values.astype(int)
                # print(data)
            for key, val in zip(keys, data):
                key_vals[key].append(val)
    return key_vals


def get_potsdam_kvs(files=None, metric='probs'):
    key_vals = defaultdict(list)
    for fn in os.listdir("potsdam/"):
        if files is None or (files is not None and fn in files):
            df = pd.read_csv(f"potsdam/{fn}", delimiter=" ", header=None)
            df = df.sort_values(5, ascending=False)
            df["key"] = df.apply(lambda row: f"{row[0]}_{row[3]}_{row[4]}", axis=1)
            keys = df["key"].values
            if metric == 'probs':
                data = df[5].values
                data = data / data.max()
            else:
                data = df.index.values.astype(int)
                # print(data)
            for key, val in zip(keys, data):
                key_vals[key].append(val)
    return key_vals


def get_dfc2022_kvs(files=None, metric='probs'):
    key_vals = defaultdict(list)
    for fn in os.listdir("dfc2022"):
        if files is None or (files is not None and fn in files):
            df = pd.read_csv(f"dfc2022/{fn}", delimiter=" ", header=None)
            df = df.sort_values(4, ascending=False)
            df["key"] = df.apply(lambda row: f"{row[0]}_{row[1]}_{row[2]}_{row[3]}", axis=1)
            keys = df["key"].values
            if metric == 'probs':
                data = df[4].values
                data = data / data.max()
            else:
                data = df.index.values.astype(int)
                # print(data)
            for key, val in zip(keys, data):
                key_vals[key].append(val)
    return key_vals


def get_results(key_vals):
    keys = []
    mean_vals = []
    std_vals = []
    # ind_methods = [[] for _ in range(6)]

    for key, vals in key_vals.items():
        keys.append(key)
        mean_vals.append(np.mean(vals))
        std_vals.append(np.std(vals))
        # for i in range(len(vals)):
        #     ind_methods[i].append(vals[i])

    keys = np.array(keys)
    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)
    # ind_methods = np.array(ind_methods)

    idxs = np.argsort(mean_vals)[::-1]
    keys = keys[idxs]
    mean_vals = mean_vals[idxs]
    std_vals = std_vals[idxs]
    # for i in range(6):
    #     ind_methods[i] = ind_methods[i][idxs]

    return keys, mean_vals, std_vals  # , ind_methods


def plot_multiple_lines(vaihingen_xs, vaihingen_mean_vals, potsdam_xs, potsdam_mean_vals,
                        dfc2022_xs, dfc2022_mean_vals):
    plt.figure(figsize=(10, 6))

    plt.plot(vaihingen_xs, vaihingen_mean_vals, label="Vaihingen")
    #plt.fill_between(vaihingen_xs, vaihingen_mean_vals-vaihingen_std_vals, vaihingen_mean_vals+vaihingen_std_vals, alpha=0.5)

    plt.plot(potsdam_xs, potsdam_mean_vals, label="Potsdam")
    #plt.fill_between(potsdam_xs, potsdam_mean_vals-potsdam_std_vals, potsdam_mean_vals+potsdam_std_vals, alpha=0.5)

    plt.plot(dfc2022_xs, dfc2022_mean_vals, label="DFC2022")
    #plt.fill_between(dfc2022_xs, dfc2022_mean_vals-dfc2022_std_vals, dfc2022_mean_vals+dfc2022_std_vals, alpha=0.5)

    plt.legend(fontsize=15)

    plt.xlabel("Patch index", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale("linear")
    plt.title("Vaihingen", fontsize=15)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    plt.close()


def smooth(xs, ys, num=250):
    spl = make_interp_spline(xs, ys, k=3)
    xnew = np.linspace(xs[0], xs[-1], num)
    power_smooth = spl(xnew)
    return xnew, power_smooth


def plot_line_results(xs, mean_vals, std_vals, title, smooth_curve=False, num=250):
    plt.figure(figsize=(10, 6), layout='compressed')

    if smooth_curve:
        _, mean_vals = smooth(xs, mean_vals, num)
        xs, std_vals = smooth(xs, std_vals, num)

    plt.plot(xs, mean_vals, label=title)
    # np.clip(mean_vals-std_vals, a_min=0, a_max=1), np.clip(mean_vals+std_vals, a_min=0, a_max=1)
    plt.fill_between(xs, mean_vals-std_vals, mean_vals+std_vals, alpha=0.5)
    # print(min(mean_vals-vaihingen_std_vals), max(mean_vals-vaihingen_std_vals))
    # print(min(mean_vals+vaihingen_std_vals), max(mean_vals+vaihingen_std_vals))

    plt.xlabel("Patch index", fontsize=24)
    plt.ylabel("Ranking Position", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.xscale('symlog', linthresh=1000)
    plt.xscale('linear')

    # plt.title(title, fontsize=15)
    ax = plt.gca()
    if title == "Vaihingen":
        plt.xticks([1, 200, 400, 600, 800, 1000, 1200, 1400])
        plt.yticks([1, 200, 400, 600, 800, 1000, 1200, 1400])
    elif title == "Potsdam":
        plt.xticks([1, 5000, 10000, 15000, 20000, 25000])
        plt.yticks([1, 5000, 10000, 15000, 20000, 25000])
    else:
        plt.xticks([1, 10000, 20000, 30000, 40000, 50000, 60000])
        plt.yticks([1, 10000, 20000, 30000, 40000, 50000, 60000])
    ax.set_xlim([0, xs[-1]])
    # ax.set_ylim([0, 1])
    ax.set_ylim([max(xs), 0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(title + "_patches_smooth_" + str(num) + ".png")  # dpi=1000
    plt.show()
    # plt.draw()
    # plt.close()


def plot_patches_v1(vaihingen_keys):
    for key in vaihingen_keys[:4]:
        with rasterio.open(f"/home/mveo/keiller_format/vaihingen/patches/images/{key}.tif") as f:
            img = f.read().transpose(1, 2, 0).astype(np.uint8)

        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        plt.close()

    for key in vaihingen_keys[-4:]:
        with rasterio.open(f"/home/mveo/keiller_format/vaihingen/patches/images/{key}.tif") as f:
            img = f.read().transpose(1, 2, 0).astype(np.uint8)

        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        plt.close()


def process_one_patch(image_name, path, dataset, save_name, patch_size):
    dataset_subpath = {'vaihingen': ["top", "dsm", "gts_for_participants"],
                       'potsdam': ["2_Ortho_RGB", "1_DSM_normalisation", "5_Labels_all"],  # "4_Ortho_RGBIR"
                       'dfc2022': ["BDORTHO",  "RGEALTI", "UrbanAtlas_colored"]
                       }

    print(image_name)
    vec = image_name.split("_")
    coord_x, coord_y = int(vec[-2]), int(vec[-1])

    if dataset == 'potsdam':
        area = '_'.join(vec[2:4])
        dsm_area = vec[2] + "_" + (vec[3] if len(vec[3]) == 2 else '0' + vec[3])
        image_path = os.path.join(path, dataset_subpath[dataset][0], 'top_potsdam_' + str(area) + '_RGB.tif')  # '_RGBIR.tif')
        label_path = os.path.join(path, dataset_subpath[dataset][2], 'top_potsdam_' + str(area) + '_label.tif')
        dsm_name = os.path.join(path, dataset_subpath[dataset][1],
                                'dsm_potsdam_0' + str(dsm_area) + '_normalized_lastools.jpg')
    elif dataset == 'vaihingen':
        area = vec[3]
        image_path = os.path.join(path, dataset_subpath[dataset][0], 'top_mosaic_09cm_' + str(area) + '.tif')
        label_path = os.path.join(path, dataset_subpath[dataset][2], 'top_mosaic_09cm_' + str(area) + '.tif')
        dsm_name = os.path.join(path, dataset_subpath[dataset][1], 'dsm_09cm_matching_' + str(area) + '.tif')
    else:  # dfc
        if vec[0] == 'Nice':
            region = vec[0]
            img_name = vec[1]
        else:
            region = '_'.join(vec[:2])
            img_name = vec[2]

        image_path = os.path.join(path, region, dataset_subpath[dataset][0], img_name + '.tif')
        label_path = os.path.join(path, region, dataset_subpath[dataset][2], img_name + '_UA2012.png')
        dsm_name = os.path.join(path, region, dataset_subpath[dataset][1], img_name + '_RGEALTI.tif')

    image = _load_image(image_path)
    image_patch = image[:, coord_x:coord_x + patch_size, coord_y:coord_y + patch_size]
    # print(image.shape, image_patch.shape)
    imageio.imwrite(save_name + '_image.png',
                    np.rollaxis(image_patch, 0, 3).astype(np.uint8))

    mask = _load_target(label_path)
    mask_patch = mask[:, coord_x:coord_x + patch_size, coord_y:coord_y + patch_size]
    # print(mask.shape, mask_patch.shape)
    imageio.imwrite(save_name + '_label.png',
                    np.rollaxis(mask_patch, 0, 3).astype(np.uint8))

    dsm = _load_image(dsm_name, shape=image.shape[1:])
    # if dataset != 'potsdam':
    #     dsm = cv2.normalize(src=dsm, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dsm_patch = dsm[:, coord_x:coord_x + patch_size, coord_y:coord_y + patch_size]
    if dataset != 'potsdam':
        dsm_patch = cv2.normalize(src=dsm_patch, dst=None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    imageio.imwrite(save_name + '_dsm.png',
                    np.repeat(np.rollaxis(dsm_patch, 0, 3), 3, 2).astype(np.uint8))


def save_patches(path, dataset_keys, dataset, save_path, patch_size=256, top=10):
    # top
    for i in range(top):
        process_one_patch(dataset_keys[i], path, dataset,
                          os.path.join(save_path, dataset, 'top_' + str(i+1) + '_' + dataset_keys[i]), patch_size)

    # bottom
    for i in range(-1, -(top+1), -1):
        process_one_patch(dataset_keys[i], path, dataset,
                          os.path.join(save_path, dataset, 'bot_' + str(-i) + '_' + dataset_keys[i]), patch_size)


def dfc22_encode_labels(path):
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

    images = glob.glob(os.path.join(path, 'UrbanAtlas', "*.tif"), recursive=True)

    for image in sorted(images):
        print(image)
        with rasterio.open(image) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                indexes=1, out_dtype="int32", resampling=Resampling.bilinear
            )
        # print(type(array), array.shape, np.bincount(array.flatten()))

        h, w = array.shape
        color_array = np.empty((h, w, 3), dtype=np.uint8)
        for j in range(len(colormap)):
            color_array[np.where(array == j)] = tuple(int(colormap[j].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

        # print('save_path', image.replace('UrbanAtlas', 'UrbanAtlas_colored').replace('tif', 'png'))
        imageio.imwrite(image.replace('UrbanAtlas', 'UrbanAtlas_colored').replace('tif', 'png'), color_array)


def plot_multiple_line_results(xs, ind_methods, mean_vals, std_vals, title, smooth_curve=False, num=250):
    plt.figure(figsize=(10, 6), layout='compressed')

    out_ind = [[] for _ in range(6)]
    if smooth_curve:
        for i in range(6):
            _, out_ind[i] = smooth(xs, ind_methods[i], num)
        _, mean_vals = smooth(xs, mean_vals, num)
        xs, std_vals = smooth(xs, std_vals, num)

    for i in range(6):
        plt.plot(xs, out_ind[i], label="Vaihingen")

    plt.plot(xs, mean_vals, label=title)
    plt.fill_between(xs, np.clip(mean_vals-std_vals, a_min=0, a_max=1),
                     np.clip(mean_vals+std_vals, a_min=0, a_max=1), alpha=0.5)
    # print(min(mean_vals-vaihingen_std_vals), max(mean_vals-vaihingen_std_vals))
    # print(min(mean_vals+vaihingen_std_vals), max(mean_vals+vaihingen_std_vals))

    plt.xlabel("Patch index", fontsize=24)
    plt.ylabel("Score", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # plt.xscale('symlog', linthresh=1000)
    plt.xscale('linear')

    # plt.title(title, fontsize=15)
    ax = plt.gca()
    ax.set_xlim([0, xs[-1]])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(title + "_patches.png")  # dpi=1000
    plt.show()
    plt.draw()
    # plt.close()


def plot_correlation(file_file, dataset, plot_colorbar=False):
    data_dict = {'ai4gg_complexity': 'LC', 'ai4gg_diversity': 'FD', 'ai4gg_hybrid': 'LC/FD',
                 dataset + '_train_coordinate_list_method1': 'FA', dataset + '_train_coordinate_list_method2': 'CB',
                 dataset + '_train_coordinate_list_hybrid': 'FA/CB'}
    df = pd.read_csv(file_file, index_col=0)

    cols = data_dict.keys()  # sorted(df.columns)

    pretty_cols = [data_dict[val] for val in cols]
    mat = np.zeros((len(cols), len(cols)), dtype=np.float32)

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            # compute rank correlation using scipy
            mat[i, j] = scipy.stats.kendalltau(df[col1], df[col2])[0]

    plt.figure()
    plt.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(np.arange(len(cols)), pretty_cols, rotation=90)
    plt.yticks(np.arange(len(cols)), pretty_cols)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for i in range(len(cols)):
        for j in range(len(cols)):
            c = "black" if mat[i, j] < 0.7 else "white"
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=c, fontsize=12)
    # plt.title("DFC2022", fontsize=15)
    if plot_colorbar:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
    plt.show()
    plt.close()


if __name__ == "__main__":
    operation = 'correlation'

    if operation == 'encoded_labels':
        # create colored maps for visualization
        dfc22_encode_labels('/mnt/DADOS_PONTOISE_1/keiller/datasets/df2022/labeled_train/Nantes_Saint-Nazaire/')
        dfc22_encode_labels('/mnt/DADOS_PONTOISE_1/keiller/datasets/df2022/labeled_train/Nice/')
    elif operation in ['line_plot', 'save_patches', 'multiple_line']:
        feat_based = ['vaihingen_train_coordinate_list_method1.txt', 'ai4gg_diversity.txt']
        label_based = ['vaihingen_train_coordinate_list_method2.txt', 'ai4gg_complexity.txt']

        vaihingen_kvs = get_vaihingen_kvs(metric='index')
        potsdam_kvs = get_potsdam_kvs(metric='index')
        dfc2022_kvs = get_dfc2022_kvs(metric='index')
        print(len(vaihingen_kvs), len(potsdam_kvs), len(dfc2022_kvs))

        vaihingen_keys, vaihingen_mean_vals, vaihingen_std_vals = get_results(vaihingen_kvs)
        potsdam_keys, potsdam_mean_vals, potsdam_std_vals = get_results(potsdam_kvs)
        dfc2022_keys, dfc2022_mean_vals, dfc2022_std_vals = get_results(dfc2022_kvs)
        print(min(vaihingen_mean_vals), max(vaihingen_mean_vals), min(vaihingen_std_vals), max(vaihingen_std_vals))
        print(min(potsdam_mean_vals), max(potsdam_mean_vals), min(potsdam_std_vals), max(potsdam_std_vals))
        print(min(dfc2022_mean_vals), max(dfc2022_mean_vals), min(dfc2022_std_vals), max(dfc2022_std_vals))

        # plot_multiple_line_results(vaihingen_xs, vaihingen_ind, vaihingen_mean_vals, vaihingen_std_vals,
        #                            title="Vaihingen", smooth_curve=True, num=20)
        if operation == 'save_patches':
            save_patches('/mnt/DADOS_PONTOISE_1/keiller/datasets/vaihingen/Vaihingen/OFFICIAL_DATASET/', vaihingen_keys,
                         'vaihingen', '/mnt/DADOS_PONTOISE_1/keiller/data-centric-rs-classification/patches_label/',
                         patch_size=256, top=10)
            save_patches('/mnt/DADOS_PONTOISE_1/keiller/datasets/potsdam/Potsdam/', potsdam_keys,
                         'potsdam', '/mnt/DADOS_PONTOISE_1/keiller/data-centric-rs-classification/patches_label/',
                         patch_size=256, top=10)
            save_patches('/mnt/DADOS_PONTOISE_1/keiller/datasets/df2022/labeled_train/', dfc2022_keys,
                         'dfc2022', '/mnt/DADOS_PONTOISE_1/keiller/data-centric-rs-classification/patches_label/',
                         patch_size=256, top=10)
        elif operation == 'multiple_line':
            vaihingen_xs = np.linspace(0, 1, len(vaihingen_keys))
            potsdam_xs = np.linspace(0, 1, len(potsdam_keys))
            dfc2022_xs = np.linspace(0, 1, len(dfc2022_keys))
            plot_multiple_lines(vaihingen_xs, vaihingen_mean_vals, potsdam_xs, potsdam_mean_vals,
                                dfc2022_xs, dfc2022_mean_vals)
        elif operation == 'line_plot':
            # used in the paper
            vaihingen_xs = np.linspace(1, len(vaihingen_keys), len(vaihingen_keys))
            potsdam_xs = np.linspace(1, len(potsdam_keys), len(potsdam_keys))
            dfc2022_xs = np.linspace(1, len(dfc2022_keys), len(dfc2022_keys))
            plot_line_results(vaihingen_xs, vaihingen_mean_vals, vaihingen_std_vals, title="Vaihingen", smooth_curve=False, num=20)
            plot_line_results(potsdam_xs, potsdam_mean_vals, potsdam_std_vals, "Potsdam", smooth_curve=False, num=20)
            plot_line_results(dfc2022_xs, dfc2022_mean_vals, dfc2022_std_vals, "DFC2022", smooth_curve=False, num=20)
    elif operation == 'correlation':
        plot_correlation('dfc2022.csv', 'dfc2022')
        plot_correlation('vaihingen.csv', 'vaihingen')
        plot_correlation('potsdam.csv', 'potsdam', True)
    else:
        raise NotImplementedError
