# MVEO Benchmarks

## Setup

First, we create an environment as follows:

```bash
mamba create -n mveo python=3.12.3
conda activate mveo
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # [Optional] (a fix to an internal driver issue)
pip install torchgeo
```

Also, install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt

# .. and the package
pip install -e .
```

Create the directories structure where raw and prepared data will be stored:
```
root/
├── raw/
│   ├── dfc22/
│   ├── vaihingen/
│   └── potsdam/
├── dfc22/
│   ├── train/
│   ├── val/
│   └── test/
├── vaihingen/
│   ├── train/
│   ├── val/
│   └── test/
└── potsdam/
    ├── train/
    ├── val/
    └── test/
```

## Data Acquisition

### DFC-22

`cd` to the `raw` directory and download/extract the data as follows:

```bash
curl -L -o train.zip "https://ieee-dataport.s3.amazonaws.com/competition/21720/labeled_train.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240529%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240529T092856Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=b3ed310dce2c4e8d1a9abf7a310a6d24f7da7332c0e4b9c77e9fad6c76e81c7e"
unzip train.zip
mv labeled_train/ train/

curl -L -o val.zip "https://ieee-dataport.s3.amazonaws.com/competition/21720/val.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240529%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240529T092856Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=6a6ac8570d854129344f483e6e45f56ffafa5f3809029196cb6e0a0900af93f9"
unzip val.zip

curl -L -o test.zip "https://ieee-dataport.s3.amazonaws.com/competition/21720/test.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20240529%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240529T092856Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=20bb7781b634fd29a516888c8ad348811e9d648a1334a4d0cd3f51673ae96cd3"
unzip test.zip
mkdir -p test; unzip test.zip -d test
```

### Potsdam

Visit this [link](https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/) and use this password to download: `CjwcipT4-P8g`.

You can extract the ZIP file:
```bash
unzip Potsdam.zip
```

Then extract all compressed files in `Potsdam`:

```bash
cd scripts/data_preparation
chmod +x extract_files.sh
./extract_files.sh raw/potsdam/Potsdam
```

### Vaihingen

Visit this [link](https://seafile.projekt.uni-hannover.de/f/6a06a837b1f349cfa749/) and use this password to download: `CjwcipT4-P8g`.

You can extract the ZIP file:
```bash
unzip Vaihingen.zip
```

Then extract all compressed files in `Vaihingen`:

```bash
cd scripts/data_preparation
chmod +x extract_files.sh
./extract_files.sh raw/vaihingen/Vaihingen
```

## Patch Extraction

### DFC-22

*Note: you need to epxort the test masks to `{root}/raw/dfc22/test/{city}/UrbanAtlas` before running the script.*

Go to the scripts directory:

```bash
cd scripts/extract_patches/
```

To export the train, validation, and test patches, run the following:

```bash
python dfc22.py \
    --indices_file ../../data/indices/dfc2022_train_val_test.csv \
    --source_dir {root}/raw/dfc22 \
    --output_dir {root}/dfc22
```

### Potsdam

```bash
# Train
python potsdam.py \
    --indices_file ../../data/indices/potsdam_train_coordinate_list.txt \
    --rgb_dir {root}/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir {root}/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir {root}/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir {root}/potsdam/train

# Validation
python potsdam.py \
    --indices_file ../../data/indices/potsdam_val_image_list.txt \
    --rgb_dir {root}/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir {root}/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir {root}/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir {root}/potsdam/val

# Testing
python potsdam.py \
    --indices_file ../../data/indices/potsdam_test_image_list.txt \
    --rgb_dir {root}/raw/potsdam/Potsdam/4_Ortho_RGBIR/4_Ortho_RGBIR \
    --dsm_dir {root}/raw/potsdam/Potsdam/1_DSM_normalisation/1_DSM_normalisation \
    --masks_dir {root}/raw/potsdam/Potsdam/5_Labels_all \
    --output_dir {root}/potsdam/test
```

### Vaihingen

```bash
# Train!
python vaihingen.py \
    --indices_file ../../data/indices/vaihingen_train_coordinate_list.txt \
    --rgb_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top \
    --dsm_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm \
    --masks_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ \
    --output_dir {root}/vaihingen/train

# Validation!
python vaihingen.py \
    --indices_file ../../data/indices/vaihingen_val_image_list.txt \
    --rgb_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top \
    --dsm_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm \
    --masks_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ \
    --output_dir {root}/vaihingen/val

# Testing!
python vaihingen.py \
    --indices_file ../../data/indices/vaihingen_test_image_list.txt \
    --rgb_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/top \
    --dsm_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen/dsm \
    --masks_dir {root}/raw/vaihingen/Vaihingen/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE/ \
    --output_dir {root}/vaihingen/test
```

## Training

Launch training as follows:

### Potsdam

```bash
$ python scripts/train.py \
        --dataset "potsdam" \
        --method_name "PotsdamDiversity" \
        --scores_file_path {root}/submissions/potsdam/diversity.csv \
        --gpu 7
```

### Vaihingen

```bash
$ python scripts/train.py \
        --dataset "vaihingen" \
        --method_name "VaihingenDiversity" \
        --scores_file_path {root}/submissions/vaihingen/diversity.csv \
        --gpu 7
```

### DFC-22

```bash
$ python scripts/train.py \
        --dataset "dfc22" \
        --method_name "DFC22Diversity" \
        --scores_file_path {root}/submissions/dfc22/diversity.csv \
        --gpu 7
```

## Evaluation

For each run, jaccard scores for each class are saved. At the end of training, you will find all of the relevant scores saved in `./results/{method_name}.txt`.

Given you have the path to the best model checkpoint, you can also evaluate using the original images in `notebooks/export_results.ipynb`. 