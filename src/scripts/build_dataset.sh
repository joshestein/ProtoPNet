#!/usr/bin/env sh

if [ $# -eq 0 ] ; then
    echo "Please provide the path to the directory where the dataset should be downloaded and extracted as an argument."
    exit 1
fi

if [ ! -d "$1" ]; then
  mkdir -p "$1"
fi

echo "Downloading dataset..."

# Download the dataset
#curl -sL https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 | tar xz -C "$1"

echo "Pre-processing dataset..."

# Run pre-processing
python src/data/cub_preprocess.py -d "$1/CUB_200_2011"
python src/data/img_aug.py -d "$1/CUB_200_2011/cub200_cropped"
