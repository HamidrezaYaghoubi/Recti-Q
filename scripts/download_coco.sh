#!/bin/bash
# Download COCO 2017 dataset
#
# This script downloads the COCO 2017 validation set and annotations.
# For training experiments, uncomment the train2017 download.
#
# Usage:
#   ./scripts/download_coco.sh
#   COCO_DIR=/path/to/coco ./scripts/download_coco.sh

set -e

# Default COCO directory
COCO_DIR="${COCO_DIR:-/fs/nexus-projects/pc_driving/yaghoubi/datasets/coco}"

echo "========================================"
echo "Downloading COCO 2017 Dataset"
echo "Target directory: $COCO_DIR"
echo "========================================"

# Create directory
mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

# Download validation images
if [ ! -d "val2017" ]; then
    echo "Downloading val2017 images..."
    wget -c http://images.cocodataset.org/zips/val2017.zip
    echo "Extracting val2017..."
    unzip -q val2017.zip
    rm val2017.zip
    echo "val2017 downloaded!"
else
    echo "val2017 already exists, skipping..."
fi

# Download annotations
if [ ! -d "annotations" ]; then
    echo "Downloading annotations..."
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "Extracting annotations..."
    unzip -q annotations_trainval2017.zip
    rm annotations_trainval2017.zip
    echo "Annotations downloaded!"
else
    echo "Annotations already exist, skipping..."
fi

# Optionally download training images (large ~18GB)
# Uncomment if needed:
# if [ ! -d "train2017" ]; then
#     echo "Downloading train2017 images..."
#     wget -c http://images.cocodataset.org/zips/train2017.zip
#     echo "Extracting train2017..."
#     unzip -q train2017.zip
#     rm train2017.zip
#     echo "train2017 downloaded!"
# fi

echo "========================================"
echo "COCO 2017 download complete!"
echo ""
echo "Directory structure:"
ls -la "$COCO_DIR"
echo ""
echo "Val2017 images: $(ls val2017 | wc -l)"
echo "========================================"
