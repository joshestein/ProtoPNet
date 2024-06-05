import os

from pathlib import Path

CONVEX_OPTIMISATION_START_EPOCH = 10
CONVEX_OPTIMISATION_STEPS = 20
NUM_TRAINING_EPOCHS = 1000
NUM_WARM_EPOCHS = 5  # How many epochs to only train additional (not pre-trained) layers

# Paths, everything relative to the src directory
src_dir = Path(__file__).resolve().parent
data_dir = src_dir.parent / "data" / "CUB_200_2011" / "cub200_cropped"
train_dir = data_dir / "train_cropped_augmented"
test_dir = data_dir / "test_cropped"
train_push_dir = data_dir / "train_cropped"
model_dir = src_dir / "models"

for dir in [train_dir, test_dir, train_push_dir, model_dir]:
    os.makedirs(dir, exist_ok=True)
