from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class CUBDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool, transform: Callable | None = None):
        super().__init__()
        image_path = "train_cropped_augmented" if train else "test_cropped"
        self.image_paths = [p for p in (data_dir / image_path).rglob("*.jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_paths[index]

        # -1 since we are zero-indexed but the image names start from 1
        # Each image name is something like <001>.<Bird name>.jpg
        label = int(image.name.split(".")[0]) - 1

        # TODO: make a PR to allow `read_image` to take a Path!
        # Open with PIL to allow `transforms.ToTensor` to succeed correctly
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        return image, label
