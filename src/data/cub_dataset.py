from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from typing import Callable


class CUBDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool, push: bool, transform: Callable | None = None):
        super().__init__()
        if train:
            image_path = "train_cropped" if push else "train_cropped_augmented"
        else:
            image_path = "test_cropped"
        self.image_paths = [p for p in (data_dir / image_path).rglob("*.jpg")]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_paths[index]

        # -1 since we are zero-indexed but the image names start from 1
        # Each image name is something like <001>.<Bird name>.jpg
        label = int(image.name.split(".")[0]) - 1

        # Open with PIL to allow `transforms.ToTensor` to succeed correctly
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        return image, label
