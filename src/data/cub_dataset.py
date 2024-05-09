from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CUBDataset(Dataset):
    def __init__(self, data_dir: Path, train: bool, transform: transforms.Compose | None = None):
        super().__init__()
        image_path = "train_cropped" if train else "test_cropped"
        self.image_paths = [p for p in Path.iterdir(data_dir / image_path) if p.is_file()]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_paths[index]
        # TODO: make a PR to allow `read_image` to take a Path!

        # Open with PIL to allow `transforms.ToTensor` to succeed correctly
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        return image
