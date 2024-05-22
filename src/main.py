from pathlib import Path

import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import NUM_TRAINING_EPOCHS
from src.data.cub_dataset import CUBDataset
from src.protopnet import ProtoPNet


def train(model: torch.nn.Module, dataloader: DataLoader):
    pass


def main():
    model = ProtoPNet()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    # TODO: augment online
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, shear=15),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_dir = Path.cwd() / "data" / "CUB-200-2011" / "CUB_200_2011" / "cub200_cropped"
    train_data = CUBDataset(data_dir, train=True, transform=train_transforms)
    test_data = CUBDataset(data_dir, train=False, transform=test_transforms)
    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=2, shuffle=True)

    for i in range(NUM_TRAINING_EPOCHS):
        train(model, train_dataloader)


if __name__ == "__main__":
    main()
