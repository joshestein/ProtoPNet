from pathlib import Path

import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import CONVEX_OPTIMISATION_START_EPOCH, CONVEX_OPTIMISATION_STEPS, NUM_TRAINING_EPOCHS, NUM_WARM_EPOCHS
from src.data.cub_dataset import CUBDataset
from src.loss import ProtoPLoss
from src.protopnet import ProtoPNet


def train(model: ProtoPNet, dataloader: DataLoader, loss_fn: ProtoPLoss, optimiser: torch.optim.Optimizer):
    model.train()

    for batch, (image, label) in enumerate(dataloader):
        output, min_distances = model(image)

        loss = loss_fn(output, label, min_distances)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{(len(dataloader)):>5d}]")


def main():
    model = ProtoPNet()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    loss = ProtoPLoss(model)

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
    warm_optimiser = torch.optim.Adam(
        [
            {"params": model.additional_layers.parameters(), "lr": 3e-3, "weight_decay": 1e-3},
            {"params": model.prototypes, "lr": 3e-3},
        ]
    )
    joint_optimiser = torch.optim.Adam(
        [
            {"params": model.pretrained_conv_net.parameters(), "lr": 1e-4, "weight_decay": 1e-3},
            {"params": model.additional_layers.parameters(), "lr": 3e-3, "weight_decay": 1e-3},
            {"params": model.prototypes, "lr": 3e-3},
        ]
    )
    last_layer_optimiser = torch.optim.Adam([{"params": model.fully_connected.parameters(), "lr": 1e-4}])

    for epoch in range(NUM_TRAINING_EPOCHS):

        if epoch < NUM_WARM_EPOCHS:
            model.warm()
            train(model, train_dataloader, loss_fn=loss, optimiser=warm_optimiser)
        else:
            model.all_layers_joint_learning()
            train(model, train_dataloader, loss_fn=loss, optimiser=joint_optimiser)

        for _step in CONVEX_OPTIMISATION_STEPS:
            model.convex_optimisation_last_layer()
            train(model, train_dataloader, optimiser=last_layer_optimiser)


if __name__ == "__main__":
    main()
