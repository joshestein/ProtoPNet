import os
from pathlib import Path

import Augmentor
from PIL import Image


def get_test_train_split_ids(data_dir: Path) -> (list[int], list[int]):
    train_ids, test_ids = [], []

    with open(data_dir / "train_test_split.txt", "r", encoding="UTF-8") as file:
        while line := file.readline():
            image_id, is_training_image = line.rstrip().split(" ")
            if is_training_image == "1":
                train_ids.append(int(image_id))
            else:
                test_ids.append(int(image_id))

    return train_ids, test_ids


def get_image_paths(data_dir: Path) -> dict[int, Path]:
    image_paths = {}

    with open(data_dir / "images.txt", "r", encoding="UTF-8") as file:
        while line := file.readline():
            image_id, image_path = line.rstrip().split(" ")
            image_paths[int(image_id)] = Path(image_path)

    return image_paths


def crop_images(train_ids: list[int], data_dir: Path, output_dir: Path):
    image_paths = get_image_paths(data_dir)

    os.makedirs(output_dir, exist_ok=True)
    train_cropped_dir = output_dir / "cub200_cropped" / "train_cropped"
    test_cropped_dir = output_dir / "cub200_cropped" / "test_cropped"
    os.makedirs(train_cropped_dir, exist_ok=True)
    os.makedirs(test_cropped_dir, exist_ok=True)

    with open(data_dir / "bounding_boxes.txt", "r", encoding="UTF-8") as file:
        while line := file.readline():
            line_data = [int(float(i)) for i in line.rstrip().split(" ")]
            image_id, x, y, width, height = line_data
            image = Image.open(data_dir / "images" / image_paths[image_id])
            image = image.crop((x, y, x + width, y + height))

            path = train_cropped_dir if image_id in train_ids else test_cropped_dir
            image.save(path / image_paths[image_id].name)


def create_augmentations(root_dir: Path):
    """Augments the input dir images with a series of rotations, skews and shears. Copied from
    https://github.com/cfchen-duke/ProtoPNet/blob/81bf2b70cb60e4f36e25e8be386eb616b7459321/img_aug.py"""

    # TODO: rotations can (and are) breaking.
    # See https://github.com/mdbloice/Augmentor/pull/258

    input_dir = root_dir / "cub200_cropped" / "train_cropped"
    output_dir = root_dir / "cub200_cropped" / "train_cropped_augmented"

    os.makedirs(output_dir, exist_ok=True)

    # Rotation
    p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p

    # Skew
    p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p

    # Shear
    p = Augmentor.Pipeline(source_directory=input_dir, output_directory=output_dir)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()


def run_preprocessing(data_dir: Path, output_dir: Path):
    train_ids, _ = get_test_train_split_ids(data_dir)
    crop_images(train_ids, data_dir, output_dir)
    create_augmentations(output_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Preprocessor for CUB-200-2011 dataset. Images are split into train/test directories and cropped "
        "according to the provided bounding boxes. Training images are augmented according to the original ProtoPNet "
        "paper."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        help="Dataset directory. Should contain the `images` directory, as well as `images.txt`, "
        "`bounding_boxes.txt`, `classes.txt` etc.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output dir where preprocessed data should be saved. Defaults to `data_dir`.",
        required=False,
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    run_preprocessing(Path(args.data_dir), Path(args.output_dir))
