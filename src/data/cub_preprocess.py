import os
from pathlib import Path

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


def crop_images(data_dir: Path, output_dir: Path):
    train_ids, test_ids = get_test_train_split_ids(data_dir)
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


def run_preprocessing(data_dir: Path, output_dir: Path):
    crop_images(data_dir, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocessor for CUB-200-2011 dataset. Images are split into train/test directories. Training "
        "images are augmented according to the original ProtoPNet paper."
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
        help="Output dir where preprocessed data should be saved.",
        required=False,
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.data_dir

    run_preprocessing(Path(args.data_dir), Path(args.output_dir))
