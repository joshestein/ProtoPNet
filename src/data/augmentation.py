import os
from pathlib import Path

import Augmentor


# TODO: rotations can (and are) breaking.
# See https://github.com/mdbloice/Augmentor/pull/258


def run_augmentation(input_dir: Path, output_dir: Path):
    """Augments the input dir images with a series of rotations, skews and shears. Copied from
    https://github.com/cfchen-duke/ProtoPNet/blob/81bf2b70cb60e4f36e25e8be386eb616b7459321/img_aug.py"""

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


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Image augmentations")
    parser.add_argument("-i", "--input_dir", default=Path.cwd() / "cub200_cropped" / "train_cropped", required=True)
    parser.add_argument(
        "-o", "--output_dir", default=Path.cwd() / "cub200_cropped" / "train_cropped_augmented", required=True
    )
    args = parser.parse_args()
    run_augmentation(Path(args.input_dir), Path(args.output_dir))
