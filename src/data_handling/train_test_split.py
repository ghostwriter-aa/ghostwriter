"""Split the author docs dataset into train, validation, and test sets.

Example usage:
python train_test_split.py \
    --input_file ../data/suitable_author_infos.ndjson \
    --output_prefix ../data/suitable_author_infos
"""

import argparse
import dataclasses
import hashlib
import json

import common.common_types as ct

# Percentage of the dataset to place in each arm.
_EXPERIMENT_ARM_TO_PERCENTAGE = {
    "train": 70,
    "val": 15,
    "test": 15,
}


def split_to_arms(author_infos: list[ct.AuthorInfo], output_prefix: str) -> None:
    arm_to_file = {arm: open(f"{output_prefix}_{arm}.ndjson", "wt") for arm in _EXPERIMENT_ARM_TO_PERCENTAGE}
    for author_info in author_infos:
        hashcode = int(hashlib.sha256(author_info.username.encode()).hexdigest(), 16) % 100
        if hashcode < _EXPERIMENT_ARM_TO_PERCENTAGE["train"]:
            arm = "train"
        elif hashcode < _EXPERIMENT_ARM_TO_PERCENTAGE["train"] + _EXPERIMENT_ARM_TO_PERCENTAGE["val"]:
            arm = "val"
        else:
            arm = "test"
        arm_to_file[arm].write(json.dumps(dataclasses.asdict(author_info)) + "\n")

    for file in arm_to_file.values():
        file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file",
        type=str,
        default="../data/suitable_author_infos.ndjson",
        help="Input NDJSON file containing AuthorInfo objects.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help=(
            "Prefix for output files, e.g., `../data/suitable_author_infos`. The suffix `_ARM.ndjson` will be added, "
            "where ARM is one of `train`, `val`, or `test`."
        ),
    )
    args = parser.parse_args()

    split_to_arms(ct.read_author_infos(args.input_file), args.output_prefix)


if __name__ == "__main__":
    main()
