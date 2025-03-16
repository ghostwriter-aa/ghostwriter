"""
Filters authors such that each persona has at least X tokens from the most frequent tokens in the dataset.
"""

import argparse
import dataclasses
import json

import common.common_types as ct
from baseline_model import utils
from common import tokenization


def main(input_file: str, output_file: str, sufficient_tokens: int, most_frequent_tokens_file: str) -> None:
    tokenizer = tokenization.get_tokenizer()
    author_infos = ct.read_author_infos(input_file)
    print(f"Loaded {len(author_infos)} authors in training set.")
    common_tokens = [
        tok[0] for tok in json.load(open(most_frequent_tokens_file, "rt"))
    ]  # Gets the 1000 most common tokens in the dataset.

    authors_with_enough_tokens_in_each_persona = utils.get_authors_with_enough_tokens_in_each_persona(
        author_infos, common_tokens, tokenizer, sufficient_tokens=sufficient_tokens
    )
    print(f"Found {len(authors_with_enough_tokens_in_each_persona)} authors with enough tokens in each persona.")

    with open(output_file, "wt") as f:
        for author in authors_with_enough_tokens_in_each_persona:
            f.write(json.dumps(dataclasses.asdict(author)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_author_infos",
        type=str,
        help="Input NDJSON file containing AuthorInfo objects.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output NDJSON file containing AuthorInfo objects,"
        "after filtering them such that each persona has at least X tokens.",
    )
    parser.add_argument(
        "--sufficient_tokens",
        type=int,
        default=500,
        help="Minimum number of tokens for a persona to be considered.",
    )
    parser.add_argument(
        "--most_frequent_tokens_file",
        type=str,
        default="../data/tiktoken_counts_top_1000.json",
        help="JSON file containing the most frequent tokens in the dataset.",
    )
    args = parser.parse_args()
    main(args.input_author_infos, args.output_file, args.sufficient_tokens, args.most_frequent_tokens_file)
