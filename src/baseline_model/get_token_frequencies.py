"""Compute the frequencies of o200k_base tokens in the training data."""

import argparse
import collections
import json

import tiktoken
import tqdm

import common.common_types as ct
from common import tokenization


def get_token_frequencies(
    author_infos: list[ct.AuthorInfo], tokenizer: tiktoken.Encoding
) -> tuple[collections.Counter[int], collections.Counter[int]]:
    """
    Return a tuple of two counters:
    - the first counts the number of occurrences of each token in the training data, and
    - the second counts the number of authors using each token.
    """
    token_frequencies: collections.Counter[int] = collections.Counter()
    unique_authors_count: collections.Counter[int] = collections.Counter()
    for author_info in tqdm.tqdm(author_infos):
        author_tokens_frequencies: collections.Counter[int] = collections.Counter()
        for token_list in tokenization.get_author_tokens_iterator(author_info, tokenizer):
            token_frequencies.update(token_list)
            author_tokens_frequencies.update(token_list)
        unique_author_tokens = set(author_tokens_frequencies)
        unique_authors_count.update(unique_author_tokens)
    return token_frequencies, unique_authors_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_author_infos",
        type=str,
        default="../data/suitable_author_infos_train.ndjson",
        help="Input NDJSON file containing AuthorInfo objects.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output JSON dictionary counting occurrences of each token."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=9999999,
        help="If specified, only the K most common tokens will be written to --output_file.",
    )
    parser.add_argument(
        "--min_author_occurrences",
        type=int,
        default=2,
        help="Minimum number of authors using a token for it to be considered.",
    )
    args = parser.parse_args()

    tokenizer = tiktoken.get_encoding("o200k_base")
    token_frequencies, unique_authors_count = get_token_frequencies(
        ct.read_author_infos(args.input_author_infos), tokenizer
    )

    # Filter out tokens that are used by fewer than `args.min_author_occurrences` authors.
    # Take the top `args.top_k` most common tokens that satisfy the above condition.
    most_common_counts = [
        (token, count)
        for token, count in token_frequencies.most_common()
        if unique_authors_count[token] >= args.min_author_occurrences
    ][: args.top_k]

    with open(args.output_file, "wt") as f:
        f.write(json.dumps(most_common_counts))

    print("Top tokens and the number of occurrences:")
    for i, (token, count) in enumerate(most_common_counts):
        print(f"#{i:2d}:  {token:8d} {count:6d}: {tokenizer.decode([token])!r}")
        if i >= 30:
            break


if __name__ == "__main__":
    main()
