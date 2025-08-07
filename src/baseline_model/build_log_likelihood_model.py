"""This script "trains" a log likelihood baseline model, in the sense that it finds the 40 most indicative tokens
to use individually for classifying between matching and mismatching authors, and assumes that these will be the best
tokens for a log-likelihood classifier."""

import argparse
import collections
import json
from typing import Sequence

import numpy as np
import tqdm
from numpy.typing import NDArray

from baseline_model import utils
from baseline_model.token_stats import TokenStats
from common import common_types as ct
from common import tokenization
from common.utils import find_best_cutoff


def find_best_cutoff_for_authors(
    authors_counters: dict[str, list[collections.Counter[int]]],
    tokens_to_use: Sequence[int],
    possible_cutoffs: NDArray[np.floating] = np.arange(0, -20, -0.05),
) -> tuple[float, float, NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculates the cutoff which maximizes accuracy when classifying between matching and mismatching authors
    using a given set of tokens.

    Args:
        authors_counters: a list of token counters for each persona of each author
        tokens_to_use: the tokens to use as input to the classifier

    Returns:
        cutoff: the best cutoff (log-likelihood) for classifying between matching and mismatching authors
        accuracy: the accuracy with the optimal cutoff
        matching_likelihoods: the log-likelihoods of the matching authors
        mismatching_likelihoods: the log-likelihoods of the mismatching authors
    """
    # Collect TokenStats for each persona of each author.
    persona_stats: list[list[TokenStats]] = []
    for personas_counters in authors_counters.values():
        personas_sparse_stats = []
        for persona_counter in personas_counters:
            personas_sparse_stats.append(TokenStats.from_counts(persona_counter, tokens_to_use))
        persona_stats.append(personas_sparse_stats)

    matching_likelihoods = [stats[0].log_likelihood(stats[1]) for stats in persona_stats]
    mismatching_likelihoods = [
        # Note that we are not calculating all mismatches, as it would be computationally heavy.
        # We consider only the five authors subsequest the current one.
        [stats1[0].log_likelihood(stats2[1]) for stats2 in persona_stats[i + 1 : min(i + 5, len(persona_stats))]]
        for i, stats1 in enumerate(persona_stats)
    ]
    mismatching_likelihoods = sum(mismatching_likelihoods, start=[])  # type: ignore[arg-type]

    matching_likelihoods_np = np.array(matching_likelihoods)
    mismatching_likelihoods_np = np.array(mismatching_likelihoods)

    cutoff, accuracy = find_best_cutoff(matching_likelihoods_np, mismatching_likelihoods_np, possible_cutoffs)
    return cutoff, accuracy, matching_likelihoods_np, mismatching_likelihoods_np


def compute_token_unigram_accuracies(
    train_authors_counters: dict[str, list[collections.Counter[int]]],
    common_tokens: list[int],
    common_token_strings: list[str],
    tokens_accuracies_file: str | None = None,
) -> list[tuple[int, str, float]]:
    """Compute accuracy for each token, when used as a single token one-gram log likelihood classifier.

    Args:
        train_authors_counters: Mapping of authors to persona token counters.
        common_tokens: List with the 1000 most frequent token ids.
        common_token_strings: Decoded string value for each token id in `common_tokens`.
        tokens_accuracies_file: When provided, the full list of accuracies (all 1000 tokens) will be saved as JSON to
            this location.

    Returns:
        A list of `(token_id, token_string, accuracy)` tuples ordered by decreasing `accuracy`.
    """
    print("Computing individual token accuracies...")
    tokens_and_accuracies: list[tuple[int, str, float]] = []  # (token_id, token_string, accuracy)
    for i in tqdm.tqdm(range(len(common_tokens))):
        _, accuracy, _, _ = find_best_cutoff_for_authors(train_authors_counters, common_tokens[i : i + 1])
        tokens_and_accuracies.append((common_tokens[i], common_token_strings[i], accuracy))

    sorted_accuracies = sorted(tokens_and_accuracies, key=lambda x: x[2], reverse=True)

    if tokens_accuracies_file is not None:
        # Structure inspired by example in `find_best_tokens.ipynb`.
        data_to_save = {
            "columns": (
                "Index in 1000 common token",
                "Distinguishing accuracy using 1-gram of this token",
                "token integer",
            ),
            "data": [(idx, acc, int(token_id)) for idx, (token_id, _token_str, acc) in enumerate(sorted_accuracies)],
        }
        with open(tokens_accuracies_file, "wt", encoding="utf-8") as f:
            json.dump(data_to_save, f)
        print(f"Saved token accuracies to {tokens_accuracies_file}.")

    return sorted_accuracies


def train_log_likelihood_model(
    tiktoken_counts_top_1000_file: str,
    suitable_author_infos_train_file: str,
    num_tokens_to_use: int,
    tokens_accuracies_file: str | None = None,
) -> ct.Model:
    """Train a log likelihood model by finding the most indicative tokens among the 1000 most commonly used tokens."""
    tokenizer = tokenization.get_tokenizer()

    # Gets the 1000 most common tokens in the dataset.
    common_tokens = [tok[0] for tok in json.load(open(tiktoken_counts_top_1000_file, "rt"))]
    common_token_strings = [tokenizer.decode([tok]) for tok in common_tokens]
    print(f"Loaded the {len(common_tokens)} most common tokens. The first 20 are:")
    print(" ".join(repr(tok) for tok in common_token_strings[:20]))

    print("Loading author token counters...")
    train_prolific_authors = ct.read_author_infos(suitable_author_infos_train_file)
    train_authors_counters = utils.username_to_persona_counters_from_author_infos(train_prolific_authors, tokenizer)

    sorted_tokens_by_accuracy = compute_token_unigram_accuracies(
        train_authors_counters,
        common_tokens,
        common_token_strings,
        tokens_accuracies_file,
    )

    top_tokens_by_accuracy = sorted_tokens_by_accuracy[:num_tokens_to_use]
    print(f"Top {num_tokens_to_use} tokens by accuracy:")
    print(" ".join(repr(tok[1]) for tok in top_tokens_by_accuracy))

    model_tokens = [ct.TiktokenToken(token_index=int(tok[0]), string=tok[1]) for tok in top_tokens_by_accuracy]
    return ct.Model(
        model_type=ct.ModelType.LOG_LIKELIHOOD_MODEL,
        model_params=ct.LogLikelihoodModelParams(tokens=model_tokens),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Builds a log likelihood baseline model.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="JSON file to which the log likelihood model will be saved.",
    )
    parser.add_argument(
        "--tiktoken_counts_top_1000_file",
        type=str,
        default="../../data/tiktoken_counts_top_1000.json",
        help=(
            "JSON file containing the 1000 most common tokens in the dataset. "
            "in the format returned by `get_token_frequencies.py`."
        ),
    )
    parser.add_argument(
        "--suitable_author_infos_train_file",
        type=str,
        default="../../data/suitable_author_infos_train.ndjson",
        help="NDJSON file containing AuthorInfo objects for the training set.",
    )
    parser.add_argument(
        "--num_tokens_to_use",
        type=int,
        default=40,
        help="Number of tokens to use in the log likelihood model.",
    )
    parser.add_argument(
        "--tokens_accuracies_file",
        type=str,
        default=None,
        help=(
            "Optional JSON file to which the ordered accuracies of the 1000 most common "
            "tokens will be saved. If omitted, the accuracies are not persisted."
        ),
    )
    args = parser.parse_args()

    model = train_log_likelihood_model(
        args.tiktoken_counts_top_1000_file,
        args.suitable_author_infos_train_file,
        args.num_tokens_to_use,
        args.tokens_accuracies_file,
    )
    with open(args.output_file, "wt", encoding="utf-8") as f:
        f.write(model.to_json_string())


if __name__ == "__main__":
    main()
