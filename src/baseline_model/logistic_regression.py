"""
Logistic regression model

Train a logistic regression model on the following features:
    [N1 * log(p1) / sum(N_i), N2 * log(p2) / sum(N_i), ...]
where p_i are from one persona, and N_i are from another persona.
Thus, the minimum performance should be that of log likelihood (which would be obtained by choosing an all-ones weights
vector).
"""

import argparse
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tiktoken
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from baseline_model import utils
from baseline_model.token_stats import TokenStats
from common import tokenization
from common.utils import random_derangement


def create_positive_and_negative_examples(
    authors_features: Dict[str, list[Counter[int]]], token_list: Sequence[int]
) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Create positive and negative examples for the logistic regression model.

    Args:
        authors_features: Dictionary from author username to a list of (two) counters, each containing the token counts
            for the two personas.
        token_list: List of tokens to use
    Returns:
        Tuple of numpy arrays, where the first array contains the features and the second array contains the labels.
    """

    x_list: List[NDArray[np.float64]] = []
    y_list: List[int] = []

    # Create positive samples (same author)
    for author, counters in authors_features.items():
        persona_1_counter, persona_2_counter = counters[0], counters[1]
        persona_2_token_stats = TokenStats.from_counts(persona_2_counter, token_list).token_count_array
        if sum(persona_2_token_stats) > 0:
            features = (
                TokenStats.from_counts(persona_1_counter, token_list).log_nonz_token_freq_with_excluded_tokens
                * persona_2_token_stats
                / sum(persona_2_token_stats)
            )
        else:
            features = np.zeros(len(persona_2_token_stats))
        if sum(persona_2_token_stats) == 0:
            print(author)
        x_list.append(features)
        y_list.append(1)

    print(f"Created {len(y_list)} positive samples")

    # Create negative samples (different authors)
    # First, collect all first and second arrays separately
    first_arrays = [features[0] for features in authors_features.values()]
    second_arrays = [features[1] for features in authors_features.values()]

    # Get derangement indices and use them to permute second arrays
    derangement = random_derangement(len(second_arrays))
    permuted_second = [second_arrays[i] for i in derangement]

    # Create negative samples using the deranged pairs
    for first, permuted in zip(first_arrays, permuted_second):
        permuted_count = TokenStats.from_counts(permuted, token_list).token_count_array
        if sum(permuted_count) > 0:
            features = (
                TokenStats.from_counts(first, token_list).log_nonz_token_freq_with_excluded_tokens
                * permuted_count
                / sum(permuted_count)
            )
        else:
            features = np.zeros(len(permuted_count))
        x_list.append(features)
        y_list.append(0)

    print(f"Created {len(y_list)} training samples (added negative examples)")

    return np.array(x_list), np.array(y_list, dtype=np.int32)


def run_logistic_regression_analysis(
    training_data_path: str,
    top_1000_tokens_success_probs_file: str,
    tokenizer: tiktoken.Encoding,
    num_tokens_to_use: int = 40,
) -> tuple[LogisticRegression, NDArray[np.float64], NDArray[np.int32], NDArray[np.float64], NDArray[np.int32]]:
    """
    Run the logistic regression analysis with the given data paths and number of tokens.

    Args:
        training_data_path: Path to the directory containing the training data files
        top_1000_tokens_success_probs_file: Path to the file containing token success probabilities
        tokenizer: Tokenizer to use
        num_tokens_to_use: Number of tokens to use when constructing training features

    Returns:
        Tuple of logistic regression model, training features, training labels, validation features, and validation
        labels.
    """
    train_validate_author_to_personas_counters = utils.get_train_validate_author_to_personas_counters(
        tokenizer, training_data_path
    )
    tokens_to_use = [
        token_int
        for _, _, token_int in utils.load_1000_most_common_tokens_sorted_by_1_gram_accuracies(
            top_1000_tokens_success_probs_file
        )
    ][:num_tokens_to_use]
    print(f"Top {num_tokens_to_use} tokens:")
    print(" ".join([repr(tokenizer.decode([tok])) for tok in tokens_to_use]))

    positive_negative_examples_pairs_of_log_nonz_probabilities = {}
    for arm in ["train", "val"]:
        positive_negative_examples_pairs_of_log_nonz_probabilities[arm] = create_positive_and_negative_examples(
            train_validate_author_to_personas_counters[arm], tokens_to_use
        )

    positive_negative_examples_pairs_of_log_nonz_probabilities = {}
    for arm in ["train", "val"]:
        positive_negative_examples_pairs_of_log_nonz_probabilities[arm] = create_positive_and_negative_examples(
            train_validate_author_to_personas_counters[arm], tokens_to_use
        )

    x_train, y_train = positive_negative_examples_pairs_of_log_nonz_probabilities["train"]
    x_validate, y_validate = positive_negative_examples_pairs_of_log_nonz_probabilities["val"]

    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(x_train, y_train)

    # Score
    print(f"Training accuracy: {lr_model.score(x_train, y_train):.4f}")
    print(f"Validation accuracy: {lr_model.score(x_validate, y_validate):.4f}")

    return lr_model, x_train, y_train, x_validate, y_validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run logistic regression.")
    parser.add_argument(
        "--top_1000_tokens_success_probs_file",
        type=str,
        help=(
            "JSON file containing the success probabilities of the 1000 most common tokens, "
            "in the format returned by `find_best_tokens.ipynb`."
        ),
    )
    parser.add_argument(
        "--suitable_author_infos_dir",
        type=str,
        help=(
            "Directory that contains the `suitable_author_infos_{train,val}.ndjson` files used "
            "as training and validation data."
        ),
    )
    parser.add_argument(
        "--num_tokens_to_use",
        type=int,
        default=40,
        help="Number of tokens to use when constructing training features.",
    )
    args = parser.parse_args()

    tokenizer = tokenization.get_tokenizer()

    run_logistic_regression_analysis(
        args.suitable_author_infos_dir,
        args.top_1000_tokens_success_probs_file,
        tokenizer,
        args.num_tokens_to_use,
    )


if __name__ == "__main__":
    main()
