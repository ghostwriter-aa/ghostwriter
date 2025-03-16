import json
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import tiktoken
from numpy.typing import NDArray

import common.common_types as ct
from baseline_model.token_stats import TokenStats
from common import tokenization
from common.common_types import AuthorInfo


def get_authors_with_enough_tokens_in_each_persona(
    author_infos: list[AuthorInfo],
    tokens_to_count: Sequence[int],
    tokenizer: tiktoken.Encoding,
    sufficient_tokens: int = 500,
) -> list[AuthorInfo]:
    """
    Return a list of authors who have at least `sufficient_tokens` tokens from the token set provided in each persona.
    """
    persona_stats = [
        [TokenStats.from_persona_info(author_info.personas[j], tokens_to_count, tokenizer) for j in range(2)]
        for author_info in author_infos
    ]
    authors_with_enough_tokens_in_each_persona = []

    print(f"Minimum number of tokens for a persona to be considered: {sufficient_tokens}")

    for author_info, persona_stat in zip(author_infos, persona_stats):
        if (
            np.sum(persona_stat[0].token_count_array) > sufficient_tokens
            and np.sum(persona_stat[1].token_count_array) > sufficient_tokens
        ):
            authors_with_enough_tokens_in_each_persona.append(author_info)

    return authors_with_enough_tokens_in_each_persona


def load_1000_most_common_tokens_sorted_by_1_gram_accuracies() -> list[tuple[int, float, int]]:
    """
    Load the top 1000 tokens with their 1-gram classification accuracies.
    Returns a sorted list of tuples of the form
    (Index in 1000 common token, Classifier accuracy using 1-gram of this token, token integer)
    ordered by the classifier accuracy.
    """
    current_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    with open(current_file_path.parent.parent / "data" / "top_1000_tokens_success_probs.json", "rt") as f:
        tokens_onegram_accuracy = json.load(f)["data"]
    return sorted(tokens_onegram_accuracy, key=lambda x: x[1], reverse=True)


def load_suitable_author_infos_train_validation() -> dict[str, list[ct.AuthorInfo]]:
    """
    Load author information for authors with many tokens, according to the saved files of the form
    `author_infos_many_tokens_{arm}.ndjson`.
    Returns a dictionary with two keys: "train" and "val". In each key, there is a list of AuthorInfo objects.
    """
    current_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    authors_with_many_tokens = {}
    for arm in ["train", "val"]:
        authors_with_many_tokens[arm] = ct.read_author_infos(
            current_file_path.parent.parent / "data" / f"author_infos_many_tokens_{arm}.ndjson"
        )
        print(f"Loaded {len(authors_with_many_tokens[arm])} authors in {arm} set with many tokens.")
    return authors_with_many_tokens


def username_to_persona_counters_from_author_infos(
    author_infos: list[ct.AuthorInfo],
    tokenizer: tiktoken.Encoding,
) -> dict[str, list[Counter[int]]]:
    """
    Returns a dictionary keyed by the author username, where the value is a list of two dictionaries, containing
    the token counts for the two personas.
    """
    authors_counters = {}
    for author_info in author_infos:
        personas = []
        for persona_info in author_info.personas:
            personas.append(tokenization.get_persona_tokens_counter(persona_info, tokenizer))
        authors_counters[author_info.username] = personas
    return authors_counters


def get_train_validate_author_to_personas_counters(
    tokenizer: tiktoken.Encoding,
) -> dict[str, dict[str, list[Counter[int]]]]:
    """
    Load the author counters for the training and validation sets.
    Returns a dictionary with two keys: "train" and "val". In each key, there is a dictionary keyed by the author
    username, where the value is a list of two dictionaries, each containing the token counts for the two personas.
    """
    authors_with_many_tokens = load_suitable_author_infos_train_validation()
    authors_counters = {}
    for arm in ["train", "val"]:
        authors_counters[arm] = username_to_persona_counters_from_author_infos(authors_with_many_tokens[arm], tokenizer)
        print(f"Created {len(authors_counters[arm])} author counters for {arm} set.")
    return authors_counters


def convert_counters_to_log_nonz_probs_in_username_to_persona_counters(
    author_to_counters_dict: dict[str, list[Counter[int]]], tokens_to_use: Sequence[int]
) -> dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]]:
    """
    Convert the token counters to log probabilities for the tokens in `tokens_to_use`.
    Inputs:
    - author_to_counters_dict: a dictionary keyed by the author username, where the value is a list of two dictionaries,
        each containing the token counts for the two personas.
    - tokens_to_use: the token set to use for the log probabilities.
    Returns a dictionary keyed by the author username, where the value is a tuple of two arrays, each containing the
    log (non zero) probabilities of the tokens in `tokens_to_use` for the two personas.
    """
    author_log_probabilities_features_dict = {}
    for author_username, author_counters in author_to_counters_dict.items():
        author_log_probabilities_features_dict[author_username] = (
            TokenStats.from_counts(author_counters[0], tokens_to_use).log_nonz_token_freq,
            TokenStats.from_counts(author_counters[1], tokens_to_use).log_nonz_token_freq,
        )
    return author_log_probabilities_features_dict


def random_derangement(n: int) -> NDArray[np.int64]:
    """
    Generate a random derangement of n elements.
    Returns an array where no element is in its original position.
    """
    np.random.seed(424242)  # For consistency across runs.
    while True:
        perm = np.random.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def create_positive_and_negative_examples_form_persona_pairs(
    author_to_features: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """
    Create balanced training data from pairs of feature arrays for each author (two arrays for two personas, each
    representing log probabilities of a persona by the same token set).
    Positive examples are concatenated feature arrays (log probabilities of two personas of the same author).
    Negative examples are concatenated feature arrays (log probabilities of two personas of different authors).
    """
    x_list: list[NDArray[np.float64]] = []
    y_list: list[int] = []

    # Create positive samples (same author)
    for features1, features2 in author_to_features.values():
        combined_features = np.concatenate([features1, features2])
        x_list.append(combined_features)
        y_list.append(1)

    print(f"Created {len(y_list)} positive samples")

    # Create negative samples (different authors)
    # First, collect all first and second arrays separately
    first_arrays = [features[0] for features in author_to_features.values()]
    second_arrays = [features[1] for features in author_to_features.values()]

    # Get derangement indices and use them to permute second arrays
    derangement = random_derangement(len(second_arrays))
    permuted_second = [second_arrays[i] for i in derangement]

    # Create negative samples using the deranged pairs
    for first, permuted in zip(first_arrays, permuted_second):
        combined_features = np.concatenate([first, permuted])
        x_list.append(combined_features)
        y_list.append(0)

    print(f"Created {len(y_list)} training samples (added negative examples)")

    # Convert lists to numpy arrays
    x = np.array(x_list)
    y = np.array(y_list, dtype=np.int32)

    return x, y
