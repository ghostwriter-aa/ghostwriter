# %% [markdown]
# # Find the best tokens to use as a classifier
# 
# The file contains the following functions and executes the following steps:
# 1. Load the 1000 most common tokens in the dataset.
# 2. Defines a function to find the best cutoff for maximizing accuracy for (samples of) two distributions (that are similar to a normal distribution) with different means.
# 3. Defines a function to plot the histogram of log-likelihoods for matching and mismatching authors.
# 4. Defines a function to find the best cutoff for maximizing accuracy when classifying between matching and mismatching authors using the given tokens.
# 5. Plots the log-likelihood histogram and classifier using the 1000 most common tokens.
# 6. Calculates the success probabilities of using each of the 1000 most common tokens as a 1-gram Log Likelihood classifier.
# 7. Saves the success probabilities to a file.
# 8. Plots the log-likelihood histogram with the "optimal" 40 tokens.
# 9. Plots a graph representing the accuracy of the classifier that uses the first $n$ best tokens.

# %%
import json
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from common import tokenization

from baseline_model.token_stats import TokenStats
from baseline_model import utils


TIKTOKEN_COUNTS_TOP_1000_FILE = "../../data/tiktoken_counts_top_1000.json"
TOP_1000_TOKENS_SUCCESS_PROBS_FILE = "../../data/top_1000_tokens_success_probs.json"

# %%
tokenizer = tokenization.get_tokenizer()

common_tokens = np.array([tok[0] for tok in json.load(open(TIKTOKEN_COUNTS_TOP_1000_FILE, "rt"))]) # Gets the 1000 most common tokens in the dataset.
common_token_strings = [tokenizer.decode([tok]) for tok in common_tokens]
print(f"Loaded the {len(common_tokens)} most common tokens. The first 20 are:")
print(" ".join(repr(tok) for tok in common_token_strings[:20]))

train_validate_author_to_personas_counters = utils.get_train_validate_author_to_personas_counters(tokenizer)


# %%
def find_best_cutoff(dist_1_samples, dist_2_samples, possible_cutoffs: np.ndarray, verbose=True) -> Tuple[int, int]:
    """
    Find the best cutoff for maximizing accuracy when classifying between samples of two distributions.
    inputs:
    - dist_1_samples: samples from the first distribution
    - dist_2_samples: samples from the second distribution
    - possible_cutoffs: the possible cutoffs to consider
    - verbose: whether to print the results
    """
    results = []
    dist_1_samples.sort()
    dist_2_samples.sort()
    for cutoff in possible_cutoffs:
        dist_1_prob = 1 - (np.searchsorted(dist_1_samples, cutoff) / len(dist_1_samples))
        dist_2_prob = 1 - (np.searchsorted(dist_2_samples, cutoff) / len(dist_2_samples))
        success_with_cutoff = 0.5 * (dist_1_prob + 1 - dist_2_prob)
        results.append((cutoff, dist_1_prob, dist_2_prob, success_with_cutoff))
    # pretty print the result with the maximum difference
    results.sort(key=lambda x: x[3], reverse=True)
    # pretty print first result
    if verbose:
        for result in results[:7]:
            print("Cutoff: {:>5}, Matching: {:.4f}, Mismatching: {:.4f}, Success Probability: {:.4f}".format(*result))
    return results[0][0], results[0][3]

# %%
def plot_likelihood_histogram(
    matching_likelihoods: NDArray[np.float32],
    mismatching_likelihoods: NDArray[np.float32],
    range_min: int = -200,
    range_max: int = 0,
    bins: int = 200,
    labels: tuple[str, str] = ("Matching", "Mismatching"),
    title: str = "Distribution of Log Likelihoods",
) -> None:
    """
    Plot histograms of log likelihoods with specified bins and range, including a tail bucket.

    Args:
        matching_likelihoods (NDArray): First set of log likelihood values
        mismatching_likelihoods (NDArray): Second set of log likelihood values
        bins (int): Number of bins
        range_min (float): Minimum value for histogram range
        range_max (float): Maximum value for histogram range
        labels (tuple): Labels for the two sets of data
    """

    bin_edges = np.linspace(range_min, range_max, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]

    matching_likelihoods = np.array(matching_likelihoods)
    matching_likelihoods = matching_likelihoods.clip(range_min, range_max)
    mismatching_likelihoods = np.array(mismatching_likelihoods)
    mismatching_likelihoods = mismatching_likelihoods.clip(range_min, range_max)

    # Calculate histograms for the main range
    matching_likelihoods_histogram, _ = np.histogram(matching_likelihoods, bins=bin_edges, density=True)
    mismatching_likelihoods_histogram, _ = np.histogram(mismatching_likelihoods, bins=bin_edges, density=True)

    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate medians
    median1 = np.median(matching_likelihoods)
    median2 = np.median(mismatching_likelihoods)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot histograms as bars
    plt.bar(bin_centers, matching_likelihoods_histogram, width=bin_width, alpha=0.5, label=labels[0])
    plt.bar(bin_centers, mismatching_likelihoods_histogram, width=bin_width, alpha=0.5, label=labels[1])

    # Add median lines
    plt.axvline(median1, color="darkblue", linestyle="--", label=f"{labels[0]} Median: {median1:.2f}")  # type: ignore
    plt.axvline(median2, color="darkred", linestyle="--", label=f"{labels[1]} Median: {median2:.2f}")  # type: ignore

    # Customize the plot
    plt.xlabel("Log Likelihood")
    plt.ylabel("Fraction of Values")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set x-axis range
    plt.xlim(range_min, range_max)

    plt.show()

# %%
def find_best_cutoff_and_plot(authors_counters: Dict[str, List[Counter]],
                              tokens_to_use,
                              title: str = '',
                              range_min: float = -20,
                              range_max: float = 0,
                              verbose: bool = True,
                              possible_cutoffs: np.ndarray = np.arange(0, -20, -0.05)
                              ) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Calculate the best cutoff for maximizing accuracy when classifying between matching and mismatching authors using the given tokens.
    inputs:
    - authors_counters: a list of counters for each persona of each author
    - tokens_to_use: the tokens to use as input to the classifier
    - verbose: whether to plot the log-likelihood histogram
    returns:
    - cutoff: the best cutoff (log-likelihood) for classifying between matching and mismatching authors
    - success: the accuracy with the optimal cutoff
    - matching_likelihoods: the log-likelihoods of the matching authors
    - mismatching_likelihoods: the log-likelihoods of the mismatching authors
    """
    good_author_personas_sparse_stats: List[List[TokenStats]] = []
    for personas_counters in authors_counters.values():
        personas_sparse_stats = []
        for persona_counter in personas_counters:
            personas_sparse_stats.append(TokenStats.from_counts(persona_counter, tokens_to_use))
        good_author_personas_sparse_stats.append(personas_sparse_stats)

    matching_likelihoods = [
        stats[0].log_likelihood(stats[1]) for stats in good_author_personas_sparse_stats
    ]
    mismatching_likelihoods = [
        # Note that we are not calculating all mismatches, as it would be computationally heavy. We consider only the five authors subsequest the current one.
        [stats1[0].log_likelihood(stats2[1]) for stats2 in good_author_personas_sparse_stats[i + 1: min(i + 5, len(good_author_personas_sparse_stats))]]
        for i, stats1 in enumerate(good_author_personas_sparse_stats)
    ]
    mismatching_likelihoods = sum(mismatching_likelihoods, start=[])

    matching_likelihoods = np.array(matching_likelihoods)
    mismatching_likelihoods = np.array(mismatching_likelihoods)

    if verbose:
        plot_likelihood_histogram(matching_likelihoods, mismatching_likelihoods, title=title or "Distribution of Log Likelihoods",
                                  range_min=range_min, range_max=range_max, bins=100)

    cutoff, success = find_best_cutoff(matching_likelihoods, mismatching_likelihoods, possible_cutoffs=possible_cutoffs, verbose=verbose)

    return cutoff, success, matching_likelihoods, mismatching_likelihoods

# %% [markdown]
# ## Plot the log-likelihood ratio

# %% [markdown]
# $LL = \frac{\sum N_i \log p_i}{\sum N_i}$ where:
# 
# $N_i$: token count in target persona
# 
# $p_i$: token probability in source persona

# %%
N = 1000
arm = "val"
cutoff, success, matching_likelihoods, mismatching_likelihoods = find_best_cutoff_and_plot(
    train_validate_author_to_personas_counters[arm], common_tokens[:N], range_min=-13, range_max=-7,
    title=f"Log likelihood histogram from {N} most common tokens ({arm} set)"
)

# %%
# calculate the success probabilities of using each of the 1000 most common tokens as a 1-gram Log Likelihood classifier
success_probs = []
for i in tqdm(range(1000)):
    _, success, _, _ = find_best_cutoff_and_plot(train_validate_author_to_personas_counters["train"], common_tokens[i:i + 1], verbose=False)
    success_probs.append((i, success))

sorted_success_probs = sorted(success_probs, key=lambda x: x[1], reverse=True)
    
sorted_success_probs_with_token = {
    "columns": ("Index in 1000 common token", "Distinguishing success probability using 1-gram of this token", "token integer"),
    "data": [(s[0], s[1], int(common_tokens[s[0]])) for s in sorted_success_probs]
}
with open(TOP_1000_TOKENS_SUCCESS_PROBS_FILE, "wt") as f:
    json.dump(sorted_success_probs_with_token, f)

# %%
with open(TOP_1000_TOKENS_SUCCESS_PROBS_FILE, "rt") as f:
    success_probs_with_token = json.load(f)
sorted_success_probs = sorted([(s[0], s[1]) for s in success_probs_with_token["data"]], key=lambda x: x[1], reverse=True)

# %% [markdown]
# Here is a list of the top 20 tokens achieving highest accuracy as individual classifiers:

# %%
for i in range(20):
    print("{:.4f}: {!r}".format(sorted_success_probs[i][1], tokenizer.decode([common_tokens[sorted_success_probs[i][0]]])))

# %% [markdown]
# ### Log-likelihood histogram with the "optimal" 40 tokens
# These are the 40 tokens that give the best accuracy when used as individual classifiers.
# 

# %%
N = 40
arm = "train"
training_cutoff, success_of_training_cutoff, _, _ = find_best_cutoff_and_plot(
    train_validate_author_to_personas_counters[arm],
    common_tokens[[ssp[0] for ssp in sorted_success_probs[:N]]],
    range_min=-15,
    range_max=-5,
    title=f"Distribution from the {N} highest-information tokens ({arm} set)",
    verbose=False
)
print(f"Training cutoff: {training_cutoff}, success: {success_of_training_cutoff}")

# %%
N = 60
arm = "val"
res = find_best_cutoff_and_plot(train_validate_author_to_personas_counters[arm], common_tokens[[ssp[0] for ssp in sorted_success_probs[:N]]],
                                range_min=-15, range_max=-5,
                                title=f"Distribution from the {N} highest-information tokens ({arm} set)",
                                possible_cutoffs=np.array([training_cutoff]))

# %% [markdown]
# ### A graph representing the accuracy of the classifier that uses the first $n$ best tokens
# "Best" in the sense that their 1-grams are the best lone classifiers.

# %%
first_best_tokens_success = []
for i in tqdm(range(100)):
    training_cutoff_temp, _, _, _ = find_best_cutoff_and_plot(train_validate_author_to_personas_counters["train"], common_tokens[[ssp[0] for ssp in sorted_success_probs[:i]]], verbose=False)
    _, validation_success, _, _ = find_best_cutoff_and_plot(train_validate_author_to_personas_counters["train"], common_tokens[[ssp[0] for ssp in sorted_success_probs[:i]]], possible_cutoffs=np.array([training_cutoff_temp]), verbose=False)
    first_best_tokens_success.append((i, validation_success))


# %%
first_best_tokens_success_arr = np.array(first_best_tokens_success)

plt.plot(first_best_tokens_success_arr[:,0], first_best_tokens_success_arr[:,1])
plt.xlabel("Number of tokens used by classifier")
plt.ylabel("Accuracy")
plt.title("Accuracy by number of high-information tokens used")
plt.show()


