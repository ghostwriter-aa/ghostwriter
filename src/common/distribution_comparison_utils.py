from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def find_best_cutoff(
    dist_1_samples: np.ndarray[Any, np.dtype[np.float64]] | Sequence[float],
    dist_2_samples: np.ndarray[Any, np.dtype[np.float64]] | Sequence[float],
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Find the best cutoff for maximizing accuracy when classifying between samples of two distributions.
    Args:
      dist_1_samples: samples from the first distribution
      dist_2_samples: samples from the second distribution
      verbose: whether to print the results

    Returns:
        - cutoff: the best cutoff for maximizing accuracy
        - success_probability: the success probability at the best cutoff
    """
    cutoff_and_succ_prob = []
    # Make copies of the arrays to ensure they are writable
    dist_1_samples = np.sort(dist_1_samples)
    dist_2_samples = np.sort(dist_2_samples)

    std_dist_1 = np.std(dist_1_samples)
    std_dist_2 = np.std(dist_2_samples)

    min_range = min(dist_1_samples.min() - 3 * std_dist_1, dist_2_samples.min() - 3 * std_dist_2)
    max_range = max(dist_1_samples.max() + 3 * std_dist_1, dist_2_samples.max() + 3 * std_dist_2)

    possible_cutoffs = np.linspace(min_range, max_range, 100)

    for cutoff in possible_cutoffs:
        dist_1_success_prob = 1 - (np.searchsorted(dist_1_samples, cutoff) / len(dist_1_samples))
        dist_2_success_prob = 1 - (np.searchsorted(dist_2_samples, cutoff) / len(dist_2_samples))
        success_with_cutoff = 0.5 * (dist_1_success_prob + 1 - dist_2_success_prob)
        cutoff_and_succ_prob.append((cutoff, success_with_cutoff))

    # Sort results by success probability in descending order
    cutoff_and_succ_prob.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print("\nDistribution Statistics:")
        print(
            f"Distribution 1 - Mean: {np.mean(dist_1_samples):.4f}, Median: {np.median(dist_1_samples):.4f}, "
            f"Std: {std_dist_1:.4f}"
        )
        print(
            f"Distribution 2 - Mean: {np.mean(dist_2_samples):.4f}, Median: {np.median(dist_2_samples):.4f}, "
            f"Std: {std_dist_2:.4f}"
        )
        print("\nBest five cutoffs:")
        for result in cutoff_and_succ_prob[:5]:
            print("Cutoff: {:>5}, Success Probability: {:.4f}".format(*result))
    return cutoff_and_succ_prob[0][0], cutoff_and_succ_prob[0][1]


def plot_distributions(
    dist_1_samples: np.ndarray[Any, np.dtype[np.float64]],
    dist_2_samples: np.ndarray[Any, np.dtype[np.float64]],
    title: str,
    xlabel: str,
    ylabel: str,
    label_1: str,
    label_2: str,
) -> None:
    """Plot two distributions with their histograms, normal approximations, and medians.

    Args:
        dist_1_samples: Samples from the first distribution
        dist_2_samples: Samples from the second distribution
        title: Title of the plot
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        label_1: Label for the first distribution
        label_2: Label for the second distribution
    """
    mean_dist_1, median_dist_1, std_dist_1 = np.mean(dist_1_samples), np.median(dist_1_samples), np.std(dist_1_samples)
    mean_dist_2, median_dist_2, std_dist_2 = np.mean(dist_2_samples), np.median(dist_2_samples), np.std(dist_2_samples)

    plt.figure(figsize=(10, 6))

    # Plot histograms
    hist1 = plt.hist(dist_1_samples, bins=20, alpha=0.6, label=label_1, density=True)
    hist2 = plt.hist(dist_2_samples, bins=20, alpha=0.6, label=label_2, density=True)

    # Get colors from histograms
    color1 = hist1[2][0].get_facecolor()  # type: ignore
    color2 = hist2[2][0].get_facecolor()  # type: ignore

    # Plot normal distributions
    x = np.linspace(
        min(dist_1_samples.min(), dist_2_samples.min()), max(dist_1_samples.max(), dist_2_samples.max()), 100
    )
    plt.plot(
        x,
        1 / (std_dist_1 * np.sqrt(2 * np.pi)) * np.exp(-((x - mean_dist_1) ** 2) / (2 * std_dist_1**2)),
        color=color1,
        linestyle=":",
        alpha=0.8,
    )
    plt.plot(
        x,
        1 / (std_dist_2 * np.sqrt(2 * np.pi)) * np.exp(-((x - mean_dist_2) ** 2) / (2 * std_dist_2**2)),
        color=color2,
        linestyle=":",
        alpha=0.8,
    )

    # Plot medians
    plt.axvline(x=median_dist_1, color=color1, linestyle=":", alpha=0.8, label=f"{label_1} median")
    plt.axvline(x=median_dist_2, color=color2, linestyle=":", alpha=0.8, label=f"{label_2} median")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
