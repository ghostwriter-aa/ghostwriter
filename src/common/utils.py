import numpy as np
from numpy.typing import NDArray


def random_derangement(n: int, seed: int = 424242) -> NDArray[np.int64]:
    """
    Generate a random derangement of n elements.
    Returns an array where no element is in its original position.
    """
    np.random.seed(seed)  # For consistency across runs.
    while True:
        perm = np.random.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


def find_best_cutoff(
    dist_1_samples: NDArray[np.floating],
    dist_2_samples: NDArray[np.floating],
    possible_cutoffs: NDArray[np.floating],
) -> tuple[float, float]:
    """
    Finds the best cutoff for maximizing accuracy when classifying between samples of two distributions.
    Args:
        dist_1_samples: samples from the first distribution
        dist_2_samples: samples from the second distribution
        possible_cutoffs: the possible cutoffs to consider
    Returns:
        cutoff: The cutoff which maximizes accuracy.
        accuracy: The accuracy obtained with said cutoff.
    """
    results = []
    dist_1_samples.sort()
    dist_2_samples.sort()
    for cutoff in possible_cutoffs:
        dist_1_prob = 1 - (np.searchsorted(dist_1_samples, cutoff) / len(dist_1_samples))
        dist_2_prob = 1 - (np.searchsorted(dist_2_samples, cutoff) / len(dist_2_samples))
        success_with_cutoff = 0.5 * (dist_1_prob + 1 - dist_2_prob)
        results.append((cutoff, dist_1_prob, dist_2_prob, success_with_cutoff))
    results.sort(key=lambda x: abs(x[3] - 0.5), reverse=True)
    return results[0][0], results[0][3]
