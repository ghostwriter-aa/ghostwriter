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
