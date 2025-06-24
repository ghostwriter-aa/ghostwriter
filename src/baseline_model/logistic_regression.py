# %% [markdown]
# # Logistic regression model
# Train a logistic regression model on the following features:
# $
# [\frac{N_1 \log p_1}{\sum_i N_i}, ...]
# $
# where $p_i$ are from one persona, and $N_i$ are from another persona.
# 
# Thus, the minimum performance should be that of log likelihood (as the logistic regression model can recover all ones coefficients).  

# %%
import importlib
from typing import Dict, Tuple, List, Sequence
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from baseline_model.token_stats import TokenStats
from baseline_model import utils
from common import tokenization    

importlib.reload(utils)

tokenizer = tokenization.get_tokenizer()

# %%
def create_positive_and_negative_examples(
    authors_features: Dict[str, Tuple[Counter[int], Counter[int]]],
    token_list: Sequence[int]
) -> Tuple[NDArray[np.float64], NDArray[np.int32]]:
    
    X_list: List[NDArray[np.float64]] = []
    y_list: List[int] = []
    
    # Create positive samples (same author)
    for author, (counter_1, counter_2) in authors_features.items():
        count_2 = (TokenStats.from_counts(counter_2, token_list).token_count_array)
        features = (
                TokenStats.from_counts(counter_1, token_list).log_nonz_token_freq_with_excluded_tokens * count_2 / sum(count_2)
                if sum(count_2) > 0 else np.zeros(len(count_2))
        )
        if sum(count_2) == 0:
            print(author)
        X_list.append(features)
        y_list.append(1)
        
    print(f"Created {len(y_list)} positive samples")

    # Create negative samples (different authors)
    # First, collect all first and second arrays separately
    first_arrays = [features[0] for features in authors_features.values()]
    second_arrays = [features[1] for features in authors_features.values()]
    
    # Get derangement indices and use them to permute second arrays
    derangement = utils.random_derangement(len(second_arrays))
    permuted_second = [second_arrays[i] for i in derangement]
    
    # Create negative samples using the deranged pairs
    for first, permuted in zip(first_arrays, permuted_second):
        permuted_count = (TokenStats.from_counts(permuted, token_list).token_count_array)
        features = TokenStats.from_counts(first, token_list).log_nonz_token_freq_with_excluded_tokens * permuted_count / sum(permuted_count)\
            if sum(permuted_count) > 0 else np.zeros(len(permuted_count))
        X_list.append(features)
        y_list.append(0)
    
    print(f"Created {len(y_list)} training samples (added negative examples)")
    
    # Convert lists to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list, dtype=np.int32)
    
    return X, y
    

# %%
train_validate_author_to_personas_counters = utils.get_train_validate_author_to_personas_counters(tokenization.get_tokenizer())
forty_tokens_to_use = [
    token_int for _, _, token_int in utils.load_1000_most_common_tokens_sorted_by_1_gram_accuracies()
][:40]
print("Top 40 tokens:")
print(" ".join([repr(tokenizer.decode([forty_tokens_to_use[i]])) for i in range(len(forty_tokens_to_use))]))


positive_negative_examples_pairs_of_log_nonz_probabilities = {}
for arm in ["train", "val"]:
    positive_negative_examples_pairs_of_log_nonz_probabilities[arm] = create_positive_and_negative_examples(train_validate_author_to_personas_counters[arm], forty_tokens_to_use)

# %%
X_train, y_train = positive_negative_examples_pairs_of_log_nonz_probabilities["train"]
X_validate, y_validate = positive_negative_examples_pairs_of_log_nonz_probabilities["val"]

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# Score
print(f"Training accuracy: {lr_model.score(X_train, y_train):.4f}")
print(f"Validation accuracy: {lr_model.score(X_validate, y_validate):.4f}")


