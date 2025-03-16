#%% md
# # Decision Trees
# Train a decision tree classifier. The data are 80 log probabilities of the best 40 individual single token LL classifiers, one set of 40 for each persona.
# 
# Positive examples are created by pairing the log probabilities of the two personas of each author.
# 
# Negative examples are created by pairing the log probabilities of the two personas of different authors.
# 
# There are the same number of positive and negative examples.
#%%
import importlib

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier

from baseline_model import utils
from common import tokenization    

importlib.reload(utils)

tokenizer = tokenization.get_tokenizer()
#%%
train_validate_author_to_personas_counters = utils.get_train_validate_author_to_personas_counters(tokenizer)
forty_tokens_to_use = [token_int for _, _, token_int in utils.load_1000_most_common_tokens_sorted_by_1_gram_accuracies()][:40]
print("Top 40 tokens:")
print(" ".join([repr(tokenizer.decode([forty_tokens_to_use[i]])) for i in range(len(forty_tokens_to_use))]))


positive_negative_examples_pairs_of_log_nonz_probabilities = {}
for arm in ["train", "val"]:
    author_to_log_nonz_persona_pairs = utils.convert_counters_to_log_nonz_probs_in_username_to_persona_counters(train_validate_author_to_personas_counters[arm], forty_tokens_to_use)
    positive_negative_examples_pairs_of_log_nonz_probabilities[arm] = utils.create_positive_and_negative_examples_form_persona_pairs(author_to_log_nonz_persona_pairs)
#%%
def boosted_decision_tree(
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int32],
        X_validate: NDArray[np.float64],
        y_validate: NDArray[np.int32],
        n_estimators: int = 100,
        max_depth: int = 3
) -> tuple[GradientBoostingClassifier, float, float]:
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    training_accuracy = model.score(X_train, y_train)
    validation_accuracy = model.score(X_validate, y_validate)

    return model, training_accuracy, validation_accuracy
#%%
for trees in [100]:
    for depth in [2,3,4]:
        model, training_acc, val_acc = boosted_decision_tree(
            positive_negative_examples_pairs_of_log_nonz_probabilities["train"][0],
            positive_negative_examples_pairs_of_log_nonz_probabilities["train"][1],
            positive_negative_examples_pairs_of_log_nonz_probabilities["val"][0],
            positive_negative_examples_pairs_of_log_nonz_probabilities["val"][1],
            n_estimators=trees,
            max_depth=depth
        )
        print(f"Training accuracy: {training_acc:.4f} Validation accuracy: {val_acc:.4f} with {trees} trees and depth {depth}")

#%%
