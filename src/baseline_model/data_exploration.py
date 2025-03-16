#%% md
# # Data exploration - plot token probabilities of personas of an author
#%%
import json

import matplotlib.pyplot as plt
import numpy as np
import tiktoken

from baseline_model import utils
from common import tokenization

from baseline_model.token_stats import TokenStats
#%%
plt.rcParams["font.family"] = "Arial Unicode MS"
#%%
def plot_comparison_token_stats(
    stats1: TokenStats,
    stats2: TokenStats,
    tokenizer: tiktoken.Encoding,
    title: str = "",
    label1: str = "",
    label2: str = "",
    start_token: int = 0,
    end_token: int = 20,
) -> None:
    """
    Plot the comparison of token statistics between two TokenStats objects.
    the X axis being the tokens, and the bars represent the frequency of occurrence of each token in the token sets (red
    and blue bars for the two TokenStats objects).
    """
    if list(stats1.token_count.keys()) != list(stats2.token_count.keys()):
        raise ValueError("The token sets of the two TokenStats objects are different.")
    width = 0.4
    x_pos = np.arange(end_token - start_token)
    stats = [stats1, stats2]
    labels = [label1, label2]
    colors = ["skyblue", "lightcoral"]
    xs = [x_pos - width / 2, x_pos + width / 2]
    plt.figure(figsize=(15, 5))
    for stat, color, x, label in zip(stats, colors, xs, labels):
        plt.bar(x, stat.token_freq[start_token:end_token], width, color=color, alpha=0.8, label=label)
        plt.errorbar(
            x,
            stat.token_freq[start_token:end_token],
            yerr=[stat.lower_ci[start_token:end_token], stat.upper_ci[start_token:end_token]],
            fmt="none",
            color="black",
            capsize=1,
            linewidth=0.5,
            capthick=0.5,
        )
    plt.ylabel("Occurrence frequency")
    plt.ylim(0, 0.3)
    plt.xticks(
        x_pos,
        [repr(tokenizer.decode([s])) for s in list(stats1.token_count.keys())[start_token:end_token]],
        rotation=90,
    )
    if label1 or label2:
        plt.legend()
    if title:
        plt.title(title)
    plt.show()

#%%
train_validate_author_to_personas_counters = utils.get_train_validate_author_to_personas_counters(tokenization.get_tokenizer())
with open("../../data/tiktoken_counts_top_1000.json") as f:
    tokens_to_use =[token for token, _ in json.load(f)]
#%%
author_to_persona_stats = {}
for author_username, author_counters in train_validate_author_to_personas_counters["train"].items():
    author_to_persona_stats[author_username] = (
        TokenStats.from_counts(author_counters[0], tokens_to_use),
        TokenStats.from_counts(author_counters[1], tokens_to_use),
    )
#%%
author_index = 42 # Change this to see the token stats comparison of different authors
author_names = list(author_to_persona_stats.keys())
author_name = author_names[author_index]
personas_stats = author_to_persona_stats[author_name]
plot_comparison_token_stats(personas_stats[0], personas_stats[1], tokenization.get_tokenizer(), title=author_name, label1="persona_1", label2="persona_2")
#%%
