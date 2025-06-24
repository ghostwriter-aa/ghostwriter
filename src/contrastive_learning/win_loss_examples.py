# %% [markdown]
# Get win and loss examples from the validation set - trained using contrastive learning.

# %%
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from common import common_types as ct

# %%
same_author_sims = np.asarray(json.load(open("../../summary_data/same_author_sims_val.json")))
diff_author_sims = np.asarray(json.load(open("../../summary_data/different_author_sims_val.json")))

# %%
author_infos = ct.read_author_infos("../../data/suitable_author_infos_val.ndjson")

# %%
plt.figure(figsize=(6, 4))

# Plot both histograms with transparency (alpha) for overlay effect
sns.histplot(same_author_sims, alpha=0.6, label='Same author', color='C0', edgecolor='none', bins=40)
sns.histplot(diff_author_sims, alpha=0.6, label='Different author', color='C1', edgecolor='none', bins=40)

# Customize the plot
plt.xlabel('Similarity score')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)

# Set x-axis limits to [0,1] since your data is in this range
plt.xlim(-0.5, 1)

plt.tight_layout()
plt.show()

# %%
def print_persona_comments_and_submissions(persona: ct.PersonaInfo, num_docs_to_show: int = 5) -> None:
    shown = 0
    for comment in persona.comments:
        print(f"    Comment: {comment.body!r}")
        shown += 1
        if shown >= num_docs_to_show:
            return
    for submission in persona.submissions:
        print(f"    Post: {submission.title!r}: {submission.selftext!r}")
        shown += 1
        if shown >= num_docs_to_show:
            return


# %%
# Examples of correct matches with high scores
threshold = 0.7
num_matches = np.sum(same_author_sims > threshold)
num_mismatches = np.sum(diff_author_sims > threshold)
print(f"Number of matches with score > {threshold}: {num_matches}")
print(f"Number of mismatches with score > {threshold}: {num_mismatches}")
print(f"Accuracy: {num_matches / (num_matches + num_mismatches) * 100:.1f}%")
wins_idx = np.where(same_author_sims > threshold)

for win_idx in wins_idx[0][:5]:
    win_author = author_infos[win_idx]
    print(f"Author: {win_author.username} - Similarity {same_author_sims[win_idx]:.3f} - Persona 0 - Subreddit {win_author.personas[0].subreddit}")
    print_persona_comments_and_submissions(win_author.personas[0])
    print(f"Author: {win_author.username} - Similarity {same_author_sims[win_idx]:.3f} - Persona 1 - Subreddit {win_author.personas[1].subreddit}")
    print_persona_comments_and_submissions(win_author.personas[1])
    print("=" * 50)

# %%
# Examples of correct matches with low scores
threshold = 0.0
num_matches = np.sum(same_author_sims < threshold)
num_mismatches = np.sum(diff_author_sims < threshold)
print(f"Number of matches with score < {threshold}: {num_matches}")
print(f"Number of mismatches with score < {threshold}: {num_mismatches}")
print(f"Accuracy: {num_matches / (num_matches + num_mismatches) * 100:.1f}%")
losses_idx = np.where(same_author_sims < threshold)

for loss_idx in losses_idx[0][:5]:
    loss_author = author_infos[loss_idx]
    print(f"Author: {loss_author.username} - Similarity {same_author_sims[loss_idx]:.3f} - Persona 0 - Subreddit {loss_author.personas[0].subreddit}")
    print_persona_comments_and_submissions(loss_author.personas[0])
    print(f"Author: {loss_author.username} - Similarity {same_author_sims[loss_idx]:.3f} - Persona 1 - Subreddit {loss_author.personas[1].subreddit}")
    print_persona_comments_and_submissions(loss_author.personas[1])
    print("=" * 50)

# %%
# Examples of incorrect matches with high scores
threshold = 0.70
num_matches = np.sum(same_author_sims > threshold)
num_mismatches = np.sum(diff_author_sims > threshold)
print(f"Number of matches with score > {threshold}: {num_matches}")
print(f"Number of mismatches with score > {threshold}: {num_mismatches}")
print(f"Accuracy: {num_matches / (num_matches + num_mismatches) * 100:.1f}%")
losses_idx = np.where(diff_author_sims > threshold)

for loss_idx in losses_idx[0][:5]:
    loss_author1 = author_infos[loss_idx]
    loss_author2 = author_infos[loss_idx+1]
    print(f"Author {loss_idx}: {loss_author1.username} - Similarity {diff_author_sims[loss_idx]:.3f} - Persona 1 - Subreddit {loss_author1.personas[0].subreddit}")
    print_persona_comments_and_submissions(loss_author1.personas[0], num_docs_to_show=5)
    print(f"Author {loss_idx+1}: {loss_author2.username} - Similarity {diff_author_sims[loss_idx]:.3f} - Persona 0 - Subreddit {loss_author2.personas[1].subreddit}")
    print_persona_comments_and_submissions(loss_author2.personas[1], num_docs_to_show=5)
    print("=" * 50)

# %%



