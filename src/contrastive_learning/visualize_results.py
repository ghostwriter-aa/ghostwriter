# %%
import subprocess

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics.pairwise import cosine_similarity


from contrastive_learning.text_sim_clr import TextSimCLR, compute_feature_representation
from common.common_types import AuthorEmbeddingCollection
from contrastive_learning.contrastive_learner import create_tensor_pairs
from common.distribution_comparison_utils import plot_distributions, find_best_cutoff

# %% [markdown]
# ### Note:
# This notebook assumes you have trained a model with `python3 contrastive_learning/contrastive_learner.py` (See the 
# readme of contrastive learning).
# 
# After training, there will be a new version of the trained models under the directory `../../data/checkpoints/TextSimCLR/lightning_logs/`. For the first trained model, it will be 0.

# %%
# Load embeddings
with open("../../data/embeddings/e5_embedding_train.json", "rt") as f:
    train_embeddings = AuthorEmbeddingCollection.from_json(f.read())

with open("../../data/embeddings/e5_embedding_val.json", "rt") as f:
    val_embeddings = AuthorEmbeddingCollection.from_json(f.read())

# Create tensor pairs from embeddings
train_persona_pairs_tensors = create_tensor_pairs(train_embeddings.author_embeddings)
val_persona_pairs_tensors = create_tensor_pairs(val_embeddings.author_embeddings)

# %%
VERSION = -1 # Get the latest version, -1 for the latest version

# Get the latest version directory
result = subprocess.run(['ls', '../../data/checkpoints/TextSimCLR/lightning_logs/'], 
                       capture_output=True, text=True)
versions = result.stdout.strip().split('\n')
latest_version = sorted(versions, key=lambda x: int(x.split('_')[1]))[VERSION]

# Get the latest checkpoint file
result = subprocess.run(['ls', f'../../data/checkpoints/TextSimCLR/lightning_logs/{latest_version}/checkpoints/'], 
                       capture_output=True, text=True)
checkpoints = result.stdout.strip().split('\n')
latest_checkpoint = sorted(checkpoints)[-1]  # Get the last checkpoint

# Extract epoch and step from checkpoint filename
# Format is typically: epoch=X-step=Y.ckpt
epoch_step = latest_checkpoint.split('.')[0]  # Remove .ckpt
epoch = int(epoch_step.split('=')[1].split('-')[0])
step = int(epoch_step.split('=')[2])

print(f"Using version: {latest_version}")
print(f"Epoch: {epoch}")
print(f"Step: {step}")

# Update the constants
MODEL_VERSION = latest_version
EPOCH = epoch
STEP = step

model = TextSimCLR.load_from_checkpoint(f"../../data/checkpoints/TextSimCLR/lightning_logs/{MODEL_VERSION}/checkpoints/epoch={EPOCH}-step={STEP}.ckpt")
print(model)

# %%
df = pd.read_csv(f'../../data/checkpoints/TextSimCLR/lightning_logs/{MODEL_VERSION}/metrics.csv')

# Create plots of training and validation metrics over time
plt.figure(figsize=(15, 5))

# Plot 1: Top-5 Accuracy
plt.subplot(1, 2, 1)
plt.plot(df['step'], df['train_acc_top5'], label='Train Acc Top5', marker='o', markersize=3)
plt.plot(df['step'], df['val_acc_top5'], label='Val Acc Top5', marker='x', markersize=3)
plt.title('Top-5 Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Loss values
plt.subplot(1, 2, 2)
plt.plot(df['step'], df['train_loss'], label='Train Loss', marker='o', markersize=3)
plt.plot(df['step'], df['val_loss'], label='Val Loss', marker='x', markersize=3)
plt.title('Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# %%
# Compute feature representations for the validation set
val_persona_pair_feature_representations = compute_feature_representation(model, val_persona_pairs_tensors)

# %%
def get_cosine_sims(embeddings: list[list[torch.Tensor]])-> tuple[np.ndarray, np.ndarray]:

    # shape: authors, personas (2), embeddings   (dimensions)
    embeddings = np.asarray(embeddings)

    # Calculate cosine similarity between each author's pair of embeddings
    # Reshape from (authors, 2, embeddings) to (authors*2, embeddings)
    flattened_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    cosine_sims = cosine_similarity(flattened_embeddings)

    # Get first off-diagonal entries (entries right below diagonal)
    first_off_diag = np.diag(cosine_sims, k=-1)
    same_author_sims = first_off_diag[::2]
    diff_author_sims = first_off_diag[1::2].tolist()
    for i in range(2, len(cosine_sims)):
        diff_author_sims.extend(cosine_sims[i, :i-1])

    diff_author_sims = np.array(diff_author_sims)

    return same_author_sims, diff_author_sims

# %%
same_author_sims, diff_author_sims = get_cosine_sims(val_persona_pair_feature_representations)
plot_distributions(same_author_sims, diff_author_sims, "Distribution of Cosine Similarities WITH feature representations", "Cosine Similarity", "Density", "Same Author", "Different Authors")
best_cutoff, best_success_rate = find_best_cutoff(same_author_sims, diff_author_sims)



# %% [markdown]
# We have ~3700 validation samples, and a success probability of at least 0.8. We expect a variation of this success prob of about
# $\sqrt{\frac{(0.8)(1-0.8)}{3700}}<0.007$.
# Below, we can indeed see that this is the case.

# %%
same_author_sims, diff_author_sims = get_cosine_sims(val_persona_pairs_tensors)
plot_distributions(same_author_sims, diff_author_sims, "Distribution of Cosine Similarities Between Persona Embeddings using the plain E5 model", "Cosine Similarity", "Density", "Same Author", "Different Authors")
best_cutoff, best_success_rate = find_best_cutoff(same_author_sims, diff_author_sims)


