#%% md
# # Data Exploration
# - Plot 2D histogram of the log likelihoods of a token for the two personas of each author.
# - Create a plot of the first three most influential components of the PCA of positive and negative examples. The data is of the log probabilities of the top 40 tokens for two personas (of the same and different authors).
# 
#%%
import importlib

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

from baseline_model.token_stats import TokenStats
from baseline_model import utils
from common import tokenization    

importlib.reload(utils)

tokenizer = tokenization.get_tokenizer()
#%%
train_validate_author_to_personas_counters = utils.get_train_validate_author_to_personas_counters(tokenizer)
forty_tokens_to_use = [token_int for _, _, token_int in utils.load_1000_most_common_tokens_sorted_by_1_gram_accuracies()][:40]
print("Top 40 tokens:")
print(" ".join([repr(tokenizer.decode([forty_tokens_to_use[i]])) for i in range(len(forty_tokens_to_use))]))
#%%
author_train_stats = [
    (TokenStats.from_counts(train_validate_author_to_personas_counters["train"][author_username][0], forty_tokens_to_use).log_nonz_token_freq_with_excluded_tokens,
     TokenStats.from_counts(train_validate_author_to_personas_counters["train"][author_username][1], forty_tokens_to_use).log_nonz_token_freq_with_excluded_tokens)
    for author_username in train_validate_author_to_personas_counters["train"].keys()
]
#%%
for j in range(3):
    fig = plt.figure(figsize=(5, 5), facecolor='white')
    fig.patch.set_facecolor('white')
    plt.hist2d([author_train_stats[i][0][j] for i in range(len(author_train_stats))], [author_train_stats[i][1][j] for i in range(len(author_train_stats))], bins=30)
    plt.set_cmap('gray_r')
    plt.title(f'Log probability correlation for token {j + 1} [{tokenizer.decode([forty_tokens_to_use[j]])}]')
    plt.xlabel('Persona 1 log probability')
    plt.ylabel('Persona 2 log probability')
    plt.show()
#%% md
# ## PCA analysis
#%%
def plot3d(
        X,
        labels=None,
        title='Plot of three components',
        components=(0, 1, 2),
):
    labels = labels if labels is not None else {'C1': '1st Axis', 'C2': '2nd Axis', 'C3': '3rd Axis'}
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'C1': X[:, components[0]],
        'C2': X[:, components[1]],
        'C3': X[:, components[2]],
        'Label': ['Class ' + str(label) for label in labels]
    })
    
    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        plot_df, 
        x='C1', 
        y='C2', 
        z='C3',
        color='Label',
        color_discrete_map={'Class 0': 'blue', 'Class 1': 'red'},
        title=title,
        labels=labels,
        size_max=0.01  # This controls the maximum marker size
    )
    
    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='1st Component',
            yaxis_title='2nd Component',
            zaxis_title='3rd Component'
        ),
        legend_title_text='Classes'
    )
    
    # adjust the height of the plot
    
    fig.update_layout(height=500)
    
    # Show the plot
    fig.show()
#%%
def analyze_and_visualize_svd_interactive(X, labels):
    """
    Perform SVD/PCA analysis and create interactive 3D visualization
    
    Parameters:
    X: numpy array of shape (n_samples, n_features)
    labels: numpy array of shape (n_samples,) with binary values
    """
    # Perform PCA
    pca = PCA()
    X_transformed = pca.fit_transform(X)
    
    print(f"\nCumulative variance explained by top 3 components: "
          f"{np.sum(pca.explained_variance_ratio_[:3]*100):.2f}%")
    
    print("\nVariance explained by each of the top 5 components:")
    # print in one line using list comprehension
    print("    ".join([f"Component {i}: {var:.4f} ({var*100:.2f}%)" for i, var in enumerate(pca.explained_variance_ratio_[:5], 1)]))
    
    
    plot3d(X_transformed, labels)
#%%
author_to_log_nonz_persona_pairs = utils.convert_counters_to_log_nonz_probs_in_username_to_persona_counters(train_validate_author_to_personas_counters["train"], forty_tokens_to_use)
positive_negative_examples_pairs_of_log_nonz_probabilities = utils.create_positive_and_negative_examples_form_persona_pairs(author_to_log_nonz_persona_pairs)
#%%
analyze_and_visualize_svd_interactive(positive_negative_examples_pairs_of_log_nonz_probabilities[0], positive_negative_examples_pairs_of_log_nonz_probabilities[1])
#%%
positive_negative_examples_pairs_of_log_nonz_probabilities[0]
#%%
    