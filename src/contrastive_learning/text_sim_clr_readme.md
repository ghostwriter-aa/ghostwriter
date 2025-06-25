## Loss function

Code is adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html

We will consider a loss function called InfoNCE loss.

In a batch if B pairs of feature vectors are given, we will have 2B feature vectors in total.

Given a pair of feature vectors $z_i$ and $z_j$ _of the same author_, the InfoNCE loss is defined as:

$$\ell_{i,j}=-\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2B}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)}=-\text{sim}(z_i,z_j)/\tau+\log\left[\sum_{k=1}^{2B}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)\right]$$

Here,

- $ \text{sim}(z_i,z_j) = \frac{z_i^\top \cdot z_j}{||z_i||\cdot||z_j||} $
- $\tau$ is a temperature parameter
- In both sums, we sum over all other $2B-1$ persona feature vectors except $z_i$, the persona we want to match.

In the code, we will sum over all $\sum_{i,j|author(i)=author(j)}\ell_{i,j}$ and minimize this sum.
