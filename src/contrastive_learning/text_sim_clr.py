from typing import Any, List, Tuple, cast

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


class TextSimCLR(pl.LightningModule):
    lr: float
    temperature: float
    weight_decay: float
    max_epochs: int
    g: nn.Sequential
    hparams: Any  # type: ignore[attr-defined,unused-ignore]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        lr: float,
        temperature: float,
        weight_decay: float,
        max_epochs: int = 15,
    ):
        """
        Initialize the TextSimCLR model.

        Args:
            input_dim: Dimension of the input embeddings
            hidden_dim: Dimension of the hidden layer. If 0, no hidden layer is used
            output_dim: Dimension of the output embeddings
            lr: Learning rate
            temperature: Temperature parameter for the InfoNCE loss
            weight_decay: Weight decay for the optimizer
            max_epochs: Maximum number of epochs to train for
        """
        super().__init__()
        # Note: save_hyperparameters() is a pytorch lightning function that saves the hyperparameters (like temperature)
        # automagically.
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        # Apply Linear->ReLU->Linear on persona embedding(s) to get the feature vector / head
        if hidden_dim == 0:
            self.g = nn.Sequential(
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.g = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim),
            )

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch: tuple[torch.Tensor, torch.Tensor], mode: str = "train") -> torch.Tensor:
        """
        Input: batch - a pair of feature tensors.
        If the batch is of size B, each persona has an embedding tensor of shape S, the batch has shape (2, B, S)
        """
        text_vecs = torch.cat(batch, dim=0)
        # Here, the shape of text_vecs is (B*2, S)
        # If we signify authors by a_1,...,a_B, and personas by p_1,p_2, then
        # the variable text_vecs will be [a_1_p_1, a_2_p_1, ..., a_B_p_1, a_1_p_2, a_2_p_2, ..., a_B_p_2]
        # Apply g() to the text vectors. Let F be the shape of the output of g(). Then, feats has shape (B*2, F)
        feats = self.g(text_vecs)

        # Calculate cosine similarity.
        # Here, feats[:,None,:] is reshaping feats to (B*2, 1, F), and feats[None,:,:] is reshaping feats to (1, B*2, F)
        # Then, cos_sim has shape (B*2, B*2)
        cos_sim = cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity of a feature vector to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        # For each row, calculate the loss.
        # That is, there will be 2B losses - two for each pair of feature vectors of the same author
        #  (once thinking of the first feature vector as the one to match, and second of the second feature vector of
        #  the author as the one to match)
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        # Average over the above losses.
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],  # Put in the first position the positive example
                cos_sim.masked_fill(pos_mask, -9e15),
            ],
            dim=-1,
        )
        # (Arg)Sort each row, and then take the index of the positive example (which initially was in the first
        # position)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.info_nce_loss(batch, mode="val")


def compute_persona_pairs_feature_representation(
    model: TextSimCLR, persona_pairs: list[list[torch.Tensor]]
) -> NDArray[np.float64]:
    """
    Compute the feature representations / heads of the given persona pairs.

    Args:
        model: The `TextSimCLR` model that maps raw persona embeddings
            to the feature space (trained with contrastive learning).
        persona_pairs: A list of entries of the type [persona_1_embedding, persona_2_embedding], where
            persona_1_embedding and persona_2_embedding are the persona embeddings of an author.

    Returns:
        A numpy array of shape (num_authors, 2, F), where F is the feature space dimension
    """
    if len(persona_pairs) == 0:
        return np.array([])

    # Ensure every author has exactly two persona tensors.
    if not all(len(pair) == 2 for pair in persona_pairs):
        raise ValueError("Each element in persona_pairs must contain exactly two persona tensors")

    # Flatten all persona tensors and project in a single batch.
    # The shape of the tensor will be (2 * num_authors, D), where D is the persona embedding dimension
    flattened_personas = []
    for pair in persona_pairs:
        flattened_personas.extend(pair)
    tensor_flattened_personas = torch.stack(flattened_personas, dim=0)

    # `projected_all` will have shape (2 * num_authors, F), where F is the feature space / head dimension
    projected_all = compute_feature_representation(model, tensor_flattened_personas).cpu().numpy()

    # Reshape to (num_authors, 2, F)
    num_authors = len(persona_pairs)
    return projected_all.reshape(num_authors, 2, projected_all.shape[1])


def compute_feature_representation(model: TextSimCLR, embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute the feature representation / head for a batch of persona embeddings.

    Args:
        model: The `TextSimCLR` model that maps raw persona embeddings
            to the feature space (trained with contrastive learning).
        embeddings: A **2-D** tensor of shape (N, D) containing the
            raw persona embeddings (N personas, each with embedding dimension D)
            after applying an off the shelf embedding model.

    Returns:
        A feature representation vector / head after applying `g`, the network trained with
        contrastive learning.
        It is a tensor of shape (N, F) where F is the dimensions of the feature space / head.
    """
    with torch.no_grad():
        input_tensor = embeddings.to(model.device)
        # In the PyTorch type stubs, nn.Module.__call__ is declared to return Any.
        # In our case, the last layer of `model.g` is a Linear layer, which returns a tensor of type torch.Tensor.
        # So, we can cast the result to a torch.Tensor.
        return cast(torch.Tensor, model.g(input_tensor)).cpu()
