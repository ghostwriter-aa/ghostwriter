import argparse
import os
from pathlib import Path
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from common.common_types import AuthorEmbedding, AuthorEmbeddingCollection
from common.gpu_utils import detect_device
from contrastive_learning.author_text_pair_dataset import AuthorTextPairsDataset
from contrastive_learning.text_sim_clr import TextSimCLR

NUM_WORKERS: int = os.cpu_count()  # type: ignore
CURRENT_FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


def setup_environment() -> str:
    """Setup the environment for reproducible results."""
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = detect_device()
    print("Number of CPU cores:", NUM_WORKERS)
    return device


def create_tensor_pairs(author_embeddings: List[AuthorEmbedding]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert author embeddings into tensor pairs for contrastive learning.

    Args:
        author_embeddings: List of AuthorEmbedding objects

    Returns:
        List of tuples (tensor1, tensor2) where each tensor is a persona embedding
    """
    tensor_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for author in author_embeddings:
        if len(author.persona_embeddings) != 2:
            raise ValueError(
                f"Author {author.username} has {len(author.persona_embeddings)} persona embeddings, expected 2"
            )
        # Convert numpy arrays to torch tensors
        tensor1 = torch.tensor(author.persona_embeddings[0].embedding, dtype=torch.float32)
        tensor2 = torch.tensor(author.persona_embeddings[1].embedding, dtype=torch.float32)
        tensor_pairs.append((tensor1, tensor2))
    return tensor_pairs


def train_simclr(
    train_author_embeddings: List[AuthorEmbedding],
    val_author_embeddings: List[AuthorEmbedding],
    batch_size: int,
    max_epochs: int,
    **kwargs: Any,
) -> Tuple[TextSimCLR, pl.Trainer]:
    """Train the SimCLR model."""
    # Setup environment
    device = setup_environment()

    # Map device string to accelerator type
    if device == "cuda":
        accelerator = "gpu"
    elif device == "mps":
        accelerator = "mps"
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CURRENT_FILE_PATH.parent.parent / "data" / "checkpoints", "TextSimCLR"),
        accelerator=accelerator,
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
    )

    # Convert author embeddings to tensor pairs
    train_tensor_pairs = create_tensor_pairs(train_author_embeddings)
    val_tensor_pairs = create_tensor_pairs(val_author_embeddings)

    # Create datasets
    train_dataset = AuthorTextPairsDataset(train_tensor_pairs)
    val_dataset = AuthorTextPairsDataset(val_tensor_pairs)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS
    )

    pl.seed_everything(42)  # To be reproducible
    model = TextSimCLR(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = TextSimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)  # type: ignore
    return model, trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TextSimCLR model on author embeddings.")
    parser.add_argument(
        "--train-embedding-file",
        type=str,
        default=CURRENT_FILE_PATH.parent.parent / "data" / "embeddings" / "e5_embedding_train_new.json",
        help="Path to the training embedding file.",
    )
    parser.add_argument(
        "--val-embedding-file",
        type=str,
        default=CURRENT_FILE_PATH.parent.parent / "data" / "embeddings" / "e5_embedding_val_new.json",
        help="Path to the validation embedding file.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Dimension of the hidden layer.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=128,
        help="Dimension of the output embedding.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of epochs to train for.",
    )
    # Add other arguments like batch_size, epochs, etc., if needed
    args = parser.parse_args()

    # Load and prepare data
    print(f"Loading train embeddings from {args.train_embedding_file}...")
    with open(args.train_embedding_file, "rt") as f:
        train_persona_pairs_tensors = AuthorEmbeddingCollection.from_json(f.read())

    print(f"Loading validation embeddings from {args.val_embedding_file}...")
    with open(args.val_embedding_file, "rt") as f:
        val_persona_pairs_tensors = AuthorEmbeddingCollection.from_json(f.read())

    assert train_persona_pairs_tensors.embedding_dim == val_persona_pairs_tensors.embedding_dim

    # Train model
    train_simclr(
        train_persona_pairs_tensors.author_embeddings,
        val_persona_pairs_tensors.author_embeddings,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        input_dim=train_persona_pairs_tensors.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        lr=1e-3,
        temperature=0.07,
        weight_decay=1e-4,
    )


if __name__ == "__main__":
    main()
