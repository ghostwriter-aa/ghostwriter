import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

from common import common_types as ct
from contrastive_learning import embeddings

CURRENT_FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to lists
        return json.JSONEncoder.default(self, obj)


def main(
    embedder_type: str,
    output_files_path_pattern: str,
    train_file: str,
    val_file: str,
    authors_chunk_size: int,
    batch_size: int,
) -> None:
    """
    Create embeddings for authors in train and validation sets using specified embedder.

    Args:
        embedder_type: Type of embedder to use
        output_files_path_pattern: Name pattern for output files. The {arm} placeholder will be replaced with
                                   "train" or "val"
        train_file: Path to the file containing training author information
        val_file: Path to the file containing validation author information
        authors_chunk_size: Number of authors to embed in each chunk. Note: the reason for this is explained in
                            AveragePostsStrategy.compute_authors_embeddings.
        batch_size: Batch size for embedding. Choose according to the memory of the GPU.
    """
    # Select embedder based on type
    embedder: embeddings.Embedder
    if embedder_type.lower() == "nomic":
        embedder = embeddings.NomicEmbedder()
    elif embedder_type.lower() == "e5":
        embedder = embeddings.E5Embedder()
    elif embedder_type.lower() == "e5-multilingual-large-instruct":
        embedder = embeddings.E5MultilingualLargeInstructEmbedder()
    else:
        raise ValueError(
            f"Unsupported embedder type: {embedder_type}. "
            f"Currently only 'nomic', 'e5' and 'e5-multilingual-large-instruct' are supported."
        )

    print("Loading author information...")
    authors_by_arm = {"train": ct.read_author_infos(Path(train_file)), "val": ct.read_author_infos(Path(val_file))}

    embedding_strategy = embeddings.AveragePostsStrategy()

    print(f"Creating embeddings using {embedder_type} embedder...")
    text_embeddings = embeddings.author_infos_to_embeddings(
        authors_by_arm, embedder, embedding_strategy, authors_chunk_size, batch_size
    )

    # Create the output directory if it doesn't exist
    output_dir, output_filename = os.path.split(output_files_path_pattern)
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings for both train and validation sets
    for arm, author_embedding_collection in text_embeddings.items():
        # Format the output filename by replacing {arm} with the current arm value
        if "{arm}" in output_filename:
            formatted_filename = output_filename.replace("{arm}", arm)
        else:
            # If there's no {arm} placeholder, insert it before the file extension
            name, ext = os.path.splitext(output_filename)
            formatted_filename = f"{name}_{arm}{ext}"

        output_path = os.path.join(output_dir, formatted_filename)
        print(f"Saving {arm} embeddings to {output_path}...")

        with open(output_path, "wt") as f:
            json.dump(dataclasses.asdict(author_embedding_collection), f, indent=2, cls=NumpyEncoder)

    print("Embeddings created and saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings for authors")
    parser.add_argument(
        "--embedder",
        type=str,
        default="nomic",
        help="Type of embedder to use (currently only 'nomic' and 'e5' are supported)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filename pattern. '{arm}' will be replaced with 'train' or 'val'",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=CURRENT_FILE_PATH.parent.parent / "data" / "suitable_author_infos_train.ndjson",
        help="Path to the file containing training author information",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=CURRENT_FILE_PATH.parent.parent / "data" / "suitable_author_infos_val.ndjson",
        help="Path to the file containing validation author information",
    )
    parser.add_argument(
        "--authors-chunk-size",
        type=int,
        default=256,
        help="Number of authors to embed in each chunk. Note: the reason for this is explained in "
        "AveragePostsStrategy.compute_authors_embeddings. In short, chunking the authors leads to faster runtime.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for embedding. Choose according to the memory of the GPU. ",
    )

    args = parser.parse_args()
    main(args.embedder, args.output, args.train_file, args.val_file, args.authors_chunk_size, args.batch_size)
