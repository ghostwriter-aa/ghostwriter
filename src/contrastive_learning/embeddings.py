from abc import ABC, abstractmethod

import numpy as np
from nomic import embed as nomic_embed  # type: ignore
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import common.common_types as ct
from common.gpu_utils import detect_device


def get_persona_texts(persona: ct.PersonaInfo) -> list[str]:
    """
    Get all texts from a persona.
    - For submissions, concatenate title and selftext
    - For comments, use the body
    """
    texts = []
    for submission in persona.submissions:
        selftext = "\n\n" + submission.selftext if submission.selftext else ""
        texts.append(submission.title + selftext)
    for comment in persona.comments:
        texts.append(comment.body)
    return texts


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed_texts(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        """
        Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.
        Returns:
            Numpy array of embeddings with shape (num_texts, embedding_dim)
        """


class NomicEmbedder(Embedder):
    """Embedder using the Nomic model."""

    def __init__(self) -> None:
        self.device = detect_device()

    def embed_texts(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        # Map torch device names to nomic device names
        device_mapping = {
            "cuda": "gpu",
            "mps": "gpu",  # Apple Silicon GPU
            "cpu": "cpu",
        }
        nomic_device = device_mapping.get(self.device, "cpu")

        output = nomic_embed.text(
            texts=texts,
            model="nomic-embed-text-v1.5",
            task_type="classification",
            inference_mode="local",
            dimensionality=768,
            device=nomic_device,
        )
        return np.array(output["embeddings"], dtype=np.float32)


class E5Embedder(Embedder):
    """Embedder using the E5 model."""

    def __init__(self) -> None:
        device = detect_device()
        self.model = SentenceTransformer("intfloat/e5-base-v2", device=device)

    def embed_texts(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        texts = ["passage: " + text for text in texts]
        return self.model.encode(
            sentences=texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 500,
            device=str(self.model.device),
        )


class E5MultilingualLargeInstructEmbedder(Embedder):
    """Embedder using the E5 multilingual large instruct model."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    def create_instruction(self, query: str) -> str:
        # Note: Documentation says "Each query must come with a one-sentence instruction that describes the task".
        # The below works within the errors of margin to a one sentence instruction that was also used.
        task_description = (
            "Generate an embedding that captures the distinctive writing style, linguistic patterns, and authorial "
            "voice of this text for the purpose of identifying and attributing authorship. Focus on stylistic features "
            "such as vocabulary choice, sentence structure, tone, rhetorical patterns, and other linguistic "
            "fingerprints that are characteristic of the author's writing style."
        )
        return f"Instruct: {task_description}\nQuery: {query}"

    def embed_texts(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        documents = [self.create_instruction(text) for text in texts]
        return self.model.encode(
            sentences=documents,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 500,
            device=str(self.model.device),
        )


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""

    @abstractmethod
    def compute_single_persona_embedding(
        self, persona: ct.PersonaInfo, embedder: Embedder, batch_size: int
    ) -> NDArray[np.float32]:
        """
        Compute an embedding for a persona using the provided embedder.

        Args:
            persona: The persona to compute embeddings for
            embedder: The embedder to use for computing embeddings
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.
        Returns:
            A single embedding vector as a numpy array
        """

    @abstractmethod
    def compute_authors_embeddings(
        self, authors: list[ct.AuthorInfo], embedder: Embedder, authors_chunk_size: int, batch_size: int
    ) -> list[ct.AuthorEmbedding]:
        """
        Compute an embedding for a list of authors using the provided embedder.

        Args:
            authors: The list of authors to compute embeddings for
            embedder: The embedder to use for computing embeddings
            authors_chunk_size: Number of authors to embed in each chunk. Note: the reason for this is explained in
                                AveragePostsStrategy.compute_authors_embeddings.
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.
        Returns:
            A list of AuthorEmbedding objects
        """


class AveragePostsStrategy(EmbeddingStrategy):
    """Strategy that computes the average embedding of all posts and comments of a persona."""

    def compute_single_persona_embedding(
        self, persona: ct.PersonaInfo, embedder: Embedder, batch_size: int
    ) -> NDArray[np.float32]:
        embeddings = embedder.embed_texts(get_persona_texts(persona), batch_size)
        return np.mean(embeddings, axis=0, dtype=np.float32)  # type: ignore

    def compute_authors_embeddings(
        self,
        authors: list[ct.AuthorInfo],
        embedder: Embedder,
        authors_chunk_size: int,
        batch_size: int,
    ) -> list[ct.AuthorEmbedding]:
        """
        Compute embeddings for a list of authors.

        Note on the implementaiton: we are sending the embedder chunks of authors, but also have batch size that is used
        by the embedder. Practically, when using e5 models, and giving the embedder the entire list of author texts, it
        ran slower on an ec2 g5.xlarge instance.

        Args:
            authors: The list of authors to compute embeddings for
            embedder: The embedder to use for computing embeddings
            authors_chunk_size: Number of authors to embed in each chunk.
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.

        Returns:
            A list of AuthorEmbedding objects
        """
        all_author_embeddings = []

        for chunk_start in tqdm(
            range(0, len(authors), authors_chunk_size),
            desc=f"Processing in chunks of {authors_chunk_size} authors",
        ):
            chunk_end = min(chunk_start + authors_chunk_size, len(authors))
            author_chunk = authors[chunk_start:chunk_end]

            # Collect all texts from all personas in this chunk and keep track of mapping
            chunk_texts = []
            text_to_persona_map = []

            for author_idx, author in enumerate(author_chunk):
                for persona_idx, persona in enumerate(author.personas):
                    persona_texts = get_persona_texts(persona)
                    for text in persona_texts:
                        chunk_texts.append(text)
                        text_to_persona_map.append((author_idx, persona_idx))

            chunk_embeddings_array: NDArray[np.float32] = embedder.embed_texts(chunk_texts, batch_size)

            # Group embeddings by persona and compute averages
            text_to_persona_map_array: NDArray[np.int32] = np.array(text_to_persona_map, dtype=np.int32)

            for author_idx, author in enumerate(author_chunk):
                persona_embeddings = []

                for persona_idx, persona in enumerate(author.personas):
                    # Find all embeddings for this specific persona
                    mask = (text_to_persona_map_array[:, 0] == author_idx) & (
                        text_to_persona_map_array[:, 1] == persona_idx
                    )
                    persona_embedding_indices = np.where(mask)[0]

                    avg_embedding: NDArray[np.float32]
                    if len(persona_embedding_indices) > 0:
                        # Compute average embedding for this persona
                        persona_embeddings_matrix = chunk_embeddings_array[persona_embedding_indices]
                        avg_embedding = np.asarray(
                            np.mean(persona_embeddings_matrix, axis=0, dtype=np.float32), dtype=np.float32
                        )
                    else:
                        # Handle case where persona has no texts
                        avg_embedding = np.zeros(chunk_embeddings_array.shape[1], dtype=np.float32)

                    persona_embeddings.append(ct.PersonaEmbedding(subreddit=persona.subreddit, embedding=avg_embedding))

                all_author_embeddings.append(
                    ct.AuthorEmbedding(username=author.username, persona_embeddings=persona_embeddings)
                )

        return all_author_embeddings


def author_infos_to_embeddings(
    authors_by_arm: dict[str, list[ct.AuthorInfo]],
    embedder: Embedder,
    embedding_strategy: EmbeddingStrategy,
    authors_chunk_size: int,
    batch_size: int,
) -> dict[str, ct.AuthorEmbeddingCollection]:
    """
    Create embeddings for all authors in the training and validation sets.

    Args:
        authors_by_arm: Dictionary with keys "train" and "val" and values lists of AuthorInfo objects
        embedder: Embedder object that computes embeddings for personas
        embedding_strategy: Strategy object that computes embeddings for personas
        authors_chunk_size: Number of authors to embed in each chunk. Note: the reason for this is explained in
                            AveragePostsStrategy.compute_authors_embeddings.
        batch_size: Batch size for embedding. Choose according to the memory of the GPU.
    Returns:
        Dictionary mapping each arm ("train" or "val") to its corresponding AuthorEmbeddingCollection
    """
    # Get embedding dimension from a sample embedding
    sample_text = "sample text"
    embedding_dim = len(embedder.embed_texts([sample_text], batch_size)[0])

    arm_embeddings = {}
    for arm, authors in authors_by_arm.items():
        author_embeddings = embedding_strategy.compute_authors_embeddings(
            authors, embedder, authors_chunk_size, batch_size
        )

        # Create collection for this arm
        arm_embeddings[arm] = ct.AuthorEmbeddingCollection(
            embedding_dim=embedding_dim,
            embedding_strategy=embedding_strategy.__class__.__name__,
            embedding_model=embedder.__class__.__name__,
            author_embeddings=author_embeddings,
        )

    return arm_embeddings
