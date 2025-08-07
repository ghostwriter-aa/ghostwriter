from abc import ABC, abstractmethod
from typing import Mapping

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


EMBEDDING_MODEL_CLASSES: Mapping[str, type[Embedder]] = {
    "nomic": NomicEmbedder,
    "e5": E5Embedder,
    "e5-multilingual-large-instruct": E5MultilingualLargeInstructEmbedder,
}


class EmbeddingStrategy(ABC):
    """Abstract base class for embedding strategies."""

    @abstractmethod
    def compute_single_persona_embedding(
        self, persona: ct.PersonaInfo, embedder: Embedder, batch_size: int
    ) -> ct.PersonaEmbedding:
        """
        Compute an embedding for a persona using the provided embedder.

        Args:
            persona: The persona to compute embeddings for
            embedder: The embedder to use for computing embeddings
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.
        Returns:
            A PersonaEmbedding object that contains the averaged embedding vector and metadata.
        """

    @abstractmethod
    def compute_authors_embeddings(
        self, authors: list[ct.AuthorInfo], embedder: Embedder, persona_chunk_size: int, batch_size: int
    ) -> list[ct.AuthorEmbedding]:
        """
        Compute an embedding for a list of authors using the provided embedder.

        Args:
            authors: The list of authors to compute embeddings for
            embedder: The embedder to use for computing embeddings
            persona_chunk_size: Number of personas to embed in each chunk. Note: the reason for this is explained in
                                AveragePostsStrategy.compute_persona_embeddings.
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.
        Returns:
            A list of AuthorEmbedding objects
        """


class AveragePostsStrategy(EmbeddingStrategy):
    """Strategy that computes the average embedding of all posts and comments of a persona."""

    def compute_single_persona_embedding(
        self, persona: ct.PersonaInfo, embedder: Embedder, batch_size: int
    ) -> ct.PersonaEmbedding:
        embeddings = embedder.embed_texts(get_persona_texts(persona), batch_size)
        return ct.PersonaEmbedding(subreddit=persona.subreddit, embedding=np.mean(embeddings, axis=0, dtype=np.float32))

    def _persona_list_to_texts_and_index_mapping(
        self, personas: list[ct.PersonaInfo]
    ) -> tuple[list[str], NDArray[np.int32]]:
        """
        Convert a list of personas to a list of texts and a mapping from text to persona index in the original list.
        """
        texts = []
        text_to_persona_map = []
        for persona_idx, persona in enumerate(personas):
            persona_texts = get_persona_texts(persona)
            for text in persona_texts:
                texts.append(text)
                text_to_persona_map.append(persona_idx)
        return texts, np.array(text_to_persona_map, dtype=np.int32)

    def _average_each_persona(
        self,
        chunk_embeddings_array: NDArray[np.float32],
        text_to_persona_map_array: NDArray[np.int32],
        personas: list[ct.PersonaInfo],
    ) -> list[ct.PersonaEmbedding]:
        personas_embeddings = []
        for persona_idx, persona in enumerate(personas):
            # Find all embeddings for this specific persona
            mask = text_to_persona_map_array == persona_idx
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
            personas_embeddings.append(ct.PersonaEmbedding(subreddit=persona.subreddit, embedding=avg_embedding))

        return personas_embeddings

    def compute_persona_embeddings(
        self, personas: list[ct.PersonaInfo], embedder: Embedder, persona_chunk_size: int, batch_size: int
    ) -> list[ct.PersonaEmbedding]:
        """
        Compute embeddings for a list of personas.

        Note on the implementaiton: we are sending the embedder chunks of personas, but also have batch size that is
        used by the embedder. Practically, when using e5 models, and giving the embedder the entire list of persona
        texts, it ran slower on an ec2 g5.xlarge instance (when running according to the readme, embedding
        suitable_author_infos_train.ndjson, it takes ±1 hour, while using maximal persona chunk size, it takes more
        than 2 hours).

        Args:
            personas: The list of personas to compute embeddings for
            embedder: The embedder to use for computing embeddings
            persona_chunk_size: Number of personas to embed in each chunk.
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.
        Returns:
            A list of PersonaEmbedding objects
        """

        all_persona_embeddings: list[ct.PersonaEmbedding] = []

        for chunk_start in tqdm(
            range(0, len(personas), persona_chunk_size),
            desc=f"Processing in chunks of {persona_chunk_size} personas",
        ):
            chunk_end = min(chunk_start + persona_chunk_size, len(personas))
            personas_chunk = personas[chunk_start:chunk_end]

            texts, text_to_persona_index = self._persona_list_to_texts_and_index_mapping(personas_chunk)

            chunk_embeddings_array: NDArray[np.float32] = embedder.embed_texts(texts, batch_size)

            all_persona_embeddings.extend(
                self._average_each_persona(chunk_embeddings_array, text_to_persona_index, personas_chunk)
            )

        return all_persona_embeddings

    def compute_authors_embeddings(
        self,
        authors: list[ct.AuthorInfo],
        embedder: Embedder,
        persona_chunk_size: int,
        batch_size: int,
    ) -> list[ct.AuthorEmbedding]:
        """
        Compute embeddings for a list of authors.

        Args:
            authors: The list of authors to compute embeddings for
            embedder: The embedder to use for computing embeddings
            persona_chunk_size: Number of personas to embed in each chunk.
            batch_size: Batch size for embedding. Choose according to the memory of the GPU.

        Returns:
            A list of AuthorEmbedding objects
        """
        all_author_embeddings: list[ct.AuthorEmbedding] = []

        # Flatten to a list of personas
        personas = []
        for author in authors:
            personas.extend(author.personas)

        persona_embeddings = self.compute_persona_embeddings(personas, embedder, persona_chunk_size, batch_size)

        # Unflatten to a list of author embeddings
        current_persona_idx = 0
        for author in authors:
            persona_of_author_embeddings = []
            for persona in author.personas:
                persona_embedding = persona_embeddings[current_persona_idx]
                persona_of_author_embeddings.append(persona_embedding)
                current_persona_idx += 1
            all_author_embeddings.append(
                ct.AuthorEmbedding(username=author.username, persona_embeddings=persona_of_author_embeddings)
            )

        return all_author_embeddings


def author_infos_to_embeddings(
    authors_by_arm: dict[str, list[ct.AuthorInfo]],
    embedder: Embedder,
    embedding_strategy: EmbeddingStrategy,
    persona_chunk_size: int,
    batch_size: int,
) -> dict[str, ct.AuthorEmbeddingCollection]:
    """
    Create embeddings for all authors in the training and validation sets.

    Args:
        authors_by_arm: Dictionary with keys "train" and "val" and values lists of AuthorInfo objects
        embedder: Embedder object that computes embeddings for personas
        embedding_strategy: Strategy object that computes embeddings for personas
        persona_chunk_size: Number of personas to embed in each chunk. Note: the reason for this is explained in
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
            authors, embedder, persona_chunk_size, batch_size
        )

        # Create collection for this arm
        arm_embeddings[arm] = ct.AuthorEmbeddingCollection(
            embedding_dim=embedding_dim,
            embedding_strategy=embedding_strategy.__class__.__name__,
            embedding_model=embedder.__class__.__name__,
            author_embeddings=author_embeddings,
        )

    return arm_embeddings
