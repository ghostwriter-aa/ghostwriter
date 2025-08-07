import numpy as np
from numpy.typing import NDArray

from common import common_types as ct
from contrastive_learning.embeddings import AveragePostsStrategy, Embedder


class DummyEmbedder(Embedder):
    """Simple embedder for testing which knows how to embed texts that are integers."""

    def embed_texts(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        texts_embeddings = np.zeros((len(texts), 4), dtype=np.float32)
        try:
            for i, text in enumerate(texts):
                texts_embeddings[i][int(text) % 4] = 1.0
        except Exception as e:
            print("This embedder can only embed texts that are integers.")
            raise e
        return texts_embeddings


def test_compute_authors_embeddings() -> None:
    """`compute_authors_embeddings` should preserve persona grouping and dims."""

    # Create personas spread across three authors (1, 2 and 5 personas)
    personas = [
        ct.PersonaInfo(
            subreddit=f"subreddit{i}",
            comments=[ct.CommentInfo(body=f"{i}", permalink="")],
            submissions=[],
        )
        for i in range(1, 9)
    ]

    authors = [
        ct.AuthorInfo(username="author1", personas=personas[0:1]),
        ct.AuthorInfo(username="author2", personas=personas[1:3]),
        ct.AuthorInfo(username="author3", personas=personas[3:8]),
    ]

    expected_embeddings = [
        [np.array([0.0, 1.0, 0.0, 0.0])],
        [np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])],
        [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ],
    ]

    strategy = AveragePostsStrategy()
    embedder = DummyEmbedder()

    author_embeddings = strategy.compute_authors_embeddings(authors, embedder, persona_chunk_size=7, batch_size=7)

    assert len(author_embeddings) == 3

    for i, author_embedding in enumerate(author_embeddings):
        assert len(author_embedding.persona_embeddings) == len(expected_embeddings[i])
        for persona_emb_idx, persona_emb in enumerate(author_embedding.persona_embeddings):
            assert np.allclose(persona_emb.embedding, expected_embeddings[i][persona_emb_idx])
