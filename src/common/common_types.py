"""Basic datatypes used across the project."""

import dataclasses
import enum
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from common.json_serialization import JsonSerializable


@dataclasses.dataclass(frozen=True)
class SubmissionInfo:
    title: str
    selftext: str
    url: str


@dataclasses.dataclass(frozen=True)
class CommentInfo:
    body: str
    permalink: str


@dataclasses.dataclass
class PersonaInfo:
    """Information about a user's posts on a particular topic (i.e., in a specific subreddit)."""

    subreddit: str
    submissions: list[SubmissionInfo] = dataclasses.field(default_factory=list)
    comments: list[CommentInfo] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AuthorInfo(JsonSerializable):
    username: str
    personas: list[PersonaInfo]


def read_author_infos(filename: str | Path) -> list[AuthorInfo]:
    """Reads a list of AuthorInfo objects from an NDJSON file."""
    with open(filename, "rt") as f:
        return [AuthorInfo.from_json(line) for line in f]


@dataclasses.dataclass(frozen=True)
class PersonaId:
    author: str
    persona: str


@dataclasses.dataclass
class PersonaEmbedding:
    """Persona embedding for an author."""

    subreddit: str
    embedding: NDArray[np.float32]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonaEmbedding":
        return cls(subreddit=data["subreddit"], embedding=np.array(data["embedding"], dtype=np.float32))


@dataclasses.dataclass
class AuthorEmbedding:
    """Author persona embeddings."""

    username: str
    persona_embeddings: list[PersonaEmbedding]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthorEmbedding":
        return cls(
            username=data["username"],
            persona_embeddings=[PersonaEmbedding.from_dict(pe) for pe in data["persona_embeddings"]],
        )


@dataclasses.dataclass
class AuthorEmbeddingCollection:
    """Information about all authors embeddings.

    Attributes:
        embedding_dim: Dimension of the embedding vectors
        embedding_strategy: How to handle multiple posts and comments for a single author.
            Right now only "AveragePostsStrategy" is supported.
        embedding_model: Model used to generate the embeddings, e.g. "E5Embedder", or "NomicEmbedder".
        author_embeddings: List of author embeddings
    """

    embedding_dim: int
    embedding_strategy: str
    embedding_model: str
    author_embeddings: list[AuthorEmbedding]

    @classmethod
    def from_json(cls, json_str: str) -> "AuthorEmbeddingCollection":
        data = json.loads(json_str)
        return cls(
            embedding_dim=data["embedding_dim"],
            embedding_strategy=data["embedding_strategy"],
            embedding_model=data["embedding_model"],
            author_embeddings=[AuthorEmbedding.from_dict(ae) for ae in data["author_embeddings"]],
        )


class ModelType(enum.Enum):
    LOG_LIKELIHOOD_MODEL = "log_likelihood_model"
    EMBEDDING_MODEL = "embedding_model"


@dataclasses.dataclass
class TiktokenToken:
    """A token in the Tiktoken encoding.

    Attributes:
        token_index: The Tiktoken token ID.
        string: The string representation of the token.
    """

    token_index: int
    string: str


@dataclasses.dataclass
class AbstractModelParams(JsonSerializable):
    """Abstract base class for model parameters."""


@dataclasses.dataclass
class LogLikelihoodModelParams(AbstractModelParams):
    """Parameters for a log-likelihood model.

    Attributes:
        tokens: List of Tiktoken tokens used by the model.
    """

    tokens: list[TiktokenToken]


@dataclasses.dataclass
class EmbeddingModelParams(AbstractModelParams):
    """Parameters for an embedding model.

    Attributes:
        embedding_model: The model used to generate the embeddings (e.g. "e5", See `EMBEDDING_MODEL_CLASSES` in
            `embeddings.py`).
        embedding_strategy: How to handle multiple posts and comments for a single author. Right now, the strategy is to
            average the embeddings of the posts and comments for a single author, and this field documents this current
            strategy.
        embedding_model_checkpoint_path: Path to the checkpoint file for the embedding model.
    """

    embedding_model: str
    embedding_strategy: str
    embedding_model_checkpoint_path: str


_MODEL_PARAMS_CLASSES: dict[ModelType, type[AbstractModelParams]] = {
    ModelType.LOG_LIKELIHOOD_MODEL: LogLikelihoodModelParams,
    ModelType.EMBEDDING_MODEL: EmbeddingModelParams,
}


@dataclasses.dataclass
class Model(JsonSerializable):
    """Stores information about a trained model."""

    model_type: ModelType
    # Data type for `model_params` is determined by `model_type`.
    model_params: AbstractModelParams

    @classmethod
    def from_json(cls, json_dict: str | Mapping[str, Any]) -> "Model":
        """Loads a Model object from a JSON string or dictionary.

        Requires special handling because the type of `model_params` depends on `model_type`.
        """
        if isinstance(json_dict, str):
            parsed_json = json.loads(json_dict)
        else:
            parsed_json = json_dict

        model_type = ModelType[parsed_json["model_type"]]
        model_params_class = _MODEL_PARAMS_CLASSES[model_type]
        model_params = model_params_class.from_json(parsed_json["model_params"])

        return cls(model_type=model_type, model_params=model_params)
