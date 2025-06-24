"""Basic datatypes used across the project."""

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import dacite
import numpy as np
from numpy.typing import NDArray


# Used to help MyPy understand that the from_json method is added to the class.
class JsonReaderInterface:
    @classmethod
    def from_json(cls, json_dict: str | Mapping[str, Any]) -> "JsonReaderInterface":
        raise NotImplementedError


def jsonreader(cls: type) -> JsonReaderInterface:
    @classmethod  # type: ignore
    def from_json(cls, json_dict: str | Mapping[str, Any]):
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict)
        return dacite.from_dict(data_class=cls, data=json_dict)  # type: ignore

    cls.from_json = from_json  # type: ignore
    return cls  # type: ignore


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


@jsonreader
@dataclasses.dataclass
class AuthorInfo:
    username: str
    personas: list[PersonaInfo]


def read_author_infos(filename: str | Path) -> list[AuthorInfo]:
    """Reads a list of AuthorInfo objects from an NDJSON file."""
    with open(filename, "rt") as f:
        return [AuthorInfo.from_json(line) for line in f]  # type: ignore


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
