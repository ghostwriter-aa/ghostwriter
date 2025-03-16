"""Basic datatypes used across the project."""

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import dacite


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
