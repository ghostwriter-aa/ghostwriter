"""Utilities for converting text strings to lists of tokens. A thin wrapper around `tiktoken`."""

from collections import Counter
from typing import Iterator

import tiktoken

import common.common_types as ct


def get_tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def get_persona_tokens_counter(persona: ct.PersonaInfo, tokenizer: tiktoken.Encoding) -> Counter[int]:
    """Returns a Counter of tokens in the persona's submissions and comments."""
    counter: Counter[int] = Counter()
    for list_of_tokens in get_persona_tokens_iterator(persona, tokenizer):
        counter.update(list_of_tokens)
    return counter


def get_author_tokens_counter(author: ct.AuthorInfo, tokenizer: tiktoken.Encoding) -> Counter[int]:
    """Returns a Counter of tokens in all the author's personas."""
    counter: Counter[int] = Counter()
    for list_of_tokens in get_author_tokens_iterator(author, tokenizer):
        counter.update(list_of_tokens)
    return counter


def get_persona_tokens_iterator(persona: ct.PersonaInfo, tokenizer: tiktoken.Encoding) -> Iterator[list[int]]:
    """Yields zero or more token *lists*, one for each text segment (comment, submission, or submission title)."""
    for submission in persona.submissions:
        yield tokenizer.encode(submission.title)
        yield tokenizer.encode(submission.selftext)
    for comment in persona.comments:
        yield tokenizer.encode(comment.body)


def get_author_tokens_iterator(author: ct.AuthorInfo, tokenizer: tiktoken.Encoding) -> Iterator[list[int]]:
    """Yields zero or more token *lists*, one for each text segment (comment, submission, or submission title)."""
    for persona in author.personas:
        yield from get_persona_tokens_iterator(persona, tokenizer)
