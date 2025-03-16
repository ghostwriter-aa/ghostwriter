import dataclasses
from collections import Counter
from functools import cached_property
from typing import Sequence

import numpy as np
import tiktoken
from numpy.typing import NDArray

import common.common_types as ct
from common import tokenization


@dataclasses.dataclass(frozen=True)
class TokenStats:
    """1-gram statistics of a subset of commonly used tokens (referred to as the "token set").

    Functions which return arrays refer to tokens by insertion order into `token_count`, which should match lexical
    order in the full set of all tokens.
    """

    token_count: Counter[int]  # Number of occurrences of each token.
    excluded_tokens_counter: int  # Number of occurrences of any of the tokens not in `token_count.keys()`.

    @cached_property
    def token_count_array(self) -> NDArray[np.int32]:
        """Number of occurrences of each token in the token set.

        Used for efficient computation requiring repeated token count lookups.
        """
        return np.array(list(self.token_count.values()), dtype=np.int32)

    @cached_property
    def num_tokens_in_set(self) -> int:
        """Number of tokens observed of the types in `token_count`."""
        return sum(self.token_count.values())

    @cached_property
    def token_freq(self) -> NDArray[np.float32]:
        """Token frequencies, not including tokens outside the token set."""
        if self.num_tokens_in_set == 0:
            return self.token_count_array.astype(np.float32)
        return (self.token_count_array / self.num_tokens_in_set).astype(np.float32)

    @cached_property
    def token_freq_with_excluded_tokens(self) -> NDArray[np.float32]:
        """Token frequencies, including tokens outside the token set."""
        num_total_tokens_with_excluded = self.num_tokens_in_set + self.excluded_tokens_counter
        if num_total_tokens_with_excluded == 0:
            return self.token_count_array.astype(np.float32)
        return (self.token_count_array / num_total_tokens_with_excluded).astype(np.float32)

    @cached_property
    def log_nonz_token_freq(self) -> NDArray[np.float32]:
        """Log token frequencies, adjusted so that unobserved tokens are assumed to have been observed 0.5 times.

        Frequencies do not include tokens outside the token set.
        """
        adj_token_count = self.token_count_array + 0.5
        return np.log2(adj_token_count) - np.log2(np.sum(adj_token_count))  # type: ignore

    @cached_property
    def log_nonz_token_freq_with_excluded_tokens(self) -> NDArray[np.float32]:
        """Log token frequencies, adjusted so that unobserved tokens are assumed to have been observed 0.5 times.

        Frequencies *include* tokens outside the token set.
        """
        adj_token_count = np.concat([self.token_count_array + 0.5, [self.excluded_tokens_counter + 0.5]])
        log_freq = np.log2(adj_token_count) - np.log2(np.sum(adj_token_count))
        return log_freq[:-1]  # type: ignore

    @property
    def lower_ci(self) -> NDArray[np.float32]:
        """Lower 95% confidence interval for the token frequency."""
        total_tokens = self.num_tokens_in_set
        token_freq = self.token_freq
        wilson_ci_halfwidth = (
            1
            / (1 + 1.96**2 / total_tokens)
            * 1.96
            / (2 * total_tokens)
            * np.sqrt(4 * total_tokens * token_freq * (1 - token_freq) + 1.96**2)
        )
        wilson_ci_lower_bound = (
            1 / (1 + 1.96**2 / total_tokens) * (token_freq + 1.96**2 / 2 / total_tokens) - wilson_ci_halfwidth
        )
        return np.clip(wilson_ci_lower_bound, 0, 1)  # type: ignore

    @property
    def upper_ci(self) -> NDArray[np.float32]:
        """Upper 95% confidence interval for the token frequency."""
        total_tokens = self.num_tokens_in_set
        token_freq = self.token_freq
        wilson_ci_halfwidth = (
            1
            / (1 + 1.96**2 / total_tokens)
            * 1.96
            / (2 * total_tokens)
            * np.sqrt(4 * total_tokens * token_freq * (1 - token_freq) + 1.96**2)
        )
        wilson_ci_upper_bound = (
            1 / (1 + 1.96**2 / total_tokens) * (token_freq + 1.96**2 / 2 / total_tokens) + wilson_ci_halfwidth
        )
        return np.clip(wilson_ci_upper_bound, 0, 1)  # type: ignore

    @classmethod
    def from_author_info(
        cls, author_info: ct.AuthorInfo, tokens_to_count: Sequence[int], tokenizer: tiktoken.Encoding
    ) -> "TokenStats":
        """Returns a TokenStats object counting only the tokens `tokens_to_count` in the author's personas."""
        return cls.from_counts(tokenization.get_author_tokens_counter(author_info, tokenizer), tokens_to_count)

    @classmethod
    def from_persona_info(
        cls, persona_info: ct.PersonaInfo, tokens_to_count: Sequence[int], tokenizer: tiktoken.Encoding
    ) -> "TokenStats":
        """Returns a TokenStats object counting only the tokens `tokens_to_count` in the persona's text."""
        return cls.from_counts(tokenization.get_persona_tokens_counter(persona_info, tokenizer), tokens_to_count)

    @classmethod
    def from_counts(cls, token_counter: Counter[int], tokens_to_count: Sequence[int]) -> "TokenStats":
        """Returns a TokenStats object counting only the tokens `tokens_to_count` in the given Counter.

        The returned object's Counter will contain exactly the tokens in `tokens_to_count`, in order.
        """
        counter: Counter[int] = Counter()
        for token in tokens_to_count:
            counter[token] = token_counter.get(token, 0)
        excluded_tokens = sum(token_counter.values()) - sum(counter.values())
        return TokenStats(token_count=counter, excluded_tokens_counter=excluded_tokens)

    def log_likelihood(self, observation_token_stats: "TokenStats") -> np.float32:
        if list(observation_token_stats.token_count.keys()) != list(self.token_count.keys()):
            raise ValueError("The token sets of the two TokenStats objects are different.")
        total_obs_tokens = observation_token_stats.num_tokens_in_set
        if total_obs_tokens == 0:
            return np.float32(0)
        return (  # type: ignore
            np.sum(observation_token_stats.token_count_array * self.log_nonz_token_freq_with_excluded_tokens)
            / total_obs_tokens
        )
