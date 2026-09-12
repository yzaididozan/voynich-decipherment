"""Token-level Markov model with a character-trigram open-vocabulary base.

VOYAGER uses this model to test whether previous Voynich token identities add
held-out predictive information beyond within-token local glyph statistics.

For a token y and token history h, the model uses interpolated Witten-Bell
recursion over token identities:

    P(y | h) =
        c(h, y) / (N(h) + T(h))
        + T(h) / (N(h) + T(h)) * P(y | suffix(h))

At the token-unigram level, the recursion backs off to the exact matched
character 3-gram that generates a token independently and terminates it with
<EOT>.

This gives the token model open-vocabulary support: unseen token types and
tokens containing validation-only glyphs still receive probability from the
character model. The base is intentionally not renormalized: it is exactly
the same token-reset glyph+EOT distribution used as the matched comparator.
Its support includes immediate <EOT> (an empty generated token), although the
analytical corpus itself contains only non-empty tokens. Keeping that support
unchanged makes likelihoods directly comparable.

Frozen usage in the companion validation runner:
- conventional S0 ZL/STA1 tokens;
- token context orders 1 and 2 only;
- locus-bounded sequences;
- tokens containing Z1 are excluded and reset token context;
- train fits parameters; validation selects context order;
- locked test is not accessed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.analysis.tier0 import AnalyticalLocus, UNKNOWN_STA1
from src.models.ngram import BOS, BaselineSequence, WittenBellNGram


EOT = "<EOT>"
TOKEN_BOS = "<TOKEN_BOS>"


Token = Tuple[str, ...]
History = Tuple[object, ...]


@dataclass(frozen=True)
class TokenMarkovExample:
    leaf_group: str
    folio: str
    locus: str
    token_index: int
    segment_index: int
    glyphs: Token


def atomic_leaf_group(locus: AnalyticalLocus) -> str:
    if locus.folio == "fRos" or locus.physical_leaf in (85, 86):
        return "f85+f86"
    if locus.physical_leaf is None:
        raise ValueError(
            f"No atomic leaf group for {locus.folio} {locus.locus}"
        )
    return f"f{locus.physical_leaf}"


def extract_token_sequences(
    loci: Sequence[AnalyticalLocus],
) -> Tuple[List[List[TokenMarkovExample]], dict]:
    """Return readable within-locus token segments.

    A token containing transcription-unknown Z1 is excluded and terminates the
    current segment. This prevents the model from predicting across unreadable
    spans.
    """
    sequences: List[List[TokenMarkovExample]] = []

    included_tokens = 0
    included_glyphs = 0
    excluded_tokens = 0
    excluded_glyphs = 0
    readable_segments = 0

    for locus in loci:
        leaf_group = atomic_leaf_group(locus)
        current: List[TokenMarkovExample] = []
        segment_index = 0

        def flush() -> None:
            nonlocal current, readable_segments, segment_index
            if current:
                sequences.append(current)
                readable_segments += 1
                current = []
                segment_index += 1

        for token_index, token in enumerate(locus.tokens):
            if not token:
                continue

            if UNKNOWN_STA1 in token:
                excluded_tokens += 1
                excluded_glyphs += len(token)
                flush()
                continue

            glyphs = tuple(token)
            included_tokens += 1
            included_glyphs += len(glyphs)
            current.append(
                TokenMarkovExample(
                    leaf_group=leaf_group,
                    folio=locus.folio,
                    locus=locus.locus,
                    token_index=token_index,
                    segment_index=segment_index,
                    glyphs=glyphs,
                )
            )

        flush()

    return sequences, {
        "included_tokens": included_tokens,
        "included_glyphs": included_glyphs,
        "excluded_tokens_with_Z1": excluded_tokens,
        "excluded_glyphs_in_Z1_tokens": excluded_glyphs,
        "readable_segments": readable_segments,
        "unknown_resets_context": True,
    }


def build_matched_trigram_examples(
    sequences: Sequence[Sequence[TokenMarkovExample]],
) -> List[BaselineSequence]:
    """Represent each evaluated token as independent glyphs + explicit EOT."""
    examples: List[BaselineSequence] = []

    for sequence in sequences:
        for example in sequence:
            examples.append(
                BaselineSequence(
                    leaf_group=example.leaf_group,
                    folio=example.folio,
                    locus=example.locus,
                    symbols=example.glyphs + (EOT,),
                    is_boundary=(
                        (False,) * len(example.glyphs)
                        + (True,)
                    ),
                )
            )

    return examples


class TokenMarkovModel:
    """Interpolated token Markov model backed by a character n-gram."""

    def __init__(
        self,
        *,
        max_context: int = 2,
        char_base_order: int = 3,
    ):
        if max_context < 1:
            raise ValueError("max_context must be >= 1")
        if char_base_order < 1:
            raise ValueError("char_base_order must be >= 1")

        self.max_context = int(max_context)
        self.char_base_order = int(char_base_order)

        self.char_base_model = WittenBellNGram(
            max_order=self.char_base_order
        )

        # index == history length
        self._counts: List[Dict[History, Counter]] = [
            defaultdict(Counter)
            for _ in range(self.max_context + 1)
        ]

        self._seen_tokens: set[Token] = set()
        self._training_tokens = 0
        self._training_glyphs = 0
        self._fitted = False

    @property
    def is_fit(self) -> bool:
        return self._fitted

    @property
    def training_tokens(self) -> int:
        return self._training_tokens

    @property
    def training_glyphs(self) -> int:
        return self._training_glyphs

    @property
    def training_events(self) -> int:
        return self._training_tokens + self._training_glyphs

    @property
    def seen_tokens(self) -> Tuple[Token, ...]:
        return tuple(sorted(self._seen_tokens))

    def fit(
        self,
        token_sequences: Sequence[Sequence[Sequence[str]]],
    ) -> "TokenMarkovModel":
        if self._fitted:
            raise RuntimeError("Model has already been fit")

        normalized_sequences: List[Tuple[Token, ...]] = []

        for sequence in token_sequences:
            normalized = tuple(
                tuple(token)
                for token in sequence
                if token
            )
            if normalized:
                normalized_sequences.append(normalized)

        if not normalized_sequences:
            raise ValueError(
                "Cannot fit token Markov model to an empty corpus"
            )

        # Fit the exact matched character base: each token is an independent
        # sequence of glyphs followed by one explicit terminal event.
        char_sequences = []
        for sequence in normalized_sequences:
            for token in sequence:
                if EOT in token:
                    raise ValueError(
                        "Input glyph sequence collides with reserved <EOT>"
                    )
                char_sequences.append(token + (EOT,))
                self._seen_tokens.add(token)
                self._training_tokens += 1
                self._training_glyphs += len(token)

        self.char_base_model.fit(char_sequences)

        # Token-level counts. BOS padding is history only and is never emitted.
        for sequence in normalized_sequences:
            padded: Tuple[object, ...] = (
                (TOKEN_BOS,) * self.max_context
                + tuple(sequence)
            )

            for i, token in enumerate(sequence):
                position = self.max_context + i

                for history_len in range(self.max_context + 1):
                    if history_len == 0:
                        history: History = ()
                    else:
                        history = tuple(
                            padded[
                                position - history_len:
                                position
                            ]
                        )

                    self._counts[history_len][history][token] += 1

        self._fitted = True
        return self

    def _character_token_probability(
        self,
        token: Sequence[str],
    ) -> float:
        if not self._fitted:
            raise RuntimeError("Model is not fit")
        if not token:
            return 0.0

        symbols = tuple(token) + (EOT,)
        padded = (
            (BOS,) * (self.char_base_order - 1)
            + symbols
        )

        probability = 1.0

        for i, symbol in enumerate(symbols):
            position = self.char_base_order - 1 + i
            if self.char_base_order == 1:
                history: Tuple[str, ...] = ()
            else:
                history = tuple(
                    padded[
                        position - (self.char_base_order - 1):
                        position
                    ]
                )

            probability *= self.char_base_model.probability(
                symbol,
                history,
                order=self.char_base_order,
            )

        return probability

    def character_token_probability(
        self,
        token: Sequence[str],
    ) -> float:
        """Public matched character-base probability for one non-empty token."""
        return self._character_token_probability(token)

    def _prob_recursive(
        self,
        token: Token,
        history: History,
    ) -> float:
        if not history:
            counts = self._counts[0][()]
            n = sum(counts.values())
            t = len(counts)

            if n <= 0 or t <= 0:
                raise RuntimeError(
                    "Invalid token-unigram count state"
                )

            observed = counts.get(token, 0)
            lower = self._character_token_probability(token)

            return (
                observed / (n + t)
                + (t / (n + t)) * lower
            )

        mapping = self._counts[len(history)]
        counts = mapping.get(history)

        if not counts:
            return self._prob_recursive(
                token,
                history[1:],
            )

        n = sum(counts.values())
        t = len(counts)
        observed = counts.get(token, 0)
        lower = self._prob_recursive(
            token,
            history[1:],
        )

        return (
            observed / (n + t)
            + (t / (n + t)) * lower
        )

    def token_probability(
        self,
        token: Sequence[str],
        previous_tokens: Sequence[Sequence[str]],
        *,
        context_order: int,
    ) -> float:
        if not self._fitted:
            raise RuntimeError("Model is not fit")
        if not token:
            raise ValueError("Cannot score an empty token")
        if context_order < 1 or context_order > self.max_context:
            raise ValueError(
                f"context_order must be within 1..{self.max_context}; "
                f"found {context_order}"
            )

        normalized_token = tuple(token)
        normalized_previous = [
            tuple(item)
            for item in previous_tokens
        ]

        if len(normalized_previous) >= context_order:
            history: History = tuple(
                normalized_previous[-context_order:]
            )
        else:
            missing = context_order - len(normalized_previous)
            history = (
                (TOKEN_BOS,) * missing
                + tuple(normalized_previous)
            )

        probability = self._prob_recursive(
            normalized_token,
            history,
        )

        if probability <= 0.0 or not math.isfinite(probability):
            raise RuntimeError(
                "Token Markov model produced a non-positive or "
                "non-finite probability"
            )

        return probability

    def score_token(
        self,
        token: Sequence[str],
        previous_tokens: Sequence[Sequence[str]],
        *,
        context_order: int,
    ) -> Tuple[float, int, bool]:
        probability = self.token_probability(
            token,
            previous_tokens,
            context_order=context_order,
        )

        base_vocabulary = set(
            self.char_base_model.seen_vocabulary
        )
        oov_glyphs = sum(
            glyph not in base_vocabulary
            for glyph in token
        )
        token_oov = tuple(token) not in self._seen_tokens

        return math.log(probability), oov_glyphs, token_oov

    def context_summary(self) -> List[dict]:
        if not self._fitted:
            raise RuntimeError("Model is not fit")

        rows = []

        for history_len in range(self.max_context + 1):
            mapping = self._counts[history_len]
            transitions = sum(
                sum(counter.values())
                for counter in mapping.values()
            )
            distinct_edges = sum(
                len(counter)
                for counter in mapping.values()
            )

            rows.append(
                {
                    "history_length": history_len,
                    "token_ngram_order": history_len + 1,
                    "contexts": len(mapping),
                    "transitions": transitions,
                    "distinct_context_next_edges": distinct_edges,
                }
            )

        return rows


def evaluate_token_markov(
    model: TokenMarkovModel,
    sequences: Sequence[Sequence[TokenMarkovExample]],
    *,
    context_order: int,
) -> Tuple[dict, List[dict], List[dict]]:
    """Teacher-forced held-out token likelihood by physical leaf and locus."""
    locus_acc: Dict[Tuple[str, str, str], dict] = defaultdict(
        lambda: {
            "total_bits": 0.0,
            "tokens": 0,
            "glyphs": 0,
            "events": 0,
            "oov_tokens": 0,
            "oov_glyphs": 0,
        }
    )

    for sequence in sequences:
        history: List[Token] = []

        for example in sequence:
            log_probability, oov_glyphs, token_oov = model.score_token(
                example.glyphs,
                history,
                context_order=context_order,
            )

            bits = -log_probability / math.log(2.0)

            key = (
                example.leaf_group,
                example.folio,
                example.locus,
            )
            acc = locus_acc[key]
            acc["total_bits"] += bits
            acc["tokens"] += 1
            acc["glyphs"] += len(example.glyphs)
            acc["events"] += len(example.glyphs) + 1
            acc["oov_tokens"] += int(token_oov)
            acc["oov_glyphs"] += oov_glyphs

            history.append(example.glyphs)

    locus_rows: List[dict] = []

    for (
        leaf_group,
        folio,
        locus,
    ), acc in sorted(locus_acc.items()):
        locus_rows.append(
            {
                "model": "token_markov",
                "context_order": context_order,
                "leaf_group": leaf_group,
                "folio": folio,
                "locus": locus,
                **acc,
                "bits_per_event": (
                    acc["total_bits"] / acc["events"]
                ),
                "bits_per_token": (
                    acc["total_bits"] / acc["tokens"]
                ),
            }
        )

    leaf_acc: Dict[str, dict] = defaultdict(
        lambda: {
            "total_bits": 0.0,
            "tokens": 0,
            "glyphs": 0,
            "events": 0,
            "oov_tokens": 0,
            "oov_glyphs": 0,
        }
    )

    for row in locus_rows:
        acc = leaf_acc[row["leaf_group"]]

        for field in (
            "total_bits",
            "tokens",
            "glyphs",
            "events",
            "oov_tokens",
            "oov_glyphs",
        ):
            acc[field] += row[field]

    leaf_rows: List[dict] = []

    for leaf_group, acc in sorted(leaf_acc.items()):
        leaf_rows.append(
            {
                "model": "token_markov",
                "context_order": context_order,
                "leaf_group": leaf_group,
                **acc,
                "bits_per_event": (
                    acc["total_bits"] / acc["events"]
                ),
                "bits_per_token": (
                    acc["total_bits"] / acc["tokens"]
                ),
            }
        )

    total_bits = sum(row["total_bits"] for row in leaf_rows)
    total_tokens = sum(row["tokens"] for row in leaf_rows)
    total_glyphs = sum(row["glyphs"] for row in leaf_rows)
    total_events = sum(row["events"] for row in leaf_rows)
    total_oov_tokens = sum(
        row["oov_tokens"] for row in leaf_rows
    )
    total_oov_glyphs = sum(
        row["oov_glyphs"] for row in leaf_rows
    )

    aggregate = {
        "model": "token_markov",
        "context_order": context_order,
        "token_ngram_order": context_order + 1,
        "total_bits": total_bits,
        "tokens": total_tokens,
        "glyphs": total_glyphs,
        "events": total_events,
        "bits_per_event": total_bits / total_events,
        "perplexity_per_event": 2 ** (total_bits / total_events),
        "bits_per_token": total_bits / total_tokens,
        "oov_tokens": total_oov_tokens,
        "oov_glyphs": total_oov_glyphs,
    }

    return aggregate, leaf_rows, locus_rows
