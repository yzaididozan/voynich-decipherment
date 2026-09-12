"""Local copy-edit / self-copy generative model for VOYAGER.

The model is deliberately interpretable.  A token is generated either by:

1. innovation:
   draw a non-empty token length from a smoothed training distribution and
   emit glyphs independently from a smoothed training glyph distribution; or

2. local copy-edit:
   choose one of the previous W readable tokens in the same locus segment,
   then pass it through a probabilistic edit channel.

The edit channel is a proper distribution over non-empty target strings.
For each source-token gap it emits a geometric number of insertions.  Each
source glyph is then either deleted or emits one target glyph.  Emission is
either an exact copy or a draw from the training glyph distribution.

Training estimates channel parameters from TRAIN ONLY.  For parameter
estimation, each training token with context is paired to the most similar
previous token within W using normalized Levenshtein distance; ties prefer
lower raw edit distance and then the nearer source.  Validation likelihood
does NOT choose a source using the target.  It marginalizes over all available
previous sources using a train-estimated distance prior.

Tokens containing the transcription-unknown STA1 symbol Z1 are excluded and
reset local copy context.  The locked test is not accessed by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.analysis.tier0 import AnalyticalLocus, UNKNOWN_STA1
from src.models.ngram import BaselineSequence, UNK


EOT = "<EOT>"


@dataclass(frozen=True)
class CopyTokenExample:
    leaf_group: str
    folio: str
    locus: str
    token_index: int
    segment_index: int
    glyphs: Tuple[str, ...]


@dataclass(frozen=True)
class CopyEditFitResult:
    source_window: int
    training_tokens: int
    training_glyphs: int
    inferred_pairs: int
    insertion_count: int
    deletion_count: int
    copy_count: int
    substitution_count: int
    p_insert: float
    p_delete: float
    p_copy: float
    rho_copy: float
    rho_iterations: int
    rho_converged: bool

    @property
    def training_events(self) -> int:
        return self.training_glyphs + self.training_tokens


def atomic_leaf_group(locus: AnalyticalLocus) -> str:
    if locus.folio == "fRos" or locus.physical_leaf in (85, 86):
        return "f85+f86"
    if locus.physical_leaf is None:
        raise ValueError(
            f"No atomic leaf group for {locus.folio} {locus.locus}"
        )
    return f"f{locus.physical_leaf}"


def extract_copy_sequences(
    loci: Sequence[AnalyticalLocus],
) -> Tuple[List[List[CopyTokenExample]], dict]:
    """Build readable within-locus token segments.

    A token containing Z1 is excluded and breaks the self-copy context so the
    model never copies across an unreadable token.
    """
    sequences: List[List[CopyTokenExample]] = []
    excluded_tokens = 0
    excluded_glyphs = 0
    included_glyphs = 0
    segment_count = 0

    for locus in loci:
        current: List[CopyTokenExample] = []
        segment_index = 0
        leaf_group = atomic_leaf_group(locus)

        def flush() -> None:
            nonlocal current, segment_count, segment_index
            if current:
                sequences.append(current)
                segment_count += 1
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
            included_glyphs += len(glyphs)
            current.append(
                CopyTokenExample(
                    leaf_group=leaf_group,
                    folio=locus.folio,
                    locus=locus.locus,
                    token_index=token_index,
                    segment_index=segment_index,
                    glyphs=glyphs,
                )
            )

        flush()

    included_tokens = sum(len(sequence) for sequence in sequences)

    return sequences, {
        "included_tokens": included_tokens,
        "included_glyphs": included_glyphs,
        "excluded_tokens_with_Z1": excluded_tokens,
        "excluded_glyphs_in_Z1_tokens": excluded_glyphs,
        "readable_segments": segment_count,
        "unknown_resets_context": True,
    }


def levenshtein_alignment(
    source: Sequence[str],
    target: Sequence[str],
) -> Tuple[int, List[Tuple[str, Optional[str], Optional[str]]]]:
    """Return deterministic unit-cost Levenshtein alignment.

    Tie order is diagonal (match/substitution), deletion, insertion.
    """
    source = tuple(source)
    target = tuple(target)
    m = len(source)
    n = len(target)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back: List[List[Optional[str]]] = [
        [None] * (n + 1)
        for _ in range(m + 1)
    ]

    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            substitution_cost = 0 if source[i - 1] == target[j - 1] else 1
            diagonal = dp[i - 1][j - 1] + substitution_cost
            deletion = dp[i - 1][j] + 1
            insertion = dp[i][j - 1] + 1

            best = min(diagonal, deletion, insertion)
            dp[i][j] = best

            if diagonal == best:
                back[i][j] = "M" if substitution_cost == 0 else "S"
            elif deletion == best:
                back[i][j] = "D"
            else:
                back[i][j] = "I"

    operations: List[Tuple[str, Optional[str], Optional[str]]] = []
    i = m
    j = n

    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            operations.append((op, source[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            operations.append((op, source[i - 1], None))
            i -= 1
        elif op == "I":
            operations.append((op, None, target[j - 1]))
            j -= 1
        else:
            raise RuntimeError("Invalid Levenshtein backtrace state")

    operations.reverse()
    return dp[m][n], operations


def normalized_levenshtein(
    source: Sequence[str],
    target: Sequence[str],
) -> Tuple[float, int]:
    distance, _ = levenshtein_alignment(source, target)
    denominator = max(len(source), len(target), 1)
    return distance / denominator, distance


class LocalCopyEditModel:
    """Mixture of token innovation and local self-copy with edits."""

    def __init__(
        self,
        source_window: int,
        *,
        max_token_length: int = 64,
        pseudocount: float = 0.5,
        rho_max_iterations: int = 100,
        rho_tolerance: float = 1e-10,
    ):
        if source_window < 1:
            raise ValueError("source_window must be >= 1")
        if max_token_length < 1:
            raise ValueError("max_token_length must be >= 1")
        if pseudocount <= 0:
            raise ValueError("pseudocount must be > 0")
        if rho_max_iterations < 1:
            raise ValueError("rho_max_iterations must be >= 1")
        if rho_tolerance <= 0:
            raise ValueError("rho_tolerance must be > 0")

        self.source_window = int(source_window)
        self.max_token_length = int(max_token_length)
        self.pseudocount = float(pseudocount)
        self.rho_max_iterations = int(rho_max_iterations)
        self.rho_tolerance = float(rho_tolerance)

        self.vocabulary: Tuple[str, ...] = ()
        self.glyph_probabilities: Optional[np.ndarray] = None
        self.length_probabilities: Optional[np.ndarray] = None
        self.distance_probabilities: Optional[np.ndarray] = None
        self.symbol_to_index: Dict[str, int] = {}

        self.p_insert: Optional[float] = None
        self.p_delete: Optional[float] = None
        self.p_copy: Optional[float] = None
        self.rho_copy: Optional[float] = None
        self.fit_result: Optional[CopyEditFitResult] = None

    @property
    def is_fit(self) -> bool:
        return (
            self.glyph_probabilities is not None
            and self.length_probabilities is not None
            and self.distance_probabilities is not None
            and self.p_insert is not None
            and self.p_delete is not None
            and self.p_copy is not None
            and self.rho_copy is not None
            and self.fit_result is not None
        )

    def _map_glyph(self, glyph: str) -> str:
        return glyph if glyph in self.symbol_to_index else UNK

    def _map_token(self, token: Sequence[str]) -> Tuple[str, ...]:
        return tuple(self._map_glyph(glyph) for glyph in token)

    def _glyph_probability(self, glyph: str) -> float:
        if self.glyph_probabilities is None:
            raise RuntimeError("Model is not fit")
        mapped = self._map_glyph(glyph)
        return float(
            self.glyph_probabilities[self.symbol_to_index[mapped]]
        )

    def _fit_innovation(
        self,
        token_sequences: Sequence[Sequence[Sequence[str]]],
    ) -> Tuple[int, int]:
        tokens = [
            tuple(token)
            for sequence in token_sequences
            for token in sequence
            if token
        ]
        if not tokens:
            raise ValueError("Cannot fit copy-edit model to no tokens")

        max_observed = max(len(token) for token in tokens)
        if max_observed > self.max_token_length:
            raise ValueError(
                "Training token exceeds configured maximum length: "
                f"observed={max_observed}, max={self.max_token_length}"
            )

        observed = sorted(
            {
                glyph
                for token in tokens
                for glyph in token
                if glyph != UNK
            }
        )
        if not observed:
            raise ValueError("Training tokens have no glyphs")

        self.vocabulary = tuple(observed) + (UNK,)
        self.symbol_to_index = {
            glyph: index
            for index, glyph in enumerate(self.vocabulary)
        }

        glyph_counts = np.full(
            len(self.vocabulary),
            self.pseudocount,
            dtype=np.float64,
        )
        for token in tokens:
            for glyph in token:
                glyph_counts[self.symbol_to_index[glyph]] += 1.0
        self.glyph_probabilities = glyph_counts / glyph_counts.sum()

        length_counts = np.full(
            self.max_token_length,
            self.pseudocount,
            dtype=np.float64,
        )
        for token in tokens:
            length_counts[len(token) - 1] += 1.0
        self.length_probabilities = length_counts / length_counts.sum()

        return len(tokens), sum(len(token) for token in tokens)

    def _select_training_source(
        self,
        target: Sequence[str],
        history: Sequence[Sequence[str]],
    ) -> Tuple[int, Tuple[str, ...]]:
        candidates = []
        max_distance = min(self.source_window, len(history))

        for distance in range(1, max_distance + 1):
            source = tuple(history[-distance])
            normalized, raw = normalized_levenshtein(source, target)
            candidates.append(
                (normalized, raw, distance, source)
            )

        if not candidates:
            raise ValueError("No training source candidates")

        _, _, distance, source = min(
            candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        return distance, source

    def _fit_edit_channel(
        self,
        token_sequences: Sequence[Sequence[Sequence[str]]],
    ) -> Tuple[int, int, int, int, int, np.ndarray]:
        insertions = 0
        deletions = 0
        copies = 0
        substitutions = 0
        source_chars = 0
        gap_trials = 0
        pair_count = 0
        distance_counts = np.full(
            self.source_window,
            self.pseudocount,
            dtype=np.float64,
        )

        for sequence in token_sequences:
            history: List[Tuple[str, ...]] = []

            for target in sequence:
                target = tuple(target)

                if history:
                    distance, source = self._select_training_source(
                        target,
                        history,
                    )
                    _, operations = levenshtein_alignment(source, target)

                    for op, _, _ in operations:
                        if op == "I":
                            insertions += 1
                        elif op == "D":
                            deletions += 1
                        elif op == "M":
                            copies += 1
                        elif op == "S":
                            substitutions += 1
                        else:
                            raise RuntimeError(f"Unknown edit op: {op}")

                    source_chars += len(source)
                    gap_trials += len(source) + 1
                    pair_count += 1
                    distance_counts[distance - 1] += 1.0

                history.append(target)

        if pair_count == 0:
            raise ValueError(
                "Training corpus has no tokens with local copy context"
            )

        pc = self.pseudocount

        # Geometric insert count per source gap:
        # P(K=k)=(1-p_insert)*p_insert**k
        self.p_insert = (
            (insertions + pc)
            / (insertions + gap_trials + 2.0 * pc)
        )

        self.p_delete = (
            (deletions + pc)
            / (source_chars + 2.0 * pc)
        )

        emitted = copies + substitutions
        self.p_copy = (
            (copies + pc)
            / (emitted + 2.0 * pc)
        )

        self.distance_probabilities = (
            distance_counts / distance_counts.sum()
        )

        return (
            pair_count,
            insertions,
            deletions,
            copies,
            substitutions,
            distance_counts,
        )

    def innovation_probability(
        self,
        token: Sequence[str],
    ) -> float:
        if self.length_probabilities is None:
            raise RuntimeError("Model is not fit")
        if not token:
            return 0.0
        if len(token) > self.max_token_length:
            return 0.0

        mapped = self._map_token(token)
        probability = float(
            self.length_probabilities[len(mapped) - 1]
        )

        for glyph in mapped:
            probability *= self._glyph_probability(glyph)

        return probability

    def _raw_edit_probability(
        self,
        source: Sequence[str],
        target: Sequence[str],
    ) -> float:
        if (
            self.p_insert is None
            or self.p_delete is None
            or self.p_copy is None
        ):
            raise RuntimeError("Edit channel is not fit")

        source = self._map_token(source)
        target = self._map_token(target)
        n = len(target)

        dp = [0.0] * (n + 1)
        dp[0] = 1.0

        for source_glyph in source:
            gap = [0.0] * (n + 1)

            for start in range(n + 1):
                if dp[start] == 0.0:
                    continue

                probability = dp[start] * (1.0 - self.p_insert)
                gap[start] += probability

                for end in range(start + 1, n + 1):
                    probability *= (
                        self.p_insert
                        * self._glyph_probability(target[end - 1])
                    )
                    gap[end] += probability

            next_dp = [0.0] * (n + 1)

            for consumed in range(n + 1):
                if gap[consumed] == 0.0:
                    continue

                next_dp[consumed] += (
                    gap[consumed] * self.p_delete
                )

                if consumed < n:
                    target_glyph = target[consumed]
                    emission = (1.0 - self.p_delete) * (
                        self.p_copy
                        * float(target_glyph == source_glyph)
                        + (1.0 - self.p_copy)
                        * self._glyph_probability(target_glyph)
                    )
                    next_dp[consumed + 1] += (
                        gap[consumed] * emission
                    )

            dp = next_dp

        total = 0.0

        for start in range(n + 1):
            if dp[start] == 0.0:
                continue

            probability = dp[start] * (1.0 - self.p_insert)
            for end in range(start + 1, n + 1):
                probability *= (
                    self.p_insert
                    * self._glyph_probability(target[end - 1])
                )

            total += probability

        return total

    def edit_probability(
        self,
        source: Sequence[str],
        target: Sequence[str],
    ) -> float:
        """P(non-empty target | source), conditioning out empty output."""
        if not target:
            return 0.0

        raw = self._raw_edit_probability(source, target)
        empty = self._raw_edit_probability(source, ())

        denominator = 1.0 - empty
        if denominator <= 0.0:
            raise RuntimeError(
                "Edit channel assigns all mass to empty output"
            )

        return raw / denominator

    def local_copy_probability(
        self,
        token: Sequence[str],
        history: Sequence[Sequence[str]],
    ) -> float:
        if self.distance_probabilities is None:
            raise RuntimeError("Model is not fit")
        if not history:
            return 0.0

        k = min(self.source_window, len(history))
        weights = self.distance_probabilities[:k]
        normalizer = float(weights.sum())
        if normalizer <= 0.0:
            raise RuntimeError("Invalid source-distance distribution")

        probability = 0.0

        for distance in range(1, k + 1):
            weight = float(weights[distance - 1]) / normalizer
            source = history[-distance]
            probability += weight * self.edit_probability(
                source,
                token,
            )

        return probability

    def _fit_rho(
        self,
        token_sequences: Sequence[Sequence[Sequence[str]]],
    ) -> Tuple[float, int, bool]:
        rho = 0.5
        pc = self.pseudocount
        converged = False
        iterations = 0

        comparable: List[Tuple[float, float]] = []

        for sequence in token_sequences:
            history: List[Tuple[str, ...]] = []

            for target in sequence:
                target = tuple(target)

                if history:
                    innovation = self.innovation_probability(target)
                    local_copy = self.local_copy_probability(
                        target,
                        history,
                    )
                    comparable.append((innovation, local_copy))

                history.append(target)

        if not comparable:
            raise ValueError("No training tokens available for rho fitting")

        for iteration in range(1, self.rho_max_iterations + 1):
            responsibility_sum = 0.0

            for innovation, local_copy in comparable:
                copy_term = rho * local_copy
                innovation_term = (1.0 - rho) * innovation
                denominator = copy_term + innovation_term

                if denominator <= 0.0:
                    raise RuntimeError(
                        "Zero train likelihood while fitting rho"
                    )

                responsibility_sum += copy_term / denominator

            updated = (
                responsibility_sum + pc
            ) / (
                len(comparable) + 2.0 * pc
            )

            iterations = iteration

            if abs(updated - rho) < self.rho_tolerance:
                rho = updated
                converged = True
                break

            rho = updated

        return rho, iterations, converged

    def fit(
        self,
        token_sequences: Sequence[Sequence[Sequence[str]]],
    ) -> CopyEditFitResult:
        training_tokens, training_glyphs = self._fit_innovation(
            token_sequences
        )

        (
            pair_count,
            insertions,
            deletions,
            copies,
            substitutions,
            _,
        ) = self._fit_edit_channel(token_sequences)

        rho, rho_iterations, rho_converged = self._fit_rho(
            token_sequences
        )
        self.rho_copy = float(rho)

        assert self.p_insert is not None
        assert self.p_delete is not None
        assert self.p_copy is not None

        result = CopyEditFitResult(
            source_window=self.source_window,
            training_tokens=training_tokens,
            training_glyphs=training_glyphs,
            inferred_pairs=pair_count,
            insertion_count=insertions,
            deletion_count=deletions,
            copy_count=copies,
            substitution_count=substitutions,
            p_insert=float(self.p_insert),
            p_delete=float(self.p_delete),
            p_copy=float(self.p_copy),
            rho_copy=float(self.rho_copy),
            rho_iterations=rho_iterations,
            rho_converged=rho_converged,
        )
        self.fit_result = result
        return result

    def score_token(
        self,
        token: Sequence[str],
        history: Sequence[Sequence[str]],
    ) -> Tuple[float, int, dict]:
        if not self.is_fit:
            raise RuntimeError("Copy-edit model must be fit before scoring")
        if not token:
            raise ValueError("Cannot score an empty token")
        if len(token) > self.max_token_length:
            raise ValueError(
                "Validation token exceeds configured maximum length: "
                f"observed={len(token)}, max={self.max_token_length}"
            )

        oov = sum(
            glyph not in self.symbol_to_index
            for glyph in token
        )

        innovation = self.innovation_probability(token)

        if history:
            local_copy = self.local_copy_probability(
                token,
                history,
            )
            assert self.rho_copy is not None
            total = (
                self.rho_copy * local_copy
                + (1.0 - self.rho_copy) * innovation
            )
            posterior_copy = (
                self.rho_copy * local_copy / total
                if total > 0.0
                else 0.0
            )
        else:
            local_copy = 0.0
            total = innovation
            posterior_copy = 0.0

        if total <= 0.0 or not math.isfinite(total):
            raise RuntimeError(
                "Non-positive or non-finite token probability"
            )

        return math.log(total), oov, {
            "innovation_probability": innovation,
            "local_copy_probability": local_copy,
            "posterior_copy": posterior_copy,
            "has_copy_context": bool(history),
            "source_candidates": min(
                len(history),
                self.source_window,
            ),
        }

    def parameters(self) -> dict:
        if not self.is_fit:
            raise RuntimeError("Copy-edit model is not fit")
        assert self.glyph_probabilities is not None
        assert self.length_probabilities is not None
        assert self.distance_probabilities is not None
        assert self.p_insert is not None
        assert self.p_delete is not None
        assert self.p_copy is not None
        assert self.rho_copy is not None

        return {
            "vocabulary": np.asarray(self.vocabulary, dtype=str),
            "glyph_probabilities": self.glyph_probabilities.copy(),
            "length_probabilities": self.length_probabilities.copy(),
            "distance_probabilities": self.distance_probabilities.copy(),
            "p_insert": np.asarray(self.p_insert),
            "p_delete": np.asarray(self.p_delete),
            "p_copy": np.asarray(self.p_copy),
            "rho_copy": np.asarray(self.rho_copy),
            "source_window": np.asarray(self.source_window),
            "max_token_length": np.asarray(self.max_token_length),
        }


def evaluate_copy_edit(
    model: LocalCopyEditModel,
    sequences: Sequence[Sequence[CopyTokenExample]],
) -> Tuple[dict, List[dict], List[dict]]:
    """Teacher-forced held-out likelihood using only previous observed tokens."""
    locus_acc: Dict[Tuple[str, str, str], dict] = defaultdict(
        lambda: {
            "total_bits": 0.0,
            "tokens": 0,
            "glyphs": 0,
            "events": 0,
            "oov_glyphs": 0,
            "tokens_with_copy_context": 0,
            "source_candidates_total": 0,
            "copy_posterior_sum": 0.0,
        }
    )

    for sequence in sequences:
        history: List[Tuple[str, ...]] = []

        for example in sequence:
            log_probability, oov, diagnostics = model.score_token(
                example.glyphs,
                history,
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
            acc["oov_glyphs"] += oov

            if diagnostics["has_copy_context"]:
                acc["tokens_with_copy_context"] += 1
                acc["source_candidates_total"] += diagnostics[
                    "source_candidates"
                ]
                acc["copy_posterior_sum"] += diagnostics[
                    "posterior_copy"
                ]

            history.append(example.glyphs)

    locus_rows: List[dict] = []
    for (
        leaf_group,
        folio,
        locus,
    ), acc in sorted(locus_acc.items()):
        context_tokens = acc["tokens_with_copy_context"]
        locus_rows.append(
            {
                "model": "local_copy_edit",
                "source_window": model.source_window,
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
                "mean_copy_posterior_with_context": (
                    acc["copy_posterior_sum"] / context_tokens
                    if context_tokens
                    else None
                ),
            }
        )

    leaf_acc: Dict[str, dict] = defaultdict(
        lambda: {
            "total_bits": 0.0,
            "tokens": 0,
            "glyphs": 0,
            "events": 0,
            "oov_glyphs": 0,
            "tokens_with_copy_context": 0,
            "source_candidates_total": 0,
            "copy_posterior_sum": 0.0,
        }
    )

    for row in locus_rows:
        acc = leaf_acc[row["leaf_group"]]
        for field in (
            "total_bits",
            "tokens",
            "glyphs",
            "events",
            "oov_glyphs",
            "tokens_with_copy_context",
            "source_candidates_total",
            "copy_posterior_sum",
        ):
            acc[field] += row[field]

    leaf_rows: List[dict] = []
    for leaf_group, acc in sorted(leaf_acc.items()):
        context_tokens = acc["tokens_with_copy_context"]
        leaf_rows.append(
            {
                "model": "local_copy_edit",
                "source_window": model.source_window,
                "leaf_group": leaf_group,
                **acc,
                "bits_per_event": (
                    acc["total_bits"] / acc["events"]
                ),
                "bits_per_token": (
                    acc["total_bits"] / acc["tokens"]
                ),
                "mean_copy_posterior_with_context": (
                    acc["copy_posterior_sum"] / context_tokens
                    if context_tokens
                    else None
                ),
            }
        )

    total_bits = sum(row["total_bits"] for row in leaf_rows)
    total_tokens = sum(row["tokens"] for row in leaf_rows)
    total_glyphs = sum(row["glyphs"] for row in leaf_rows)
    total_events = sum(row["events"] for row in leaf_rows)
    total_oov = sum(row["oov_glyphs"] for row in leaf_rows)
    total_context = sum(
        row["tokens_with_copy_context"]
        for row in leaf_rows
    )
    total_candidates = sum(
        row["source_candidates_total"]
        for row in leaf_rows
    )
    total_copy_posterior = sum(
        row["copy_posterior_sum"]
        for row in leaf_rows
    )

    aggregate = {
        "model": "local_copy_edit",
        "source_window": model.source_window,
        "total_bits": total_bits,
        "tokens": total_tokens,
        "glyphs": total_glyphs,
        "events": total_events,
        "bits_per_event": total_bits / total_events,
        "perplexity_per_event": 2 ** (total_bits / total_events),
        "bits_per_token": total_bits / total_tokens,
        "oov_glyphs": total_oov,
        "tokens_with_copy_context": total_context,
        "source_candidates_total": total_candidates,
        "mean_copy_posterior_with_context": (
            total_copy_posterior / total_context
            if total_context
            else None
        ),
    }

    return aggregate, leaf_rows, locus_rows


def build_matched_trigram_examples(
    sequences: Sequence[Sequence[CopyTokenExample]],
) -> List[BaselineSequence]:
    """Represent the exact evaluated tokens as token-reset glyphs + EOT."""
    output: List[BaselineSequence] = []

    for sequence in sequences:
        for example in sequence:
            output.append(
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

    return output
