"""Finite-template slot grammar for VOYAGER.

This model is a generative latent-class model over conventionally segmented
Voynich tokens. It is designed to test whether token structure is better
explained by a small inventory of recurrent position-sensitive templates than
by a purely local Markov process.

Generative story
----------------
For each token independently:

1. Draw token length L from a smoothed categorical length distribution.
2. Draw a latent template z from categorical mixture weights.
3. Map each token position deterministically to one of S relative slots
   spanning token-initial to token-final position.
4. Emit each glyph independently from the categorical distribution associated
   with template z and that relative slot.

Thus

    P(x_1...x_L)
      = P(L) * sum_z pi_z prod_j theta[z, slot(j,L), x_j].

The model is a proper probability distribution over token strings up to the
configured maximum length. Validation-only glyphs map to a fixed ``<UNK>``
emission bucket.

Methodological constraints
--------------------------
- model fitting uses frozen training physical leaves only;
- random restart selection uses TRAINING likelihood only;
- validation selects template count only from a frozen grid;
- unreadable ``Z1`` tokens are excluded as transcription-uncertain units;
- no locked-test input is required or accessed.

For an apples-to-apples local comparator, the runner also evaluates a matched
3-gram on independently reset token sequences with an explicit ``<EOT>``
terminal event. The slot model's token-length factor is the generative
counterpart of this termination information. Both are normalized by
glyph_count + token_count for matched bits/event reporting.
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
class SlotTokenExample:
    leaf_group: str
    folio: str
    locus: str
    token_index: int
    glyphs: Tuple[str, ...]


@dataclass(frozen=True)
class SlotGrammarFitResult:
    templates: int
    seed: int
    iterations: int
    converged: bool
    training_log_likelihood_nats: float
    training_tokens: int
    training_glyphs: int

    @property
    def training_bits_per_token(self) -> float:
        return (
            -self.training_log_likelihood_nats
            / math.log(2.0)
            / self.training_tokens
        )

    @property
    def training_bits_per_event(self) -> float:
        events = self.training_glyphs + self.training_tokens
        return (
            -self.training_log_likelihood_nats
            / math.log(2.0)
            / events
        )


def relative_slot_indices(length: int, slot_count: int) -> np.ndarray:
    """Map token positions onto fixed relative slots.

    The first glyph always maps to slot 0 and the final glyph always maps to
    slot S-1 when length > 1. Interior glyphs are placed at nearest equally
    spaced relative slots. Long tokens may place multiple glyphs in the same
    slot; their emissions remain conditionally independent given the template.
    """
    if length < 1:
        raise ValueError("Token length must be >= 1")
    if slot_count < 2:
        raise ValueError("slot_count must be >= 2")

    if length == 1:
        return np.asarray([slot_count // 2], dtype=np.int32)

    return np.rint(
        np.linspace(0, slot_count - 1, num=length)
    ).astype(np.int32)


def atomic_leaf_group(locus: AnalyticalLocus) -> str:
    if locus.folio == "fRos" or locus.physical_leaf in (85, 86):
        return "f85+f86"
    if locus.physical_leaf is None:
        raise ValueError(
            f"No atomic leaf group for {locus.folio} {locus.locus}"
        )
    return f"f{locus.physical_leaf}"


def extract_slot_tokens(
    loci: Sequence[AnalyticalLocus],
) -> Tuple[List[SlotTokenExample], dict]:
    """Extract fully readable tokens for the finite-template model."""
    examples: List[SlotTokenExample] = []
    excluded_tokens = 0
    excluded_glyphs = 0

    for locus in loci:
        leaf_group = atomic_leaf_group(locus)

        for token_index, token in enumerate(locus.tokens):
            if not token:
                continue

            if UNKNOWN_STA1 in token:
                excluded_tokens += 1
                excluded_glyphs += len(token)
                continue

            examples.append(
                SlotTokenExample(
                    leaf_group=leaf_group,
                    folio=locus.folio,
                    locus=locus.locus,
                    token_index=token_index,
                    glyphs=tuple(token),
                )
            )

    diagnostics = {
        "included_tokens": len(examples),
        "included_glyphs": sum(len(x.glyphs) for x in examples),
        "excluded_tokens_with_Z1": excluded_tokens,
        "excluded_glyphs_in_Z1_tokens": excluded_glyphs,
    }
    return examples, diagnostics


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(
        float(np.exp(values - maximum).sum())
    )


class FiniteTemplateSlotGrammar:
    """Mixture of position-sensitive finite token templates."""

    def __init__(
        self,
        templates: int,
        *,
        slot_count: int = 8,
        max_token_length: int = 64,
        pseudocount: float = 0.5,
        max_iterations: int = 50,
        min_iterations: int = 5,
        tolerance_bits_per_token: float = 1e-5,
    ):
        if templates < 1:
            raise ValueError("templates must be >= 1")
        if slot_count < 2:
            raise ValueError("slot_count must be >= 2")
        if max_token_length < 1:
            raise ValueError("max_token_length must be >= 1")
        if pseudocount <= 0:
            raise ValueError("pseudocount must be > 0")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if min_iterations < 1 or min_iterations > max_iterations:
            raise ValueError(
                "min_iterations must be within 1..max_iterations"
            )
        if tolerance_bits_per_token <= 0:
            raise ValueError(
                "tolerance_bits_per_token must be > 0"
            )

        self.templates = int(templates)
        self.slot_count = int(slot_count)
        self.max_token_length = int(max_token_length)
        self.pseudocount = float(pseudocount)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.tolerance_bits_per_token = float(
            tolerance_bits_per_token
        )

        self.vocabulary: Tuple[str, ...] = ()
        self.symbol_to_index: Dict[str, int] = {}

        self.length_probabilities: Optional[np.ndarray] = None
        self.mixture_weights: Optional[np.ndarray] = None
        self.emission: Optional[np.ndarray] = None
        self.fit_result: Optional[SlotGrammarFitResult] = None

    @property
    def is_fit(self) -> bool:
        return (
            self.length_probabilities is not None
            and self.mixture_weights is not None
            and self.emission is not None
            and self.fit_result is not None
        )

    def _prepare_training_tokens(
        self,
        tokens: Sequence[Sequence[str]],
    ) -> Tuple[
        List[Tuple[np.ndarray, int]],
        int,
        int,
        np.ndarray,
    ]:
        materialized = [
            tuple(token)
            for token in tokens
            if token
        ]
        if not materialized:
            raise ValueError("Cannot fit slot grammar to no tokens")

        max_observed = max(len(token) for token in materialized)
        if max_observed > self.max_token_length:
            raise ValueError(
                "Training token exceeds configured maximum length: "
                f"observed={max_observed}, "
                f"max={self.max_token_length}"
            )

        observed = sorted(
            {
                glyph
                for token in materialized
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

        token_counter = Counter(materialized)
        encoded: List[Tuple[np.ndarray, int]] = []

        glyph_counts = np.zeros(
            len(self.vocabulary),
            dtype=np.float64,
        )

        for token, frequency in sorted(token_counter.items()):
            array = np.fromiter(
                (
                    self.symbol_to_index.get(
                        glyph,
                        self.symbol_to_index[UNK],
                    )
                    for glyph in token
                ),
                dtype=np.int32,
                count=len(token),
            )
            encoded.append((array, frequency))
            glyph_counts += np.bincount(
                array,
                minlength=len(self.vocabulary),
            ) * frequency

        training_tokens = len(materialized)
        training_glyphs = sum(len(token) for token in materialized)

        return (
            encoded,
            training_tokens,
            training_glyphs,
            glyph_counts,
        )

    def _fit_length_distribution(
        self,
        encoded: Sequence[Tuple[np.ndarray, int]],
    ) -> None:
        counts = np.zeros(
            self.max_token_length,
            dtype=np.float64,
        )

        for token, frequency in encoded:
            counts[len(token) - 1] += frequency

        counts += self.pseudocount
        self.length_probabilities = counts / counts.sum()

    def _initialize_parameters(
        self,
        glyph_counts: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> None:
        k = self.templates
        s = self.slot_count
        v = len(self.vocabulary)

        self.mixture_weights = rng.dirichlet(
            np.ones(k, dtype=np.float64)
        )

        global_probs = (
            glyph_counts + self.pseudocount
        )
        global_probs /= global_probs.sum()

        # Random but data-scaled initialization. The concentration is fixed,
        # not validation-tuned. It starts each slot near the global inventory
        # while allowing enough variation for EM to separate templates.
        concentration = 20.0
        alpha = (
            global_probs * concentration
            + 0.05
        )

        emission = np.empty(
            (k, s, v),
            dtype=np.float64,
        )
        for template in range(k):
            for slot in range(s):
                emission[template, slot] = rng.dirichlet(alpha)

        self.emission = emission

    def _token_component_log_probabilities(
        self,
        token: np.ndarray,
    ) -> np.ndarray:
        assert self.mixture_weights is not None
        assert self.emission is not None

        slots = relative_slot_indices(
            len(token),
            self.slot_count,
        )

        log_prob = np.log(self.mixture_weights)

        for position, glyph_index in enumerate(token):
            slot = slots[position]
            log_prob = (
                log_prob
                + np.log(
                    self.emission[
                        :,
                        slot,
                        glyph_index,
                    ]
                )
            )

        return log_prob

    def _e_step(
        self,
        encoded: Sequence[Tuple[np.ndarray, int]],
    ) -> Tuple[
        float,
        np.ndarray,
        np.ndarray,
    ]:
        assert self.length_probabilities is not None
        assert self.mixture_weights is not None
        assert self.emission is not None

        k = self.templates
        s = self.slot_count
        v = len(self.vocabulary)

        mixture_counts = np.zeros(k, dtype=np.float64)
        emission_counts = np.zeros(
            (k, s, v),
            dtype=np.float64,
        )

        total_log_likelihood = 0.0

        for token, frequency in encoded:
            length_log_prob = math.log(
                float(
                    self.length_probabilities[len(token) - 1]
                )
            )

            component_log = (
                self._token_component_log_probabilities(token)
            )
            mixture_log_prob = _logsumexp(component_log)

            total_log_likelihood += frequency * (
                length_log_prob + mixture_log_prob
            )

            posterior = np.exp(
                component_log - mixture_log_prob
            )

            weighted_posterior = posterior * frequency
            mixture_counts += weighted_posterior

            slots = relative_slot_indices(
                len(token),
                self.slot_count,
            )

            for position, glyph_index in enumerate(token):
                slot = slots[position]
                emission_counts[
                    :,
                    slot,
                    glyph_index,
                ] += weighted_posterior

        return (
            total_log_likelihood,
            mixture_counts,
            emission_counts,
        )

    def _m_step(
        self,
        mixture_counts: np.ndarray,
        emission_counts: np.ndarray,
    ) -> None:
        mixture = mixture_counts + self.pseudocount
        self.mixture_weights = mixture / mixture.sum()

        emission = emission_counts + self.pseudocount
        self.emission = (
            emission
            / emission.sum(axis=2, keepdims=True)
        )

    def _score_encoded_token(
        self,
        token: np.ndarray,
    ) -> float:
        assert self.length_probabilities is not None

        if len(token) > self.max_token_length:
            return float("-inf")

        length_log_prob = math.log(
            float(
                self.length_probabilities[len(token) - 1]
            )
        )
        component_log = (
            self._token_component_log_probabilities(token)
        )

        return length_log_prob + _logsumexp(component_log)

    def fit(
        self,
        tokens: Sequence[Sequence[str]],
        *,
        seed: int,
    ) -> SlotGrammarFitResult:
        (
            encoded,
            training_tokens,
            training_glyphs,
            glyph_counts,
        ) = self._prepare_training_tokens(tokens)

        self._fit_length_distribution(encoded)

        rng = np.random.default_rng(seed)
        self._initialize_parameters(
            glyph_counts,
            rng=rng,
        )

        previous_ll: Optional[float] = None
        converged = False
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            (
                log_likelihood,
                mixture_counts,
                emission_counts,
            ) = self._e_step(encoded)

            iterations = iteration

            if (
                previous_ll is not None
                and iteration >= self.min_iterations
            ):
                delta_bits_per_token = (
                    abs(log_likelihood - previous_ll)
                    / math.log(2.0)
                    / training_tokens
                )
                if (
                    delta_bits_per_token
                    < self.tolerance_bits_per_token
                ):
                    converged = True
                    break

            self._m_step(
                mixture_counts,
                emission_counts,
            )
            previous_ll = log_likelihood

        final_ll = 0.0
        for token, frequency in encoded:
            final_ll += (
                frequency * self._score_encoded_token(token)
            )

        result = SlotGrammarFitResult(
            templates=self.templates,
            seed=int(seed),
            iterations=iterations,
            converged=converged,
            training_log_likelihood_nats=float(final_ll),
            training_tokens=training_tokens,
            training_glyphs=training_glyphs,
        )
        self.fit_result = result
        return result

    def encode_token(
        self,
        token: Sequence[str],
    ) -> Tuple[np.ndarray, int]:
        if not self.symbol_to_index:
            raise RuntimeError("Slot grammar is not fit")

        if len(token) > self.max_token_length:
            raise ValueError(
                "Validation token exceeds configured maximum length: "
                f"observed={len(token)}, "
                f"max={self.max_token_length}"
            )

        unk_index = self.symbol_to_index[UNK]
        oov = sum(
            glyph not in self.symbol_to_index
            for glyph in token
        )
        encoded = np.fromiter(
            (
                self.symbol_to_index.get(glyph, unk_index)
                for glyph in token
            ),
            dtype=np.int32,
            count=len(token),
        )
        return encoded, oov

    def score_token(
        self,
        token: Sequence[str],
    ) -> Tuple[float, int]:
        if not self.is_fit:
            raise RuntimeError("Slot grammar must be fit before scoring")

        encoded, oov = self.encode_token(token)
        return self._score_encoded_token(encoded), oov

    def parameters(self) -> dict:
        if not self.is_fit:
            raise RuntimeError("Slot grammar is not fit")
        assert self.length_probabilities is not None
        assert self.mixture_weights is not None
        assert self.emission is not None

        return {
            "vocabulary": np.asarray(self.vocabulary, dtype=str),
            "length_probabilities": self.length_probabilities.copy(),
            "mixture_weights": self.mixture_weights.copy(),
            "emission": self.emission.copy(),
        }


def evaluate_slot_grammar(
    model: FiniteTemplateSlotGrammar,
    examples: Sequence[SlotTokenExample],
    *,
    templates: int,
    restart: int,
    seed: int,
) -> Tuple[dict, List[dict], List[dict]]:
    """Evaluate one train-selected finite-template model on validation."""
    locus_acc: Dict[Tuple[str, str, str], dict] = defaultdict(
        lambda: {
            "total_bits": 0.0,
            "tokens": 0,
            "glyphs": 0,
            "events": 0,
            "oov_glyphs": 0,
        }
    )

    for example in examples:
        log_likelihood_nats, oov = model.score_token(
            example.glyphs
        )

        if not math.isfinite(log_likelihood_nats):
            raise RuntimeError(
                "Non-finite slot-grammar likelihood for "
                f"{example.folio} {example.locus} "
                f"token {example.token_index}"
            )

        bits = -log_likelihood_nats / math.log(2.0)

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

    locus_rows: List[dict] = []
    for (
        leaf_group,
        folio,
        locus,
    ), acc in sorted(locus_acc.items()):
        locus_rows.append(
            {
                "model": "finite_template_slot_grammar",
                "templates": templates,
                "restart": restart,
                "seed": seed,
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
            "oov_glyphs",
        ):
            acc[field] += row[field]

    leaf_rows: List[dict] = []
    for leaf_group, acc in sorted(leaf_acc.items()):
        leaf_rows.append(
            {
                "model": "finite_template_slot_grammar",
                "templates": templates,
                "restart": restart,
                "seed": seed,
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
    total_oov = sum(row["oov_glyphs"] for row in leaf_rows)

    aggregate = {
        "model": "finite_template_slot_grammar",
        "templates": templates,
        "restart": restart,
        "seed": seed,
        "total_bits": total_bits,
        "tokens": total_tokens,
        "glyphs": total_glyphs,
        "events": total_events,
        "bits_per_event": total_bits / total_events,
        "perplexity_per_event": 2 ** (total_bits / total_events),
        "bits_per_token": total_bits / total_tokens,
        "oov_glyphs": total_oov,
    }

    return aggregate, leaf_rows, locus_rows


def build_matched_trigram_examples(
    examples: Sequence[SlotTokenExample],
) -> List[BaselineSequence]:
    """Represent each token as glyphs + explicit EOT for matched comparison."""
    return [
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
        for example in examples
    ]
