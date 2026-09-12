"""Categorical HMM baseline for VOYAGER.

The HMM consumes the same frozen event streams used by the Tier 1 n-gram
baselines. It is a generative latent-state model trained with Baum-Welch EM.

Methodological constraints
--------------------------
- Training sequences come only from the frozen training physical leaves.
- Validation is used only to select the number of hidden states from the
  preregistered grid.
- For each hidden-state count, random initialization is resolved using
  training likelihood only: the training-best restart is the one evaluated
  on validation.
- The locked test is neither required nor accessed.
- Unreadable ``Z1`` symbols have already split/reset sequences upstream.
- Validation-only symbols map to a fixed ``<UNK>`` emission bucket.
- Symmetric 0.5 pseudo-counts are fixed before validation and are not tuned.

The implementation uses scaled forward-backward recursions and batched NumPy
operations. Log likelihoods are accumulated in natural-log units internally
and converted to bits for reported metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.models.ngram import BaselineSequence, UNK


@dataclass(frozen=True)
class HMMFitResult:
    hidden_states: int
    seed: int
    iterations: int
    converged: bool
    training_log_likelihood_nats: float
    training_events: int

    @property
    def training_bits_per_event(self) -> float:
        return (
            -self.training_log_likelihood_nats
            / math.log(2.0)
            / self.training_events
        )


class CategoricalHMM:
    """Finite-state categorical HMM with deterministic seeded EM."""

    def __init__(
        self,
        hidden_states: int,
        *,
        pseudocount: float = 0.5,
        max_iterations: int = 50,
        min_iterations: int = 5,
        tolerance_bits_per_event: float = 1e-5,
        batch_size: int = 256,
    ):
        if hidden_states < 1:
            raise ValueError("hidden_states must be >= 1")
        if pseudocount <= 0:
            raise ValueError("pseudocount must be > 0")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if min_iterations < 1 or min_iterations > max_iterations:
            raise ValueError(
                "min_iterations must be within 1..max_iterations"
            )
        if tolerance_bits_per_event <= 0:
            raise ValueError("tolerance_bits_per_event must be > 0")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.hidden_states = hidden_states
        self.pseudocount = float(pseudocount)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.tolerance_bits_per_event = float(
            tolerance_bits_per_event
        )
        self.batch_size = int(batch_size)

        self.vocabulary: Tuple[str, ...] = ()
        self.symbol_to_index: Dict[str, int] = {}

        self.initial: Optional[np.ndarray] = None
        self.transition: Optional[np.ndarray] = None
        self.emission: Optional[np.ndarray] = None
        self.fit_result: Optional[HMMFitResult] = None

    @property
    def is_fit(self) -> bool:
        return (
            self.initial is not None
            and self.transition is not None
            and self.emission is not None
            and self.fit_result is not None
        )

    def _initialize_parameters(
        self,
        vocabulary_size: int,
        rng: np.random.Generator,
    ) -> None:
        k = self.hidden_states
        self.initial = rng.dirichlet(np.ones(k))
        self.transition = rng.dirichlet(
            np.ones(k),
            size=k,
        )
        self.emission = rng.dirichlet(
            np.ones(vocabulary_size),
            size=k,
        )

    def _prepare_training_sequences(
        self,
        sequences: Sequence[Sequence[str]],
    ) -> Tuple[List[np.ndarray], int]:
        materialized = [
            tuple(sequence)
            for sequence in sequences
            if sequence
        ]
        if not materialized:
            raise ValueError("Cannot fit HMM to an empty corpus")

        observed = sorted(
            {
                symbol
                for sequence in materialized
                for symbol in sequence
                if symbol != UNK
            }
        )
        if not observed:
            raise ValueError("Training corpus has no observed symbols")

        self.vocabulary = tuple(observed) + (UNK,)
        self.symbol_to_index = {
            symbol: index
            for index, symbol in enumerate(self.vocabulary)
        }

        arrays = [
            np.fromiter(
                (
                    self.symbol_to_index.get(symbol, self.symbol_to_index[UNK])
                    for symbol in sequence
                ),
                dtype=np.int32,
                count=len(sequence),
            )
            for sequence in materialized
        ]
        events = sum(len(array) for array in arrays)
        return arrays, events

    def _encode_sequences(
        self,
        sequences: Sequence[Sequence[str]],
    ) -> Tuple[List[np.ndarray], int]:
        if not self.symbol_to_index:
            raise RuntimeError("HMM vocabulary is not initialized")
        unk_index = self.symbol_to_index[UNK]

        arrays = [
            np.fromiter(
                (
                    self.symbol_to_index.get(symbol, unk_index)
                    for symbol in sequence
                ),
                dtype=np.int32,
                count=len(sequence),
            )
            for sequence in sequences
            if sequence
        ]
        oov = sum(
            symbol not in self.symbol_to_index
            for sequence in sequences
            for symbol in sequence
        )
        return arrays, oov

    @staticmethod
    def _pad_batch(
        batch: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        lengths = np.asarray(
            [len(sequence) for sequence in batch],
            dtype=np.int32,
        )
        max_length = int(lengths.max())
        padded = np.zeros(
            (len(batch), max_length),
            dtype=np.int32,
        )
        for row, sequence in enumerate(batch):
            padded[row, : len(sequence)] = sequence
        return padded, lengths

    def _forward_batch(
        self,
        observations: np.ndarray,
        lengths: np.ndarray,
        *,
        store_alpha: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        assert self.initial is not None
        assert self.transition is not None
        assert self.emission is not None

        batch_size, max_length = observations.shape
        k = self.hidden_states

        alpha_store = (
            np.zeros((batch_size, max_length, k), dtype=np.float64)
            if store_alpha
            else None
        )
        scale_store = (
            np.ones((batch_size, max_length), dtype=np.float64)
            if store_alpha
            else None
        )

        emission_t = self.emission[:, observations[:, 0]].T
        alpha = self.initial[None, :] * emission_t
        scale = alpha.sum(axis=1)
        if np.any(scale <= 0):
            raise RuntimeError("Non-positive forward scale at t=0")
        alpha /= scale[:, None]

        log_likelihood = np.log(scale)

        if store_alpha:
            alpha_store[:, 0, :] = alpha
            scale_store[:, 0] = scale

        for t in range(1, max_length):
            active = lengths > t
            if not np.any(active):
                break

            active_obs = observations[active, t]
            emission_t = self.emission[:, active_obs].T
            next_alpha = (
                alpha[active] @ self.transition
            ) * emission_t
            scale = next_alpha.sum(axis=1)
            if np.any(scale <= 0):
                raise RuntimeError(
                    f"Non-positive forward scale at t={t}"
                )
            next_alpha /= scale[:, None]

            alpha[active] = next_alpha
            log_likelihood[active] += np.log(scale)

            if store_alpha:
                alpha_store[active, t, :] = next_alpha
                scale_store[active, t] = scale

        return log_likelihood, alpha_store, scale_store

    def _expectation_batch(
        self,
        observations: np.ndarray,
        lengths: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        assert self.transition is not None
        assert self.emission is not None

        log_likelihood, alpha_store, scale_store = self._forward_batch(
            observations,
            lengths,
            store_alpha=True,
        )
        assert alpha_store is not None
        assert scale_store is not None

        batch_size, max_length = observations.shape
        k = self.hidden_states
        v = len(self.vocabulary)

        initial_counts = np.zeros(k, dtype=np.float64)
        transition_counts = np.zeros((k, k), dtype=np.float64)
        emission_counts = np.zeros((k, v), dtype=np.float64)

        # For each row, beta is conceptually 1 at its own final valid time.
        beta = np.ones((batch_size, k), dtype=np.float64)

        for t in range(max_length - 1, -1, -1):
            active = lengths > t
            if not np.any(active):
                continue

            alpha_t = alpha_store[active, t, :]
            beta_t = beta[active]

            gamma = alpha_t * beta_t
            gamma_sum = gamma.sum(axis=1)
            gamma /= gamma_sum[:, None]

            active_obs = observations[active, t]
            for state in range(k):
                emission_counts[state] += np.bincount(
                    active_obs,
                    weights=gamma[:, state],
                    minlength=v,
                )

            if t == 0:
                initial_counts += gamma.sum(axis=0)
                continue

            has_previous = lengths > t
            if not np.any(has_previous):
                continue

            obs_t = observations[has_previous, t]
            beta_current = beta[has_previous]
            right = (
                self.emission[:, obs_t].T
                * beta_current
                / scale_store[has_previous, t, None]
            )
            left = alpha_store[has_previous, t - 1, :]

            transition_counts += (
                self.transition * (left.T @ right)
            )

            beta_previous = right @ self.transition.T
            beta[has_previous] = beta_previous

        return (
            log_likelihood,
            initial_counts,
            transition_counts,
            emission_counts,
        )

    def _e_step(
        self,
        arrays: Sequence[np.ndarray],
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        k = self.hidden_states
        v = len(self.vocabulary)

        initial_counts = np.zeros(k, dtype=np.float64)
        transition_counts = np.zeros((k, k), dtype=np.float64)
        emission_counts = np.zeros((k, v), dtype=np.float64)
        total_log_likelihood = 0.0

        for start in range(0, len(arrays), self.batch_size):
            batch = arrays[start : start + self.batch_size]
            observations, lengths = self._pad_batch(batch)
            (
                batch_ll,
                batch_initial,
                batch_transition,
                batch_emission,
            ) = self._expectation_batch(observations, lengths)

            total_log_likelihood += float(batch_ll.sum())
            initial_counts += batch_initial
            transition_counts += batch_transition
            emission_counts += batch_emission

        return (
            total_log_likelihood,
            initial_counts,
            transition_counts,
            emission_counts,
        )

    def _m_step(
        self,
        initial_counts: np.ndarray,
        transition_counts: np.ndarray,
        emission_counts: np.ndarray,
    ) -> None:
        pseudo = self.pseudocount

        initial = initial_counts + pseudo
        self.initial = initial / initial.sum()

        transition = transition_counts + pseudo
        self.transition = (
            transition
            / transition.sum(axis=1, keepdims=True)
        )

        emission = emission_counts + pseudo
        self.emission = (
            emission
            / emission.sum(axis=1, keepdims=True)
        )

    def _score_encoded(
        self,
        arrays: Sequence[np.ndarray],
    ) -> np.ndarray:
        scores: List[np.ndarray] = []
        for start in range(0, len(arrays), self.batch_size):
            batch = arrays[start : start + self.batch_size]
            observations, lengths = self._pad_batch(batch)
            batch_ll, _, _ = self._forward_batch(
                observations,
                lengths,
                store_alpha=False,
            )
            scores.append(batch_ll)

        if not scores:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(scores)

    def fit(
        self,
        sequences: Sequence[Sequence[str]],
        *,
        seed: int,
    ) -> HMMFitResult:
        arrays, training_events = self._prepare_training_sequences(
            sequences
        )

        rng = np.random.default_rng(seed)
        self._initialize_parameters(
            len(self.vocabulary),
            rng,
        )

        previous_ll: Optional[float] = None
        converged = False
        iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            (
                log_likelihood,
                initial_counts,
                transition_counts,
                emission_counts,
            ) = self._e_step(arrays)

            iterations = iteration

            if previous_ll is not None and iteration >= self.min_iterations:
                delta_bits_per_event = (
                    abs(log_likelihood - previous_ll)
                    / math.log(2.0)
                    / training_events
                )
                if delta_bits_per_event < self.tolerance_bits_per_event:
                    converged = True
                    break

            self._m_step(
                initial_counts,
                transition_counts,
                emission_counts,
            )
            previous_ll = log_likelihood

        # Score the actual final parameter values. This also makes the
        # reported training likelihood comparable across restarts.
        final_ll = float(self._score_encoded(arrays).sum())

        result = HMMFitResult(
            hidden_states=self.hidden_states,
            seed=int(seed),
            iterations=iterations,
            converged=converged,
            training_log_likelihood_nats=final_ll,
            training_events=training_events,
        )
        self.fit_result = result
        return result

    def score_sequences(
        self,
        sequences: Sequence[Sequence[str]],
    ) -> Tuple[np.ndarray, int]:
        if not self.is_fit:
            raise RuntimeError("HMM must be fit before scoring")

        materialized = [tuple(sequence) for sequence in sequences if sequence]
        arrays, oov = self._encode_sequences(materialized)
        return self._score_encoded(arrays), oov

    def parameters(self) -> dict:
        if not self.is_fit:
            raise RuntimeError("HMM is not fit")
        assert self.initial is not None
        assert self.transition is not None
        assert self.emission is not None
        return {
            "vocabulary": np.asarray(self.vocabulary, dtype=str),
            "initial": self.initial.copy(),
            "transition": self.transition.copy(),
            "emission": self.emission.copy(),
        }


def evaluate_hmm(
    model: CategoricalHMM,
    examples: Sequence[BaselineSequence],
    *,
    view: str,
    hidden_states: int,
    restart: int,
    seed: int,
) -> Tuple[dict, List[dict], List[dict]]:
    """Evaluate one train-selected HMM restart on validation."""
    nonempty = [example for example in examples if example.symbols]
    scores_nats, _ = model.score_sequences(
        [example.symbols for example in nonempty]
    )
    if len(scores_nats) != len(nonempty):
        raise RuntimeError("Sequence score count mismatch")

    locus_acc: Dict[Tuple[str, str, str], dict] = {}

    for example, log_likelihood_nats in zip(nonempty, scores_nats):
        key = (example.leaf_group, example.folio, example.locus)
        row = locus_acc.setdefault(
            key,
            {
                "total_bits": 0.0,
                "total_events": 0,
                "oov_events": 0,
            },
        )
        row["total_bits"] += (
            -float(log_likelihood_nats) / math.log(2.0)
        )
        row["total_events"] += len(example.symbols)
        row["oov_events"] += sum(
            symbol not in model.symbol_to_index
            for symbol in example.symbols
        )

    locus_rows = []
    for (leaf_group, folio, locus), acc in sorted(locus_acc.items()):
        locus_rows.append(
            {
                "view": view,
                "hidden_states": hidden_states,
                "restart": restart,
                "seed": seed,
                "leaf_group": leaf_group,
                "folio": folio,
                "locus": locus,
                **acc,
                "bits_per_event": (
                    acc["total_bits"] / acc["total_events"]
                ),
            }
        )

    leaf_acc: Dict[str, dict] = {}
    for row in locus_rows:
        acc = leaf_acc.setdefault(
            row["leaf_group"],
            {
                "total_bits": 0.0,
                "total_events": 0,
                "oov_events": 0,
            },
        )
        acc["total_bits"] += row["total_bits"]
        acc["total_events"] += row["total_events"]
        acc["oov_events"] += row["oov_events"]

    leaf_rows = []
    for leaf_group, acc in sorted(leaf_acc.items()):
        leaf_rows.append(
            {
                "view": view,
                "hidden_states": hidden_states,
                "restart": restart,
                "seed": seed,
                "leaf_group": leaf_group,
                **acc,
                "bits_per_event": (
                    acc["total_bits"] / acc["total_events"]
                ),
            }
        )

    total_bits = sum(row["total_bits"] for row in leaf_rows)
    total_events = sum(row["total_events"] for row in leaf_rows)
    total_oov = sum(row["oov_events"] for row in leaf_rows)

    aggregate = {
        "view": view,
        "hidden_states": hidden_states,
        "restart": restart,
        "seed": seed,
        "smoothing": "symmetric_pseudocount_0.5",
        "total_bits": total_bits,
        "total_events": total_events,
        "bits_per_event": total_bits / total_events,
        "perplexity_per_event": 2 ** (total_bits / total_events),
        "oov_events": total_oov,
    }

    return aggregate, leaf_rows, locus_rows
