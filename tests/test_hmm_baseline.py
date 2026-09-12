from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.hmm import CategoricalHMM, evaluate_hmm
from src.models.ngram import BaselineSequence


def test_one_state_forward_matches_iid_probability():
    model = CategoricalHMM(
        1,
        max_iterations=5,
        min_iterations=2,
        batch_size=4,
    )
    fit = model.fit(
        [
            ("A", "A", "B"),
            ("A", "B"),
        ],
        seed=123,
    )
    assert math.isfinite(fit.training_log_likelihood_nats)

    scores, oov = model.score_sequences(
        [("A", "B", "A")]
    )
    assert len(scores) == 1
    assert oov == 0

    params = model.parameters()
    vocab = list(params["vocabulary"])
    a = vocab.index("A")
    b = vocab.index("B")
    expected = (
        math.log(params["emission"][0, a])
        + math.log(params["emission"][0, b])
        + math.log(params["emission"][0, a])
    )
    assert scores[0] == pytest.approx(expected, abs=1e-10)


def test_parameter_rows_are_normalized():
    model = CategoricalHMM(
        3,
        max_iterations=6,
        min_iterations=2,
        batch_size=4,
    )
    model.fit(
        [
            ("A", "B", "A", "B"),
            ("A", "C", "A", "C"),
            ("B", "A", "B", "A"),
        ],
        seed=7,
    )
    params = model.parameters()

    assert params["initial"].sum() == pytest.approx(1.0)
    assert np.allclose(
        params["transition"].sum(axis=1),
        1.0,
    )
    assert np.allclose(
        params["emission"].sum(axis=1),
        1.0,
    )
    assert np.all(params["transition"] > 0)
    assert np.all(params["emission"] > 0)


def test_same_seed_is_deterministic():
    sequences = [
        ("A", "B", "A", "B"),
        ("B", "A", "B", "A"),
        ("A", "B", "B", "A"),
    ]

    first = CategoricalHMM(
        2,
        max_iterations=7,
        min_iterations=2,
        batch_size=2,
    )
    second = CategoricalHMM(
        2,
        max_iterations=7,
        min_iterations=2,
        batch_size=2,
    )

    fit_a = first.fit(sequences, seed=99)
    fit_b = second.fit(sequences, seed=99)

    assert fit_a.training_log_likelihood_nats == pytest.approx(
        fit_b.training_log_likelihood_nats,
        abs=1e-12,
    )
    assert np.allclose(
        first.parameters()["transition"],
        second.parameters()["transition"],
    )
    assert np.allclose(
        first.parameters()["emission"],
        second.parameters()["emission"],
    )


def test_oov_maps_to_unk_and_remains_finite():
    model = CategoricalHMM(
        2,
        max_iterations=6,
        min_iterations=2,
    )
    model.fit(
        [("A", "B", "A"), ("B", "A", "B")],
        seed=11,
    )

    scores, oov = model.score_sequences(
        [("A", "NEVER_SEEN", "B")]
    )

    assert oov == 1
    assert np.isfinite(scores[0])


def test_two_state_hmm_can_model_alternating_structure():
    train = [
        ("A", "B") * 8,
        ("B", "A") * 8,
    ] * 12

    one = CategoricalHMM(
        1,
        max_iterations=10,
        min_iterations=3,
    )
    two = CategoricalHMM(
        2,
        max_iterations=20,
        min_iterations=3,
    )

    one.fit(train, seed=5)

    # Multiple deterministic starts make this synthetic test robust to a
    # locally symmetric two-state initialization.
    candidates = []
    for seed in (5, 6, 7):
        candidate = CategoricalHMM(
            2,
            max_iterations=20,
            min_iterations=3,
        )
        fit = candidate.fit(train, seed=seed)
        candidates.append((fit.training_log_likelihood_nats, candidate))

    two = max(candidates, key=lambda item: item[0])[1]

    score_one, _ = one.score_sequences(train)
    score_two, _ = two.score_sequences(train)

    assert score_two.sum() > score_one.sum()


def test_evaluate_hmm_preserves_leaf_pairing():
    model = CategoricalHMM(
        2,
        max_iterations=8,
        min_iterations=2,
    )
    model.fit(
        [
            ("A", "B", "A"),
            ("A", "B", "A"),
            ("B", "A", "B"),
        ],
        seed=22,
    )

    examples = [
        BaselineSequence(
            leaf_group="f10",
            folio="f10r",
            locus="f10r.1,+P0",
            symbols=("A", "B", "A"),
            is_boundary=(False, False, False),
        ),
        BaselineSequence(
            leaf_group="f11",
            folio="f11r",
            locus="f11r.1,+P0",
            symbols=("B", "A", "B"),
            is_boundary=(False, False, False),
        ),
    ]

    aggregate, leaves, loci = evaluate_hmm(
        model,
        examples,
        view="space_free",
        hidden_states=2,
        restart=0,
        seed=22,
    )

    assert aggregate["total_events"] == 6
    assert len(leaves) == 2
    assert len(loci) == 2
    assert {row["leaf_group"] for row in leaves} == {"f10", "f11"}
    assert all(math.isfinite(row["bits_per_event"]) for row in leaves)
