from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.slot_grammar import (
    FiniteTemplateSlotGrammar,
    SlotTokenExample,
    build_matched_trigram_examples,
    evaluate_slot_grammar,
    relative_slot_indices,
)


def test_relative_slots_anchor_initial_and_final():
    slots = relative_slot_indices(5, 8)
    assert slots[0] == 0
    assert slots[-1] == 7
    assert np.all(np.diff(slots) >= 0)

    singleton = relative_slot_indices(1, 8)
    assert singleton.tolist() == [4]


def test_single_template_parameters_normalize():
    model = FiniteTemplateSlotGrammar(
        1,
        slot_count=8,
        max_token_length=16,
        max_iterations=6,
        min_iterations=2,
    )
    model.fit(
        [
            ("A", "B"),
            ("A", "C"),
            ("A", "B"),
        ],
        seed=7,
    )

    params = model.parameters()

    assert params["length_probabilities"].sum() == pytest.approx(1.0)
    assert params["mixture_weights"].sum() == pytest.approx(1.0)
    assert np.allclose(
        params["emission"].sum(axis=2),
        1.0,
    )
    assert np.all(params["emission"] > 0)


def test_same_seed_is_deterministic():
    tokens = [
        ("A", "B"),
        ("A", "C"),
        ("D", "B"),
        ("D", "C"),
    ] * 3

    first = FiniteTemplateSlotGrammar(
        2,
        max_token_length=16,
        max_iterations=8,
        min_iterations=2,
    )
    second = FiniteTemplateSlotGrammar(
        2,
        max_token_length=16,
        max_iterations=8,
        min_iterations=2,
    )

    fit_a = first.fit(tokens, seed=123)
    fit_b = second.fit(tokens, seed=123)

    assert fit_a.training_log_likelihood_nats == pytest.approx(
        fit_b.training_log_likelihood_nats,
        abs=1e-12,
    )
    assert np.allclose(
        first.parameters()["mixture_weights"],
        second.parameters()["mixture_weights"],
    )
    assert np.allclose(
        first.parameters()["emission"],
        second.parameters()["emission"],
    )


def test_oov_glyph_maps_to_unk_and_scores_finitely():
    model = FiniteTemplateSlotGrammar(
        2,
        max_token_length=16,
        max_iterations=8,
        min_iterations=2,
    )
    model.fit(
        [
            ("A", "B"),
            ("C", "B"),
            ("A", "C"),
        ],
        seed=5,
    )

    score, oov = model.score_token(
        ("A", "NEVER_SEEN"),
    )

    assert oov == 1
    assert math.isfinite(score)


def test_latent_templates_help_on_two_token_families():
    # Families differ jointly at initial/final slots. A one-template model
    # factorizes those positions and cannot capture their correlation.
    train = (
        [("A", "X")] * 50
        + [("B", "Y")] * 50
    )

    one = FiniteTemplateSlotGrammar(
        1,
        max_token_length=8,
        max_iterations=12,
        min_iterations=3,
    )
    fit_one = one.fit(train, seed=1)

    candidates = []
    for seed in (1, 2, 3):
        model = FiniteTemplateSlotGrammar(
            2,
            max_token_length=8,
            max_iterations=20,
            min_iterations=3,
        )
        fit = model.fit(train, seed=seed)
        candidates.append((fit.training_log_likelihood_nats, model))

    best_two_ll, _ = max(candidates, key=lambda item: item[0])

    assert best_two_ll > fit_one.training_log_likelihood_nats


def test_evaluation_preserves_leaf_pairing():
    model = FiniteTemplateSlotGrammar(
        2,
        max_token_length=16,
        max_iterations=8,
        min_iterations=2,
    )
    model.fit(
        [
            ("A", "B"),
            ("A", "C"),
            ("D", "B"),
            ("D", "C"),
        ],
        seed=12,
    )

    examples = [
        SlotTokenExample(
            leaf_group="f10",
            folio="f10r",
            locus="f10r.1,+P0",
            token_index=0,
            glyphs=("A", "B"),
        ),
        SlotTokenExample(
            leaf_group="f11",
            folio="f11r",
            locus="f11r.1,+P0",
            token_index=0,
            glyphs=("D", "C"),
        ),
    ]

    aggregate, leaves, loci = evaluate_slot_grammar(
        model,
        examples,
        templates=2,
        restart=0,
        seed=12,
    )

    assert aggregate["tokens"] == 2
    assert aggregate["glyphs"] == 4
    assert aggregate["events"] == 6
    assert len(leaves) == 2
    assert len(loci) == 2
    assert {row["leaf_group"] for row in leaves} == {"f10", "f11"}


def test_matched_trigram_adds_one_eot_per_token():
    examples = [
        SlotTokenExample(
            leaf_group="f1",
            folio="f1r",
            locus="f1r.1,+P0",
            token_index=0,
            glyphs=("A", "B", "C"),
        )
    ]

    converted = build_matched_trigram_examples(examples)

    assert len(converted) == 1
    assert converted[0].symbols == ("A", "B", "C", "<EOT>")
    assert converted[0].is_boundary == (
        False,
        False,
        False,
        True,
    )
