from __future__ import annotations

import math

import pytest

from src.analysis.tier0 import AnalyticalLocus
from src.models.copy_edit import (
    EOT,
    LocalCopyEditModel,
    build_matched_trigram_examples,
    evaluate_copy_edit,
    extract_copy_sequences,
    levenshtein_alignment,
)


def make_example_locus(tokens):
    return AnalyticalLocus(
        folio="f1r",
        physical_leaf=1,
        recto_verso="r",
        quire=1,
        section="herbal",
        currier="A",
        scribe="1",
        locus="f1r.1,+P0",
        locus_type="P",
        paragraph=1,
        line=1,
        source_line_number=1,
        tokens=tuple(tuple(x) for x in tokens),
    )


def test_levenshtein_alignment_counts_expected_edits():
    distance, operations = levenshtein_alignment(
        ("A", "B", "C"),
        ("A", "X", "C", "D"),
    )

    assert distance == 2
    assert sum(op == "M" for op, _, _ in operations) == 2
    assert sum(op == "S" for op, _, _ in operations) == 1
    assert sum(op == "I" for op, _, _ in operations) == 1
    assert sum(op == "D" for op, _, _ in operations) == 0


def test_unknown_token_is_excluded_and_resets_context():
    locus = make_example_locus(
        [
            ("A",),
            ("A", "B"),
            ("Z1",),
            ("C",),
            ("C", "D"),
        ]
    )

    sequences, diagnostics = extract_copy_sequences([locus])

    assert diagnostics["included_tokens"] == 4
    assert diagnostics["excluded_tokens_with_Z1"] == 1
    assert len(sequences) == 2
    assert [len(sequence) for sequence in sequences] == [2, 2]
    assert sequences[0][0].glyphs == ("A",)
    assert sequences[1][0].glyphs == ("C",)


def test_fit_is_deterministic():
    train = [
        [
            ("A", "B"),
            ("A", "B"),
            ("A", "C"),
            ("A", "B"),
        ],
        [
            ("D", "E"),
            ("D", "E"),
            ("D", "F"),
        ],
    ]

    first = LocalCopyEditModel(4)
    second = LocalCopyEditModel(4)

    fit_a = first.fit(train)
    fit_b = second.fit(train)

    assert fit_a == fit_b
    assert first.parameters()["rho_copy"] == pytest.approx(
        second.parameters()["rho_copy"]
    )


def test_exact_copy_gets_higher_edit_probability_than_unrelated():
    train = [
        [
            ("A", "B"),
            ("A", "B"),
            ("A", "B"),
            ("A", "C"),
            ("A", "B"),
            ("A", "B"),
        ]
    ]

    model = LocalCopyEditModel(4)
    model.fit(train)

    exact = model.edit_probability(
        ("A", "B"),
        ("A", "B"),
    )
    unrelated = model.edit_probability(
        ("A", "B"),
        ("C", "C"),
    )

    assert exact > unrelated
    assert exact > 0.0
    assert unrelated > 0.0


def test_score_is_finite_for_training_oov_validation_glyph():
    train = [
        [
            ("A", "B"),
            ("A", "B"),
            ("A", "C"),
        ]
    ]
    model = LocalCopyEditModel(2)
    model.fit(train)

    logp, oov, diagnostics = model.score_token(
        ("A", "NEVER_SEEN"),
        [("A", "B")],
    )

    assert math.isfinite(logp)
    assert oov == 1
    assert diagnostics["has_copy_context"] is True


def test_local_context_can_raise_probability_for_repeated_token():
    train = [
        [
            ("A", "B"),
            ("A", "B"),
            ("A", "B"),
            ("A", "C"),
            ("A", "B"),
        ]
    ]
    model = LocalCopyEditModel(4)
    model.fit(train)

    repeated_logp, _, _ = model.score_token(
        ("A", "B"),
        [("A", "B")],
    )
    no_context_logp, _, _ = model.score_token(
        ("A", "B"),
        [],
    )

    assert repeated_logp > no_context_logp


def test_matched_trigram_adds_one_eot_per_token():
    locus = make_example_locus(
        [
            ("A", "B"),
            ("C",),
        ]
    )
    sequences, _ = extract_copy_sequences([locus])

    examples = build_matched_trigram_examples(sequences)

    assert len(examples) == 2
    assert examples[0].symbols == ("A", "B", EOT)
    assert examples[0].is_boundary == (False, False, True)
    assert examples[1].symbols == ("C", EOT)


def test_evaluation_preserves_event_denominator_and_leaf_pairing():
    loci = [
        make_example_locus(
            [
                ("A", "B"),
                ("A", "B"),
                ("A", "C"),
            ]
        )
    ]
    sequences, _ = extract_copy_sequences(loci)

    model = LocalCopyEditModel(2)
    model.fit(
        [
            [
                ("A", "B"),
                ("A", "B"),
                ("A", "C"),
                ("A", "B"),
            ]
        ]
    )

    aggregate, leaves, locus_rows = evaluate_copy_edit(
        model,
        sequences,
    )

    assert aggregate["tokens"] == 3
    assert aggregate["glyphs"] == 6
    assert aggregate["events"] == 9
    assert len(leaves) == 1
    assert leaves[0]["leaf_group"] == "f1"
    assert len(locus_rows) == 1


def test_source_window_limits_number_of_candidates():
    train = [
        [
            ("A",),
            ("B",),
            ("C",),
            ("D",),
            ("E",),
        ]
    ]
    model = LocalCopyEditModel(2)
    model.fit(train)

    _, _, diagnostics = model.score_token(
        ("E",),
        [("A",), ("B",), ("C",), ("D",)],
    )

    assert diagnostics["source_candidates"] == 2
