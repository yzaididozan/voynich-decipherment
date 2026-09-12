from __future__ import annotations

import math

import pytest

from src.analysis.tier0 import AnalyticalLocus
from src.models.token_markov import (
    EOT,
    TOKEN_BOS,
    TokenMarkovModel,
    build_matched_trigram_examples,
    evaluate_token_markov,
    extract_token_sequences,
)


def make_locus(tokens, *, leaf=1, locus="f1r.1,+P0"):
    return AnalyticalLocus(
        folio=f"f{leaf}r",
        physical_leaf=leaf,
        recto_verso="r",
        quire=1,
        section="herbal",
        currier="A",
        scribe="1",
        locus=locus,
        locus_type="P",
        paragraph=1,
        line=1,
        source_line_number=1,
        tokens=tuple(tuple(x) for x in tokens),
    )


def test_unknown_token_is_excluded_and_resets_context():
    locus = make_locus(
        [
            ("A",),
            ("B",),
            ("Z1",),
            ("C",),
            ("D",),
        ]
    )

    sequences, diagnostics = extract_token_sequences([locus])

    assert diagnostics["included_tokens"] == 4
    assert diagnostics["excluded_tokens_with_Z1"] == 1
    assert len(sequences) == 2
    assert [len(sequence) for sequence in sequences] == [2, 2]
    assert sequences[0][0].glyphs == ("A",)
    assert sequences[1][0].glyphs == ("C",)


def test_matched_trigram_adds_exactly_one_eot_per_token():
    locus = make_locus(
        [
            ("A", "B"),
            ("C",),
        ]
    )
    sequences, _ = extract_token_sequences([locus])

    examples = build_matched_trigram_examples(sequences)

    assert len(examples) == 2
    assert examples[0].symbols == ("A", "B", EOT)
    assert examples[0].is_boundary == (False, False, True)
    assert examples[1].symbols == ("C", EOT)
    assert examples[1].is_boundary == (False, True)


def test_fit_context_summary_has_expected_transition_counts():
    train = [
        [
            ("A",),
            ("B",),
            ("C",),
        ],
        [
            ("A",),
            ("B",),
        ],
    ]

    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(train)

    rows = {
        row["history_length"]: row
        for row in model.context_summary()
    }

    assert rows[0]["transitions"] == 5
    assert rows[1]["transitions"] == 5
    assert rows[2]["transitions"] == 5
    assert model.training_tokens == 5
    assert model.training_glyphs == 5


def test_repeated_transition_gets_more_probability_than_unseen_transition():
    train = [
        [
            ("A",),
            ("B",),
            ("A",),
            ("B",),
            ("A",),
            ("B",),
        ],
        [
            ("C",),
            ("D",),
            ("C",),
            ("D",),
        ],
    ]

    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(train)

    seen = model.token_probability(
        ("B",),
        [("A",)],
        context_order=1,
    )
    unseen = model.token_probability(
        ("D",),
        [("A",)],
        context_order=1,
    )

    assert seen > unseen
    assert seen > 0.0
    assert unseen > 0.0


def test_second_order_context_can_distinguish_same_immediate_predecessor():
    train = []

    for _ in range(20):
        train.append(
            [
                ("A",),
                ("X",),
                ("B",),
            ]
        )
        train.append(
            [
                ("C",),
                ("X",),
                ("D",),
            ]
        )

    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(train)

    p_b_given_ax = model.token_probability(
        ("B",),
        [("A",), ("X",)],
        context_order=2,
    )
    p_d_given_ax = model.token_probability(
        ("D",),
        [("A",), ("X",)],
        context_order=2,
    )

    assert p_b_given_ax > p_d_given_ax


def test_unseen_token_and_unseen_glyph_have_finite_probability():
    train = [
        [
            ("A", "B"),
            ("B", "A"),
            ("A", "B"),
        ]
    ]

    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(train)

    probability = model.token_probability(
        ("A", "NEVER_SEEN"),
        [("A", "B")],
        context_order=1,
    )

    assert math.isfinite(probability)
    assert probability > 0.0


def test_bos_padding_is_used_for_initial_token():
    train = [
        [
            ("A",),
            ("B",),
        ],
        [
            ("A",),
            ("C",),
        ],
        [
            ("A",),
            ("D",),
        ],
    ]

    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(train)

    initial_a = model.token_probability(
        ("A",),
        [],
        context_order=2,
    )
    initial_b = model.token_probability(
        ("B",),
        [],
        context_order=2,
    )

    assert initial_a > initial_b


def test_invalid_context_order_is_rejected():
    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(
        [
            [
                ("A",),
                ("B",),
            ]
        ]
    )

    with pytest.raises(ValueError):
        model.token_probability(
            ("A",),
            [],
            context_order=0,
        )

    with pytest.raises(ValueError):
        model.token_probability(
            ("A",),
            [],
            context_order=3,
        )


def test_evaluation_preserves_event_denominator_and_leaf_pairing():
    loci = [
        make_locus(
            [
                ("A", "B"),
                ("A", "B"),
                ("A", "C"),
            ],
            leaf=1,
        ),
        make_locus(
            [
                ("A",),
                ("B",),
            ],
            leaf=2,
            locus="f2r.1,+P0",
        ),
    ]

    sequences, _ = extract_token_sequences(loci)

    train = [
        [
            ("A", "B"),
            ("A", "B"),
            ("A", "C"),
            ("A", "B"),
        ],
        [
            ("A",),
            ("B",),
            ("A",),
        ],
    ]

    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(train)

    aggregate, leaves, locus_rows = evaluate_token_markov(
        model,
        sequences,
        context_order=1,
    )

    # 5 tokens, 8 glyphs, one EOT-equivalent event per token.
    assert aggregate["tokens"] == 5
    assert aggregate["glyphs"] == 8
    assert aggregate["events"] == 13
    assert {row["leaf_group"] for row in leaves} == {"f1", "f2"}
    assert len(locus_rows) == 2


def test_character_base_probability_is_positive_for_nonempty_token():
    model = TokenMarkovModel(
        max_context=2,
        char_base_order=3,
    ).fit(
        [
            [
                ("A",),
                ("B",),
                ("A",),
            ]
        ]
    )

    assert model.character_token_probability(("A",)) > 0.0
    assert model.character_token_probability(("Z",)) > 0.0
