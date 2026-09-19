"""Tests for Phase 3A.D1 token-Markov failure localization.

Intended repository destination:
    tests/test_token_markov_failure_localization.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.token_markov_failure_localization import (
    CIPHER_BOS,
    PLAIN_BOS,
    assign_upper_bound_bin,
    build_training_statistics,
    compact_transition_rows,
    decode_variant_index_from_key,
    deterministic_frequency_ranks,
    line_position_class,
    read_selected_plaintext_units,
    score_validation_contributions,
    summarize_language,
    summarize_rows,
    summarize_top_frequency_sets,
    verify_aggregate_parity,
)


BINS = {
    "token_frequency": [
        {"label": "0", "upper": 0},
        {"label": "1", "upper": 1},
        {"label": "2-3", "upper": 3},
        {"label": "4+", "upper": None},
    ],
    "bigram_frequency": [
        {"label": "0", "upper": 0},
        {"label": "1", "upper": 1},
        {"label": "2+", "upper": None},
    ],
    "probability": [
        {"label": "0", "upper": 0.0},
        {"label": "(0,.5]", "upper": 0.5},
        {"label": "(.5,1]", "upper": 1.0},
    ],
    "token_length": [
        {"label": "1-4", "upper": 4},
        {"label": "5+", "upper": None},
    ],
}


def key_1x16():
    return {
        "active_slot_a": ["A0001"],
        "active_slot_b": [
            f"B{i:04d}"
            for i in range(1, 17)
        ],
    }


def key_2x8():
    return {
        "active_slot_a": [
            "A0001",
            "A0002",
        ],
        "active_slot_b": [
            f"B{i:04d}"
            for i in range(1, 9)
        ],
    }


def key_4x4():
    return {
        "active_slot_a": [
            f"A{i:04d}"
            for i in range(1, 5)
        ],
        "active_slot_b": [
            f"B{i:04d}"
            for i in range(1, 5)
        ],
    }


def make_token(base, variant):
    return (
        base,
        "A0001",
        f"B{variant + 1:04d}",
    )


def small_units():
    plain_train = [
        (1, ("alpha", "beta", "alpha")),
        (2, ("beta", "alpha", "gamma")),
    ]
    cipher_train = [
        (
            1,
            (
                make_token("M_ALPHA", 0),
                make_token("M_BETA", 1),
                make_token("M_ALPHA", 2),
            ),
        ),
        (
            2,
            (
                make_token("M_BETA", 3),
                make_token("M_ALPHA", 4),
                make_token("M_GAMMA", 5),
            ),
        ),
    ]

    plain_validation = [
        (3, ("alpha", "beta", "delta")),
        (4, ("beta", "alpha")),
    ]
    cipher_validation = [
        (
            3,
            (
                make_token("M_ALPHA", 0),
                make_token("M_BETA", 6),
                make_token("M_DELTA", 7),
            ),
        ),
        (
            4,
            (
                make_token("M_BETA", 3),
                make_token("M_ALPHA", 8),
            ),
        ),
    ]

    return (
        plain_train,
        cipher_train,
        plain_validation,
        cipher_validation,
    )


def test_assign_upper_bound_bin():
    specs = BINS["token_frequency"]

    assert assign_upper_bound_bin(0, specs) == "0"
    assert assign_upper_bound_bin(1, specs) == "1"
    assert assign_upper_bound_bin(2, specs) == "2-3"
    assert assign_upper_bound_bin(3, specs) == "2-3"
    assert assign_upper_bound_bin(99, specs) == "4+"

    with pytest.raises(ValueError):
        assign_upper_bound_bin(-1, specs)


def test_line_position_class():
    assert line_position_class(0, 1) == "singleton"
    assert line_position_class(0, 3) == "initial"
    assert line_position_class(1, 3) == "medial"
    assert line_position_class(2, 3) == "final"

    with pytest.raises(ValueError):
        line_position_class(3, 3)


def test_variant_decoding_all_frozen_factorizations():
    # Index 9.
    assert decode_variant_index_from_key(
        ("M", "A0001", "B0010"),
        key=key_1x16(),
        factorization="1x16",
    ) == 9

    # 2x8: row=1, col=1 -> 9.
    assert decode_variant_index_from_key(
        ("M", "A0002", "B0002"),
        key=key_2x8(),
        factorization="2x8",
    ) == 9

    # 4x4: row=2, col=1 -> 9.
    assert decode_variant_index_from_key(
        ("M", "A0003", "B0002"),
        key=key_4x4(),
        factorization="4x4",
    ) == 9


def test_read_selected_plaintext_units_only_returns_selected_rows(tmp_path: Path):
    path = tmp_path / "source.txt"
    path.write_text(
        "alpha beta\n"
        "reserved material is not selected\n"
        "gamma delta\n",
        encoding="utf-8",
    )

    rows = read_selected_plaintext_units(
        path,
        selected_indices=[1, 3],
        expected_line_count=3,
    )

    assert rows == [
        (1, ("alpha", "beta")),
        (3, ("gamma", "delta")),
    ]


def test_deterministic_frequency_ranks():
    from collections import Counter

    ranks = deterministic_frequency_ranks(
        Counter(
            {
                "b": 3,
                "a": 3,
                "c": 1,
            }
        )
    )

    assert ranks["a"] == 1
    assert ranks["b"] == 2
    assert ranks["c"] == 3


def test_training_statistics_use_train_only():
    (
        plain_train,
        cipher_train,
        _,
        _,
    ) = small_units()

    stats = build_training_statistics(
        plain_train,
        cipher_train,
        key=key_1x16(),
        factorization="1x16",
    )

    assert stats.plain_token_counts["alpha"] == 3
    assert stats.plain_token_counts["beta"] == 2
    assert stats.plain_token_counts["gamma"] == 1
    assert stats.plain_token_counts["delta"] == 0

    assert stats.plain_bigram_counts[
        (PLAIN_BOS, "alpha")
    ] == 1
    assert stats.plain_bigram_counts[
        (PLAIN_BOS, "beta")
    ] == 1
    assert stats.cipher_bigram_counts[
        (CIPHER_BOS, make_token("M_ALPHA", 0))
    ] == 1


def test_score_validation_contributions_reconstructs_one_run():
    (
        plain_train,
        cipher_train,
        plain_validation,
        cipher_validation,
    ) = small_units()

    rows, aggregate = score_validation_contributions(
        source_id="synthetic",
        language="Synthetic",
        factorization="1x16",
        seed=7,
        plain_train_units=plain_train,
        plain_validation_units=plain_validation,
        cipher_train_units=cipher_train,
        cipher_validation_units=cipher_validation,
        key=key_1x16(),
        selected_context_order=1,
        max_context=2,
        char_base_order=3,
        bins=BINS,
        top_n_frequency_sets=[1, 2],
    )

    assert len(rows) == 5
    assert aggregate["validation_tokens"] == 5
    assert aggregate["validation_events"] == sum(
        len(token) + 1
        for _, line in cipher_validation
        for token in line
    )

    # "delta" never occurred in TRAIN.
    delta_row = next(
        row
        for row in rows
        if row["plaintext_token"] == "delta"
    )
    assert delta_row[
        "train_plain_current_frequency"
    ] == 0
    assert delta_row[
        "train_plain_current_frequency_bin"
    ] == "0"

    assert rows[0]["line_position_class"] == "initial"
    assert rows[2]["line_position_class"] == "final"


def test_verify_aggregate_parity_exact_pass():
    aggregate = {
        "source_id": "x",
        "language": "X",
        "factorization": "1x16",
        "seed": 1,
        "selected_context_order": 1,
        "validation_tokens": 3,
        "validation_events": 10,
        "recomputed_competitor_bits_per_event": 2.0,
        "recomputed_trigram_bits_per_event": 1.75,
        "recomputed_delta_bits_per_event": 0.25,
        "total_token_markov_bits": 20.0,
        "total_matched_trigram_bits": 17.5,
        "total_delta_bits": 2.5,
    }
    stored = {
        "competitor_bits_per_event": 2.0,
        "trigram_bits_per_event": 1.75,
        "delta_competitor_minus_trigram_bits_per_event": 0.25,
    }

    result = verify_aggregate_parity(
        aggregate,
        stored,
        tolerance=1e-12,
    )

    assert result["parity_pass"] is True
    assert result["delta_abs_error"] == 0.0


def test_verify_aggregate_parity_fails_mismatch():
    aggregate = {
        "source_id": "x",
        "language": "X",
        "factorization": "1x16",
        "seed": 1,
        "selected_context_order": 1,
        "validation_tokens": 3,
        "validation_events": 10,
        "recomputed_competitor_bits_per_event": 2.0,
        "recomputed_trigram_bits_per_event": 1.75,
        "recomputed_delta_bits_per_event": 0.25,
        "total_token_markov_bits": 20.0,
        "total_matched_trigram_bits": 17.5,
        "total_delta_bits": 2.5,
    }
    stored = {
        "competitor_bits_per_event": 2.0,
        "trigram_bits_per_event": 1.75,
        "delta_competitor_minus_trigram_bits_per_event": 0.20,
    }

    with pytest.raises(ValueError):
        verify_aggregate_parity(
            aggregate,
            stored,
            tolerance=1e-12,
        )


def _summary_rows_fixture():
    return [
        {
            "language": "X",
            "factorization": "1x16",
            "seed": 1,
            "bin": "low",
            "events": 2,
            "delta_bits": -2.0,
            "token_markov_better": True,
            "token_markov_advantage_bits": 2.0,
            "trigram_advantage_bits": 0.0,
        },
        {
            "language": "X",
            "factorization": "1x16",
            "seed": 1,
            "bin": "high",
            "events": 8,
            "delta_bits": 4.0,
            "token_markov_better": False,
            "token_markov_advantage_bits": 0.0,
            "trigram_advantage_bits": 4.0,
        },
        {
            "language": "X",
            "factorization": "1x16",
            "seed": 2,
            "bin": "low",
            "events": 2,
            "delta_bits": -1.0,
            "token_markov_better": True,
            "token_markov_advantage_bits": 1.0,
            "trigram_advantage_bits": 0.0,
        },
    ]


def test_summarize_rows_uses_ratio_of_sums_and_negative_mass_share():
    summary = summarize_rows(
        _summary_rows_fixture(),
        group_fields=(
            "language",
            "factorization",
            "bin",
        ),
    )

    low = next(
        row
        for row in summary
        if row["bin"] == "low"
    )

    assert low["tokens"] == 2
    assert low["events"] == 4
    assert low[
        "delta_bits_per_event_ratio_of_sums"
    ] == pytest.approx(-0.75)
    assert low[
        "share_of_language_token_markov_advantage_bits"
    ] == pytest.approx(1.0)


def test_summarize_top_frequency_sets():
    rows = []
    for seed in (1, 2):
        rows.append(
            {
                "language": "X",
                "factorization": "1x16",
                "seed": seed,
                "events": 2,
                "delta_bits": -1.0,
                "token_markov_better": True,
                "token_markov_advantage_bits": 1.0,
                "trigram_advantage_bits": 0.0,
                "plain_current_top_8": True,
                "plain_current_top_16": True,
            }
        )

    summary = summarize_top_frequency_sets(
        rows,
        top_n_frequency_sets=[8, 16],
    )

    assert {
        row["top_n"]
        for row in summary
    } == {8, 16}
    assert all(
        row["top_n_membership"]
        == "in_top_n"
        for row in summary
    )


def test_compact_transition_rows_excludes_line_initial_tokens():
    rows = [
        {
            "source_id": "x",
            "language": "X",
            "factorization": "1x16",
            "seed": 1,
            "selected_context_order": 1,
            "line_index": 1,
            "token_index": index,
            "previous_plaintext_token": "p",
            "plaintext_token": "c",
            "previous_variant_index": -1,
            "current_variant_index": 1,
            "delta_bits": 0.1,
            "events": 4,
            "delta_bits_per_event": 0.025,
            "token_markov_better": False,
            "train_plain_previous_frequency": 1,
            "train_plain_current_frequency": 2,
            "train_plain_bigram_frequency": 1,
            "train_plain_conditional_probability": 0.5,
            "train_plain_previous_distinct_next_types": 2,
            "train_plain_previous_transition_concentration": 0.5,
            "train_cipher_previous_frequency": 1,
            "train_cipher_current_frequency": 1,
            "train_cipher_bigram_frequency": 0,
            "train_cipher_conditional_probability": 0.0,
            "train_cipher_previous_distinct_next_types": 1,
            "train_cipher_previous_transition_concentration": 1.0,
            "train_variant_bigram_frequency": 0,
            "train_variant_conditional_probability": 0.0,
        }
        for index in (0, 1)
    ]

    compact = compact_transition_rows(
        rows
    )

    assert len(compact) == 1
    assert compact[0]["token_index"] == 1


def test_summarize_language_marks_status():
    rows = []
    for seed in (1, 2):
        rows.append(
            {
                "language": "X",
                "factorization": "1x16",
                "seed": seed,
                "events": 4,
                "delta_bits": -1.0,
                "token_markov_better": True,
                "token_markov_advantage_bits": 1.0,
                "trigram_advantage_bits": 0.0,
                "line_position_class": "medial",
            }
        )

    summary = summarize_language(
        rows,
        relation_status_by_language={
            "X": "nonrobust"
        },
    )

    assert len(summary) == 1
    assert summary[0][
        "phase3a5_token_relation_status"
    ] == "nonrobust"
    assert summary[0][
        "mean_seed_delta_bits_per_event"
    ] == pytest.approx(-0.25)
