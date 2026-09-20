"""Tests for Phase 3A.6 recurrence-avoiding surface allocation.

Intended repository destination:
    tests/test_recurrence_avoiding_surface_allocation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ciphers import base_form_suffix_extended as phase3a4
from src.ciphers.recurrence_avoiding_surface_allocation import (
    VARIANT_COUNT,
    build_master_key,
    decode_token,
    decode_variant_index,
    encode_base,
    encode_token_for_occurrence,
    generate_control,
    strip_suffix,
    variant_index_for_occurrence,
    variant_permutation,
    write_generated_control,
)


def small_lines():
    return [
        ("alpha", "alpha", "beta", "alpha"),
        ("alpha", "gamma", "alpha", "beta"),
        ("alpha", "beta", "alpha", "delta"),
        ("alpha", "alpha", "beta", "epsilon"),
        ("alpha", "beta", "alpha", "zeta"),
        ("alpha", "beta", "alpha", "eta"),
        ("alpha", "beta", "theta"),
        ("alpha", "beta", "iota"),
        ("alpha", "beta", "validation"),
        ("alpha", "validation", "beta"),
        ("reserved", "alpha"),
        ("reserved", "beta"),
    ]


def split():
    return {
        "train": list(range(1, 9)),
        "validation": [9, 10],
        "reserved": [11, 12],
    }


def test_variant_count_is_frozen_to_sixteen():
    assert VARIANT_COUNT == 16


def test_master_key_inherits_exact_phase3a4_K16_base_and_suffix_inventory():
    train = small_lines()[:8]
    seed = 40814041438

    parent = phase3a4.build_master_key(
        train,
        seed=seed,
        master_suffix_count=16,
    )
    current = build_master_key(
        train,
        seed=seed,
    )

    assert current["mono_forward"] == parent["mono_forward"]
    assert current["mono_reverse"] == parent["mono_reverse"]
    assert (
        current["train_character_alphabet"]
        == parent["train_character_alphabet"]
    )
    assert current["master_suffixes"] == parent["master_suffixes"]
    assert (
        current["parent_phase3a4_master_key_sha256"]
        == parent["master_key_sha256"]
    )


def test_variant_permutation_is_deterministic_complete_and_base_specific():
    key = build_master_key(
        small_lines()[:8],
        seed=17,
    )
    alpha = encode_base(
        "alpha",
        key=key,
    )
    beta = encode_base(
        "beta",
        key=key,
    )

    a1 = variant_permutation(
        alpha,
        seed=17,
    )
    a2 = variant_permutation(
        alpha,
        seed=17,
    )
    b = variant_permutation(
        beta,
        seed=17,
    )

    assert a1 == a2
    assert sorted(a1) == list(range(16))
    assert len(set(a1)) == 16

    # Different encoded bases should almost certainly have different permutations;
    # this is deterministic for the frozen test fixture.
    assert a1 != b


def test_seed_changes_variant_permutation():
    key = build_master_key(
        small_lines()[:8],
        seed=17,
    )
    base = encode_base(
        "alpha",
        key=key,
    )

    assert (
        variant_permutation(
            base,
            seed=17,
        )
        != variant_permutation(
            base,
            seed=18,
        )
    )


def test_first_sixteen_occurrences_use_each_variant_exactly_once():
    key = build_master_key(
        small_lines()[:8],
        seed=23,
    )
    base = encode_base(
        "alpha",
        key=key,
    )

    first = [
        variant_index_for_occurrence(
            base,
            seed=23,
            occurrence_number=n,
        )
        for n in range(16)
    ]

    assert sorted(first) == list(range(16))
    assert len(set(first)) == 16


def test_cycle_repeats_only_after_all_sixteen_variants():
    key = build_master_key(
        small_lines()[:8],
        seed=29,
    )
    base = encode_base(
        "alpha",
        key=key,
    )

    first = [
        variant_index_for_occurrence(
            base,
            seed=29,
            occurrence_number=n,
        )
        for n in range(16)
    ]
    second = [
        variant_index_for_occurrence(
            base,
            seed=29,
            occurrence_number=n,
        )
        for n in range(16, 32)
    ]

    assert first == second


def test_encoding_preserves_base_adds_one_suffix_and_roundtrips():
    key = build_master_key(
        small_lines()[:8],
        seed=31,
    )
    base = encode_base(
        "alpha",
        key=key,
    )

    for occurrence_number in (
        0,
        1,
        15,
        16,
        31,
    ):
        cipher = encode_token_for_occurrence(
            "alpha",
            key=key,
            seed=31,
            occurrence_number=occurrence_number,
        )

        assert len(cipher) == len(base) + 1
        assert strip_suffix(
            cipher,
            key=key,
        ) == base
        assert decode_token(
            cipher,
            key=key,
        ) == "alpha"
        assert (
            decode_variant_index(
                cipher,
                key=key,
            )
            == variant_index_for_occurrence(
                base,
                seed=31,
                occurrence_number=occurrence_number,
            )
        )


def test_generate_control_is_deterministic_and_reserved_rows_are_opaque():
    parts = split()

    kwargs = dict(
        source_lines=small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=40814041438,
    )

    first = generate_control(
        **kwargs
    )
    second = generate_control(
        **kwargs
    )

    assert first.rows == second.rows
    assert first.key == second.key
    assert first.summary == second.summary

    for line_index in parts["reserved"]:
        assert first.rows[
            line_index - 1
        ] == {
            "line_index": line_index,
            "reserved_test_opaque": True,
        }


def test_validation_continues_frozen_train_occurrence_state():
    source = [
        ("alpha", "alpha", "alpha"),
        ("alpha", "alpha", "alpha"),
        ("alpha", "alpha"),
        ("alpha", "alpha"),
        ("reserved",),
    ]

    generated = generate_control(
        source,
        train_indices=[1, 2],
        validation_indices=[3, 4],
        reserved_indices=[5],
        seed=37,
    )

    # Six TRAIN occurrences followed by four VALIDATION occurrences should use
    # ten distinct variants of alpha: validation cannot restart at occurrence 0.
    train_tokens = [
        tuple(token)
        for row in generated.rows[:2]
        for token in row["tokens"]
    ]
    validation_tokens = [
        tuple(token)
        for row in generated.rows[2:4]
        for token in row["tokens"]
    ]

    train_variants = {
        decode_variant_index(
            token,
            key=generated.key,
        )
        for token in train_tokens
    }
    validation_variants = {
        decode_variant_index(
            token,
            key=generated.key,
        )
        for token in validation_tokens
    }

    assert len(train_variants) == 6
    assert len(validation_variants) == 4
    assert train_variants.isdisjoint(
        validation_variants
    )

    assert generated.summary[
        "validation_continues_frozen_train_state"
    ] is True
    assert generated.summary[
        "continuation_diagnostics"
    ][
        "validation_avoidable_collisions"
    ] == 0


def test_avoid_reuse_until_sixteen_total_occurrences_then_reuse_is_allowed():
    source = [
        tuple(["alpha"] * 14),
        tuple(["alpha"] * 4),
        ("reserved",),
    ]

    generated = generate_control(
        source,
        train_indices=[1],
        validation_indices=[2],
        reserved_indices=[3],
        seed=41,
    )

    train_tokens = {
        tuple(token)
        for token in generated.rows[0]["tokens"]
    }
    validation_tokens = [
        tuple(token)
        for token in generated.rows[1]["tokens"]
    ]

    # With 14 TRAIN occurrences, first two VALIDATION forms complete the
    # unused suffix inventory; only the next two are forced to reuse TRAIN forms.
    assert validation_tokens[0] not in train_tokens
    assert validation_tokens[1] not in train_tokens
    assert validation_tokens[2] in train_tokens
    assert validation_tokens[3] in train_tokens

    diag = generated.summary[
        "continuation_diagnostics"
    ]
    assert diag[
        "validation_avoidable_collision_opportunities"
    ] == 2
    assert diag[
        "validation_avoidable_collisions"
    ] == 0
    assert diag[
        "validation_unavoidable_train_surface_reuses"
    ] == 2


def test_train_uses_min_count_16_unique_variants_for_each_base():
    source = [
        tuple(["alpha"] * 20),
        ("beta", "beta", "beta"),
        ("alpha", "beta"),
        ("reserved",),
    ]

    generated = generate_control(
        source,
        train_indices=[1, 2],
        validation_indices=[3],
        reserved_indices=[4],
        seed=43,
    )

    alpha_tokens = [
        tuple(token)
        for token in generated.rows[0]["tokens"]
    ]
    beta_tokens = [
        tuple(token)
        for token in generated.rows[1]["tokens"]
    ]

    assert len(
        {
            decode_variant_index(
                token,
                key=generated.key,
            )
            for token in alpha_tokens
        }
    ) == 16

    assert len(
        {
            decode_variant_index(
                token,
                key=generated.key,
            )
            for token in beta_tokens
        }
    ) == 3


def test_validation_unseen_character_is_reversible_without_refitting():
    source = [
        ("aaaa", "bbbb"),
        ("aaaa", "bbbb"),
        ("zzzz",),
        ("reserved",),
    ]

    generated = generate_control(
        source,
        train_indices=[1, 2],
        validation_indices=[3],
        reserved_indices=[4],
        seed=47,
    )

    token = generated.rows[2][
        "tokens"
    ][0]

    assert all(
        glyph.startswith("U")
        for glyph in token[:-1]
    )
    assert decode_token(
        token,
        key=generated.key,
    ) == "zzzz"


def test_required_diagnostics_are_exact():
    parts = split()
    generated = generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=53,
    )

    assert generated.summary[
        "round_trip_verified_train_validation"
    ] is True
    assert generated.summary[
        "base_form_preservation_verified_train_validation"
    ] is True
    assert generated.summary[
        "token_length_control_verified_train_validation"
    ] is True
    assert generated.summary[
        "balanced_without_replacement_schedule_verified"
    ] is True
    assert generated.summary[
        "reserved_test_encoded"
    ] is False

    for split_name in (
        "train_diagnostics",
        "validation_diagnostics",
    ):
        diag = generated.summary[
            split_name
        ]
        assert diag[
            "round_trip_accuracy"
        ] == 1.0
        assert diag[
            "base_form_preservation_fraction"
        ] == 1.0
        assert diag[
            "token_length_plus_one_accuracy"
        ] == 1.0
        assert diag[
            "balanced_schedule_accuracy"
        ] == 1.0


def test_write_generated_control_preserves_opaque_reserved_rows(tmp_path: Path):
    parts = split()
    generated = generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=59,
    )

    output = tmp_path / "control"
    artifacts = write_generated_control(
        output,
        generated,
        provenance={
            "test": True,
        },
    )

    assert set(artifacts) == {
        "corpus.jsonl",
        "key.json",
        "summary.json",
    }

    rows = [
        json.loads(line)
        for line in (
            output
            / "corpus.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert rows[10] == {
        "line_index": 11,
        "reserved_test_opaque": True,
    }
    assert rows[11] == {
        "line_index": 12,
        "reserved_test_opaque": True,
    }


def test_negative_occurrence_number_is_rejected():
    key = build_master_key(
        small_lines()[:8],
        seed=61,
    )
    base = encode_base(
        "alpha",
        key=key,
    )

    with pytest.raises(ValueError):
        variant_index_for_occurrence(
            base,
            seed=61,
            occurrence_number=-1,
        )
