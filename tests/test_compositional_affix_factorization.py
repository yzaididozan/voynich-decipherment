"""Tests for Phase 3A.5 compositional affix factorization.

Intended repository destination:
    tests/test_compositional_affix_factorization.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ciphers import base_form_suffix_extended as phase3a4
from src.ciphers.compositional_affix_factorization import (
    FACTORIZATIONS,
    active_factor_states,
    build_master_key,
    decode_token,
    decode_variant_index,
    encode_base,
    encode_token,
    factorization_spec,
    generate_control,
    strip_affix,
    variant_index_choice,
    variant_pair,
    write_generated_control,
)
from src.evaluation.level2_predictive_controls import (
    read_control_units,
)


def small_lines():
    return [
        ("alpha", "beta", "alpha", "gamma"),
        ("alpha", "beta", "delta"),
        ("alpha", "beta", "epsilon"),
        ("alpha", "beta", "zeta"),
        ("alpha", "beta", "eta"),
        ("alpha", "beta", "theta"),
        ("alpha", "beta", "iota"),
        ("alpha", "beta", "kappa"),
        ("alpha", "beta", "validation"),
        ("beta", "alpha", "validation"),
        ("reserved", "alpha"),
        ("reserved", "beta"),
    ]


def split():
    return {
        "train": list(range(1, 9)),
        "validation": [9, 10],
        "reserved": [11, 12],
    }


def test_factorization_grid_is_exact_and_each_has_sixteen_pairs():
    assert FACTORIZATIONS == {
        "1x16": (1, 16),
        "2x8": (2, 8),
        "4x4": (4, 4),
    }

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        rows, cols = factorization_spec(
            label
        )
        assert rows * cols == 16


def test_master_key_inherits_exact_phase3a4_base_mapping():
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

    assert (
        current["mono_forward"]
        == parent["mono_forward"]
    )
    assert (
        current["mono_reverse"]
        == parent["mono_reverse"]
    )
    assert (
        current["train_character_alphabet"]
        == parent["train_character_alphabet"]
    )
    assert (
        current[
            "parent_phase3a4_master_key_sha256"
        ]
        == parent["master_key_sha256"]
    )


def test_master_factor_orders_are_deterministic_and_seed_sensitive():
    train = small_lines()[:8]

    a = build_master_key(
        train,
        seed=7,
    )
    b = build_master_key(
        train,
        seed=7,
    )
    c = build_master_key(
        train,
        seed=8,
    )

    assert a == b
    assert (
        a["master_key_sha256"]
        == b["master_key_sha256"]
    )
    assert (
        a["master_key_sha256"]
        != c["master_key_sha256"]
    )
    assert len(
        set(a["slot_a_order"])
    ) == 4
    assert len(
        set(a["slot_b_order"])
    ) == 16


def test_each_factorization_maps_all_16_indices_bijectively():
    key = build_master_key(
        small_lines()[:8],
        seed=11,
    )

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        pairs = [
            variant_pair(
                key,
                factorization=label,
                variant_index=index,
            )
            for index in range(16)
        ]

        assert len(
            set(pairs)
        ) == 16

        for index, pair in enumerate(
            pairs
        ):
            assert decode_variant_index(
                pair,
                key=key,
                factorization=label,
            ) == index


def test_active_factor_states_use_shared_master_prefixes():
    key = build_master_key(
        small_lines()[:8],
        seed=19,
    )

    a1, b16 = active_factor_states(
        key,
        factorization="1x16",
    )
    a2, b8 = active_factor_states(
        key,
        factorization="2x8",
    )
    a4, b4 = active_factor_states(
        key,
        factorization="4x4",
    )

    assert a1 == a2[:1]
    assert a2 == a4[:2]
    assert b4 == b8[:4]
    assert b8 == b16[:8]


def test_variant_schedule_exactly_matches_phase3a4_k16():
    for seed in (
        7,
        40814041438,
    ):
        for line_index in (
            1,
            3,
            99,
        ):
            for token_index in (
                0,
                2,
                11,
            ):
                assert variant_index_choice(
                    occurrence_seed=seed,
                    line_index=line_index,
                    token_index=token_index,
                ) == phase3a4.suffix_choice_index(
                    occurrence_seed=seed,
                    line_index=line_index,
                    token_index=token_index,
                    variant_count=16,
                )


def test_every_factorization_preserves_base_and_adds_exactly_two_glyphs():
    key = build_master_key(
        small_lines()[:8],
        seed=23,
    )
    plaintext = "alpha"
    base = encode_base(
        plaintext,
        key=key,
    )

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        token = encode_token(
            plaintext,
            key=key,
            factorization=label,
            occurrence_seed=23,
            line_index=3,
            token_index=2,
        )

        assert len(token) == len(
            base
        ) + 2
        assert strip_affix(
            token,
            key=key,
            factorization=label,
        ) == base
        assert decode_token(
            token,
            key=key,
            factorization=label,
        ) == plaintext


def test_plaintext_identity_is_excluded_from_variant_choice():
    key = build_master_key(
        small_lines()[:8],
        seed=29,
    )

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        left = encode_token(
            "alpha",
            key=key,
            factorization=label,
            occurrence_seed=29,
            line_index=5,
            token_index=3,
        )
        right = encode_token(
            "beta",
            key=key,
            factorization=label,
            occurrence_seed=29,
            line_index=5,
            token_index=3,
        )

        assert left[:-2] != right[:-2]
        assert left[-2:] == right[-2:]


def test_same_occurrence_has_same_abstract_variant_index_across_factorizations():
    key = build_master_key(
        small_lines()[:8],
        seed=31,
    )

    observed = []

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        token = encode_token(
            "alpha",
            key=key,
            factorization=label,
            occurrence_seed=31,
            line_index=7,
            token_index=4,
        )
        observed.append(
            decode_variant_index(
                token[-2:],
                key=key,
                factorization=label,
            )
        )

    assert observed[0] == observed[1] == observed[2]


def test_all_16_surface_variants_are_realizable_for_repeated_word():
    train = [
        tuple(
            ["alpha"] * 1024
        ),
        tuple(
            ["beta"] * 20
        ),
    ]
    key = build_master_key(
        train,
        seed=40814041438,
    )

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        surfaces = {
            encode_token(
                "alpha",
                key=key,
                factorization=label,
                occurrence_seed=40814041438,
                line_index=1,
                token_index=i,
            )
            for i in range(1024)
        }

        assert len(surfaces) == 16
        assert len(
            {
                token[:-2]
                for token in surfaces
            }
        ) == 1


def test_abstract_identity_partition_is_identical_across_factorizations():
    parts = split()
    generated = {}

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        generated[
            label
        ] = generate_control(
            small_lines(),
            train_indices=parts[
                "train"
            ],
            validation_indices=parts[
                "validation"
            ],
            reserved_indices=parts[
                "reserved"
            ],
            seed=40814041438,
            factorization=label,
        )

    for split_name in (
        "train_diagnostics",
        "validation_diagnostics",
    ):
        fingerprints = {
            generated[label].summary[
                split_name
            ][
                "abstract_token_identity_sha256"
            ]
            for label in generated
        }
        recurrence = {
            generated[label].summary[
                split_name
            ][
                "exact_cipher_token_recurrence_fraction"
            ]
            for label in generated
        }
        unique_tokens = {
            generated[label].summary[
                split_name
            ][
                "unique_cipher_tokens"
            ]
            for label in generated
        }

        assert len(
            fingerprints
        ) == 1
        assert len(
            recurrence
        ) == 1
        assert len(
            unique_tokens
        ) == 1


def test_same_surface_probability_is_identical_across_factorizations():
    source = [
        tuple(
            ["alpha", "beta"] * 300
        ),
        tuple(
            ["alpha", "beta"] * 300
        ),
        tuple(
            ["alpha", "beta"] * 150
        ),
        ("reserved",),
    ]

    probabilities = []

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        generated = generate_control(
            source,
            train_indices=[1, 2],
            validation_indices=[3],
            reserved_indices=[4],
            seed=40814041438,
            factorization=label,
        )
        probabilities.append(
            generated.summary[
                "validation_diagnostics"
            ][
                "same_surface_probability_given_same_plaintext_word"
            ]
        )

    assert (
        probabilities[0]
        == probabilities[1]
        == probabilities[2]
    )


def test_generator_is_deterministic_and_reserved_rows_are_opaque():
    parts = split()
    kwargs = dict(
        source_lines=small_lines(),
        train_indices=parts[
            "train"
        ],
        validation_indices=parts[
            "validation"
        ],
        reserved_indices=parts[
            "reserved"
        ],
        seed=40814041438,
        factorization="4x4",
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

    assert first.summary[
        "reserved_test_encoded"
    ] is False
    assert first.summary[
        "round_trip_verified_train_validation"
    ] is True
    assert first.summary[
        "base_form_preservation_verified_train_validation"
    ] is True
    assert first.summary[
        "token_length_control_verified_train_validation"
    ] is True
    assert first.summary[
        "phase3a4_k16_variant_schedule_verified_train_validation"
    ] is True

    for line_index in parts[
        "reserved"
    ]:
        row = first.rows[
            line_index - 1
        ]
        assert row == {
            "line_index": line_index,
            "reserved_test_opaque": True,
        }
        assert "tokens" not in row


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
        seed=7,
        factorization="4x4",
    )

    token = generated.rows[
        2
    ][
        "tokens"
    ][0]

    assert all(
        glyph.startswith(
            "U"
        )
        for glyph in token[
            :-2
        ]
    )
    assert decode_token(
        token,
        key=generated.key,
        factorization="4x4",
    ) == "zzzz"


def test_diagnostics_confirm_fixed_coverage_length_and_schedule():
    parts = split()

    for label in (
        "1x16",
        "2x8",
        "4x4",
    ):
        generated = generate_control(
            small_lines(),
            train_indices=parts[
                "train"
            ],
            validation_indices=parts[
                "validation"
            ],
            reserved_indices=parts[
                "reserved"
            ],
            seed=37,
            factorization=label,
        )

        for split_name in (
            "train_diagnostics",
            "validation_diagnostics",
        ):
            diag = generated.summary[
                split_name
            ]
            assert diag[
                "token_coverage_fraction"
            ] == 1.0
            assert diag[
                "base_form_recovery_accuracy"
            ] == 1.0
            assert diag[
                "base_form_preservation_fraction"
            ] == 1.0
            assert diag[
                "token_length_plus_two_accuracy"
            ] == 1.0
            assert diag[
                "phase3a4_k16_variant_schedule_accuracy"
            ] == 1.0
            assert diag[
                "plaintext_types_with_multiple_observed_base_forms"
            ] == 0
            assert diag[
                "nominal_variant_entropy_bits"
            ] == 4.0


def test_generated_jsonl_is_compatible_with_level2_reader(
    tmp_path: Path,
):
    parts = split()

    generated = generate_control(
        small_lines(),
        train_indices=parts[
            "train"
        ],
        validation_indices=parts[
            "validation"
        ],
        reserved_indices=parts[
            "reserved"
        ],
        seed=40814041438,
        factorization="2x8",
    )

    out = tmp_path / "run"
    write_generated_control(
        out,
        generated,
        provenance={
            "test": True
        },
    )

    train_units, validation_units = read_control_units(
        out / "corpus.jsonl",
        train_indices=parts[
            "train"
        ],
        validation_indices=parts[
            "validation"
        ],
        expected_line_count=len(
            small_lines()
        ),
    )

    assert len(
        train_units
    ) == len(
        parts["train"]
    )
    assert len(
        validation_units
    ) == len(
        parts["validation"]
    )

    raw_rows = (
        out / "corpus.jsonl"
    ).read_text(
        encoding="utf-8"
    ).splitlines()

    reserved_payload = json.loads(
        raw_rows[10]
    )
    assert reserved_payload[
        "reserved_test_opaque"
    ] is True
    assert "tokens" not in reserved_payload


def test_invalid_factorization_is_rejected():
    key = build_master_key(
        small_lines()[:8],
        seed=5,
    )

    try:
        variant_pair(
            key,
            factorization="8x2",
            variant_index=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Post-hoc factorization must not be accepted"
        )
