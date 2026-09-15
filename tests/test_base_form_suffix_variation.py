"""Tests for Phase 3A.3 base-form-preserving suffix variation.

Intended repository destination:
    tests/test_base_form_suffix_variation.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ciphers.base_form_suffix_variation import (
    active_suffixes,
    build_master_key,
    decode_token,
    encode_base,
    encode_token,
    generate_control,
    strip_suffix,
    suffix_choice_index,
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


def test_master_key_is_deterministic_and_seed_sensitive():
    train = small_lines()[:8]

    a = build_master_key(
        train,
        seed=40814041438,
    )
    b = build_master_key(
        train,
        seed=40814041438,
    )
    c = build_master_key(
        train,
        seed=40814041439,
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


def test_active_suffix_sets_are_nested_prefixes():
    key = build_master_key(
        small_lines()[:8],
        seed=7,
    )

    k1 = active_suffixes(
        key,
        variant_count=1,
    )
    k2 = active_suffixes(
        key,
        variant_count=2,
    )
    k4 = active_suffixes(
        key,
        variant_count=4,
    )

    assert k1 == k2[:1]
    assert k2 == k4[:2]
    assert len(set(k4)) == 4
    assert all(
        suffix.startswith("V")
        for suffix in k4
    )


def test_every_k_adds_exactly_one_suffix_and_preserves_same_base():
    key = build_master_key(
        small_lines()[:8],
        seed=11,
    )
    plaintext = "alpha"
    expected_base = encode_base(
        plaintext,
        key=key,
    )

    for k in (1, 2, 4):
        encoded = encode_token(
            plaintext,
            key=key,
            variant_count=k,
            occurrence_seed=11,
            line_index=3,
            token_index=2,
        )

        assert len(encoded) == len(expected_base) + 1
        assert strip_suffix(
            encoded,
            key=key,
        ) == expected_base
        assert decode_token(
            encoded,
            key=key,
        ) == plaintext


def test_suffix_choice_excludes_plaintext_token_identity():
    key = build_master_key(
        small_lines()[:8],
        seed=19,
    )

    left = encode_token(
        "alpha",
        key=key,
        variant_count=4,
        occurrence_seed=19,
        line_index=5,
        token_index=3,
    )
    right = encode_token(
        "beta",
        key=key,
        variant_count=4,
        occurrence_seed=19,
        line_index=5,
        token_index=3,
    )

    # Different words have different bases, but the suffix choice for the same
    # occurrence coordinates is identical because token identity is not hashed.
    assert left[:-1] != right[:-1]
    assert left[-1] == right[-1]

    assert suffix_choice_index(
        occurrence_seed=19,
        line_index=5,
        token_index=3,
        variant_count=4,
    ) == suffix_choice_index(
        occurrence_seed=19,
        line_index=5,
        token_index=3,
        variant_count=4,
    )


def test_k1_is_fixed_and_k4_realizes_multiple_surfaces_for_repeated_word():
    train = [
        tuple(["alpha"] * 150),
        tuple(["beta"] * 20),
    ]
    key = build_master_key(
        train,
        seed=40814041438,
    )

    k1 = {
        encode_token(
            "alpha",
            key=key,
            variant_count=1,
            occurrence_seed=40814041438,
            line_index=1,
            token_index=i,
        )
        for i in range(150)
    }
    k4 = {
        encode_token(
            "alpha",
            key=key,
            variant_count=4,
            occurrence_seed=40814041438,
            line_index=1,
            token_index=i,
        )
        for i in range(150)
    }

    assert len(k1) == 1
    assert len(k4) == 4

    bases = {
        token[:-1]
        for token in k4
    }
    assert len(bases) == 1


def test_base_mapping_is_identical_across_k_for_same_seed():
    parts = split()

    generated = {}
    for k in (1, 2, 4):
        generated[k] = generate_control(
            small_lines(),
            train_indices=parts["train"],
            validation_indices=parts["validation"],
            reserved_indices=parts["reserved"],
            seed=123,
            variant_count=k,
        )

    assert (
        generated[1].key["master_key_sha256"]
        == generated[2].key["master_key_sha256"]
        == generated[4].key["master_key_sha256"]
    )
    assert (
        generated[1].key["mono_forward"]
        == generated[2].key["mono_forward"]
        == generated[4].key["mono_forward"]
    )
    assert (
        generated[1].key["master_suffixes"]
        == generated[2].key["master_suffixes"]
        == generated[4].key["master_suffixes"]
    )


def test_generator_is_deterministic_and_reserved_lines_are_opaque():
    parts = split()
    kwargs = dict(
        source_lines=small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=40814041438,
        variant_count=4,
    )

    first = generate_control(**kwargs)
    second = generate_control(**kwargs)

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

    for line_index in parts["reserved"]:
        row = first.rows[line_index - 1]
        assert row == {
            "line_index": line_index,
            "reserved_test_opaque": True,
        }
        assert "tokens" not in row


def test_validation_unseen_character_is_reversible_without_refitting_key():
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
        variant_count=4,
    )

    validation_cipher = (
        generated.rows[2]["tokens"][0]
    )
    validation_base = validation_cipher[:-1]

    assert all(
        glyph.startswith("U")
        for glyph in validation_base
    )
    assert validation_cipher[-1].startswith("V")
    assert decode_token(
        validation_cipher,
        key=generated.key,
    ) == "zzzz"


def test_all_train_and_validation_tokens_have_full_mechanism_coverage():
    parts = split()

    for k in (1, 2, 4):
        generated = generate_control(
            small_lines(),
            train_indices=parts["train"],
            validation_indices=parts["validation"],
            reserved_indices=parts["reserved"],
            seed=17,
            variant_count=k,
        )

        assert generated.summary[
            "train_diagnostics"
        ]["token_coverage_fraction"] == 1.0
        assert generated.summary[
            "validation_diagnostics"
        ]["token_coverage_fraction"] == 1.0

        assert generated.summary[
            "train_diagnostics"
        ]["base_form_recovery_accuracy"] == 1.0
        assert generated.summary[
            "validation_diagnostics"
        ]["base_form_recovery_accuracy"] == 1.0

        assert generated.summary[
            "validation_diagnostics"
        ]["plaintext_types_with_multiple_observed_base_forms"] == 0


def test_same_surface_probability_decreases_with_k_on_repeated_corpus():
    # Repeated words give enough same-word pairs to test the designed trend.
    source = [
        tuple(["alpha", "beta"] * 50),
        tuple(["alpha", "beta"] * 50),
        tuple(["alpha", "beta"] * 20),
        ("reserved",),
    ]

    probabilities = []

    for k in (1, 2, 4):
        generated = generate_control(
            source,
            train_indices=[1, 2],
            validation_indices=[3],
            reserved_indices=[4],
            seed=40814041438,
            variant_count=k,
        )
        probabilities.append(
            generated.summary[
                "validation_diagnostics"
            ][
                "same_surface_probability_given_same_plaintext_word"
            ]
        )

    assert probabilities[0] == 1.0
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_generated_jsonl_is_compatible_with_level2_reader_and_skips_reserved(
    tmp_path: Path,
):
    parts = split()

    generated = generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=40814041438,
        variant_count=4,
    )

    out = tmp_path / "run"
    write_generated_control(
        out,
        generated,
        provenance={"test": True},
    )

    train_units, validation_units = read_control_units(
        out / "corpus.jsonl",
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        expected_line_count=len(
            small_lines()
        ),
    )

    assert len(train_units) == len(
        parts["train"]
    )
    assert len(validation_units) == len(
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


def test_suffix_entropy_and_length_diagnostics_are_well_formed():
    parts = split()
    generated = generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=31,
        variant_count=4,
    )

    diag = generated.summary[
        "validation_diagnostics"
    ]

    assert diag[
        "token_length_plus_one_accuracy"
    ] == 1.0
    assert diag[
        "base_form_preservation_fraction"
    ] == 1.0
    assert 0.0 <= diag[
        "suffix_entropy_fraction_of_max"
    ] <= 1.0
    assert diag[
        "active_suffix_count"
    ] == 4

    suffix_counts = json.loads(
        diag["active_suffix_counts_json"]
    )
    assert len(suffix_counts) == 4
    assert sum(suffix_counts.values()) == diag["tokens"]
