"""Tests for Phase 3A.4 extended base-form surface multiplicity.

Intended repository destination:
    tests/test_base_form_suffix_extended.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ciphers import base_form_suffix_variation as phase3a3
from src.ciphers.base_form_suffix_extended import (
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


def test_extended_key_inherits_exact_phase3a3_base_and_first_four_suffixes():
    train = small_lines()[:8]
    seed = 40814041438

    parent = phase3a3.build_master_key(
        train,
        seed=seed,
        master_suffix_count=4,
    )
    extended = build_master_key(
        train,
        seed=seed,
        master_suffix_count=16,
    )

    assert (
        extended["mono_forward"]
        == parent["mono_forward"]
    )
    assert (
        extended["mono_reverse"]
        == parent["mono_reverse"]
    )
    assert (
        extended["train_character_alphabet"]
        == parent["train_character_alphabet"]
    )
    assert (
        extended["master_suffixes"][:4]
        == parent["master_suffixes"]
    )
    assert (
        extended["legacy_phase3a3_suffixes"]
        == parent["master_suffixes"]
    )
    assert (
        extended["parent_phase3a3_master_key_sha256"]
        == parent["master_key_sha256"]
    )


def test_extension_has_sixteen_unique_suffixes_and_preserves_legacy_namespace():
    key = build_master_key(
        small_lines()[:8],
        seed=7,
    )

    suffixes = key["master_suffixes"]

    assert len(suffixes) == 16
    assert len(set(suffixes)) == 16
    assert set(suffixes[:4]) == {
        "V0001",
        "V0002",
        "V0003",
        "V0004",
    }
    assert set(suffixes[4:]) == {
        f"V{i:04d}"
        for i in range(5, 17)
    }


def test_active_suffix_sets_are_nested_4_8_16_prefixes():
    key = build_master_key(
        small_lines()[:8],
        seed=19,
    )

    k4 = active_suffixes(
        key,
        variant_count=4,
    )
    k8 = active_suffixes(
        key,
        variant_count=8,
    )
    k16 = active_suffixes(
        key,
        variant_count=16,
    )

    assert k4 == k8[:4]
    assert k8 == k16[:8]
    assert len(k4) == 4
    assert len(k8) == 8
    assert len(k16) == 16


def test_k4_token_generation_exactly_matches_phase3a3():
    train = small_lines()[:8]
    seed = 40814041438

    parent_key = phase3a3.build_master_key(
        train,
        seed=seed,
        master_suffix_count=4,
    )
    extended_key = build_master_key(
        train,
        seed=seed,
        master_suffix_count=16,
    )

    words = ["alpha", "beta", "gamma", "theta"]

    for line_index in range(1, 9):
        for token_index, word in enumerate(words):
            parent_token = phase3a3.encode_token(
                word,
                key=parent_key,
                variant_count=4,
                occurrence_seed=seed,
                line_index=line_index,
                token_index=token_index,
            )
            extended_token = encode_token(
                word,
                key=extended_key,
                variant_count=4,
                occurrence_seed=seed,
                line_index=line_index,
                token_index=token_index,
            )
            assert extended_token == parent_token


def test_k4_full_generated_rows_exactly_match_phase3a3():
    parts = split()
    seed = 40814041438

    parent = phase3a3.generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=seed,
        variant_count=4,
        master_suffix_count=4,
    )
    extended = generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=seed,
        variant_count=4,
        master_suffix_count=16,
    )

    assert extended.rows == parent.rows


def test_k4_written_corpus_is_byte_identical_to_phase3a3(tmp_path: Path):
    parts = split()
    seed = 40814041438

    parent = phase3a3.generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=seed,
        variant_count=4,
        master_suffix_count=4,
    )
    extended = generate_control(
        small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=seed,
        variant_count=4,
        master_suffix_count=16,
    )

    parent_dir = tmp_path / "parent"
    extended_dir = tmp_path / "extended"

    phase3a3.write_generated_control(
        parent_dir,
        parent,
        provenance={"test": "parent"},
    )
    write_generated_control(
        extended_dir,
        extended,
        provenance={"test": "extended"},
    )

    assert (
        (parent_dir / "corpus.jsonl").read_bytes()
        == (extended_dir / "corpus.jsonl").read_bytes()
    )


def test_suffix_choice_hash_namespace_is_exact_phase3a3_at_k4():
    for seed in (7, 40814041438):
        for line_index in (1, 3, 99):
            for token_index in (0, 2, 11):
                assert suffix_choice_index(
                    occurrence_seed=seed,
                    line_index=line_index,
                    token_index=token_index,
                    variant_count=4,
                ) == phase3a3.suffix_choice_index(
                    occurrence_seed=seed,
                    line_index=line_index,
                    token_index=token_index,
                    variant_count=4,
                )


def test_every_k_preserves_base_and_adds_exactly_one_suffix():
    key = build_master_key(
        small_lines()[:8],
        seed=11,
    )
    plaintext = "alpha"
    expected_base = encode_base(
        plaintext,
        key=key,
    )

    for k in (4, 8, 16):
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


def test_suffix_choice_excludes_plaintext_identity():
    key = build_master_key(
        small_lines()[:8],
        seed=23,
    )

    left = encode_token(
        "alpha",
        key=key,
        variant_count=16,
        occurrence_seed=23,
        line_index=5,
        token_index=3,
    )
    right = encode_token(
        "beta",
        key=key,
        variant_count=16,
        occurrence_seed=23,
        line_index=5,
        token_index=3,
    )

    assert left[:-1] != right[:-1]
    assert left[-1] == right[-1]


def test_higher_k_realizes_more_surfaces_for_heavily_repeated_word():
    train = [
        tuple(["alpha"] * 512),
        tuple(["beta"] * 20),
    ]
    key = build_master_key(
        train,
        seed=40814041438,
    )

    realized = {}
    for k in (4, 8, 16):
        realized[k] = {
            encode_token(
                "alpha",
                key=key,
                variant_count=k,
                occurrence_seed=40814041438,
                line_index=1,
                token_index=i,
            )
            for i in range(512)
        }

    assert len(realized[4]) == 4
    assert len(realized[8]) == 8
    assert len(realized[16]) == 16

    for surfaces in realized.values():
        assert len(
            {surface[:-1] for surface in surfaces}
        ) == 1


def test_master_key_is_identical_across_k_generation():
    parts = split()

    generated = {}
    for k in (4, 8, 16):
        generated[k] = generate_control(
            small_lines(),
            train_indices=parts["train"],
            validation_indices=parts["validation"],
            reserved_indices=parts["reserved"],
            seed=123,
            variant_count=k,
        )

    assert (
        generated[4].key["master_key_sha256"]
        == generated[8].key["master_key_sha256"]
        == generated[16].key["master_key_sha256"]
    )
    assert (
        generated[4].key[
            "parent_phase3a3_master_key_sha256"
        ]
        == generated[8].key[
            "parent_phase3a3_master_key_sha256"
        ]
        == generated[16].key[
            "parent_phase3a3_master_key_sha256"
        ]
    )


def test_generator_is_deterministic_and_reserved_lines_are_opaque():
    parts = split()
    kwargs = dict(
        source_lines=small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=40814041438,
        variant_count=16,
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
        variant_count=16,
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


def test_all_train_validation_tokens_have_full_coverage_and_preserved_base():
    parts = split()

    for k in (4, 8, 16):
        generated = generate_control(
            small_lines(),
            train_indices=parts["train"],
            validation_indices=parts["validation"],
            reserved_indices=parts["reserved"],
            seed=17,
            variant_count=k,
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
                "token_length_plus_one_accuracy"
            ] == 1.0
            assert diag[
                "plaintext_types_with_multiple_observed_base_forms"
            ] == 0


def test_same_surface_probability_decreases_with_extended_k():
    source = [
        tuple(["alpha", "beta"] * 200),
        tuple(["alpha", "beta"] * 200),
        tuple(["alpha", "beta"] * 100),
        ("reserved",),
    ]

    probabilities = []

    for k in (4, 8, 16):
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
        variant_count=16,
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


def test_k_above_16_is_rejected():
    key = build_master_key(
        small_lines()[:8],
        seed=5,
    )

    try:
        active_suffixes(
            key,
            variant_count=32,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Phase 3A.4 must not permit post-hoc K > 16"
        )
