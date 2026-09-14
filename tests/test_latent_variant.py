"""Tests for Phase 3A related lexical-variant controls.

Intended repository destination:
    tests/test_latent_variant.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ciphers.latent_variant import (
    build_mechanism_key,
    decode_token,
    encode_token,
    generate_control,
    select_target_words,
    write_generated_control,
)
from src.evaluation.level2_predictive_controls import (
    read_control_units,
)


def small_lines():
    return [
        ("alpha", "beta", "alpha", "gamma"),
        ("alpha", "beta", "delta"),
        ("alpha", "beta", "alpha"),
        ("beta", "alpha", "epsilon"),
        ("alpha", "beta", "zeta"),
        ("alpha", "beta", "eta"),
        ("alpha", "beta", "theta"),
        ("alpha", "beta", "iota"),
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


def test_target_words_use_train_frequency_then_lexical_tie_break():
    lines = [
        ("b", "a", "b", "a"),
        ("c", "c"),
        ("d",),
    ]
    # a, b, c all have count 2; lexical order breaks the tie.
    assert select_target_words(
        lines,
        top_n=3,
    ) == ["a", "b", "c"]


def test_variant_families_are_collision_free_and_change_final_position_only():
    lines = small_lines()[:8]
    key = build_mechanism_key(
        lines,
        seed=40814041438,
        top_n=2,
    )

    seen = {}
    for word, family in key["families"].items():
        variants = family["variants_max_k4"]
        assert len(variants) == 4

        canonical = variants[0]
        for variant in variants:
            assert len(variant) == 2
            assert variant[0] == canonical[0]

        assert len(
            {tuple(variant) for variant in variants}
        ) == 4

        for variant in variants:
            variant_tuple = tuple(variant)
            assert variant_tuple not in seen
            seen[variant_tuple] = word
            assert decode_token(
                variant,
                key=key,
            ) == word

    assert len(seen) == 8


def test_k1_is_fixed_but_k4_permits_related_surface_variation():
    # Give one target word many occurrences so deterministic occurrence hashing
    # has ample opportunity to realize multiple K=4 variants.
    train = [
        tuple(["alpha"] * 100),
        tuple(["beta"] * 80),
    ]
    key = build_mechanism_key(
        train,
        seed=40814041438,
        top_n=2,
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
        for i in range(100)
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
        for i in range(100)
    }

    assert len(k1) == 1
    assert 2 <= len(k4) <= 4

    canonical = next(iter(k1))
    for variant in k4:
        assert variant[0] == canonical[0]
        assert sum(
            left != right
            for left, right in zip(
                canonical,
                variant,
            )
        ) <= 1


def test_generator_is_deterministic_and_does_not_encode_reserved_lines():
    parts = split()
    kwargs = dict(
        source_lines=small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=40814041438,
        variant_count=4,
        top_n=2,
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

    for line_index in parts["reserved"]:
        row = first.rows[line_index - 1]
        assert row == {
            "line_index": line_index,
            "reserved_test_opaque": True,
        }
        assert "tokens" not in row


def test_validation_unseen_character_is_reversible_without_refitting_key():
    source = [
        ("aaaa", "aaaa"),
        ("aaaa",),
        ("z",),
        ("reserved",),
    ]

    generated = generate_control(
        source,
        train_indices=[1, 2],
        validation_indices=[3],
        reserved_indices=[4],
        seed=7,
        variant_count=1,
        top_n=1,
    )

    validation_cipher = generated.rows[2]["tokens"][0]
    assert validation_cipher[0].startswith("U")
    assert decode_token(
        validation_cipher,
        key=generated.key,
    ) == "z"


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
        variant_count=2,
        top_n=2,
    )

    out = tmp_path / "run"
    write_generated_control(
        out,
        generated,
        provenance={
            "test": True,
        },
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

    # Reserved rows are intentionally opaque; the reader succeeds because it
    # streams past them before JSON token parsing.
    raw_rows = (
        out / "corpus.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    reserved_payload = json.loads(
        raw_rows[10]
    )
    assert reserved_payload[
        "reserved_test_opaque"
    ] is True
    assert "tokens" not in reserved_payload


def test_increasing_k_does_not_change_code_length_or_code_symbol_inventory():
    lines = small_lines()[:8]
    key = build_mechanism_key(
        lines,
        seed=99,
        top_n=2,
    )

    all_symbols = set(
        key["code_symbols"]
    )

    for word, family in key["families"].items():
        for k in (1, 2, 4):
            active = family[
                "variants_max_k4"
            ][:k]
            assert all(
                len(variant) == 2
                for variant in active
            )
            assert all(
                glyph in all_symbols
                for variant in active
                for glyph in variant
            )
