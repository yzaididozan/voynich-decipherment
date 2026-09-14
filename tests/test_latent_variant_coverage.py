"""Tests for Phase 3A.2 lexical-coverage dose-response controls.

Intended repository destination:
    tests/test_latent_variant_coverage.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ciphers.latent_variant_coverage import (
    active_target_words,
    build_master_key,
    decode_token,
    encode_token,
    generate_control,
    ranked_train_words,
    write_generated_control,
)
from src.evaluation.level2_predictive_controls import (
    read_control_units,
)


def small_lines():
    return [
        ("alpha", "beta", "alpha", "gamma", "delta"),
        ("alpha", "beta", "epsilon", "zeta"),
        ("alpha", "beta", "eta", "theta"),
        ("alpha", "beta", "iota", "kappa"),
        ("alpha", "beta", "lambda", "mu"),
        ("alpha", "beta", "nu", "xi"),
        ("alpha", "beta", "omicron", "pi"),
        ("alpha", "beta", "rho", "sigma"),
        ("alpha", "beta", "tau", "validation"),
        ("beta", "alpha", "upsilon", "validation"),
        ("reserved", "alpha"),
        ("reserved", "beta"),
    ]


def split():
    return {
        "train": list(range(1, 9)),
        "validation": [9, 10],
        "reserved": [11, 12],
    }


def test_ranking_uses_train_frequency_then_lexical_tie_break():
    lines = [
        ("b", "a", "b", "a"),
        ("c", "c"),
        ("d",),
    ]
    # a, b, c each occur twice; lexical order breaks the tie.
    assert ranked_train_words(lines)[:4] == [
        "a", "b", "c", "d"
    ]


def test_master_key_has_unique_three_glyph_family_ids_and_four_variants():
    lines = small_lines()[:8]
    key = build_master_key(
        lines,
        seed=40814041438,
        master_target_count=8,
    )

    family_ids = []
    surfaces = {}

    for word in key[
        "master_target_words_frequency_ranked"
    ]:
        family = key["families"][word]
        family_id = tuple(family["family_id"])
        family_ids.append(family_id)

        variants = family["variants_k4"]
        assert len(variants) == 4
        assert len(
            {tuple(v) for v in variants}
        ) == 4

        for variant in variants:
            assert len(variant) == 4
            assert tuple(variant[:3]) == family_id

            surface = tuple(variant)
            assert surface not in surfaces
            surfaces[surface] = word

            assert decode_token(
                variant,
                key=key,
            ) == word

        # Every pair differs only at final position.
        for i, left in enumerate(variants):
            for right in variants[i + 1 :]:
                assert left[:3] == right[:3]
                assert left[3] != right[3]

    assert len(set(family_ids)) == 8
    assert len(surfaces) == 8 * 4


def test_active_sets_are_frequency_ranked_nested_prefixes():
    lines = small_lines()[:8]
    key = build_master_key(
        lines,
        seed=7,
        master_target_count=10,
    )

    n2 = active_target_words(
        key,
        target_count=2,
    )
    n5 = active_target_words(
        key,
        target_count=5,
    )
    n10 = active_target_words(
        key,
        target_count=10,
    )

    assert n2 == n5[:2]
    assert n5 == n10[:5]


def test_shared_word_mapping_and_occurrence_variant_are_invariant_across_n():
    lines = small_lines()[:8]
    key = build_master_key(
        lines,
        seed=40814041438,
        master_target_count=10,
    )

    shared_word = active_target_words(
        key,
        target_count=2,
    )[0]

    outputs = []
    for n in (2, 5, 10):
        outputs.append(
            encode_token(
                shared_word,
                key=key,
                target_count=n,
                occurrence_seed=40814041438,
                line_index=3,
                token_index=4,
            )
        )

    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0][:3] == tuple(
        key["families"][shared_word]["family_id"]
    )


def test_inactive_master_word_uses_fallback_until_coverage_reaches_it():
    lines = small_lines()[:8]
    key = build_master_key(
        lines,
        seed=99,
        master_target_count=10,
    )

    word = active_target_words(
        key,
        target_count=10,
    )[7]

    low_n = encode_token(
        word,
        key=key,
        target_count=2,
        occurrence_seed=99,
        line_index=1,
        token_index=0,
    )
    high_n = encode_token(
        word,
        key=key,
        target_count=10,
        occurrence_seed=99,
        line_index=1,
        token_index=0,
    )

    assert all(
        glyph.startswith("M")
        for glyph in low_n
    )
    assert len(high_n) == 4
    assert all(
        glyph.startswith("N")
        for glyph in high_n
    )

    assert decode_token(
        low_n,
        key=key,
    ) == word
    assert decode_token(
        high_n,
        key=key,
    ) == word


def test_master_codebook_fingerprint_is_deterministic():
    lines = small_lines()[:8]

    a = build_master_key(
        lines,
        seed=123,
        master_target_count=10,
    )
    b = build_master_key(
        lines,
        seed=123,
        master_target_count=10,
    )
    c = build_master_key(
        lines,
        seed=124,
        master_target_count=10,
    )

    assert (
        a["master_codebook_sha256"]
        == b["master_codebook_sha256"]
    )
    assert (
        a["master_codebook_sha256"]
        != c["master_codebook_sha256"]
    )


def test_generator_is_deterministic_and_reserved_lines_are_opaque():
    parts = split()
    kwargs = dict(
        source_lines=small_lines(),
        train_indices=parts["train"],
        validation_indices=parts["validation"],
        reserved_indices=parts["reserved"],
        seed=40814041438,
        target_count=4,
        master_target_count=10,
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
        ("aaaa", "bbbb", "cccc"),
        ("aaaa", "bbbb", "dddd"),
        ("zzzz",),
        ("reserved",),
    ]

    generated = generate_control(
        source,
        train_indices=[1, 2],
        validation_indices=[3],
        reserved_indices=[4],
        seed=7,
        target_count=1,
        master_target_count=4,
    )

    validation_cipher = (
        generated.rows[2]["tokens"][0]
    )
    assert all(
        glyph.startswith("U")
        for glyph in validation_cipher
    )
    assert decode_token(
        validation_cipher,
        key=generated.key,
    ) == "zzzz"


def test_increasing_n_increases_or_preserves_targeted_validation_coverage():
    # TRAIN establishes a ranked target list; validation contains examples
    # spanning both high- and lower-frequency TRAIN types.
    source = [
        ("a", "a", "a", "b", "c", "d"),
        ("a", "a", "b", "b", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("a", "b", "c", "d", "e", "f"),
        ("reserved",),
    ]

    train = list(range(1, 9))
    validation = [9, 10]
    reserved = [11]

    coverages = []
    for n in (1, 3, 6):
        generated = generate_control(
            source,
            train_indices=train,
            validation_indices=validation,
            reserved_indices=reserved,
            seed=11,
            target_count=n,
            master_target_count=6,
        )
        coverages.append(
            generated.summary[
                "validation_diagnostics"
            ][
                "targeted_token_fraction"
            ]
        )

    assert coverages[0] <= coverages[1] <= coverages[2]
    assert coverages[0] < coverages[2]


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
        target_count=4,
        master_target_count=10,
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


def test_target_variants_have_pairwise_hamming_distance_one():
    lines = small_lines()[:8]
    key = build_master_key(
        lines,
        seed=19,
        master_target_count=5,
    )

    for word in active_target_words(
        key,
        target_count=5,
    ):
        variants = key[
            "families"
        ][word]["variants_k4"]

        for i, left in enumerate(variants):
            for right in variants[i + 1 :]:
                distance = sum(
                    a != b
                    for a, b in zip(
                        left,
                        right,
                    )
                )
                assert distance == 1
