from __future__ import annotations

from collections import Counter

import pytest

from src.ciphers.historical_controls import (
    FAMILIES,
    _columnar_decrypt,
    _columnar_encrypt,
    count_ciphertext,
    decode_result,
    encode_homophonic,
    encode_monoalphabetic,
    encode_nomenclator,
    encode_null_insertion,
    encode_syllabic_hybrid,
    encode_transposition,
    encode_verbose,
    generate_family,
    normalize_plaintext,
    plaintext_alphabet,
    verify_round_trip,
)


TEXT = """Arma virumque cano.
Arma cano; arma virum.
Gallia est omnis divisa in partes tres.
"""


@pytest.fixture
def corpus():
    return normalize_plaintext(TEXT)


def test_normalization_is_minimal_and_line_preserving():
    corpus = normalize_plaintext(
        "Ārma, VIRUMQUE! 123\nGallia-est.\n"
    )
    assert corpus == (
        ("ārma", "virumque"),
        ("gallia", "est"),
    )


def test_all_frozen_families_are_declared():
    assert FAMILIES == (
        "monoalphabetic",
        "homophonic",
        "verbose",
        "null_insertion",
        "nomenclator",
        "transposition",
        "syllabic_hybrid",
    )


def test_monoalphabetic_round_trip_and_single_glyph_per_char(corpus):
    result = encode_monoalphabetic(
        corpus,
        seed=40814041438,
    )
    verify_round_trip(corpus, result)

    plain_chars = sum(
        len(token) for line in corpus for token in line
    )
    cipher = count_ciphertext(result.ciphertext)

    assert cipher["glyphs"] == plain_chars
    assert cipher["alphabet_size"] == len(
        plaintext_alphabet(corpus)
    )


def test_homophonic_round_trip_and_uses_multiple_variants(corpus):
    repeated = normalize_plaintext(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    )
    result = encode_homophonic(
        repeated,
        seed=40814041438,
        homophones_per_symbol=3,
    )
    verify_round_trip(repeated, result)

    used = {
        glyph
        for line in result.ciphertext
        for token in line
        for glyph in token
    }
    assert len(used) == 3


def test_verbose_round_trip_and_doubles_character_count(corpus):
    result = encode_verbose(
        corpus,
        seed=40814041438,
        codeword_length=2,
    )
    verify_round_trip(corpus, result)

    plain_chars = sum(
        len(token) for line in corpus for token in line
    )
    assert count_ciphertext(result.ciphertext)["glyphs"] == 2 * plain_chars


def test_null_insertion_round_trip_and_inserts_nulls(corpus):
    result = encode_null_insertion(
        corpus,
        seed=40814041438,
        insertion_probability=0.5,
        null_symbol_count=3,
    )
    verify_round_trip(corpus, result)

    assert result.diagnostics["inserted_nulls"] > 0

    nulls = set(result.key["null_symbols"])
    observed = {
        glyph
        for line in result.ciphertext
        for token in line
        for glyph in token
    }
    assert observed & nulls


def test_nomenclator_round_trip_and_codes_frequent_words(corpus):
    result = encode_nomenclator(
        corpus,
        seed=40814041438,
        top_word_count=2,
        codeword_length=2,
    )
    verify_round_trip(corpus, result)

    assert "arma" in result.key["word_codes"]
    assert result.diagnostics["coded_word_occurrences"] >= 3


def test_columnar_transposition_inverse():
    sequence = tuple(f"G{i}" for i in range(13))
    order = [2, 0, 4, 1, 3]

    encrypted = _columnar_encrypt(
        sequence,
        width=5,
        column_order=order,
    )
    decrypted = _columnar_decrypt(
        encrypted,
        width=5,
        column_order=order,
    )

    assert encrypted != sequence
    assert decrypted == sequence


def test_transposition_round_trip_and_preserves_token_lengths(corpus):
    result = encode_transposition(
        corpus,
        seed=40814041438,
        width=5,
    )
    verify_round_trip(corpus, result)

    for plain_line, cipher_line in zip(
        corpus,
        result.ciphertext,
    ):
        assert [len(token) for token in plain_line] == [
            len(token) for token in cipher_line
        ]


def test_syllabic_hybrid_round_trip_and_compresses_bigrams():
    corpus = normalize_plaintext(
        "banana bandana banana bandana\n"
    )
    result = encode_syllabic_hybrid(
        corpus,
        seed=40814041438,
        top_bigram_count=4,
        minimum_bigram_count=2,
    )
    verify_round_trip(corpus, result)

    plain_chars = sum(
        len(token) for line in corpus for token in line
    )
    cipher_glyphs = count_ciphertext(
        result.ciphertext
    )["glyphs"]

    assert result.diagnostics["bigram_encoded_occurrences"] > 0
    assert cipher_glyphs < plain_chars


def test_same_seed_is_deterministic_for_every_family(corpus):
    parameters = {
        "monoalphabetic": {},
        "homophonic": {
            "homophones_per_symbol": 3,
        },
        "verbose": {
            "codeword_length": 2,
        },
        "null_insertion": {
            "insertion_probability": 0.2,
            "null_symbol_count": 3,
        },
        "nomenclator": {
            "top_word_count": 4,
            "codeword_length": 2,
        },
        "transposition": {
            "width": 5,
        },
        "syllabic_hybrid": {
            "top_bigram_count": 4,
            "minimum_bigram_count": 2,
        },
    }

    for family in FAMILIES:
        first = generate_family(
            corpus,
            family=family,
            seed=40814041438,
            parameters=parameters[family],
        )
        second = generate_family(
            corpus,
            family=family,
            seed=40814041438,
            parameters=parameters[family],
        )

        assert first.ciphertext == second.ciphertext
        assert first.key == second.key
        verify_round_trip(corpus, first)


def test_different_seed_changes_stochastic_or_keyed_outputs(corpus):
    parameters = {
        "monoalphabetic": {},
        "homophonic": {"homophones_per_symbol": 3},
        "verbose": {"codeword_length": 2},
        "null_insertion": {
            "insertion_probability": 0.2,
            "null_symbol_count": 3,
        },
        "nomenclator": {
            "top_word_count": 4,
            "codeword_length": 2,
        },
        "transposition": {"width": 5},
        "syllabic_hybrid": {
            "top_bigram_count": 4,
            "minimum_bigram_count": 2,
        },
    }

    changed = 0

    for family in FAMILIES:
        first = generate_family(
            corpus,
            family=family,
            seed=40814041438,
            parameters=parameters[family],
        )
        second = generate_family(
            corpus,
            family=family,
            seed=40814041439,
            parameters=parameters[family],
        )
        if first.ciphertext != second.ciphertext:
            changed += 1

    assert changed >= 5


def test_decoder_dispatch_round_trips_all_families(corpus):
    parameters = {
        "monoalphabetic": {},
        "homophonic": {"homophones_per_symbol": 3},
        "verbose": {"codeword_length": 2},
        "null_insertion": {
            "insertion_probability": 0.2,
            "null_symbol_count": 3,
        },
        "nomenclator": {
            "top_word_count": 4,
            "codeword_length": 2,
        },
        "transposition": {"width": 5},
        "syllabic_hybrid": {
            "top_bigram_count": 4,
            "minimum_bigram_count": 2,
        },
    }

    for family in FAMILIES:
        result = generate_family(
            corpus,
            family=family,
            seed=40814041438,
            parameters=parameters[family],
        )
        assert decode_result(result) == corpus
