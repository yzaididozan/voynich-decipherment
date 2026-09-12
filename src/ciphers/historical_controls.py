"""Deterministic historical-cipher control generators for VOYAGER.

These are *control families*, not claims that the Voynich Manuscript uses any
one of these exact systems.  The goal is to transform independent plaintext
corpora through simple, reproducible mechanisms that span historically
relevant design ideas:

- monoalphabetic substitution;
- homophonic substitution;
- verbose substitution;
- null insertion;
- nomenclator-like word codes with alphabetic fallback;
- columnar transposition;
- frequent-bigram ("syllabic") hybrid encoding.

Cipher glyphs are abstract atomic labels (G0001, G0002, ...).  This avoids
smuggling visual resemblance to Voynich glyphs into the control.

Every family is mechanically reversible under its saved key.  The
transposition control restores the original token-length partition after
line-level columnar transposition solely so token-based structural metrics can
be applied to the synthetic control.  That segmentation restoration is a
benchmarking convention, not a historical claim.

The module uses only Python's standard library and never touches Voynich
validation/test membership.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
import re
import unicodedata
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


PlainToken = str
PlainLine = Tuple[PlainToken, ...]
CipherToken = Tuple[str, ...]
CipherLine = Tuple[CipherToken, ...]
PlainCorpus = Tuple[PlainLine, ...]
CipherCorpus = Tuple[CipherLine, ...]


FAMILIES = (
    "monoalphabetic",
    "homophonic",
    "verbose",
    "null_insertion",
    "nomenclator",
    "transposition",
    "syllabic_hybrid",
)


@dataclass(frozen=True)
class ControlResult:
    family: str
    seed: int
    ciphertext: CipherCorpus
    key: dict
    diagnostics: dict


class GlyphAllocator:
    """Allocate globally unique atomic synthetic glyph labels."""

    def __init__(self, prefix: str = "G"):
        self.prefix = prefix
        self._next = 1

    def take(self, count: int) -> Tuple[str, ...]:
        if count < 0:
            raise ValueError("count must be >= 0")
        result = tuple(
            f"{self.prefix}{index:04d}"
            for index in range(self._next, self._next + count)
        )
        self._next += count
        return result


def normalize_plaintext(text: str) -> PlainCorpus:
    """Normalize UTF-8 text into lowercase Unicode-letter tokens.

    Rules are intentionally minimal:
    - NFKC Unicode normalization;
    - lowercase;
    - contiguous alphabetic Unicode characters form tokens;
    - punctuation/digits are separators;
    - original non-empty line boundaries are preserved when they contain text.

    No spelling modernization, accent stripping, abbreviation expansion, or
    language-specific normalization occurs.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    lines: List[PlainLine] = []

    for raw_line in normalized.splitlines():
        tokens: List[str] = []
        current: List[str] = []

        for char in raw_line:
            if char.isalpha():
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []

        if current:
            tokens.append("".join(current))

        if tokens:
            lines.append(tuple(tokens))

    if not lines:
        raise ValueError("Plaintext contains no alphabetic tokens")

    return tuple(lines)


def flatten_plaintext(corpus: PlainCorpus) -> Tuple[str, ...]:
    return tuple(
        char
        for line in corpus
        for token in line
        for char in token
    )


def plaintext_alphabet(corpus: PlainCorpus) -> Tuple[str, ...]:
    return tuple(sorted(set(flatten_plaintext(corpus))))


def plaintext_word_counts(corpus: PlainCorpus) -> Counter:
    return Counter(
        token
        for line in corpus
        for token in line
    )


def plaintext_bigram_counts(corpus: PlainCorpus) -> Counter:
    counts: Counter = Counter()
    for line in corpus:
        for token in line:
            counts.update(
                token[i : i + 2]
                for i in range(len(token) - 1)
            )
    return counts


def count_plaintext(corpus: PlainCorpus) -> dict:
    tokens = [token for line in corpus for token in line]
    return {
        "lines": len(corpus),
        "tokens": len(tokens),
        "characters": sum(len(token) for token in tokens),
        "unique_tokens": len(set(tokens)),
        "alphabet_size": len(
            set(char for token in tokens for char in token)
        ),
    }


def count_ciphertext(corpus: CipherCorpus) -> dict:
    tokens = [token for line in corpus for token in line]
    glyphs = [glyph for token in tokens for glyph in token]
    return {
        "lines": len(corpus),
        "tokens": len(tokens),
        "glyphs": len(glyphs),
        "unique_tokens": len(set(tokens)),
        "alphabet_size": len(set(glyphs)),
    }


def ciphertext_to_text(corpus: CipherCorpus) -> str:
    """Human-readable serialization; '-' joins atomic glyphs within tokens."""
    return "\n".join(
        " ".join("-".join(token) for token in line)
        for line in corpus
    ) + "\n"


def _shuffle_mapping(
    symbols: Sequence[str],
    glyphs: Sequence[str],
    rng: random.Random,
) -> Dict[str, str]:
    if len(symbols) != len(glyphs):
        raise ValueError("symbols and glyphs must have equal length")

    ordered_symbols = list(symbols)
    ordered_glyphs = list(glyphs)
    rng.shuffle(ordered_glyphs)

    return dict(zip(ordered_symbols, ordered_glyphs))


def _mono_key(
    corpus: PlainCorpus,
    allocator: GlyphAllocator,
    rng: random.Random,
) -> Dict[str, str]:
    alphabet = plaintext_alphabet(corpus)
    glyphs = allocator.take(len(alphabet))
    return _shuffle_mapping(alphabet, glyphs, rng)


def _encode_char_mapping(
    corpus: PlainCorpus,
    mapping: Mapping[str, str],
) -> CipherCorpus:
    return tuple(
        tuple(
            tuple(mapping[char] for char in token)
            for token in line
        )
        for line in corpus
    )


def _inverse_single_mapping(
    mapping: Mapping[str, str],
) -> Dict[str, str]:
    inverse = {cipher: plain for plain, cipher in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Mapping is not injective")
    return inverse


def _decode_char_mapping(
    ciphertext: CipherCorpus,
    inverse: Mapping[str, str],
) -> PlainCorpus:
    return tuple(
        tuple(
            "".join(inverse[glyph] for glyph in token)
            for token in line
        )
        for line in ciphertext
    )


def encode_monoalphabetic(
    corpus: PlainCorpus,
    *,
    seed: int,
) -> ControlResult:
    rng = random.Random(seed)
    allocator = GlyphAllocator()
    mapping = _mono_key(corpus, allocator, rng)
    ciphertext = _encode_char_mapping(corpus, mapping)

    return ControlResult(
        family="monoalphabetic",
        seed=seed,
        ciphertext=ciphertext,
        key={
            "mapping": mapping,
            "word_boundaries_preserved": True,
        },
        diagnostics={},
    )


def decode_monoalphabetic(result: ControlResult) -> PlainCorpus:
    inverse = _inverse_single_mapping(result.key["mapping"])
    return _decode_char_mapping(result.ciphertext, inverse)


def encode_homophonic(
    corpus: PlainCorpus,
    *,
    seed: int,
    homophones_per_symbol: int,
) -> ControlResult:
    if homophones_per_symbol < 2:
        raise ValueError("homophones_per_symbol must be >= 2")

    rng = random.Random(seed)
    allocator = GlyphAllocator()
    alphabet = plaintext_alphabet(corpus)

    pools: Dict[str, Tuple[str, ...]] = {}
    for symbol in alphabet:
        pools[symbol] = allocator.take(homophones_per_symbol)

    # Shuffle each pool so seed affects the concrete key as well as draws.
    pools = {
        symbol: tuple(rng.sample(list(pool), len(pool)))
        for symbol, pool in pools.items()
    }

    ciphertext: List[CipherLine] = []
    usage: Counter = Counter()

    for line in corpus:
        out_line: List[CipherToken] = []
        for token in line:
            out_token = []
            for char in token:
                glyph = rng.choice(pools[char])
                out_token.append(glyph)
                usage[glyph] += 1
            out_line.append(tuple(out_token))
        ciphertext.append(tuple(out_line))

    return ControlResult(
        family="homophonic",
        seed=seed,
        ciphertext=tuple(ciphertext),
        key={
            "pools": {
                symbol: list(pool)
                for symbol, pool in pools.items()
            },
            "homophones_per_symbol": homophones_per_symbol,
            "selection": "uniform_per_occurrence",
            "word_boundaries_preserved": True,
        },
        diagnostics={
            "used_cipher_symbols": len(usage),
        },
    )


def decode_homophonic(result: ControlResult) -> PlainCorpus:
    inverse: Dict[str, str] = {}
    for plain, pool in result.key["pools"].items():
        for glyph in pool:
            if glyph in inverse:
                raise ValueError("Homophonic pools overlap")
            inverse[glyph] = plain
    return _decode_char_mapping(result.ciphertext, inverse)


def _unique_codewords(
    item_count: int,
    *,
    codeword_length: int,
    allocator: GlyphAllocator,
    rng: random.Random,
    minimum_base_size: int = 2,
) -> Tuple[Tuple[str, ...], ...]:
    if item_count < 1:
        return ()
    if codeword_length < 1:
        raise ValueError("codeword_length must be >= 1")

    base_size = max(
        minimum_base_size,
        int(math.ceil(item_count ** (1.0 / codeword_length))),
    )

    while base_size ** codeword_length < item_count:
        base_size += 1

    base = allocator.take(base_size)

    codewords: List[Tuple[str, ...]] = [()]
    for _ in range(codeword_length):
        expanded: List[Tuple[str, ...]] = []
        for prefix in codewords:
            for glyph in base:
                expanded.append(prefix + (glyph,))
        codewords = expanded

    rng.shuffle(codewords)
    return tuple(codewords[:item_count])


def encode_verbose(
    corpus: PlainCorpus,
    *,
    seed: int,
    codeword_length: int,
) -> ControlResult:
    if codeword_length < 2:
        raise ValueError("verbose codeword_length must be >= 2")

    rng = random.Random(seed)
    allocator = GlyphAllocator()
    alphabet = plaintext_alphabet(corpus)
    codewords = _unique_codewords(
        len(alphabet),
        codeword_length=codeword_length,
        allocator=allocator,
        rng=rng,
    )
    mapping = dict(zip(alphabet, codewords))

    ciphertext = tuple(
        tuple(
            tuple(
                glyph
                for char in token
                for glyph in mapping[char]
            )
            for token in line
        )
        for line in corpus
    )

    return ControlResult(
        family="verbose",
        seed=seed,
        ciphertext=ciphertext,
        key={
            "mapping": {
                plain: list(codeword)
                for plain, codeword in mapping.items()
            },
            "codeword_length": codeword_length,
            "word_boundaries_preserved": True,
        },
        diagnostics={},
    )


def decode_verbose(result: ControlResult) -> PlainCorpus:
    length = int(result.key["codeword_length"])
    inverse = {
        tuple(codeword): plain
        for plain, codeword in result.key["mapping"].items()
    }

    decoded: List[PlainLine] = []

    for line in result.ciphertext:
        out_line: List[str] = []
        for token in line:
            if len(token) % length != 0:
                raise ValueError(
                    "Verbose ciphertext token length is not divisible by "
                    "codeword length"
                )
            chars = []
            for start in range(0, len(token), length):
                codeword = tuple(token[start : start + length])
                chars.append(inverse[codeword])
            out_line.append("".join(chars))
        decoded.append(tuple(out_line))

    return tuple(decoded)


def encode_null_insertion(
    corpus: PlainCorpus,
    *,
    seed: int,
    insertion_probability: float,
    null_symbol_count: int,
) -> ControlResult:
    if not 0.0 <= insertion_probability < 1.0:
        raise ValueError(
            "insertion_probability must be in [0, 1)"
        )
    if null_symbol_count < 1:
        raise ValueError("null_symbol_count must be >= 1")

    rng = random.Random(seed)
    allocator = GlyphAllocator()
    mapping = _mono_key(corpus, allocator, rng)
    null_symbols = allocator.take(null_symbol_count)

    inserted = 0
    ciphertext: List[CipherLine] = []

    for line in corpus:
        out_line: List[CipherToken] = []
        for token in line:
            out_token: List[str] = []
            for char in token:
                out_token.append(mapping[char])
                if rng.random() < insertion_probability:
                    out_token.append(rng.choice(null_symbols))
                    inserted += 1
            out_line.append(tuple(out_token))
        ciphertext.append(tuple(out_line))

    return ControlResult(
        family="null_insertion",
        seed=seed,
        ciphertext=tuple(ciphertext),
        key={
            "mapping": mapping,
            "null_symbols": list(null_symbols),
            "insertion_probability": insertion_probability,
            "insertion_position": "after_real_glyph",
            "max_nulls_per_real_glyph": 1,
            "word_boundaries_preserved": True,
        },
        diagnostics={
            "inserted_nulls": inserted,
        },
    )


def decode_null_insertion(result: ControlResult) -> PlainCorpus:
    inverse = _inverse_single_mapping(result.key["mapping"])
    nulls = set(result.key["null_symbols"])

    cleaned: CipherCorpus = tuple(
        tuple(
            tuple(
                glyph
                for glyph in token
                if glyph not in nulls
            )
            for token in line
        )
        for line in result.ciphertext
    )
    return _decode_char_mapping(cleaned, inverse)


def encode_nomenclator(
    corpus: PlainCorpus,
    *,
    seed: int,
    top_word_count: int,
    codeword_length: int,
) -> ControlResult:
    if top_word_count < 1:
        raise ValueError("top_word_count must be >= 1")
    if codeword_length < 1:
        raise ValueError("codeword_length must be >= 1")

    rng = random.Random(seed)
    allocator = GlyphAllocator()

    frequencies = plaintext_word_counts(corpus)
    ranked_words = sorted(
        frequencies,
        key=lambda word: (-frequencies[word], word),
    )
    selected_words = ranked_words[:top_word_count]

    # Reserve a disjoint nomenclator glyph alphabet before the fallback
    # alphabetic mapping.
    codewords = _unique_codewords(
        len(selected_words),
        codeword_length=codeword_length,
        allocator=allocator,
        rng=rng,
    )
    word_codes = dict(zip(selected_words, codewords))
    fallback_mapping = _mono_key(corpus, allocator, rng)

    ciphertext: List[CipherLine] = []
    coded_occurrences = 0

    for line in corpus:
        out_line: List[CipherToken] = []
        for token in line:
            if token in word_codes:
                out_line.append(word_codes[token])
                coded_occurrences += 1
            else:
                out_line.append(
                    tuple(fallback_mapping[char] for char in token)
                )
        ciphertext.append(tuple(out_line))

    return ControlResult(
        family="nomenclator",
        seed=seed,
        ciphertext=tuple(ciphertext),
        key={
            "word_codes": {
                word: list(codeword)
                for word, codeword in word_codes.items()
            },
            "fallback_mapping": fallback_mapping,
            "top_word_count_requested": top_word_count,
            "top_word_count_actual": len(selected_words),
            "codeword_length": codeword_length,
            "selection": "highest_source_word_frequency_then_lexicographic",
            "word_boundaries_preserved": True,
        },
        diagnostics={
            "coded_word_occurrences": coded_occurrences,
            "coded_word_types": len(selected_words),
        },
    )


def decode_nomenclator(result: ControlResult) -> PlainCorpus:
    word_inverse = {
        tuple(codeword): word
        for word, codeword in result.key["word_codes"].items()
    }
    char_inverse = _inverse_single_mapping(
        result.key["fallback_mapping"]
    )

    decoded: List[PlainLine] = []

    for line in result.ciphertext:
        out_line: List[str] = []
        for token in line:
            token_tuple = tuple(token)
            if token_tuple in word_inverse:
                out_line.append(word_inverse[token_tuple])
            else:
                out_line.append(
                    "".join(char_inverse[glyph] for glyph in token_tuple)
                )
        decoded.append(tuple(out_line))

    return tuple(decoded)


def _columnar_encrypt(
    sequence: Sequence[str],
    *,
    width: int,
    column_order: Sequence[int],
) -> Tuple[str, ...]:
    if width < 2:
        raise ValueError("width must be >= 2")
    if sorted(column_order) != list(range(width)):
        raise ValueError("column_order must be a permutation of width")

    rows = [
        tuple(sequence[start : start + width])
        for start in range(0, len(sequence), width)
    ]

    output: List[str] = []
    for column in column_order:
        for row in rows:
            if column < len(row):
                output.append(row[column])

    return tuple(output)


def _columnar_decrypt(
    ciphertext: Sequence[str],
    *,
    width: int,
    column_order: Sequence[int],
) -> Tuple[str, ...]:
    n = len(ciphertext)
    if n == 0:
        return ()

    row_count = int(math.ceil(n / width))
    full_columns = n % width
    if full_columns == 0:
        column_lengths = [row_count] * width
    else:
        column_lengths = [
            row_count if column < full_columns else row_count - 1
            for column in range(width)
        ]

    columns: Dict[int, Tuple[str, ...]] = {}
    cursor = 0
    for column in column_order:
        length = column_lengths[column]
        columns[column] = tuple(
            ciphertext[cursor : cursor + length]
        )
        cursor += length

    output: List[str] = []
    for row in range(row_count):
        for column in range(width):
            column_values = columns[column]
            if row < len(column_values):
                output.append(column_values[row])

    if len(output) != n:
        raise RuntimeError("Columnar transposition inverse length mismatch")

    return tuple(output)


def _partition_by_lengths(
    sequence: Sequence[str],
    lengths: Sequence[int],
) -> Tuple[Tuple[str, ...], ...]:
    output = []
    cursor = 0
    for length in lengths:
        output.append(tuple(sequence[cursor : cursor + length]))
        cursor += length
    if cursor != len(sequence):
        raise ValueError("Partition lengths do not consume sequence")
    return tuple(output)


def encode_transposition(
    corpus: PlainCorpus,
    *,
    seed: int,
    width: int,
) -> ControlResult:
    if width < 2:
        raise ValueError("transposition width must be >= 2")

    rng = random.Random(seed)
    allocator = GlyphAllocator()
    mapping = _mono_key(corpus, allocator, rng)

    column_order = list(range(width))
    rng.shuffle(column_order)

    ciphertext: List[CipherLine] = []
    line_token_lengths: List[List[int]] = []

    for line in corpus:
        mapped_tokens = [
            tuple(mapping[char] for char in token)
            for token in line
        ]
        lengths = [len(token) for token in mapped_tokens]
        flat = tuple(
            glyph
            for token in mapped_tokens
            for glyph in token
        )

        transposed = _columnar_encrypt(
            flat,
            width=width,
            column_order=column_order,
        )
        restored_partition = _partition_by_lengths(
            transposed,
            lengths,
        )

        ciphertext.append(restored_partition)
        line_token_lengths.append(lengths)

    return ControlResult(
        family="transposition",
        seed=seed,
        ciphertext=tuple(ciphertext),
        key={
            "mapping": mapping,
            "width": width,
            "column_order": column_order,
            "line_token_lengths": line_token_lengths,
            "transposition_scope": "within_line_after_monoalphabetic_substitution",
            "output_segmentation": (
                "original token lengths restored for benchmark comparability"
            ),
        },
        diagnostics={},
    )


def decode_transposition(result: ControlResult) -> PlainCorpus:
    inverse = _inverse_single_mapping(result.key["mapping"])
    width = int(result.key["width"])
    column_order = list(result.key["column_order"])
    lengths_by_line = result.key["line_token_lengths"]

    decoded: List[PlainLine] = []

    if len(lengths_by_line) != len(result.ciphertext):
        raise ValueError("Missing transposition line metadata")

    for line, lengths in zip(
        result.ciphertext,
        lengths_by_line,
    ):
        flat_cipher = tuple(
            glyph
            for token in line
            for glyph in token
        )
        original_mapped = _columnar_decrypt(
            flat_cipher,
            width=width,
            column_order=column_order,
        )
        mapped_tokens = _partition_by_lengths(
            original_mapped,
            lengths,
        )
        decoded.append(
            tuple(
                "".join(inverse[glyph] for glyph in token)
                for token in mapped_tokens
            )
        )

    return tuple(decoded)


def _selected_bigrams(
    corpus: PlainCorpus,
    *,
    top_bigram_count: int,
    minimum_bigram_count: int,
) -> Tuple[str, ...]:
    if top_bigram_count < 1:
        raise ValueError("top_bigram_count must be >= 1")
    if minimum_bigram_count < 1:
        raise ValueError("minimum_bigram_count must be >= 1")

    counts = plaintext_bigram_counts(corpus)
    eligible = [
        bigram
        for bigram, count in counts.items()
        if count >= minimum_bigram_count
    ]
    eligible.sort(
        key=lambda bigram: (-counts[bigram], bigram)
    )
    return tuple(eligible[:top_bigram_count])


def encode_syllabic_hybrid(
    corpus: PlainCorpus,
    *,
    seed: int,
    top_bigram_count: int,
    minimum_bigram_count: int,
) -> ControlResult:
    rng = random.Random(seed)
    allocator = GlyphAllocator()

    bigrams = _selected_bigrams(
        corpus,
        top_bigram_count=top_bigram_count,
        minimum_bigram_count=minimum_bigram_count,
    )
    bigram_glyphs = allocator.take(len(bigrams))
    shuffled_bigram_glyphs = list(bigram_glyphs)
    rng.shuffle(shuffled_bigram_glyphs)
    bigram_mapping = dict(
        zip(bigrams, shuffled_bigram_glyphs)
    )

    char_mapping = _mono_key(corpus, allocator, rng)

    ciphertext: List[CipherLine] = []
    bigram_occurrences = 0

    for line in corpus:
        out_line: List[CipherToken] = []
        for token in line:
            out_token: List[str] = []
            i = 0
            while i < len(token):
                if i + 1 < len(token):
                    bigram = token[i : i + 2]
                    glyph = bigram_mapping.get(bigram)
                    if glyph is not None:
                        out_token.append(glyph)
                        bigram_occurrences += 1
                        i += 2
                        continue

                out_token.append(char_mapping[token[i]])
                i += 1

            out_line.append(tuple(out_token))
        ciphertext.append(tuple(out_line))

    return ControlResult(
        family="syllabic_hybrid",
        seed=seed,
        ciphertext=tuple(ciphertext),
        key={
            "bigram_mapping": bigram_mapping,
            "char_mapping": char_mapping,
            "top_bigram_count_requested": top_bigram_count,
            "selected_bigram_count": len(bigrams),
            "minimum_bigram_count": minimum_bigram_count,
            "segmentation": (
                "greedy left-to-right frequent-bigram then single-character fallback"
            ),
            "word_boundaries_preserved": True,
        },
        diagnostics={
            "bigram_encoded_occurrences": bigram_occurrences,
        },
    )


def decode_syllabic_hybrid(result: ControlResult) -> PlainCorpus:
    inverse: Dict[str, str] = {}

    for bigram, glyph in result.key["bigram_mapping"].items():
        if glyph in inverse:
            raise ValueError("Hybrid glyph collision")
        inverse[glyph] = bigram

    for char, glyph in result.key["char_mapping"].items():
        if glyph in inverse:
            raise ValueError("Hybrid glyph collision")
        inverse[glyph] = char

    return tuple(
        tuple(
            "".join(inverse[glyph] for glyph in token)
            for token in line
        )
        for line in result.ciphertext
    )


def generate_family(
    corpus: PlainCorpus,
    *,
    family: str,
    seed: int,
    parameters: Mapping[str, object],
) -> ControlResult:
    if family == "monoalphabetic":
        return encode_monoalphabetic(corpus, seed=seed)

    if family == "homophonic":
        return encode_homophonic(
            corpus,
            seed=seed,
            homophones_per_symbol=int(
                parameters["homophones_per_symbol"]
            ),
        )

    if family == "verbose":
        return encode_verbose(
            corpus,
            seed=seed,
            codeword_length=int(parameters["codeword_length"]),
        )

    if family == "null_insertion":
        return encode_null_insertion(
            corpus,
            seed=seed,
            insertion_probability=float(
                parameters["insertion_probability"]
            ),
            null_symbol_count=int(
                parameters["null_symbol_count"]
            ),
        )

    if family == "nomenclator":
        return encode_nomenclator(
            corpus,
            seed=seed,
            top_word_count=int(parameters["top_word_count"]),
            codeword_length=int(parameters["codeword_length"]),
        )

    if family == "transposition":
        return encode_transposition(
            corpus,
            seed=seed,
            width=int(parameters["width"]),
        )

    if family == "syllabic_hybrid":
        return encode_syllabic_hybrid(
            corpus,
            seed=seed,
            top_bigram_count=int(
                parameters["top_bigram_count"]
            ),
            minimum_bigram_count=int(
                parameters["minimum_bigram_count"]
            ),
        )

    raise ValueError(f"Unknown cipher-control family: {family!r}")


def decode_result(result: ControlResult) -> PlainCorpus:
    if result.family == "monoalphabetic":
        return decode_monoalphabetic(result)
    if result.family == "homophonic":
        return decode_homophonic(result)
    if result.family == "verbose":
        return decode_verbose(result)
    if result.family == "null_insertion":
        return decode_null_insertion(result)
    if result.family == "nomenclator":
        return decode_nomenclator(result)
    if result.family == "transposition":
        return decode_transposition(result)
    if result.family == "syllabic_hybrid":
        return decode_syllabic_hybrid(result)

    raise ValueError(
        f"No decoder for family: {result.family!r}"
    )


def verify_round_trip(
    corpus: PlainCorpus,
    result: ControlResult,
) -> None:
    decoded = decode_result(result)
    if decoded != corpus:
        raise AssertionError(
            f"{result.family} failed plaintext round-trip"
        )


def result_summary(
    source_corpus: PlainCorpus,
    result: ControlResult,
) -> dict:
    return {
        "family": result.family,
        "seed": result.seed,
        "plaintext": count_plaintext(source_corpus),
        "ciphertext": count_ciphertext(result.ciphertext),
        "diagnostics": result.diagnostics,
        "round_trip_verified": True,
    }
