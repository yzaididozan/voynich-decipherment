"""Phase 3A.3 base-form-preserving suffix variation for VOYAGER.

Intended repository destination:
    src/ciphers/base_form_suffix_variation.py

Scientific intervention
-----------------------
Every plaintext token is first encoded with a TRAIN-fitted monoalphabetic
character substitution. That complete encoded base is preserved. Exactly one
suffix glyph is then appended to every token.

The exploratory grid changes only suffix multiplicity:

    K = 1, 2, 4

All K conditions have identical token length: encoded-base length + 1.
A single four-glyph master suffix ordering and a single base-character mapping
are fit for each source/seed from CONTROL TRAIN only. K activates a prefix of
the same master suffix ordering.

Suffix choice is occurrence-level and deliberately excludes plaintext token
identity:

    hash(seed, line_index, token_index) mod K

Thus the suffix does not function as an additional lexical code. Repeated
occurrences of the same plaintext word can become distinct whole ciphertext
tokens while retaining an identical complete base form.

Reserved-test source lines are never encoded. Their JSONL rows are opaque
placeholders so frozen line indices remain compatible with the Level-2 reader.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Dict, List, Mapping, Sequence, Tuple


GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]


@dataclass(frozen=True)
class GeneratedBaseFormSuffixControl:
    rows: Tuple[dict, ...]
    key: dict
    summary: dict


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_uint64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int.from_bytes(
        sha256(payload.encode("utf-8")).digest()[:8],
        "big",
    )


def read_source_lines(path: Path) -> List[PlainLine]:
    """Read the frozen normalized source without renormalizing it."""
    lines: List[PlainLine] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.rstrip("\r\n")
            if not text.strip():
                raise ValueError(
                    f"{path}:{line_number}: empty source line is not allowed"
                )
            tokens = tuple(text.split())
            if not tokens:
                raise ValueError(
                    f"{path}:{line_number}: source line has no tokens"
                )
            lines.append(tokens)

    if not lines:
        raise ValueError(f"Source is empty: {path}")

    return lines


def token_count(lines: Sequence[PlainLine]) -> int:
    return sum(len(line) for line in lines)


def validate_partition(
    line_count: int,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
) -> None:
    sets = {
        "train": set(int(x) for x in train_indices),
        "validation": set(int(x) for x in validation_indices),
        "reserved": set(int(x) for x in reserved_indices),
    }

    for name, values in sets.items():
        bad = sorted(
            x for x in values
            if x < 1 or x > line_count
        )
        if bad:
            raise ValueError(
                f"{name} contains out-of-range line indices: {bad[:10]}"
            )

    names = list(sets)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1 :]:
            overlap = sets[left_name] & sets[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name}/{right_name} overlap: "
                    f"{sorted(overlap)[:10]}"
                )

    observed = set().union(*sets.values())
    expected = set(range(1, line_count + 1))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "Split does not cover source lines exactly; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def _train_character_alphabet(
    train_lines: Sequence[PlainLine],
) -> List[str]:
    alphabet = sorted(
        {
            char
            for line in train_lines
            for token in line
            for char in token
        }
    )
    if not alphabet:
        raise ValueError("TRAIN character alphabet is empty")
    return alphabet


def build_master_key(
    train_lines: Sequence[PlainLine],
    *,
    seed: int,
    master_suffix_count: int = 4,
) -> dict:
    """Fit the base substitution and four-suffix master key on TRAIN only."""
    if master_suffix_count != 4:
        raise ValueError(
            "Phase 3A.3 v1 freezes master_suffix_count to exactly 4"
        )

    alphabet = _train_character_alphabet(train_lines)
    rng = random.Random(int(seed))

    mono_glyphs = [
        f"M{i:04d}"
        for i in range(1, len(alphabet) + 1)
    ]
    rng.shuffle(mono_glyphs)

    mono_forward = dict(zip(alphabet, mono_glyphs))
    mono_reverse = {
        glyph: char
        for char, glyph in mono_forward.items()
    }

    master_suffixes = [
        f"V{i:04d}"
        for i in range(1, master_suffix_count + 1)
    ]
    rng.shuffle(master_suffixes)

    fingerprint_payload = {
        "seed": int(seed),
        "mono_forward": mono_forward,
        "master_suffixes": master_suffixes,
    }
    master_key_sha256 = sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": "1.0",
        "mechanism": "base_form_suffix_variation",
        "fit_scope": "CONTROL TRAIN ONLY",
        "seed": int(seed),
        "mono_forward": mono_forward,
        "mono_reverse": mono_reverse,
        "train_character_alphabet": alphabet,
        "master_suffix_count": int(master_suffix_count),
        "master_suffixes": master_suffixes,
        "master_key_sha256": master_key_sha256,
        "suffix_choice_identity_inputs": [
            "seed",
            "line_index",
            "token_index",
        ],
        "suffix_choice_excludes_plaintext_token_identity": True,
        "validation_unseen_character_policy": (
            "U<hex-codepoint>, deterministic and not fit from validation"
        ),
    }


def _unicode_oov_glyph(char: str) -> str:
    return f"U{ord(char):08X}"


def _decode_unicode_oov_glyph(glyph: str) -> str:
    if not glyph.startswith("U") or len(glyph) != 9:
        raise ValueError(f"Invalid Unicode OOV glyph: {glyph!r}")
    return chr(int(glyph[1:], 16))


def encode_base(
    token: str,
    *,
    key: Mapping[str, object],
) -> GlyphToken:
    if not token:
        raise ValueError("Cannot encode empty token")

    mono_forward = key["mono_forward"]
    encoded = []

    for char in token:
        glyph = mono_forward.get(char)
        if glyph is None:
            glyph = _unicode_oov_glyph(char)
        encoded.append(glyph)

    return tuple(encoded)


def active_suffixes(
    key: Mapping[str, object],
    *,
    variant_count: int,
) -> Tuple[str, ...]:
    if variant_count not in (1, 2, 4):
        raise ValueError("variant_count must be one of 1, 2, 4")

    suffixes = tuple(key["master_suffixes"])
    if len(suffixes) != 4:
        raise ValueError("Master key must contain exactly four suffixes")

    return suffixes[: int(variant_count)]


def suffix_choice_index(
    *,
    occurrence_seed: int,
    line_index: int,
    token_index: int,
    variant_count: int,
) -> int:
    """Choose a suffix without using plaintext token identity."""
    if variant_count not in (1, 2, 4):
        raise ValueError("variant_count must be one of 1, 2, 4")

    return stable_uint64(
        "base-form-suffix-choice-v1",
        int(occurrence_seed),
        int(line_index),
        int(token_index),
    ) % int(variant_count)


def encode_token(
    token: str,
    *,
    key: Mapping[str, object],
    variant_count: int,
    occurrence_seed: int,
    line_index: int,
    token_index: int,
) -> GlyphToken:
    base = encode_base(
        token,
        key=key,
    )
    suffixes = active_suffixes(
        key,
        variant_count=int(variant_count),
    )
    choice = suffix_choice_index(
        occurrence_seed=int(occurrence_seed),
        line_index=int(line_index),
        token_index=int(token_index),
        variant_count=int(variant_count),
    )
    return (*base, suffixes[choice])


def strip_suffix(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
) -> GlyphToken:
    token = tuple(cipher_token)
    if len(token) < 2:
        raise ValueError(
            "Cipher token must contain at least one base glyph and one suffix"
        )

    suffix = token[-1]
    if suffix not in set(key["master_suffixes"]):
        raise ValueError(
            f"Cipher token has invalid suffix glyph: {suffix!r}"
        )

    return token[:-1]


def decode_base(
    base: Sequence[str],
    *,
    key: Mapping[str, object],
) -> str:
    mono_reverse = key["mono_reverse"]
    chars = []

    for glyph in base:
        if glyph in mono_reverse:
            chars.append(mono_reverse[glyph])
        elif glyph.startswith("U"):
            chars.append(_decode_unicode_oov_glyph(glyph))
        else:
            raise ValueError(
                f"Base contains undecodable glyph {glyph!r}"
            )

    return "".join(chars)


def decode_token(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
) -> str:
    return decode_base(
        strip_suffix(
            cipher_token,
            key=key,
        ),
        key=key,
    )


def _shannon_entropy_bits(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _diagnostics(
    plaintext_by_line: Mapping[int, PlainLine],
    ciphertext_by_line: Mapping[int, Sequence[GlyphToken]],
    *,
    key: Mapping[str, object],
    variant_count: int,
) -> dict:
    total_tokens = 0
    cipher_counts: Counter[GlyphToken] = Counter()
    suffix_counts: Counter[str] = Counter()
    glyph_counts: Counter[str] = Counter()

    surfaces_by_word: Dict[str, Counter[GlyphToken]] = defaultdict(Counter)
    bases_by_word: Dict[str, set[GlyphToken]] = defaultdict(set)

    decoded_ok = 0
    base_ok = 0
    length_ok = 0

    for line_index in sorted(plaintext_by_line):
        plain_line = plaintext_by_line[line_index]
        cipher_line = ciphertext_by_line[line_index]

        if len(plain_line) != len(cipher_line):
            raise RuntimeError("Token-count mismatch in diagnostics")

        for plaintext, cipher_token in zip(
            plain_line,
            cipher_line,
        ):
            total_tokens += 1
            cipher_tuple = tuple(cipher_token)
            cipher_counts[cipher_tuple] += 1
            glyph_counts.update(cipher_tuple)
            suffix_counts[cipher_tuple[-1]] += 1
            surfaces_by_word[plaintext][cipher_tuple] += 1

            expected_base = encode_base(
                plaintext,
                key=key,
            )
            observed_base = strip_suffix(
                cipher_tuple,
                key=key,
            )
            bases_by_word[plaintext].add(
                observed_base
            )

            if observed_base == expected_base:
                base_ok += 1

            if len(cipher_tuple) == len(expected_base) + 1:
                length_ok += 1

            if decode_token(
                cipher_tuple,
                key=key,
            ) == plaintext:
                decoded_ok += 1

    if decoded_ok != total_tokens:
        raise RuntimeError(
            f"Round-trip failure: {decoded_ok}/{total_tokens}"
        )
    if base_ok != total_tokens:
        raise RuntimeError(
            f"Base-form preservation failure: {base_ok}/{total_tokens}"
        )
    if length_ok != total_tokens:
        raise RuntimeError(
            f"Token-length control failure: {length_ok}/{total_tokens}"
        )

    same_surface_numerator = 0
    same_plaintext_pair_denominator = 0
    realized_surface_counts = []

    for plaintext, word_counter in surfaces_by_word.items():
        n = sum(word_counter.values())
        realized_surface_counts.append(
            len(word_counter)
        )
        if n >= 2:
            same_plaintext_pair_denominator += n * (n - 1)
            same_surface_numerator += sum(
                count * (count - 1)
                for count in word_counter.values()
            )

    same_surface_probability = (
        same_surface_numerator
        / same_plaintext_pair_denominator
        if same_plaintext_pair_denominator
        else None
    )

    words_with_multiple_bases = sum(
        len(bases) != 1
        for bases in bases_by_word.values()
    )

    active = active_suffixes(
        key,
        variant_count=int(variant_count),
    )
    active_suffix_counts = {
        suffix: int(suffix_counts.get(suffix, 0))
        for suffix in active
    }

    return {
        "tokens": total_tokens,
        "token_coverage_fraction": 1.0 if total_tokens else 0.0,
        "unique_plaintext_word_types": len(surfaces_by_word),
        "unique_cipher_tokens": len(cipher_counts),
        "exact_cipher_token_recurrence_fraction": (
            (total_tokens - len(cipher_counts)) / total_tokens
            if total_tokens else 0.0
        ),
        "cipher_glyph_alphabet_size": len(glyph_counts),
        "round_trip_accuracy": (
            decoded_ok / total_tokens
            if total_tokens else 1.0
        ),
        "base_form_recovery_accuracy": (
            base_ok / total_tokens
            if total_tokens else 1.0
        ),
        "base_form_preservation_fraction": (
            base_ok / total_tokens
            if total_tokens else 1.0
        ),
        "plaintext_types_with_multiple_observed_base_forms": (
            words_with_multiple_bases
        ),
        "token_length_plus_one_accuracy": (
            length_ok / total_tokens
            if total_tokens else 1.0
        ),
        "mean_realized_surface_forms_per_plaintext_word_present": (
            sum(realized_surface_counts) / len(realized_surface_counts)
            if realized_surface_counts else 0.0
        ),
        "max_realized_surface_forms_per_plaintext_word": (
            max(realized_surface_counts)
            if realized_surface_counts else 0
        ),
        "same_surface_probability_given_same_plaintext_word": (
            same_surface_probability
        ),
        "active_suffix_count": int(variant_count),
        "active_suffix_counts_json": json.dumps(
            active_suffix_counts,
            sort_keys=True,
        ),
        "suffix_entropy_bits": _shannon_entropy_bits(
            suffix_counts
        ),
        "suffix_entropy_fraction_of_max": (
            _shannon_entropy_bits(suffix_counts)
            / math.log2(int(variant_count))
            if int(variant_count) > 1 and total_tokens
            else 1.0
        ),
    }


def generate_control(
    source_lines: Sequence[PlainLine],
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
    seed: int,
    variant_count: int,
    master_suffix_count: int = 4,
) -> GeneratedBaseFormSuffixControl:
    """Generate one K/seed control without encoding reserved-test lines."""
    line_count = len(source_lines)
    validate_partition(
        line_count,
        train_indices,
        validation_indices,
        reserved_indices,
    )

    train_set = set(int(x) for x in train_indices)
    validation_set = set(int(x) for x in validation_indices)
    reserved_set = set(int(x) for x in reserved_indices)

    train_lines = [
        source_lines[i - 1]
        for i in sorted(train_set)
    ]

    key = build_master_key(
        train_lines,
        seed=int(seed),
        master_suffix_count=int(master_suffix_count),
    )

    # Validate K against the frozen master key before encoding anything.
    active = active_suffixes(
        key,
        variant_count=int(variant_count),
    )

    rows: List[dict] = []
    plaintext_train: Dict[int, PlainLine] = {}
    plaintext_validation: Dict[int, PlainLine] = {}
    cipher_train: Dict[int, List[GlyphToken]] = {}
    cipher_validation: Dict[int, List[GlyphToken]] = {}

    for line_index, plain_line in enumerate(source_lines, start=1):
        if line_index in reserved_set:
            rows.append(
                {
                    "line_index": line_index,
                    "reserved_test_opaque": True,
                }
            )
            continue

        cipher_line = [
            encode_token(
                plaintext,
                key=key,
                variant_count=int(variant_count),
                occurrence_seed=int(seed),
                line_index=line_index,
                token_index=token_index,
            )
            for token_index, plaintext in enumerate(plain_line)
        ]

        rows.append(
            {
                "line_index": line_index,
                "tokens": [
                    list(token)
                    for token in cipher_line
                ],
            }
        )

        if line_index in train_set:
            plaintext_train[line_index] = plain_line
            cipher_train[line_index] = cipher_line
        elif line_index in validation_set:
            plaintext_validation[line_index] = plain_line
            cipher_validation[line_index] = cipher_line
        else:
            raise RuntimeError("Line belongs to no split")

    train_diag = _diagnostics(
        plaintext_train,
        cipher_train,
        key=key,
        variant_count=int(variant_count),
    )
    validation_diag = _diagnostics(
        plaintext_validation,
        cipher_validation,
        key=key,
        variant_count=int(variant_count),
    )

    output_key = dict(key)
    output_key["variant_count"] = int(variant_count)
    output_key["active_suffixes"] = list(active)

    summary = {
        "schema_version": "1.0",
        "mechanism": "base_form_suffix_variation",
        "status": "exploratory",
        "seed": int(seed),
        "variant_count": int(variant_count),
        "master_suffix_count": int(master_suffix_count),
        "coverage": "all_train_and_validation_tokens",
        "suffix_operation": "append_exactly_one_suffix_glyph_to_every_token",
        "suffix_choice_excludes_plaintext_token_identity": True,
        "fit_scope": "CONTROL TRAIN ONLY",
        "evaluation_scope": "CONTROL VALIDATION ONLY",
        "train_lines": len(train_set),
        "validation_lines": len(validation_set),
        "reserved_lines": len(reserved_set),
        "reserved_test_encoded": False,
        "round_trip_verified_train_validation": True,
        "base_form_preservation_verified_train_validation": True,
        "token_length_control_verified_train_validation": True,
        "train_diagnostics": train_diag,
        "validation_diagnostics": validation_diag,
        "guardrails": [
            "Base substitution and four-suffix master order are fit from CONTROL TRAIN only.",
            "Every TRAIN/VALIDATION token preserves its entire monoencoded base.",
            "Every ciphertext token receives exactly one suffix at K=1,2,4.",
            "K activates a prefix of one shared four-suffix master ordering.",
            "Suffix choice uses seed, line index, and token index but not plaintext token identity.",
            "Validation cannot fit or modify the base-character mapping.",
            "Reserved-test rows are opaque placeholders and are not encoded.",
        ],
    }

    return GeneratedBaseFormSuffixControl(
        rows=tuple(rows),
        key=output_key,
        summary=summary,
    )


def write_generated_control(
    output_dir: Path,
    generated: GeneratedBaseFormSuffixControl,
    *,
    provenance: Mapping[str, object],
) -> dict:
    """Write one generated control and return artifact hashes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    key_path = output_dir / "key.json"
    summary_path = output_dir / "summary.json"

    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in generated.rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    key_payload = dict(generated.key)
    key_payload["provenance"] = dict(provenance)
    key_path.write_text(
        json.dumps(
            key_payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_payload = dict(generated.summary)
    summary_payload["provenance"] = dict(provenance)
    summary_path.write_text(
        json.dumps(
            summary_payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "corpus.jsonl": sha256_file(corpus_path),
        "key.json": sha256_file(key_path),
        "summary.json": sha256_file(summary_path),
    }

    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{digest}  {name}"
            for name, digest in artifacts.items()
        )
        + "\n",
        encoding="utf-8",
    )

    return artifacts
