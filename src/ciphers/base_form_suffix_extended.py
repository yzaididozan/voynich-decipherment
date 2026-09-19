"""Phase 3A.4 extended base-form-preserving suffix variation for VOYAGER.

Intended repository destination:
    src/ciphers/base_form_suffix_extended.py

This experiment is an explicit extension of frozen Phase 3A.3.

The scientific intervention holds the complete monoalphabetic base form,
coverage, token length, occurrence-level suffix-choice rule, source splits,
languages, seeds, and Level-2 evaluator fixed while extending suffix
multiplicity:

    K = 4, 8, 16

Bridge invariant
----------------
K=4 must reproduce frozen Phase-3A.3 K=4 ciphertext byte-for-byte. To enforce
that structurally, this module derives the monoalphabetic base mapping and the
first four suffixes directly from the Phase-3A.3 TRAIN-only master-key
constructor, then appends twelve new suffixes without altering those legacy
values.

The exact Phase-3A.3 occurrence choice remains:

    stable_uint64(
        "base-form-suffix-choice-v1",
        seed,
        line_index,
        token_index,
    ) % K

Plaintext token identity is deliberately excluded.

Reserved-test source lines are never encoded. Their JSONL rows remain opaque
placeholders so frozen line indices stay compatible with the Level-2 reader.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from src.ciphers import base_form_suffix_variation as phase3a3


GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]


@dataclass(frozen=True)
class GeneratedExtendedSuffixControl:
    rows: Tuple[dict, ...]
    key: dict
    summary: dict


def sha256_file(path: Path) -> str:
    return phase3a3.sha256_file(path)


def stable_uint64(*parts: object) -> int:
    """Use the exact frozen Phase-3A.3 stable-hash implementation."""
    return phase3a3.stable_uint64(*parts)


def read_source_lines(path: Path) -> List[PlainLine]:
    return phase3a3.read_source_lines(path)


def token_count(lines: Sequence[PlainLine]) -> int:
    return phase3a3.token_count(lines)


def validate_partition(
    line_count: int,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
) -> None:
    phase3a3.validate_partition(
        line_count,
        train_indices,
        validation_indices,
        reserved_indices,
    )


def build_master_key(
    train_lines: Sequence[PlainLine],
    *,
    seed: int,
    master_suffix_count: int = 16,
) -> dict:
    """Build an extended key whose K=4 prefix exactly matches Phase 3A.3."""
    if master_suffix_count != 16:
        raise ValueError(
            "Phase 3A.4 v1 freezes master_suffix_count to exactly 16"
        )

    # This is the critical bridge: base mapping + first four suffixes are
    # produced by the frozen Phase-3A.3 implementation itself.
    parent = phase3a3.build_master_key(
        train_lines,
        seed=int(seed),
        master_suffix_count=4,
    )

    legacy_suffixes = list(parent["master_suffixes"])
    if len(legacy_suffixes) != 4:
        raise RuntimeError(
            "Phase-3A.3 parent key did not contain exactly four suffixes"
        )

    expected_legacy_names = {
        "V0001",
        "V0002",
        "V0003",
        "V0004",
    }
    if set(legacy_suffixes) != expected_legacy_names:
        raise RuntimeError(
            "Unexpected Phase-3A.3 suffix namespace; bridge cannot be guaranteed"
        )

    # New suffix order is generated independently of the legacy RNG sequence.
    # Consequently it cannot alter the first four suffixes or the base mapping.
    extension_candidates = [
        f"V{i:04d}"
        for i in range(5, 17)
    ]
    extension_suffixes = sorted(
        extension_candidates,
        key=lambda suffix: (
            stable_uint64(
                "base-form-suffix-extension-v1",
                int(seed),
                suffix,
            ),
            suffix,
        ),
    )

    master_suffixes = [
        *legacy_suffixes,
        *extension_suffixes,
    ]
    if len(master_suffixes) != 16:
        raise RuntimeError("Extended suffix inventory is not length 16")
    if len(set(master_suffixes)) != 16:
        raise RuntimeError("Extended suffix inventory contains duplicates")

    fingerprint_payload = {
        "seed": int(seed),
        "mono_forward": parent["mono_forward"],
        "master_suffixes": master_suffixes,
        "parent_master_key_sha256": parent["master_key_sha256"],
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
        "mechanism": "base_form_suffix_extended",
        "fit_scope": "CONTROL TRAIN ONLY",
        "seed": int(seed),
        "mono_forward": dict(parent["mono_forward"]),
        "mono_reverse": dict(parent["mono_reverse"]),
        "train_character_alphabet": list(
            parent["train_character_alphabet"]
        ),
        "master_suffix_count": 16,
        "master_suffixes": master_suffixes,
        "legacy_phase3a3_suffixes": legacy_suffixes,
        "extension_suffixes": extension_suffixes,
        "parent_phase3a3_master_key_sha256": parent[
            "master_key_sha256"
        ],
        "master_key_sha256": master_key_sha256,
        "suffix_choice_identity_inputs": [
            "seed",
            "line_index",
            "token_index",
        ],
        "suffix_choice_excludes_plaintext_token_identity": True,
        "suffix_choice_hash_namespace": "base-form-suffix-choice-v1",
        "validation_unseen_character_policy": parent[
            "validation_unseen_character_policy"
        ],
        "bridge_statement": (
            "Base mapping and first four suffixes are inherited exactly from "
            "Phase 3A.3; K=4 therefore uses the same token-generation rule."
        ),
    }


def encode_base(
    token: str,
    *,
    key: Mapping[str, object],
) -> GlyphToken:
    """Encode a base using the exact Phase-3A.3 base representation."""
    return phase3a3.encode_base(
        token,
        key=key,
    )


def active_suffixes(
    key: Mapping[str, object],
    *,
    variant_count: int,
) -> Tuple[str, ...]:
    if variant_count not in (4, 8, 16):
        raise ValueError(
            "Phase 3A.4 variant_count must be one of 4, 8, 16"
        )

    suffixes = tuple(key["master_suffixes"])
    if len(suffixes) != 16:
        raise ValueError(
            "Extended master key must contain exactly sixteen suffixes"
        )

    return suffixes[: int(variant_count)]


def suffix_choice_index(
    *,
    occurrence_seed: int,
    line_index: int,
    token_index: int,
    variant_count: int,
) -> int:
    """Use the exact Phase-3A.3 hash namespace and input tuple."""
    if variant_count not in (4, 8, 16):
        raise ValueError(
            "Phase 3A.4 variant_count must be one of 4, 8, 16"
        )

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
    """Decode through the frozen Phase-3A.3 base decoder."""
    return phase3a3.decode_base(
        base,
        key=key,
    )


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

    for word_counter in surfaces_by_word.values():
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

    entropy = _shannon_entropy_bits(
        suffix_counts
    )

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
        "suffix_entropy_bits": entropy,
        "suffix_entropy_fraction_of_max": (
            entropy / math.log2(int(variant_count))
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
    master_suffix_count: int = 16,
) -> GeneratedExtendedSuffixControl:
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
        "mechanism": "base_form_suffix_extended",
        "status": "exploratory",
        "seed": int(seed),
        "variant_count": int(variant_count),
        "master_suffix_count": 16,
        "coverage": "all_train_and_validation_tokens",
        "suffix_operation": "append_exactly_one_suffix_glyph_to_every_token",
        "suffix_choice_hash_namespace": "base-form-suffix-choice-v1",
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
        "parent_phase3a3_master_key_sha256": key[
            "parent_phase3a3_master_key_sha256"
        ],
        "train_diagnostics": train_diag,
        "validation_diagnostics": validation_diag,
        "guardrails": [
            "Base mapping and first four suffixes are inherited exactly from Phase 3A.3.",
            "K=4 uses the exact Phase-3A.3 occurrence hash namespace and modulo rule.",
            "K=8 and K=16 only extend suffix multiplicity.",
            "Every TRAIN/VALIDATION token preserves its entire monoencoded base.",
            "Every ciphertext token receives exactly one suffix.",
            "Suffix choice excludes plaintext token identity.",
            "Validation cannot fit or modify the base-character mapping.",
            "Reserved-test rows are opaque placeholders and are not encoded.",
        ],
    }

    return GeneratedExtendedSuffixControl(
        rows=tuple(rows),
        key=output_key,
        summary=summary,
    )


def write_generated_control(
    output_dir: Path,
    generated: GeneratedExtendedSuffixControl,
    *,
    provenance: Mapping[str, object],
) -> dict:
    """Write in the exact corpus JSONL format used by Phase 3A.3."""
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
