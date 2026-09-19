"""Phase 3A.5 compositional affix factorization for VOYAGER.

Intended repository destination:
    src/ciphers/compositional_affix_factorization.py

Scientific intervention
-----------------------
Phase 3A.4 reached a positive global token-Markov-minus-trigram mean at K=16
but reproduced the frozen strong token-Markov relation in only 2/4 languages.

Phase 3A.5 holds the following fixed:
    * complete TRAIN-fitted monoalphabetic base form
    * 100% TRAIN/VALIDATION coverage
    * exactly 16 whole-token variants
    * exactly two appended affix glyphs
    * exact Phase-3A.4 K=16 occurrence-level variant-index schedule
    * source splits, languages, seeds, and Level-2 evaluator

It changes only how the 16 variant identities are represented as two slots:

    1 x 16
    2 x 8
    4 x 4

For a factorization R x C and variant index i in [0, 15]:

    slot_a_index = i // C
    slot_b_index = i % C

Every condition therefore has a bijection between the same 16 abstract
variant identities and 16 exact two-glyph affix pairs.

Important interpretation note
-----------------------------
With uniform occurrence-level variant indices, all three factorizations carry
the same nominal four bits of variant identity. This experiment tests whether
the *representation geometry* of those identities changes finite-order/local
predictability. It does not assume that factorization reduces the underlying
variant entropy.

Reserved-test source lines are never encoded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from src.ciphers import base_form_suffix_extended as phase3a4


GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]

FACTORIZATIONS = {
    "1x16": (1, 16),
    "2x8": (2, 8),
    "4x4": (4, 4),
}


@dataclass(frozen=True)
class GeneratedCompositionalAffixControl:
    rows: Tuple[dict, ...]
    key: dict
    summary: dict


def sha256_file(path: Path) -> str:
    return phase3a4.sha256_file(path)


def stable_uint64(*parts: object) -> int:
    return phase3a4.stable_uint64(*parts)


def read_source_lines(path: Path) -> List[PlainLine]:
    return phase3a4.read_source_lines(path)


def token_count(lines: Sequence[PlainLine]) -> int:
    return phase3a4.token_count(lines)


def validate_partition(
    line_count: int,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
) -> None:
    phase3a4.validate_partition(
        line_count,
        train_indices,
        validation_indices,
        reserved_indices,
    )


def factorization_spec(label: str) -> Tuple[int, int]:
    try:
        rows, cols = FACTORIZATIONS[label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown factorization {label!r}; expected one of "
            f"{sorted(FACTORIZATIONS)}"
        ) from exc

    if rows * cols != 16:
        raise RuntimeError(
            f"Invalid factorization {label}: {rows}x{cols} != 16"
        )
    return rows, cols


def build_master_key(
    train_lines: Sequence[PlainLine],
    *,
    seed: int,
) -> dict:
    """Inherit the exact Phase-3A.4 base mapping and create factor orders."""
    parent = phase3a4.build_master_key(
        train_lines,
        seed=int(seed),
        master_suffix_count=16,
    )

    parent_sha = parent["master_key_sha256"]

    slot_a_candidates = [
        f"A{i:04d}"
        for i in range(1, 5)
    ]
    slot_b_candidates = [
        f"B{i:04d}"
        for i in range(1, 17)
    ]

    slot_a_order = sorted(
        slot_a_candidates,
        key=lambda glyph: (
            stable_uint64(
                "compositional-affix-slot-a-order-v1",
                int(seed),
                parent_sha,
                glyph,
            ),
            glyph,
        ),
    )
    slot_b_order = sorted(
        slot_b_candidates,
        key=lambda glyph: (
            stable_uint64(
                "compositional-affix-slot-b-order-v1",
                int(seed),
                parent_sha,
                glyph,
            ),
            glyph,
        ),
    )

    payload = {
        "seed": int(seed),
        "parent_phase3a4_master_key_sha256": parent_sha,
        "mono_forward": parent["mono_forward"],
        "slot_a_order": slot_a_order,
        "slot_b_order": slot_b_order,
    }
    master_key_sha256 = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": "1.0",
        "mechanism": "compositional_affix_factorization",
        "fit_scope": "CONTROL TRAIN ONLY",
        "seed": int(seed),
        "mono_forward": dict(parent["mono_forward"]),
        "mono_reverse": dict(parent["mono_reverse"]),
        "train_character_alphabet": list(
            parent["train_character_alphabet"]
        ),
        "parent_phase3a4_master_key_sha256": parent_sha,
        "parent_phase3a4_master_suffixes": list(
            parent["master_suffixes"]
        ),
        "slot_a_order": slot_a_order,
        "slot_b_order": slot_b_order,
        "whole_token_variant_count": 16,
        "suffix_length_glyphs": 2,
        "master_key_sha256": master_key_sha256,
        "variant_choice_hash_namespace": "base-form-suffix-choice-v1",
        "variant_choice_inputs": [
            "seed",
            "line_index",
            "token_index",
        ],
        "variant_choice_excludes_plaintext_token_identity": True,
        "validation_unseen_character_policy": parent[
            "validation_unseen_character_policy"
        ],
        "information_note": (
            "Each factorization is a bijective representation of the same "
            "16 abstract variant identities."
        ),
    }


def encode_base(
    token: str,
    *,
    key: Mapping[str, object],
) -> GlyphToken:
    return phase3a4.encode_base(
        token,
        key=key,
    )


def variant_index_choice(
    *,
    occurrence_seed: int,
    line_index: int,
    token_index: int,
) -> int:
    """Use the exact Phase-3A.4 K=16 occurrence schedule."""
    return phase3a4.suffix_choice_index(
        occurrence_seed=int(occurrence_seed),
        line_index=int(line_index),
        token_index=int(token_index),
        variant_count=16,
    )


def active_factor_states(
    key: Mapping[str, object],
    *,
    factorization: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    rows, cols = factorization_spec(
        factorization
    )

    slot_a_order = tuple(
        key["slot_a_order"]
    )
    slot_b_order = tuple(
        key["slot_b_order"]
    )

    if len(slot_a_order) != 4:
        raise ValueError(
            "Master key must contain exactly four slot-A states"
        )
    if len(slot_b_order) != 16:
        raise ValueError(
            "Master key must contain exactly sixteen slot-B states"
        )

    return (
        slot_a_order[:rows],
        slot_b_order[:cols],
    )


def variant_pair(
    key: Mapping[str, object],
    *,
    factorization: str,
    variant_index: int,
) -> Tuple[str, str]:
    rows, cols = factorization_spec(
        factorization
    )
    if variant_index < 0 or variant_index >= 16:
        raise ValueError(
            "variant_index must be in [0, 15]"
        )

    slot_a, slot_b = active_factor_states(
        key,
        factorization=factorization,
    )

    a_index = int(variant_index) // cols
    b_index = int(variant_index) % cols

    if a_index >= rows:
        raise RuntimeError(
            "Variant mapping exceeded active slot-A states"
        )

    return (
        slot_a[a_index],
        slot_b[b_index],
    )


def decode_variant_index(
    affix_pair: Sequence[str],
    *,
    key: Mapping[str, object],
    factorization: str,
) -> int:
    pair = tuple(affix_pair)
    if len(pair) != 2:
        raise ValueError(
            "Compositional affix must contain exactly two glyphs"
        )

    rows, cols = factorization_spec(
        factorization
    )
    slot_a, slot_b = active_factor_states(
        key,
        factorization=factorization,
    )

    try:
        a_index = slot_a.index(pair[0])
        b_index = slot_b.index(pair[1])
    except ValueError as exc:
        raise ValueError(
            f"Affix pair {pair!r} is not valid for {factorization}"
        ) from exc

    index = a_index * cols + b_index
    if index >= rows * cols:
        raise RuntimeError(
            "Decoded variant index exceeded factorization size"
        )
    return index


def encode_token(
    token: str,
    *,
    key: Mapping[str, object],
    factorization: str,
    occurrence_seed: int,
    line_index: int,
    token_index: int,
) -> GlyphToken:
    base = encode_base(
        token,
        key=key,
    )
    variant_index = variant_index_choice(
        occurrence_seed=int(occurrence_seed),
        line_index=int(line_index),
        token_index=int(token_index),
    )
    pair = variant_pair(
        key,
        factorization=factorization,
        variant_index=variant_index,
    )
    return (*base, *pair)


def strip_affix(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
    factorization: str,
) -> GlyphToken:
    token = tuple(cipher_token)
    if len(token) < 3:
        raise ValueError(
            "Cipher token must contain at least one base glyph plus two affix glyphs"
        )

    decode_variant_index(
        token[-2:],
        key=key,
        factorization=factorization,
    )
    return token[:-2]


def decode_base(
    base: Sequence[str],
    *,
    key: Mapping[str, object],
) -> str:
    return phase3a4.decode_base(
        base,
        key=key,
    )


def decode_token(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
    factorization: str,
) -> str:
    return decode_base(
        strip_affix(
            cipher_token,
            key=key,
            factorization=factorization,
        ),
        key=key,
    )


def _shannon_entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _abstract_identity_sha256(
    plaintext_by_line: Mapping[int, PlainLine],
    ciphertext_by_line: Mapping[int, Sequence[GlyphToken]],
    *,
    key: Mapping[str, object],
    factorization: str,
) -> str:
    """Hash base form + abstract variant index, independent of factor glyphs."""
    digest = sha256()

    for line_index in sorted(plaintext_by_line):
        cipher_line = ciphertext_by_line[line_index]
        for token_index, cipher_token in enumerate(
            cipher_line
        ):
            base = strip_affix(
                cipher_token,
                key=key,
                factorization=factorization,
            )
            variant_index = decode_variant_index(
                cipher_token[-2:],
                key=key,
                factorization=factorization,
            )
            record = [
                int(line_index),
                int(token_index),
                list(base),
                int(variant_index),
            ]
            digest.update(
                json.dumps(
                    record,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
            digest.update(b"\n")

    return digest.hexdigest()


def _diagnostics(
    plaintext_by_line: Mapping[int, PlainLine],
    ciphertext_by_line: Mapping[int, Sequence[GlyphToken]],
    *,
    key: Mapping[str, object],
    factorization: str,
    occurrence_seed: int,
) -> dict:
    rows, cols = factorization_spec(
        factorization
    )

    total_tokens = 0
    cipher_counts: Counter[GlyphToken] = Counter()
    glyph_counts: Counter[str] = Counter()
    slot_a_counts: Counter[str] = Counter()
    slot_b_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    variant_counts: Counter[int] = Counter()

    surfaces_by_word: Dict[str, Counter[GlyphToken]] = defaultdict(Counter)
    bases_by_word: Dict[str, set[GlyphToken]] = defaultdict(set)

    decoded_ok = 0
    base_ok = 0
    length_ok = 0
    schedule_ok = 0

    for line_index in sorted(plaintext_by_line):
        plain_line = plaintext_by_line[line_index]
        cipher_line = ciphertext_by_line[line_index]

        if len(plain_line) != len(cipher_line):
            raise RuntimeError(
                "Token-count mismatch in diagnostics"
            )

        for token_index, (
            plaintext,
            cipher_token,
        ) in enumerate(
            zip(plain_line, cipher_line)
        ):
            total_tokens += 1
            cipher_tuple = tuple(cipher_token)
            cipher_counts[cipher_tuple] += 1
            glyph_counts.update(cipher_tuple)

            affix = tuple(
                cipher_tuple[-2:]
            )
            slot_a_counts[affix[0]] += 1
            slot_b_counts[affix[1]] += 1
            pair_counts[affix] += 1

            observed_variant = decode_variant_index(
                affix,
                key=key,
                factorization=factorization,
            )
            variant_counts[
                observed_variant
            ] += 1

            expected_variant = variant_index_choice(
                occurrence_seed=int(occurrence_seed),
                line_index=int(line_index),
                token_index=int(token_index),
            )
            if observed_variant == expected_variant:
                schedule_ok += 1

            expected_base = encode_base(
                plaintext,
                key=key,
            )
            observed_base = strip_affix(
                cipher_tuple,
                key=key,
                factorization=factorization,
            )

            bases_by_word[
                plaintext
            ].add(observed_base)
            surfaces_by_word[
                plaintext
            ][cipher_tuple] += 1

            if observed_base == expected_base:
                base_ok += 1
            if len(cipher_tuple) == len(
                expected_base
            ) + 2:
                length_ok += 1
            if decode_token(
                cipher_tuple,
                key=key,
                factorization=factorization,
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
    if schedule_ok != total_tokens:
        raise RuntimeError(
            f"Variant-schedule continuity failure: {schedule_ok}/{total_tokens}"
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

    slot_a_entropy = _shannon_entropy_bits(
        slot_a_counts
    )
    slot_b_entropy = _shannon_entropy_bits(
        slot_b_counts
    )
    joint_entropy = _shannon_entropy_bits(
        pair_counts
    )
    variant_entropy = _shannon_entropy_bits(
        variant_counts
    )

    return {
        "tokens": total_tokens,
        "factorization": factorization,
        "slot_a_states": rows,
        "slot_b_states": cols,
        "nominal_whole_token_variant_count": 16,
        "suffix_length_glyphs": 2,
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
        "token_length_plus_two_accuracy": (
            length_ok / total_tokens
            if total_tokens else 1.0
        ),
        "phase3a4_k16_variant_schedule_accuracy": (
            schedule_ok / total_tokens
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
        "realized_variant_indices": len(variant_counts),
        "realized_affix_pairs": len(pair_counts),
        "slot_a_entropy_bits": slot_a_entropy,
        "slot_b_entropy_bits": slot_b_entropy,
        "joint_affix_pair_entropy_bits": joint_entropy,
        "abstract_variant_index_entropy_bits": variant_entropy,
        "nominal_variant_entropy_bits": 4.0,
        "slot_a_counts_json": json.dumps(
            dict(sorted(slot_a_counts.items())),
            sort_keys=True,
        ),
        "slot_b_counts_json": json.dumps(
            dict(sorted(slot_b_counts.items())),
            sort_keys=True,
        ),
        "variant_index_counts_json": json.dumps(
            {
                str(k): int(v)
                for k, v in sorted(
                    variant_counts.items()
                )
            },
            sort_keys=True,
        ),
        "abstract_token_identity_sha256": (
            _abstract_identity_sha256(
                plaintext_by_line,
                ciphertext_by_line,
                key=key,
                factorization=factorization,
            )
        ),
    }


def generate_control(
    source_lines: Sequence[PlainLine],
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
    seed: int,
    factorization: str,
) -> GeneratedCompositionalAffixControl:
    """Generate one factorization/seed control without encoding reserved test."""
    factorization_spec(
        factorization
    )

    line_count = len(source_lines)
    validate_partition(
        line_count,
        train_indices,
        validation_indices,
        reserved_indices,
    )

    train_set = set(
        int(x) for x in train_indices
    )
    validation_set = set(
        int(x) for x in validation_indices
    )
    reserved_set = set(
        int(x) for x in reserved_indices
    )

    train_lines = [
        source_lines[i - 1]
        for i in sorted(train_set)
    ]

    key = build_master_key(
        train_lines,
        seed=int(seed),
    )

    active_a, active_b = active_factor_states(
        key,
        factorization=factorization,
    )

    rows: List[dict] = []
    plaintext_train: Dict[int, PlainLine] = {}
    plaintext_validation: Dict[int, PlainLine] = {}
    cipher_train: Dict[int, List[GlyphToken]] = {}
    cipher_validation: Dict[int, List[GlyphToken]] = {}

    for line_index, plain_line in enumerate(
        source_lines,
        start=1,
    ):
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
                factorization=factorization,
                occurrence_seed=int(seed),
                line_index=line_index,
                token_index=token_index,
            )
            for token_index, plaintext in enumerate(
                plain_line
            )
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
            plaintext_train[
                line_index
            ] = plain_line
            cipher_train[
                line_index
            ] = cipher_line
        elif line_index in validation_set:
            plaintext_validation[
                line_index
            ] = plain_line
            cipher_validation[
                line_index
            ] = cipher_line
        else:
            raise RuntimeError(
                "Line belongs to no split"
            )

    train_diag = _diagnostics(
        plaintext_train,
        cipher_train,
        key=key,
        factorization=factorization,
        occurrence_seed=int(seed),
    )
    validation_diag = _diagnostics(
        plaintext_validation,
        cipher_validation,
        key=key,
        factorization=factorization,
        occurrence_seed=int(seed),
    )

    rows_count, cols_count = factorization_spec(
        factorization
    )

    output_key = dict(key)
    output_key.update(
        {
            "factorization": factorization,
            "slot_a_states": rows_count,
            "slot_b_states": cols_count,
            "active_slot_a": list(active_a),
            "active_slot_b": list(active_b),
        }
    )

    summary = {
        "schema_version": "1.0",
        "mechanism": "compositional_affix_factorization",
        "status": "exploratory",
        "seed": int(seed),
        "factorization": factorization,
        "slot_a_states": rows_count,
        "slot_b_states": cols_count,
        "whole_token_variant_count": 16,
        "suffix_length_glyphs": 2,
        "coverage": "all_train_and_validation_tokens",
        "fit_scope": "CONTROL TRAIN ONLY",
        "evaluation_scope": "CONTROL VALIDATION ONLY",
        "train_lines": len(train_set),
        "validation_lines": len(validation_set),
        "reserved_lines": len(reserved_set),
        "reserved_test_encoded": False,
        "round_trip_verified_train_validation": True,
        "base_form_preservation_verified_train_validation": True,
        "token_length_control_verified_train_validation": True,
        "phase3a4_k16_variant_schedule_verified_train_validation": True,
        "parent_phase3a4_master_key_sha256": key[
            "parent_phase3a4_master_key_sha256"
        ],
        "train_diagnostics": train_diag,
        "validation_diagnostics": validation_diag,
        "guardrails": [
            "Base mapping is inherited exactly from Phase 3A.4.",
            "The exact Phase-3A.4 K=16 occurrence variant-index schedule is retained.",
            "Every condition has exactly 16 abstract whole-token variants.",
            "Every condition appends exactly two affix glyphs.",
            "The variant-index-to-affix-pair map is bijective in every condition.",
            "Suffix choice excludes plaintext token identity.",
            "Validation cannot fit or modify the base-character mapping.",
            "Reserved-test rows are opaque placeholders and are not encoded.",
        ],
    }

    return GeneratedCompositionalAffixControl(
        rows=tuple(rows),
        key=output_key,
        summary=summary,
    )


def write_generated_control(
    output_dir: Path,
    generated: GeneratedCompositionalAffixControl,
    *,
    provenance: Mapping[str, object],
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus_path = output_dir / "corpus.jsonl"
    key_path = output_dir / "key.json"
    summary_path = output_dir / "summary.json"

    with corpus_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in generated.rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    key_payload = dict(
        generated.key
    )
    key_payload[
        "provenance"
    ] = dict(provenance)
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

    summary_payload = dict(
        generated.summary
    )
    summary_payload[
        "provenance"
    ] = dict(provenance)
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
        "corpus.jsonl": sha256_file(
            corpus_path
        ),
        "key.json": sha256_file(
            key_path
        ),
        "summary.json": sha256_file(
            summary_path
        ),
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
