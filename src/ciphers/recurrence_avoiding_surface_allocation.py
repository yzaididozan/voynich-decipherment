"""Phase 3A.6 recurrence-avoiding surface allocation for VOYAGER.

Intended repository destination:
    src/ciphers/recurrence_avoiding_surface_allocation.py

Scientific intervention
-----------------------
Phase 3A.D1 localized the remaining Phase-3A.5 token-Markov failure primarily
to exact current-token recurrence rather than exact transition recurrence.
Phase 3A.6 therefore keeps the Phase-3A.4 K=16 representation fixed and changes
only how those 16 inherited suffix variants are allocated to occurrences of an
encoded lexical base.

Parent Phase 3A.4 K=16:
    occurrence -> independent deterministic hash -> one of 16 suffixes
    (sampling with replacement at the level of repeated lexical occurrences)

Phase 3A.6:
    each encoded lexical base receives a deterministic seed-specific permutation
    of the same 16 inherited suffixes; consecutive occurrences walk through that
    permutation without replacement and cycle only after all 16 are exhausted.

TRAIN is processed first. VALIDATION starts from the frozen final TRAIN
occurrence counter for each encoded base and is then processed online in
ascending line/token order. Reserved-test rows are never encoded.

This is exploratory. It does not access Voynich test_LOCKED.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

from src.ciphers import base_form_suffix_extended as phase3a4


GlyphToken = Tuple[str, ...]
PlainLine = Tuple[str, ...]
VARIANT_COUNT = 16
ALLOCATION_NAMESPACE = "recurrence-avoiding-base-cycle-v1"


@dataclass(frozen=True)
class GeneratedRecurrenceAvoidingControl:
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


def build_master_key(
    train_lines: Sequence[PlainLine],
    *,
    seed: int,
) -> dict:
    """Inherit the exact Phase-3A.4 K=16 base map and suffix inventory."""
    parent = phase3a4.build_master_key(
        train_lines,
        seed=int(seed),
        master_suffix_count=VARIANT_COUNT,
    )

    suffixes = list(parent["master_suffixes"])
    if len(suffixes) != VARIANT_COUNT:
        raise ValueError(
            f"Phase-3A.4 parent key must expose exactly {VARIANT_COUNT} suffixes"
        )
    if len(set(suffixes)) != VARIANT_COUNT:
        raise ValueError("Phase-3A.4 parent suffix inventory is not unique")

    payload = {
        "seed": int(seed),
        "parent_phase3a4_master_key_sha256": parent["master_key_sha256"],
        "allocation_namespace": ALLOCATION_NAMESPACE,
        "variant_count": VARIANT_COUNT,
    }
    allocation_key_sha256 = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": "1.0",
        "mechanism": "recurrence_avoiding_surface_allocation",
        "fit_scope": "CONTROL TRAIN ONLY",
        "seed": int(seed),
        "mono_forward": dict(parent["mono_forward"]),
        "mono_reverse": dict(parent["mono_reverse"]),
        "train_character_alphabet": list(parent["train_character_alphabet"]),
        "master_suffixes": suffixes,
        "parent_phase3a4_master_key_sha256": parent["master_key_sha256"],
        "allocation_namespace": ALLOCATION_NAMESPACE,
        "variant_count": VARIANT_COUNT,
        "suffix_length_glyphs": 1,
        "allocation_key_sha256": allocation_key_sha256,
        "validation_unseen_character_policy": parent[
            "validation_unseen_character_policy"
        ],
    }


def encode_base(
    token: str,
    *,
    key: Mapping[str, object],
) -> GlyphToken:
    return tuple(
        phase3a4.encode_base(
            token,
            key=key,
        )
    )


def decode_base(
    base: Sequence[str],
    *,
    key: Mapping[str, object],
) -> str:
    return phase3a4.decode_base(
        base,
        key=key,
    )


def _serialized_base(base: Sequence[str]) -> str:
    return json.dumps(
        list(base),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def variant_permutation(
    base: Sequence[str],
    *,
    seed: int,
) -> Tuple[int, ...]:
    """Return a deterministic permutation of 0..15 for one encoded base."""
    serialized = _serialized_base(base)

    ordering = sorted(
        range(VARIANT_COUNT),
        key=lambda variant_index: (
            stable_uint64(
                ALLOCATION_NAMESPACE,
                int(seed),
                serialized,
                int(variant_index),
            ),
            int(variant_index),
        ),
    )

    permutation = tuple(int(x) for x in ordering)
    if sorted(permutation) != list(range(VARIANT_COUNT)):
        raise RuntimeError("Variant ordering is not a permutation of 0..15")
    return permutation


def variant_index_for_occurrence(
    base: Sequence[str],
    *,
    seed: int,
    occurrence_number: int,
) -> int:
    """Map a zero-based lexical-base occurrence number to one inherited variant."""
    if occurrence_number < 0:
        raise ValueError("occurrence_number must be non-negative")

    permutation = variant_permutation(
        base,
        seed=int(seed),
    )
    return permutation[
        int(occurrence_number) % VARIANT_COUNT
    ]


def suffix_for_variant(
    key: Mapping[str, object],
    variant_index: int,
) -> str:
    if variant_index < 0 or variant_index >= VARIANT_COUNT:
        raise ValueError("variant_index must be in [0, 15]")

    suffixes = tuple(key["master_suffixes"])
    if len(suffixes) != VARIANT_COUNT:
        raise ValueError("Key does not contain exactly 16 master suffixes")
    return suffixes[int(variant_index)]


def decode_variant_index(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
) -> int:
    token = tuple(cipher_token)
    if len(token) < 2:
        raise ValueError(
            "Cipher token must contain at least one base glyph plus one suffix"
        )

    suffixes = tuple(key["master_suffixes"])
    try:
        return suffixes.index(token[-1])
    except ValueError as exc:
        raise ValueError(
            f"Unknown recurrence-avoiding suffix: {token[-1]!r}"
        ) from exc


def encode_token_for_occurrence(
    plaintext_token: str,
    *,
    key: Mapping[str, object],
    seed: int,
    occurrence_number: int,
) -> GlyphToken:
    base = encode_base(
        plaintext_token,
        key=key,
    )
    variant_index = variant_index_for_occurrence(
        base,
        seed=int(seed),
        occurrence_number=int(occurrence_number),
    )
    suffix = suffix_for_variant(
        key,
        variant_index,
    )
    return (*base, suffix)


def strip_suffix(
    cipher_token: Sequence[str],
    *,
    key: Mapping[str, object],
) -> GlyphToken:
    token = tuple(cipher_token)
    decode_variant_index(
        token,
        key=key,
    )
    return token[:-1]


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


def _encode_phase(
    source_lines: Sequence[PlainLine],
    line_indices: Sequence[int],
    *,
    key: Mapping[str, object],
    seed: int,
    counters: MutableMapping[GlyphToken, int],
) -> Tuple[Dict[int, Tuple[GlyphToken, ...]], Dict[int, Tuple[int, ...]]]:
    """Encode selected lines online in ascending line/token order."""
    encoded: Dict[int, Tuple[GlyphToken, ...]] = {}
    occurrence_numbers: Dict[int, Tuple[int, ...]] = {}

    for line_index in sorted(int(x) for x in line_indices):
        plain_line = source_lines[
            line_index - 1
        ]
        cipher_line: List[GlyphToken] = []
        occurrence_line: List[int] = []

        for plaintext_token in plain_line:
            base = encode_base(
                plaintext_token,
                key=key,
            )
            occurrence_number = int(
                counters.get(
                    base,
                    0,
                )
            )

            cipher_token = encode_token_for_occurrence(
                plaintext_token,
                key=key,
                seed=int(seed),
                occurrence_number=occurrence_number,
            )

            counters[
                base
            ] = occurrence_number + 1
            cipher_line.append(
                cipher_token
            )
            occurrence_line.append(
                occurrence_number
            )

        encoded[
            line_index
        ] = tuple(cipher_line)
        occurrence_numbers[
            line_index
        ] = tuple(occurrence_line)

    return encoded, occurrence_numbers


def _same_surface_probability(
    plaintext_by_line: Mapping[int, PlainLine],
    cipher_by_line: Mapping[int, Sequence[GlyphToken]],
) -> float | None:
    surfaces_by_word: Dict[str, Counter] = defaultdict(Counter)

    for line_index in sorted(
        plaintext_by_line
    ):
        plain_line = plaintext_by_line[
            line_index
        ]
        cipher_line = cipher_by_line[
            line_index
        ]
        for plaintext, cipher_token in zip(
            plain_line,
            cipher_line,
        ):
            surfaces_by_word[
                plaintext
            ][tuple(cipher_token)] += 1

    numerator = 0
    denominator = 0
    for counter in surfaces_by_word.values():
        n = sum(counter.values())
        if n >= 2:
            denominator += n * (n - 1)
            numerator += sum(
                count * (count - 1)
                for count in counter.values()
            )

    if denominator == 0:
        return None
    return numerator / denominator


def _split_diagnostics(
    plaintext_by_line: Mapping[int, PlainLine],
    cipher_by_line: Mapping[int, Sequence[GlyphToken]],
    occurrence_by_line: Mapping[int, Sequence[int]],
    *,
    key: Mapping[str, object],
    seed: int,
    train_cipher_tokens: set[GlyphToken] | None,
) -> dict:
    total = 0
    decoded_ok = 0
    base_ok = 0
    length_ok = 0
    schedule_ok = 0

    exact_counts: Counter = Counter()
    variants_by_base: Dict[GlyphToken, set[int]] = defaultdict(set)
    occurrences_by_base: Counter = Counter()

    validation_seen_in_train = 0

    for line_index in sorted(
        plaintext_by_line
    ):
        plain_line = plaintext_by_line[
            line_index
        ]
        cipher_line = cipher_by_line[
            line_index
        ]
        occurrence_line = occurrence_by_line[
            line_index
        ]

        if not (
            len(plain_line)
            == len(cipher_line)
            == len(occurrence_line)
        ):
            raise RuntimeError(
                f"Token-count mismatch on line {line_index}"
            )

        for plaintext, cipher_token, occurrence_number in zip(
            plain_line,
            cipher_line,
            occurrence_line,
        ):
            total += 1
            cipher_token = tuple(
                cipher_token
            )
            exact_counts[
                cipher_token
            ] += 1

            expected_base = encode_base(
                plaintext,
                key=key,
            )
            observed_base = strip_suffix(
                cipher_token,
                key=key,
            )
            observed_variant = decode_variant_index(
                cipher_token,
                key=key,
            )
            expected_variant = variant_index_for_occurrence(
                expected_base,
                seed=int(seed),
                occurrence_number=int(occurrence_number),
            )

            occurrences_by_base[
                expected_base
            ] += 1
            variants_by_base[
                expected_base
            ].add(
                observed_variant
            )

            if observed_base == expected_base:
                base_ok += 1
            if (
                len(cipher_token)
                == len(expected_base) + 1
            ):
                length_ok += 1
            if observed_variant == expected_variant:
                schedule_ok += 1
            if (
                decode_token(
                    cipher_token,
                    key=key,
                )
                == plaintext
            ):
                decoded_ok += 1

            if (
                train_cipher_tokens is not None
                and cipher_token
                in train_cipher_tokens
            ):
                validation_seen_in_train += 1

    expected_unique_variant_failures = 0
    for base, n in occurrences_by_base.items():
        # This split-local check is exact only when occurrence numbers start at
        # a cycle boundary. TRAIN does. VALIDATION may continue a TRAIN cycle,
        # so the more important continuation invariant is checked separately.
        if train_cipher_tokens is None:
            expected_unique = min(
                int(n),
                VARIANT_COUNT,
            )
            if (
                len(
                    variants_by_base[
                        base
                    ]
                )
                != expected_unique
            ):
                expected_unique_variant_failures += 1

    if decoded_ok != total:
        raise RuntimeError(
            f"Round-trip failure: {decoded_ok}/{total}"
        )
    if base_ok != total:
        raise RuntimeError(
            f"Base-form preservation failure: {base_ok}/{total}"
        )
    if length_ok != total:
        raise RuntimeError(
            f"Token-length control failure: {length_ok}/{total}"
        )
    if schedule_ok != total:
        raise RuntimeError(
            f"Balanced schedule failure: {schedule_ok}/{total}"
        )
    if (
        train_cipher_tokens is None
        and expected_unique_variant_failures
    ):
        raise RuntimeError(
            "TRAIN did not use min(count,16) unique variants for every base"
        )

    return {
        "tokens": total,
        "unique_exact_cipher_tokens": len(
            exact_counts
        ),
        "exact_cipher_token_recurrence_fraction": (
            (total - len(exact_counts))
            / total
            if total
            else 0.0
        ),
        "round_trip_accuracy": (
            decoded_ok / total
            if total
            else 1.0
        ),
        "base_form_preservation_fraction": (
            base_ok / total
            if total
            else 1.0
        ),
        "token_length_plus_one_accuracy": (
            length_ok / total
            if total
            else 1.0
        ),
        "balanced_schedule_accuracy": (
            schedule_ok / total
            if total
            else 1.0
        ),
        "lexical_bases": len(
            occurrences_by_base
        ),
        "same_surface_probability_given_same_plaintext_word": (
            _same_surface_probability(
                plaintext_by_line,
                cipher_by_line,
            )
        ),
        "validation_exact_tokens_seen_in_train": (
            validation_seen_in_train
            if train_cipher_tokens is not None
            else None
        ),
        "validation_exact_token_seen_in_train_fraction": (
            validation_seen_in_train / total
            if (
                train_cipher_tokens is not None
                and total
            )
            else None
        ),
    }


def _continuation_diagnostics(
    *,
    source_lines: Sequence[PlainLine],
    validation_indices: Sequence[int],
    validation_cipher: Mapping[int, Sequence[GlyphToken]],
    validation_occurrence: Mapping[int, Sequence[int]],
    key: Mapping[str, object],
    seed: int,
    train_counts: Mapping[GlyphToken, int],
) -> dict:
    """Check that validation continues the TRAIN cycle without avoidable reuse."""
    train_variants_by_base: Dict[GlyphToken, set[int]] = {}

    for base, count in train_counts.items():
        train_variants_by_base[
            base
        ] = {
            variant_index_for_occurrence(
                base,
                seed=int(seed),
                occurrence_number=n,
            )
            for n in range(
                int(count)
            )
        }

    avoidable_opportunities = 0
    avoidable_collisions = 0
    unavoidable_reuse = 0
    validation_occurrence_offset: Counter = Counter()

    for line_index in sorted(
        int(x)
        for x in validation_indices
    ):
        plain_line = source_lines[
            line_index - 1
        ]
        cipher_line = validation_cipher[
            line_index
        ]
        occurrence_line = validation_occurrence[
            line_index
        ]

        for plaintext, cipher_token, occurrence_number in zip(
            plain_line,
            cipher_line,
            occurrence_line,
        ):
            base = encode_base(
                plaintext,
                key=key,
            )
            train_count = int(
                train_counts.get(
                    base,
                    0,
                )
            )
            offset = int(
                validation_occurrence_offset[
                    base
                ]
            )
            validation_occurrence_offset[
                base
            ] += 1

            if occurrence_number != train_count + offset:
                raise RuntimeError(
                    "VALIDATION occurrence counter did not continue TRAIN state"
                )

            variant = decode_variant_index(
                cipher_token,
                key=key,
            )
            train_variants = (
                train_variants_by_base.get(
                    base,
                    set(),
                )
            )

            # Reuse is avoidable precisely while the base has not yet traversed
            # all 16 cycle positions across TRAIN + earlier VALIDATION.
            reuse_avoidable = (
                train_count < VARIANT_COUNT
                and offset
                < (
                    VARIANT_COUNT
                    - train_count
                )
            )

            if reuse_avoidable:
                avoidable_opportunities += 1
                if variant in train_variants:
                    avoidable_collisions += 1
            elif variant in train_variants:
                unavoidable_reuse += 1

    if avoidable_collisions != 0:
        raise RuntimeError(
            f"Observed {avoidable_collisions} avoidable validation collisions"
        )

    return {
        "validation_avoidable_collision_opportunities": (
            avoidable_opportunities
        ),
        "validation_avoidable_collisions": (
            avoidable_collisions
        ),
        "validation_avoidable_collision_rate": (
            avoidable_collisions
            / avoidable_opportunities
            if avoidable_opportunities
            else 0.0
        ),
        "validation_unavoidable_train_surface_reuses": (
            unavoidable_reuse
        ),
    }


def generate_control(
    source_lines: Sequence[PlainLine],
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    reserved_indices: Sequence[int],
    seed: int,
) -> GeneratedRecurrenceAvoidingControl:
    """Generate one balanced K=16 control without encoding reserved test rows."""
    validate_partition(
        len(source_lines),
        train_indices,
        validation_indices,
        reserved_indices,
    )

    train_set = {
        int(x)
        for x in train_indices
    }
    validation_set = {
        int(x)
        for x in validation_indices
    }
    reserved_set = {
        int(x)
        for x in reserved_indices
    }

    train_lines_for_fit = [
        source_lines[
            line_index - 1
        ]
        for line_index in sorted(
            train_set
        )
    ]

    key = build_master_key(
        train_lines_for_fit,
        seed=int(seed),
    )

    counters: Counter = Counter()

    train_cipher, train_occurrence = _encode_phase(
        source_lines,
        sorted(train_set),
        key=key,
        seed=int(seed),
        counters=counters,
    )

    frozen_train_counts = dict(
        counters
    )

    validation_cipher, validation_occurrence = _encode_phase(
        source_lines,
        sorted(validation_set),
        key=key,
        seed=int(seed),
        counters=counters,
    )

    rows: List[dict] = []

    for line_index in range(
        1,
        len(source_lines) + 1,
    ):
        if line_index in train_set:
            rows.append(
                {
                    "line_index": line_index,
                    "tokens": [
                        list(token)
                        for token in train_cipher[
                            line_index
                        ]
                    ],
                }
            )
        elif line_index in validation_set:
            rows.append(
                {
                    "line_index": line_index,
                    "tokens": [
                        list(token)
                        for token in validation_cipher[
                            line_index
                        ]
                    ],
                }
            )
        elif line_index in reserved_set:
            rows.append(
                {
                    "line_index": line_index,
                    "reserved_test_opaque": True,
                }
            )
        else:
            raise RuntimeError(
                f"Line {line_index} belongs to no frozen split"
            )

    plain_train = {
        line_index: source_lines[
            line_index - 1
        ]
        for line_index in sorted(
            train_set
        )
    }
    plain_validation = {
        line_index: source_lines[
            line_index - 1
        ]
        for line_index in sorted(
            validation_set
        )
    }

    train_diag = _split_diagnostics(
        plain_train,
        train_cipher,
        train_occurrence,
        key=key,
        seed=int(seed),
        train_cipher_tokens=None,
    )

    train_cipher_token_set = {
        tuple(token)
        for line in train_cipher.values()
        for token in line
    }

    validation_diag = _split_diagnostics(
        plain_validation,
        validation_cipher,
        validation_occurrence,
        key=key,
        seed=int(seed),
        train_cipher_tokens=train_cipher_token_set,
    )

    continuation_diag = _continuation_diagnostics(
        source_lines=source_lines,
        validation_indices=sorted(
            validation_set
        ),
        validation_cipher=validation_cipher,
        validation_occurrence=validation_occurrence,
        key=key,
        seed=int(seed),
        train_counts=frozen_train_counts,
    )

    train_count_payload = [
        [
            list(base),
            int(count),
        ]
        for base, count in sorted(
            frozen_train_counts.items(),
            key=lambda item: repr(
                item[0]
            ),
        )
    ]
    train_state_sha256 = sha256(
        json.dumps(
            train_count_payload,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    summary = {
        "schema_version": "1.0",
        "mechanism": "recurrence_avoiding_surface_allocation",
        "status": "exploratory",
        "seed": int(seed),
        "variant_count": VARIANT_COUNT,
        "suffix_length_glyphs": 1,
        "coverage": "all_train_and_validation_tokens",
        "fit_scope": "CONTROL TRAIN ONLY",
        "evaluation_scope": "CONTROL VALIDATION ONLY",
        "train_lines": len(
            train_set
        ),
        "validation_lines": len(
            validation_set
        ),
        "reserved_lines": len(
            reserved_set
        ),
        "train_final_occurrence_state_sha256": (
            train_state_sha256
        ),
        "train_diagnostics": train_diag,
        "validation_diagnostics": validation_diag,
        "continuation_diagnostics": continuation_diag,
        "round_trip_verified_train_validation": True,
        "base_form_preservation_verified_train_validation": True,
        "token_length_control_verified_train_validation": True,
        "balanced_without_replacement_schedule_verified": True,
        "validation_continues_frozen_train_state": True,
        "reserved_test_encoded": False,
        "guardrails": [
            "Exact Phase-3A.4 TRAIN-fitted base mapping is inherited.",
            "Exact Phase-3A.4 K16 master suffix inventory/order is inherited.",
            "Every TRAIN/VALIDATION token receives exactly one suffix glyph.",
            "Each encoded base traverses all 16 variants once per 16-occurrence cycle.",
            "VALIDATION starts from the final TRAIN per-base occurrence counter.",
            "VALIDATION is processed online without look-ahead.",
            "Reserved-test rows remain opaque and are never encoded.",
        ],
    }

    return GeneratedRecurrenceAvoidingControl(
        rows=tuple(
            rows
        ),
        key=key,
        summary=summary,
    )


def write_generated_control(
    output_dir: Path,
    generated: GeneratedRecurrenceAvoidingControl,
    *,
    provenance: Mapping[str, object],
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    corpus_path = (
        output_dir
        / "corpus.jsonl"
    )
    key_path = (
        output_dir
        / "key.json"
    )
    summary_path = (
        output_dir
        / "summary.json"
    )

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
    ] = dict(
        provenance
    )
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
    ] = dict(
        provenance
    )
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

    (
        output_dir
        / "SHA256SUMS"
    ).write_text(
        "\n".join(
            f"{digest}  {name}"
            for name, digest in artifacts.items()
        )
        + "\n",
        encoding="utf-8",
    )

    return artifacts
