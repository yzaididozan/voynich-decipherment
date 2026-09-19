"""Phase 3A.D1 token-Markov failure localization for VOYAGER.

Intended repository destination:
    src/analysis/token_markov_failure_localization.py

This module performs a post-hoc *development diagnostic* on already-generated
Phase-3A.5 synthetic controls. It does not generate a new cipher and does not
touch the Voynich locked test.

For each Phase-3A.5 run, the exact frozen token Markov model is refit on the
control TRAIN units and scored on VALIDATION using the context order already
selected by Phase 3A.5. The matched comparator is the model's own frozen
token-reset character trigram base, which the Level-2 evaluator already
verified to be exactly the same distribution as its external matched trigram.

For each validation token:

    delta_bits = token_markov_bits - matched_trigram_bits

Positive values favor the matched character trigram.
Negative values favor the exact-token Markov model.

Because both models assign one token probability but share the frozen
glyphs+EOT event denominator, summing token contributions and dividing by the
sum of (len(cipher_token)+1) exactly reconstructs the frozen run-level
token-Markov relation. The runner hard-checks this equality before emitting
localization summaries.

All explanatory features are computed from TRAIN only. Validation contributes
only the item being scored and the assignment of that item into bins defined
before the diagnostic run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.models.token_markov import TokenMarkovModel


Token = Tuple[str, ...]
PlainLine = Tuple[str, ...]

PLAIN_BOS = "<LINE_BOS>"
CIPHER_BOS: Token = ("<LINE_BOS>",)
VARIANT_BOS = -1


@dataclass(frozen=True)
class TrainingStatistics:
    plain_token_counts: Counter
    plain_token_rank: Mapping[str, int]
    plain_bigram_counts: Counter
    plain_outgoing: Mapping[str, Counter]
    plain_incoming: Mapping[str, Counter]
    cipher_token_counts: Counter
    cipher_token_rank: Mapping[Token, int]
    cipher_bigram_counts: Counter
    cipher_outgoing: Mapping[Token, Counter]
    cipher_incoming: Mapping[Token, Counter]
    variant_counts: Counter
    variant_bigram_counts: Counter
    variant_outgoing: Mapping[int, Counter]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_selected_plaintext_units(
    path: Path,
    *,
    selected_indices: Sequence[int],
    expected_line_count: int,
) -> List[Tuple[int, PlainLine]]:
    """Tokenize only selected source lines.

    Lines outside ``selected_indices`` are streamed past without splitting or
    otherwise using their textual content. This mirrors the reserved-data
    discipline used by the Level-2 control reader.
    """
    selected = set(int(x) for x in selected_indices)
    if not selected:
        raise ValueError("selected_indices may not be empty")

    bad = sorted(
        index for index in selected
        if index < 1 or index > expected_line_count
    )
    if bad:
        raise ValueError(
            f"selected_indices contain out-of-range values: {bad[:10]}"
        )

    output: List[Tuple[int, PlainLine]] = []
    observed_lines = 0

    with path.open(encoding="utf-8") as handle:
        for observed_lines, raw in enumerate(handle, start=1):
            if observed_lines not in selected:
                continue

            text = raw.rstrip("\r\n")
            tokens = tuple(text.split())
            if not tokens:
                raise ValueError(
                    f"{path}:{observed_lines}: selected source line has no tokens"
                )
            output.append((observed_lines, tokens))

    if observed_lines != expected_line_count:
        raise ValueError(
            f"{path}: expected {expected_line_count} lines, found {observed_lines}"
        )

    found = {line_index for line_index, _ in output}
    if found != selected:
        missing = sorted(selected - found)
        raise ValueError(
            f"{path}: selected source rows missing: {missing[:10]}"
        )

    return output


def units_to_dict(
    units: Sequence[Tuple[int, Sequence]],
) -> Dict[int, Tuple]:
    result = {}
    for line_index, sequence in units:
        line_index = int(line_index)
        if line_index in result:
            raise ValueError(f"Duplicate unit index: {line_index}")
        result[line_index] = tuple(sequence)
    return result


def deterministic_frequency_ranks(counter: Counter) -> dict:
    """Return one-based deterministic frequency ranks.

    Ties are broken by a stable string representation. The rank is therefore
    deterministic but should not be interpreted as a tie-aware statistical
    rank.
    """
    ordered = sorted(
        counter.items(),
        key=lambda item: (
            -int(item[1]),
            repr(item[0]),
        ),
    )
    return {
        item: rank
        for rank, (item, _) in enumerate(ordered, start=1)
    }


def factorization_dimensions(label: str) -> Tuple[int, int]:
    mapping = {
        "1x16": (1, 16),
        "2x8": (2, 8),
        "4x4": (4, 4),
    }
    try:
        rows, cols = mapping[label]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Phase-3A.5 factorization: {label!r}"
        ) from exc
    return rows, cols


def decode_variant_index_from_key(
    token: Sequence[str],
    *,
    key: Mapping[str, object],
    factorization: str,
) -> int:
    """Decode the Phase-3A.5 abstract 16-way variant index."""
    token = tuple(token)
    if len(token) < 3:
        raise ValueError(
            "Phase-3A.5 token must contain at least one base glyph plus A/B affix"
        )

    rows, cols = factorization_dimensions(factorization)
    active_a = tuple(key["active_slot_a"])
    active_b = tuple(key["active_slot_b"])

    if len(active_a) != rows or len(active_b) != cols:
        raise ValueError(
            f"Key active-state dimensions do not match {factorization}: "
            f"{len(active_a)}x{len(active_b)}"
        )

    try:
        a_index = active_a.index(token[-2])
        b_index = active_b.index(token[-1])
    except ValueError as exc:
        raise ValueError(
            f"Token affix {token[-2:]!r} is not valid for {factorization}"
        ) from exc

    index = a_index * cols + b_index
    if not 0 <= index < 16:
        raise RuntimeError("Decoded variant index fell outside 0..15")
    return index


def _transition_update(
    bigram_counts: Counter,
    outgoing: Mapping,
    incoming: Mapping,
    previous,
    current,
) -> None:
    bigram_counts[(previous, current)] += 1
    outgoing[previous][current] += 1
    incoming[current][previous] += 1


def build_training_statistics(
    plain_train_units: Sequence[Tuple[int, PlainLine]],
    cipher_train_units: Sequence[Tuple[int, Sequence[Token]]],
    *,
    key: Mapping[str, object],
    factorization: str,
) -> TrainingStatistics:
    plain_by_line = units_to_dict(plain_train_units)
    cipher_by_line = units_to_dict(cipher_train_units)

    if set(plain_by_line) != set(cipher_by_line):
        raise ValueError(
            "TRAIN plaintext and ciphertext unit memberships differ"
        )

    plain_token_counts: Counter = Counter()
    plain_bigram_counts: Counter = Counter()
    plain_outgoing = defaultdict(Counter)
    plain_incoming = defaultdict(Counter)

    cipher_token_counts: Counter = Counter()
    cipher_bigram_counts: Counter = Counter()
    cipher_outgoing = defaultdict(Counter)
    cipher_incoming = defaultdict(Counter)

    variant_counts: Counter = Counter()
    variant_bigram_counts: Counter = Counter()
    variant_outgoing = defaultdict(Counter)

    for line_index in sorted(plain_by_line):
        plain_line = tuple(plain_by_line[line_index])
        cipher_line = tuple(
            tuple(token)
            for token in cipher_by_line[line_index]
        )

        if len(plain_line) != len(cipher_line):
            raise ValueError(
                f"TRAIN token-count mismatch on line {line_index}: "
                f"{len(plain_line)} != {len(cipher_line)}"
            )

        previous_plain = PLAIN_BOS
        previous_cipher = CIPHER_BOS
        previous_variant = VARIANT_BOS

        for plain_token, cipher_token in zip(
            plain_line,
            cipher_line,
        ):
            variant = decode_variant_index_from_key(
                cipher_token,
                key=key,
                factorization=factorization,
            )

            plain_token_counts[plain_token] += 1
            cipher_token_counts[cipher_token] += 1
            variant_counts[variant] += 1

            _transition_update(
                plain_bigram_counts,
                plain_outgoing,
                plain_incoming,
                previous_plain,
                plain_token,
            )
            _transition_update(
                cipher_bigram_counts,
                cipher_outgoing,
                cipher_incoming,
                previous_cipher,
                cipher_token,
            )
            variant_bigram_counts[
                (previous_variant, variant)
            ] += 1
            variant_outgoing[
                previous_variant
            ][variant] += 1

            previous_plain = plain_token
            previous_cipher = cipher_token
            previous_variant = variant

    return TrainingStatistics(
        plain_token_counts=plain_token_counts,
        plain_token_rank=deterministic_frequency_ranks(
            plain_token_counts
        ),
        plain_bigram_counts=plain_bigram_counts,
        plain_outgoing=dict(plain_outgoing),
        plain_incoming=dict(plain_incoming),
        cipher_token_counts=cipher_token_counts,
        cipher_token_rank=deterministic_frequency_ranks(
            cipher_token_counts
        ),
        cipher_bigram_counts=cipher_bigram_counts,
        cipher_outgoing=dict(cipher_outgoing),
        cipher_incoming=dict(cipher_incoming),
        variant_counts=variant_counts,
        variant_bigram_counts=variant_bigram_counts,
        variant_outgoing=dict(variant_outgoing),
    )


def assign_upper_bound_bin(
    value: float,
    specs: Sequence[Mapping[str, object]],
) -> str:
    """Assign a non-negative value to ascending upper-bound bins."""
    if value < 0:
        raise ValueError("Bin values must be non-negative")

    previous_upper: Optional[float] = None

    for index, spec in enumerate(specs):
        label = str(spec["label"])
        raw_upper = spec.get("upper")

        if raw_upper is None:
            if index != len(specs) - 1:
                raise ValueError(
                    "Open-ended bin must be the final bin"
                )
            return label

        upper = float(raw_upper)
        if previous_upper is not None and upper <= previous_upper:
            raise ValueError(
                "Bin upper bounds must be strictly increasing"
            )

        if value <= upper:
            return label
        previous_upper = upper

    raise ValueError(
        f"Value {value} exceeded the final closed bin without an open-ended bin"
    )


def line_position_class(
    token_index: int,
    line_length: int,
) -> str:
    if line_length < 1:
        raise ValueError("line_length must be positive")
    if token_index < 0 or token_index >= line_length:
        raise ValueError("token_index outside line")

    if line_length == 1:
        return "singleton"
    if token_index == 0:
        return "initial"
    if token_index == line_length - 1:
        return "final"
    return "medial"


def _transition_features(
    *,
    previous,
    current,
    token_counts: Counter,
    bigram_counts: Counter,
    outgoing: Mapping,
    incoming: Mapping,
    bos_value,
) -> dict:
    current_frequency = int(
        token_counts.get(current, 0)
    )
    previous_frequency = (
        0
        if previous == bos_value
        else int(token_counts.get(previous, 0))
    )

    bigram_frequency = int(
        bigram_counts.get(
            (previous, current),
            0,
        )
    )

    next_counter = outgoing.get(
        previous,
        Counter(),
    )
    outgoing_total = int(
        sum(next_counter.values())
    )
    distinct_next = int(
        len(next_counter)
    )
    max_next = (
        max(next_counter.values())
        if next_counter
        else 0
    )

    conditional_probability = (
        bigram_frequency / outgoing_total
        if outgoing_total
        else 0.0
    )
    transition_concentration = (
        max_next / outgoing_total
        if outgoing_total
        else 0.0
    )

    predecessor_diversity = int(
        len(
            incoming.get(
                current,
                Counter(),
            )
        )
    )

    return {
        "current_frequency": current_frequency,
        "previous_frequency": previous_frequency,
        "bigram_frequency": bigram_frequency,
        "outgoing_total": outgoing_total,
        "distinct_next_types": distinct_next,
        "conditional_probability": conditional_probability,
        "transition_concentration": transition_concentration,
        "current_predecessor_diversity": predecessor_diversity,
    }


def short_token_id(token: Sequence[str]) -> str:
    payload = json.dumps(
        list(token),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def score_validation_contributions(
    *,
    source_id: str,
    language: str,
    factorization: str,
    seed: int,
    plain_train_units: Sequence[Tuple[int, PlainLine]],
    plain_validation_units: Sequence[Tuple[int, PlainLine]],
    cipher_train_units: Sequence[Tuple[int, Sequence[Token]]],
    cipher_validation_units: Sequence[Tuple[int, Sequence[Token]]],
    key: Mapping[str, object],
    selected_context_order: int,
    max_context: int,
    char_base_order: int,
    bins: Mapping[str, Sequence[Mapping[str, object]]],
    top_n_frequency_sets: Sequence[int],
) -> Tuple[List[dict], dict]:
    """Return per-token contributions and exact run-level aggregate metrics."""
    if selected_context_order < 1:
        raise ValueError(
            "selected_context_order must be >= 1"
        )
    if selected_context_order > max_context:
        raise ValueError(
            "selected_context_order exceeds max_context"
        )

    plain_validation = units_to_dict(
        plain_validation_units
    )
    cipher_validation = units_to_dict(
        cipher_validation_units
    )

    if set(plain_validation) != set(cipher_validation):
        raise ValueError(
            "VALIDATION plaintext and ciphertext unit memberships differ"
        )

    train_sequences = [
        [
            tuple(token)
            for token in tokens
        ]
        for _, tokens in cipher_train_units
    ]

    model = TokenMarkovModel(
        max_context=int(max_context),
        char_base_order=int(char_base_order),
    ).fit(train_sequences)

    stats = build_training_statistics(
        plain_train_units,
        cipher_train_units,
        key=key,
        factorization=factorization,
    )

    seen_cipher_tokens = set(
        model.seen_tokens
    )
    seen_glyphs = set(
        model.char_base_model.seen_vocabulary
    )

    rows: List[dict] = []
    total_token_bits = 0.0
    total_trigram_bits = 0.0
    total_events = 0

    for line_index in sorted(
        plain_validation
    ):
        plain_line = tuple(
            plain_validation[line_index]
        )
        cipher_line = tuple(
            tuple(token)
            for token in cipher_validation[
                line_index
            ]
        )

        if len(plain_line) != len(cipher_line):
            raise ValueError(
                f"VALIDATION token-count mismatch on line {line_index}"
            )

        cipher_history: List[Token] = []
        previous_plain = PLAIN_BOS
        previous_cipher = CIPHER_BOS
        previous_variant = VARIANT_BOS

        for token_index, (
            plain_token,
            cipher_token,
        ) in enumerate(
            zip(
                plain_line,
                cipher_line,
            )
        ):
            token_probability = model.token_probability(
                cipher_token,
                cipher_history,
                context_order=int(
                    selected_context_order
                ),
            )
            trigram_probability = (
                model.character_token_probability(
                    cipher_token
                )
            )

            if (
                token_probability <= 0.0
                or trigram_probability <= 0.0
            ):
                raise RuntimeError(
                    "Scorer produced non-positive probability"
                )

            token_bits = -math.log2(
                token_probability
            )
            trigram_bits = -math.log2(
                trigram_probability
            )
            delta_bits = (
                token_bits
                - trigram_bits
            )
            events = len(
                cipher_token
            ) + 1

            current_variant = (
                decode_variant_index_from_key(
                    cipher_token,
                    key=key,
                    factorization=factorization,
                )
            )

            plain_features = (
                _transition_features(
                    previous=previous_plain,
                    current=plain_token,
                    token_counts=stats.plain_token_counts,
                    bigram_counts=stats.plain_bigram_counts,
                    outgoing=stats.plain_outgoing,
                    incoming=stats.plain_incoming,
                    bos_value=PLAIN_BOS,
                )
            )
            cipher_features = (
                _transition_features(
                    previous=previous_cipher,
                    current=cipher_token,
                    token_counts=stats.cipher_token_counts,
                    bigram_counts=stats.cipher_bigram_counts,
                    outgoing=stats.cipher_outgoing,
                    incoming=stats.cipher_incoming,
                    bos_value=CIPHER_BOS,
                )
            )

            variant_bigram_frequency = int(
                stats.variant_bigram_counts.get(
                    (
                        previous_variant,
                        current_variant,
                    ),
                    0,
                )
            )
            variant_next = (
                stats.variant_outgoing.get(
                    previous_variant,
                    Counter(),
                )
            )
            variant_outgoing_total = int(
                sum(
                    variant_next.values()
                )
            )
            variant_conditional_probability = (
                variant_bigram_frequency
                / variant_outgoing_total
                if variant_outgoing_total
                else 0.0
            )

            plain_rank = int(
                stats.plain_token_rank.get(
                    plain_token,
                    0,
                )
            )
            cipher_rank = int(
                stats.cipher_token_rank.get(
                    cipher_token,
                    0,
                )
            )

            record = {
                "source_id": source_id,
                "language": language,
                "factorization": factorization,
                "seed": int(seed),
                "selected_context_order": int(
                    selected_context_order
                ),
                "line_index": int(line_index),
                "token_index": int(token_index),
                "line_length": len(
                    plain_line
                ),
                "line_position_class": (
                    line_position_class(
                        token_index,
                        len(plain_line),
                    )
                ),
                "plaintext_token": plain_token,
                "previous_plaintext_token": (
                    previous_plain
                ),
                "plaintext_token_length": len(
                    plain_token
                ),
                "cipher_token_sha16": (
                    short_token_id(
                        cipher_token
                    )
                ),
                "previous_cipher_token_sha16": (
                    "BOS"
                    if previous_cipher
                    == CIPHER_BOS
                    else short_token_id(
                        previous_cipher
                    )
                ),
                "cipher_token_glyph_length": len(
                    cipher_token
                ),
                "current_variant_index": int(
                    current_variant
                ),
                "previous_variant_index": int(
                    previous_variant
                ),
                "train_variant_index_frequency": int(
                    stats.variant_counts.get(
                        current_variant,
                        0,
                    )
                ),
                "train_variant_bigram_frequency": (
                    variant_bigram_frequency
                ),
                "train_variant_conditional_probability": (
                    variant_conditional_probability
                ),
                "token_markov_probability": (
                    token_probability
                ),
                "matched_trigram_probability": (
                    trigram_probability
                ),
                "token_markov_bits": (
                    token_bits
                ),
                "matched_trigram_bits": (
                    trigram_bits
                ),
                "delta_bits": delta_bits,
                "events": int(events),
                "delta_bits_per_event": (
                    delta_bits
                    / events
                ),
                "token_markov_better": (
                    delta_bits < 0.0
                ),
                "trigram_better": (
                    delta_bits > 0.0
                ),
                "token_markov_advantage_bits": (
                    max(
                        -delta_bits,
                        0.0,
                    )
                ),
                "trigram_advantage_bits": (
                    max(
                        delta_bits,
                        0.0,
                    )
                ),
                "cipher_token_seen_in_train": (
                    cipher_token
                    in seen_cipher_tokens
                ),
                "cipher_oov_glyphs": sum(
                    glyph not in seen_glyphs
                    for glyph in cipher_token
                ),
                "train_plain_current_frequency": (
                    plain_features[
                        "current_frequency"
                    ]
                ),
                "train_plain_previous_frequency": (
                    plain_features[
                        "previous_frequency"
                    ]
                ),
                "train_plain_current_frequency_rank": (
                    plain_rank
                ),
                "train_plain_bigram_frequency": (
                    plain_features[
                        "bigram_frequency"
                    ]
                ),
                "train_plain_previous_outgoing_total": (
                    plain_features[
                        "outgoing_total"
                    ]
                ),
                "train_plain_previous_distinct_next_types": (
                    plain_features[
                        "distinct_next_types"
                    ]
                ),
                "train_plain_conditional_probability": (
                    plain_features[
                        "conditional_probability"
                    ]
                ),
                "train_plain_previous_transition_concentration": (
                    plain_features[
                        "transition_concentration"
                    ]
                ),
                "train_plain_current_predecessor_diversity": (
                    plain_features[
                        "current_predecessor_diversity"
                    ]
                ),
                "train_cipher_current_frequency": (
                    cipher_features[
                        "current_frequency"
                    ]
                ),
                "train_cipher_previous_frequency": (
                    cipher_features[
                        "previous_frequency"
                    ]
                ),
                "train_cipher_current_frequency_rank": (
                    cipher_rank
                ),
                "train_cipher_bigram_frequency": (
                    cipher_features[
                        "bigram_frequency"
                    ]
                ),
                "train_cipher_previous_outgoing_total": (
                    cipher_features[
                        "outgoing_total"
                    ]
                ),
                "train_cipher_previous_distinct_next_types": (
                    cipher_features[
                        "distinct_next_types"
                    ]
                ),
                "train_cipher_conditional_probability": (
                    cipher_features[
                        "conditional_probability"
                    ]
                ),
                "train_cipher_previous_transition_concentration": (
                    cipher_features[
                        "transition_concentration"
                    ]
                ),
                "train_cipher_current_predecessor_diversity": (
                    cipher_features[
                        "current_predecessor_diversity"
                    ]
                ),
            }

            record[
                "train_plain_current_frequency_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "train_plain_current_frequency"
                    ]
                ),
                bins[
                    "token_frequency"
                ],
            )
            record[
                "train_plain_bigram_frequency_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "train_plain_bigram_frequency"
                    ]
                ),
                bins[
                    "bigram_frequency"
                ],
            )
            record[
                "train_cipher_current_frequency_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "train_cipher_current_frequency"
                    ]
                ),
                bins[
                    "token_frequency"
                ],
            )
            record[
                "train_cipher_bigram_frequency_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "train_cipher_bigram_frequency"
                    ]
                ),
                bins[
                    "bigram_frequency"
                ],
            )
            record[
                "train_plain_conditional_probability_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "train_plain_conditional_probability"
                    ]
                ),
                bins[
                    "probability"
                ],
            )
            record[
                "train_cipher_conditional_probability_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "train_cipher_conditional_probability"
                    ]
                ),
                bins[
                    "probability"
                ],
            )
            record[
                "train_plain_previous_transition_concentration_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "train_plain_previous_transition_concentration"
                    ]
                ),
                bins[
                    "probability"
                ],
            )
            record[
                "plaintext_token_length_bin"
            ] = assign_upper_bound_bin(
                float(
                    record[
                        "plaintext_token_length"
                    ]
                ),
                bins[
                    "token_length"
                ],
            )

            for top_n in top_n_frequency_sets:
                record[
                    f"plain_current_top_{int(top_n)}"
                ] = bool(
                    plain_rank > 0
                    and plain_rank
                    <= int(top_n)
                )

            rows.append(
                record
            )

            total_token_bits += (
                token_bits
            )
            total_trigram_bits += (
                trigram_bits
            )
            total_events += events

            cipher_history.append(
                cipher_token
            )
            previous_plain = (
                plain_token
            )
            previous_cipher = (
                cipher_token
            )
            previous_variant = (
                current_variant
            )

    if total_events <= 0:
        raise RuntimeError(
            "No validation events were scored"
        )

    aggregate = {
        "source_id": source_id,
        "language": language,
        "factorization": factorization,
        "seed": int(seed),
        "selected_context_order": int(
            selected_context_order
        ),
        "validation_tokens": len(
            rows
        ),
        "validation_events": int(
            total_events
        ),
        "recomputed_competitor_bits_per_event": (
            total_token_bits
            / total_events
        ),
        "recomputed_trigram_bits_per_event": (
            total_trigram_bits
            / total_events
        ),
        "recomputed_delta_bits_per_event": (
            total_token_bits
            / total_events
            - total_trigram_bits
            / total_events
        ),
        "total_token_markov_bits": (
            total_token_bits
        ),
        "total_matched_trigram_bits": (
            total_trigram_bits
        ),
        "total_delta_bits": (
            total_token_bits
            - total_trigram_bits
        ),
    }

    return rows, aggregate


def verify_aggregate_parity(
    aggregate: Mapping[str, object],
    stored_relation: Mapping[str, object],
    *,
    tolerance: float,
) -> dict:
    """Verify token-level decomposition exactly reconstructs the frozen relation."""
    comparisons = {
        "competitor": (
            float(
                aggregate[
                    "recomputed_competitor_bits_per_event"
                ]
            ),
            float(
                stored_relation[
                    "competitor_bits_per_event"
                ]
            ),
        ),
        "trigram": (
            float(
                aggregate[
                    "recomputed_trigram_bits_per_event"
                ]
            ),
            float(
                stored_relation[
                    "trigram_bits_per_event"
                ]
            ),
        ),
        "delta": (
            float(
                aggregate[
                    "recomputed_delta_bits_per_event"
                ]
            ),
            float(
                stored_relation[
                    "delta_competitor_minus_trigram_bits_per_event"
                ]
            ),
        ),
    }

    errors = {
        name: abs(
            recomputed
            - stored
        )
        for name, (
            recomputed,
            stored,
        ) in comparisons.items()
    }

    passed = all(
        error <= tolerance
        for error in errors.values()
    )

    result = {
        **dict(aggregate),
        "stored_competitor_bits_per_event": (
            comparisons[
                "competitor"
            ][1]
        ),
        "stored_trigram_bits_per_event": (
            comparisons[
                "trigram"
            ][1]
        ),
        "stored_delta_bits_per_event": (
            comparisons[
                "delta"
            ][1]
        ),
        "competitor_abs_error": (
            errors[
                "competitor"
            ]
        ),
        "trigram_abs_error": (
            errors[
                "trigram"
            ]
        ),
        "delta_abs_error": (
            errors[
                "delta"
            ]
        ),
        "parity_tolerance": float(
            tolerance
        ),
        "parity_pass": bool(
            passed
        ),
    }

    if not passed:
        raise ValueError(
            "Token-level localization does not reconstruct the frozen "
            f"Phase-3A.5 relation within tolerance: {result}"
        )

    return result


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    group_fields: Sequence[str],
) -> List[dict]:
    """Summarize contributions with ratio-of-sums and seed-level dispersion."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            row[field]
            for field in group_fields
        )
        groups[key].append(row)

    output: List[dict] = []

    # Denominator for "where does token-Markov gain its bits?" within language.
    negative_mass_by_language_factor = defaultdict(float)
    for row in rows:
        negative_mass_by_language_factor[
            (
                row["language"],
                row["factorization"],
            )
        ] += float(
            row[
                "token_markov_advantage_bits"
            ]
        )

    for key in sorted(
        groups,
        key=lambda item: tuple(
            str(x)
            for x in item
        ),
    ):
        group = groups[key]

        tokens = len(group)
        events = sum(
            int(
                row[
                    "events"
                ]
            )
            for row in group
        )
        delta_bits = sum(
            float(
                row[
                    "delta_bits"
                ]
            )
            for row in group
        )
        token_advantage = sum(
            float(
                row[
                    "token_markov_advantage_bits"
                ]
            )
            for row in group
        )
        trigram_advantage = sum(
            float(
                row[
                    "trigram_advantage_bits"
                ]
            )
            for row in group
        )

        seed_groups = defaultdict(list)
        for row in group:
            seed_groups[
                int(
                    row[
                        "seed"
                    ]
                )
            ].append(row)

        run_ratios = []
        for seed_rows in seed_groups.values():
            run_events = sum(
                int(row["events"])
                for row in seed_rows
            )
            run_delta = sum(
                float(
                    row[
                        "delta_bits"
                    ]
                )
                for row in seed_rows
            )
            if run_events:
                run_ratios.append(
                    run_delta
                    / run_events
                )

        result = {
            field: value
            for field, value in zip(
                group_fields,
                key,
            )
        }

        language = str(
            group[0][
                "language"
            ]
        )
        factorization = str(
            group[0][
                "factorization"
            ]
        )
        total_negative_mass = (
            negative_mass_by_language_factor[
                (
                    language,
                    factorization,
                )
            ]
        )

        result.update(
            {
                "runs": len(
                    seed_groups
                ),
                "tokens": tokens,
                "events": events,
                "total_delta_bits": (
                    delta_bits
                ),
                "delta_bits_per_event_ratio_of_sums": (
                    delta_bits
                    / events
                    if events
                    else None
                ),
                "mean_seed_delta_bits_per_event": (
                    mean(
                        run_ratios
                    )
                    if run_ratios
                    else None
                ),
                "sd_seed_delta_bits_per_event": (
                    pstdev(
                        run_ratios
                    )
                    if len(
                        run_ratios
                    )
                    > 1
                    else 0.0
                ),
                "tokens_token_markov_better": sum(
                    bool(
                        row[
                            "token_markov_better"
                        ]
                    )
                    for row in group
                ),
                "token_markov_better_token_fraction": (
                    sum(
                        bool(
                            row[
                                "token_markov_better"
                            ]
                        )
                        for row in group
                    )
                    / tokens
                    if tokens
                    else None
                ),
                "token_markov_advantage_bits": (
                    token_advantage
                ),
                "trigram_advantage_bits": (
                    trigram_advantage
                ),
                "share_of_language_token_markov_advantage_bits": (
                    token_advantage
                    / total_negative_mass
                    if total_negative_mass
                    > 0.0
                    else 0.0
                ),
            }
        )

        output.append(
            result
        )

    return output


def summarize_language(
    rows: Sequence[Mapping[str, object]],
    *,
    relation_status_by_language: Mapping[str, str],
) -> List[dict]:
    """One descriptive row per language for the primary factorization."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            row["language"]
        ].append(row)

    output = []

    for language in sorted(
        grouped
    ):
        group = grouped[
            language
        ]
        seed_groups = defaultdict(list)
        for row in group:
            seed_groups[
                int(
                    row[
                        "seed"
                    ]
                )
            ].append(row)

        seed_delta = []
        for seed_rows in seed_groups.values():
            events = sum(
                int(row["events"])
                for row in seed_rows
            )
            delta = sum(
                float(
                    row[
                        "delta_bits"
                    ]
                )
                for row in seed_rows
            )
            seed_delta.append(
                delta / events
            )

        all_events = sum(
            int(row["events"])
            for row in group
        )
        all_delta = sum(
            float(
                row["delta_bits"]
            )
            for row in group
        )

        output.append(
            {
                "language": language,
                "phase3a5_token_relation_status": (
                    relation_status_by_language[
                        language
                    ]
                ),
                "seeds": len(
                    seed_groups
                ),
                "tokens": len(
                    group
                ),
                "events": all_events,
                "pooled_delta_bits_per_event": (
                    all_delta
                    / all_events
                ),
                "mean_seed_delta_bits_per_event": mean(
                    seed_delta
                ),
                "sd_seed_delta_bits_per_event": (
                    pstdev(
                        seed_delta
                    )
                    if len(
                        seed_delta
                    )
                    > 1
                    else 0.0
                ),
                "token_markov_better_token_fraction": (
                    sum(
                        bool(
                            row[
                                "token_markov_better"
                            ]
                        )
                        for row in group
                    )
                    / len(group)
                ),
                "token_markov_advantage_bits": sum(
                    float(
                        row[
                            "token_markov_advantage_bits"
                        ]
                    )
                    for row in group
                ),
                "trigram_advantage_bits": sum(
                    float(
                        row[
                            "trigram_advantage_bits"
                        ]
                    )
                    for row in group
                ),
                "line_initial_token_fraction": (
                    sum(
                        row[
                            "line_position_class"
                        ]
                        in {
                            "initial",
                            "singleton",
                        }
                        for row in group
                    )
                    / len(group)
                ),
            }
        )

    return output


def summarize_top_frequency_sets(
    rows: Sequence[Mapping[str, object]],
    *,
    top_n_frequency_sets: Sequence[int],
) -> List[dict]:
    expanded = []

    for top_n in top_n_frequency_sets:
        field = (
            f"plain_current_top_{int(top_n)}"
        )
        for row in rows:
            copy = dict(row)
            copy["top_n"] = int(
                top_n
            )
            copy[
                "top_n_membership"
            ] = (
                "in_top_n"
                if bool(
                    row[
                        field
                    ]
                )
                else "outside_top_n"
            )
            expanded.append(
                copy
            )

    return summarize_rows(
        expanded,
        group_fields=(
            "language",
            "factorization",
            "top_n",
            "top_n_membership",
        ),
    )


def compact_transition_rows(
    rows: Sequence[Mapping[str, object]],
) -> List[dict]:
    fields = [
        "source_id",
        "language",
        "factorization",
        "seed",
        "selected_context_order",
        "line_index",
        "token_index",
        "previous_plaintext_token",
        "plaintext_token",
        "previous_variant_index",
        "current_variant_index",
        "delta_bits",
        "events",
        "delta_bits_per_event",
        "token_markov_better",
        "train_plain_previous_frequency",
        "train_plain_current_frequency",
        "train_plain_bigram_frequency",
        "train_plain_conditional_probability",
        "train_plain_previous_distinct_next_types",
        "train_plain_previous_transition_concentration",
        "train_cipher_previous_frequency",
        "train_cipher_current_frequency",
        "train_cipher_bigram_frequency",
        "train_cipher_conditional_probability",
        "train_cipher_previous_distinct_next_types",
        "train_cipher_previous_transition_concentration",
        "train_variant_bigram_frequency",
        "train_variant_conditional_probability",
    ]

    output = []
    for row in rows:
        if int(
            row[
                "token_index"
            ]
        ) == 0:
            continue
        output.append(
            {
                field: row[
                    field
                ]
                for field in fields
            }
        )
    return output
