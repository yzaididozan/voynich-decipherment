"""Matched Phase 6A null: independently reshuffle the K=16 cycle each cycle.

Repository destination:
    src/ciphers/recurrence_cycle_shuffle_null.py

This null inherits the frozen recurrence-aware K=16 mechanism's:
- TRAIN-fitted monoalphabetic base map,
- 16 unique one-glyph suffixes,
- one suffix glyph per token,
- exactly-once use of every suffix within each block of 16 occurrences
  of one encoded lexical base.

It changes only the across-cycle allocation rule.  The frozen candidate reuses
the same base-specific permutation every cycle.  This null derives a new
deterministic base/seed/cycle-specific permutation for each 16-occurrence cycle.

The module is deterministic and contains no Voynich access or scoring logic.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence, Tuple

from src.ciphers import recurrence_avoiding_surface_allocation as fixed_cycle


GlyphToken = Tuple[str, ...]
VARIANT_COUNT = fixed_cycle.VARIANT_COUNT
ALLOCATION_NAMESPACE = "phase6a-cycle-shuffle-null-v1"


def _serialized_base(base: Sequence[str]) -> str:
    return json.dumps(
        list(base),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def cycle_permutation(
    base: Sequence[str],
    *,
    seed: int,
    cycle_number: int,
) -> Tuple[int, ...]:
    """Return one deterministic permutation of 0..15 for a specific cycle."""
    if cycle_number < 0:
        raise ValueError("cycle_number must be non-negative")

    serialized = _serialized_base(base)
    ordering = sorted(
        range(VARIANT_COUNT),
        key=lambda variant_index: (
            fixed_cycle.stable_uint64(
                ALLOCATION_NAMESPACE,
                int(seed),
                serialized,
                int(cycle_number),
                int(variant_index),
            ),
            int(variant_index),
        ),
    )
    permutation = tuple(int(value) for value in ordering)
    if sorted(permutation) != list(range(VARIANT_COUNT)):
        raise RuntimeError("Cycle-shuffle ordering is not a permutation of 0..15")
    return permutation


def variant_index_for_occurrence(
    base: Sequence[str],
    *,
    seed: int,
    occurrence_number: int,
) -> int:
    """Map one zero-based lexical-base occurrence to a shuffled-cycle variant."""
    if occurrence_number < 0:
        raise ValueError("occurrence_number must be non-negative")

    cycle_number, cycle_position = divmod(
        int(occurrence_number),
        VARIANT_COUNT,
    )
    permutation = cycle_permutation(
        base,
        seed=int(seed),
        cycle_number=cycle_number,
    )
    return int(permutation[cycle_position])


def encode_token_for_occurrence(
    plaintext_token: str,
    *,
    key: Mapping[str, object],
    seed: int,
    occurrence_number: int,
) -> GlyphToken:
    """Encode with the frozen base map/suffix inventory and this null allocator."""
    base = tuple(
        fixed_cycle.encode_base(
            plaintext_token,
            key=key,
        )
    )
    variant_index = variant_index_for_occurrence(
        base,
        seed=int(seed),
        occurrence_number=int(occurrence_number),
    )
    suffix = fixed_cycle.suffix_for_variant(
        key,
        variant_index,
    )
    return (*base, suffix)
