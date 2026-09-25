"""Matched Phase 6A null: IID K=16 suffix allocation.

Repository destination:
    src/ciphers/recurrence_iid_null.py

This null inherits the frozen recurrence-aware K=16 mechanism's:
- TRAIN-fitted monoalphabetic base map,
- 16 unique one-glyph suffixes,
- one suffix glyph per token.

It changes only the occurrence-to-variant allocation rule.  Each occurrence
receives a deterministic pseudo-random variant in 0..15, keyed by
base/seed/occurrence number.  Unlike the frozen candidate and the cycle-shuffle
null, it does not enforce without-replacement use inside 16-occurrence blocks.

The module is deterministic and contains no Voynich access or scoring logic.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence, Tuple

from src.ciphers import recurrence_avoiding_surface_allocation as fixed_cycle


GlyphToken = Tuple[str, ...]
VARIANT_COUNT = fixed_cycle.VARIANT_COUNT
ALLOCATION_NAMESPACE = "phase6a-iid-k16-null-v1"


def _serialized_base(base: Sequence[str]) -> str:
    return json.dumps(
        list(base),
        separators=(",", ":"),
        ensure_ascii=False,
    )


def variant_index_for_occurrence(
    base: Sequence[str],
    *,
    seed: int,
    occurrence_number: int,
) -> int:
    """Map one occurrence independently to one of the inherited 16 variants."""
    if occurrence_number < 0:
        raise ValueError("occurrence_number must be non-negative")

    value = fixed_cycle.stable_uint64(
        ALLOCATION_NAMESPACE,
        int(seed),
        _serialized_base(base),
        int(occurrence_number),
    )
    return int(value % VARIANT_COUNT)


def encode_token_for_occurrence(
    plaintext_token: str,
    *,
    key: Mapping[str, object],
    seed: int,
    occurrence_number: int,
) -> GlyphToken:
    """Encode with the frozen base map/suffix inventory and this IID allocator."""
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
