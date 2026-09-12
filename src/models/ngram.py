"""Validation-only n-gram baselines for VOYAGER.

This module implements the preregistered Tier 1 unigram -> 5-gram baseline
family. Models are fit on the frozen training physical leaves and evaluated
only on the frozen validation physical leaves.

Two frozen analytical views are supported:
- ``space_free``: STA1 glyphs concatenated within each locus; conventional
  token boundaries are omitted.
- ``token_aware``: the same STA1 glyphs, with an explicit ``<WB>`` event
  between conventional S0 tokens.

Unknown transcription symbol ``Z1`` is not treated as a manuscript glyph for
prediction. It splits a locus into independent observable segments and is
neither trained on nor scored. This prevents a model from learning across an
unreadable span.

Smoothing is interpolated Witten-Bell. It has no tuned smoothing parameter.
For an observed history h,

    P(w|h) = c(hw)/(N(h)+T(h))
             + T(h)/(N(h)+T(h)) * P(w|suffix(h))

and unseen histories back off directly. The unigram distribution reserves the
standard Witten-Bell unseen mass for a single ``<UNK>`` bucket.

No locked-test file is required or accessed by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.analysis.tier0 import (
    AnalyticalLocus,
    UNKNOWN_STA1,
)
from src.data.ivtff import LocusRecord


BOS = "<BOS>"
WB = "<WB>"
UNK = "<UNK>"
VALID_VIEWS = ("space_free", "token_aware")


@dataclass(frozen=True)
class BaselineSequence:
    """One observable sequence segment derived from a validation/train locus."""

    leaf_group: str
    folio: str
    locus: str
    symbols: Tuple[str, ...]
    is_boundary: Tuple[bool, ...]


def parse_leaf_ids(text: str, *, label: str) -> Tuple[int, ...]:
    leaves: List[int] = []
    seen = set()

    for line_number, raw in enumerate(text.splitlines(), start=1):
        value = raw.strip()
        if not value:
            continue
        match = re.fullmatch(r"f(\d+)", value)
        if not match:
            raise ValueError(
                f"Invalid {label} leaf ID on line {line_number}: {value!r}"
            )
        leaf = int(match.group(1))
        if leaf in seen:
            raise ValueError(f"Duplicate {label} leaf: f{leaf}")
        seen.add(leaf)
        leaves.append(leaf)

    return tuple(sorted(leaves))


def select_records_by_leaves(
    records: Sequence[LocusRecord],
    leaves: Sequence[int],
    *,
    split_name: str,
) -> List[LocusRecord]:
    """Select records for one split without consulting any other split.

    ``fRos`` follows the linked f85/f86 unit. If neither leaf is present, fRos
    is omitted. If exactly one is present, the frozen leakage rule is broken
    and evaluation aborts.
    """
    selected_leaves = set(leaves)

    if (85 in selected_leaves) != (86 in selected_leaves):
        raise ValueError(
            f"{split_name}: frozen Rosettes leakage control violated; "
            "f85 and f86 must be assigned together."
        )

    include_ros = 85 in selected_leaves and 86 in selected_leaves
    selected: List[LocusRecord] = []

    for record in records:
        if record.physical_leaf is not None:
            if record.physical_leaf in selected_leaves:
                selected.append(record)
            continue

        if record.folio == "fRos":
            if include_ros:
                selected.append(record)
            continue

        raise ValueError(
            "Encountered a source record without a physical leaf and without "
            f"a declared selection rule: {record.folio} {record.locus}"
        )

    observed = {
        r.physical_leaf
        for r in selected
        if r.physical_leaf is not None
    }
    missing = selected_leaves - observed
    if missing:
        raise ValueError(
            f"{split_name} contains leaves with no selected records: "
            + ", ".join(f"f{x}" for x in sorted(missing))
        )

    return selected


def atomic_leaf_group(record: AnalyticalLocus) -> str:
    """Return the primary split unit for per-leaf aggregation."""
    if record.folio == "fRos" or record.physical_leaf in (85, 86):
        return "f85+f86"
    if record.physical_leaf is None:
        raise ValueError(
            f"No atomic leaf group for {record.folio} {record.locus}"
        )
    return f"f{record.physical_leaf}"


def _split_on_unknown(
    symbols: Sequence[str],
    boundaries: Sequence[bool],
    *,
    token_aware: bool,
) -> List[Tuple[Tuple[str, ...], Tuple[bool, ...]]]:
    if len(symbols) != len(boundaries):
        raise ValueError("symbol/boundary arrays differ in length")

    result: List[Tuple[Tuple[str, ...], Tuple[bool, ...]]] = []
    current_symbols: List[str] = []
    current_boundaries: List[bool] = []

    def flush() -> None:
        nonlocal current_symbols, current_boundaries

        if token_aware:
            while current_symbols and current_symbols[0] == WB:
                current_symbols.pop(0)
                current_boundaries.pop(0)
            while current_symbols and current_symbols[-1] == WB:
                current_symbols.pop()
                current_boundaries.pop()

        if current_symbols:
            result.append(
                (tuple(current_symbols), tuple(current_boundaries))
            )

        current_symbols = []
        current_boundaries = []

    for symbol, is_boundary in zip(symbols, boundaries):
        if symbol == UNKNOWN_STA1:
            flush()
            continue

        current_symbols.append(symbol)
        current_boundaries.append(is_boundary)

    flush()
    return result


def build_baseline_sequences(
    loci: Sequence[AnalyticalLocus],
    *,
    view: str,
) -> List[BaselineSequence]:
    if view not in VALID_VIEWS:
        raise ValueError(f"Unknown n-gram view: {view!r}")

    token_aware = view == "token_aware"
    output: List[BaselineSequence] = []

    for locus in loci:
        symbols: List[str] = []
        boundaries: List[bool] = []

        for token_index, token in enumerate(locus.tokens):
            if token_aware and token_index > 0:
                symbols.append(WB)
                boundaries.append(True)

            for glyph in token:
                symbols.append(glyph)
                boundaries.append(False)

        for segment_symbols, segment_boundaries in _split_on_unknown(
            symbols,
            boundaries,
            token_aware=token_aware,
        ):
            output.append(
                BaselineSequence(
                    leaf_group=atomic_leaf_group(locus),
                    folio=locus.folio,
                    locus=locus.locus,
                    symbols=segment_symbols,
                    is_boundary=segment_boundaries,
                )
            )

    return output


class WittenBellNGram:
    """Interpolated Witten-Bell model fit once through a maximum order."""

    def __init__(self, max_order: int):
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        self.max_order = max_order
        self._counts: List[Dict[Tuple[str, ...], Counter]] = [
            defaultdict(Counter)
            for _ in range(max_order)
        ]
        self._seen_vocabulary: set[str] = set()
        self._training_events = 0
        self._fitted = False

    @property
    def training_events(self) -> int:
        return self._training_events

    @property
    def seen_vocabulary(self) -> Tuple[str, ...]:
        return tuple(sorted(self._seen_vocabulary))

    @property
    def support(self) -> Tuple[str, ...]:
        return tuple(sorted(self._seen_vocabulary)) + (UNK,)

    def fit(self, sequences: Iterable[Sequence[str]]) -> "WittenBellNGram":
        if self._fitted:
            raise RuntimeError("Model has already been fit")

        for sequence in sequences:
            seq = tuple(sequence)
            if not seq:
                continue

            self._seen_vocabulary.update(seq)
            padded = (BOS,) * (self.max_order - 1) + seq

            for i, symbol in enumerate(seq):
                position = self.max_order - 1 + i
                self._training_events += 1

                for history_len in range(self.max_order):
                    if history_len == 0:
                        history: Tuple[str, ...] = ()
                    else:
                        history = tuple(
                            padded[position - history_len : position]
                        )
                    self._counts[history_len][history][symbol] += 1

        if not self._seen_vocabulary or self._training_events == 0:
            raise ValueError("Cannot fit n-gram model to an empty corpus")

        self._fitted = True
        return self

    def _map_symbol(self, symbol: str) -> str:
        if symbol in self._seen_vocabulary:
            return symbol
        return UNK

    def probability(
        self,
        symbol: str,
        history: Sequence[str],
        *,
        order: int,
    ) -> float:
        if not self._fitted:
            raise RuntimeError("Model is not fit")
        if order < 1 or order > self.max_order:
            raise ValueError(
                f"order must be within 1..{self.max_order}, found {order}"
            )

        mapped = self._map_symbol(symbol)
        history_len = order - 1
        h = tuple(history[-history_len:]) if history_len else ()

        return self._prob_recursive(mapped, h)

    def _prob_recursive(
        self,
        mapped_symbol: str,
        history: Tuple[str, ...],
    ) -> float:
        if not history:
            counts = self._counts[0][()]
            n = sum(counts.values())
            t = len(self._seen_vocabulary)
            if n <= 0 or t <= 0:
                raise RuntimeError("Invalid unigram count state")

            if mapped_symbol == UNK:
                return t / (n + t)

            return counts.get(mapped_symbol, 0) / (n + t)

        counts = self._counts[len(history)].get(history)
        if not counts:
            return self._prob_recursive(
                mapped_symbol,
                history[1:],
            )

        n = sum(counts.values())
        t = len(counts)
        lower = self._prob_recursive(
            mapped_symbol,
            history[1:],
        )
        observed = counts.get(mapped_symbol, 0)

        return (
            observed / (n + t)
            + (t / (n + t)) * lower
        )

    def context_counts(self) -> List[dict]:
        rows = []
        for history_len in range(self.max_order):
            mapping = self._counts[history_len]
            transitions = sum(
                sum(counter.values())
                for counter in mapping.values()
            )
            rows.append(
                {
                    "history_length": history_len,
                    "ngram_order": history_len + 1,
                    "contexts": len(mapping),
                    "transitions": transitions,
                }
            )
        return rows


def _eval_sequence(
    model: WittenBellNGram,
    example: BaselineSequence,
    *,
    order: int,
) -> dict:
    symbols = example.symbols
    boundaries = example.is_boundary
    padded = (BOS,) * (order - 1) + symbols

    total_bits = 0.0
    glyph_bits = 0.0
    boundary_bits = 0.0
    total_events = 0
    glyph_events = 0
    boundary_events = 0
    oov_events = 0

    for i, (symbol, is_boundary) in enumerate(
        zip(symbols, boundaries)
    ):
        position = order - 1 + i
        history = (
            ()
            if order == 1
            else padded[position - (order - 1) : position]
        )
        probability = model.probability(
            symbol,
            history,
            order=order,
        )
        if probability <= 0.0:
            raise RuntimeError(
                f"Zero probability for symbol={symbol!r}, history={history!r}"
            )

        bits = -math.log2(probability)
        total_bits += bits
        total_events += 1

        if symbol not in model._seen_vocabulary:
            oov_events += 1

        if is_boundary:
            boundary_bits += bits
            boundary_events += 1
        else:
            glyph_bits += bits
            glyph_events += 1

    return {
        "total_bits": total_bits,
        "total_events": total_events,
        "glyph_bits": glyph_bits,
        "glyph_events": glyph_events,
        "boundary_bits": boundary_bits,
        "boundary_events": boundary_events,
        "oov_events": oov_events,
    }


def evaluate_model(
    model: WittenBellNGram,
    examples: Sequence[BaselineSequence],
    *,
    orders: Sequence[int],
    view: str,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Return aggregate, per-leaf, and per-locus validation metrics."""
    order_set = tuple(sorted(set(orders)))
    if not order_set:
        raise ValueError("At least one order is required")

    locus_acc: Dict[Tuple[int, str, str, str], Counter] = defaultdict(Counter)

    for example in examples:
        for order in order_set:
            metrics = _eval_sequence(model, example, order=order)
            key = (
                order,
                example.leaf_group,
                example.folio,
                example.locus,
            )
            acc = locus_acc[key]
            for metric, value in metrics.items():
                acc[metric] += value

    locus_rows: List[dict] = []
    for (
        order,
        leaf_group,
        folio,
        locus,
    ), acc in sorted(locus_acc.items()):
        row = {
            "view": view,
            "order": order,
            "leaf_group": leaf_group,
            "folio": folio,
            "locus": locus,
            **dict(acc),
        }
        row["bits_per_event"] = (
            row["total_bits"] / row["total_events"]
            if row["total_events"]
            else None
        )
        row["glyph_bits_per_glyph"] = (
            row["glyph_bits"] / row["glyph_events"]
            if row["glyph_events"]
            else None
        )
        locus_rows.append(row)

    leaf_acc: Dict[Tuple[int, str], Counter] = defaultdict(Counter)
    for row in locus_rows:
        key = (row["order"], row["leaf_group"])
        acc = leaf_acc[key]
        for metric in (
            "total_bits",
            "total_events",
            "glyph_bits",
            "glyph_events",
            "boundary_bits",
            "boundary_events",
            "oov_events",
        ):
            acc[metric] += row[metric]

    leaf_rows: List[dict] = []
    for (order, leaf_group), acc in sorted(leaf_acc.items()):
        row = {
            "view": view,
            "order": order,
            "leaf_group": leaf_group,
            **dict(acc),
        }
        row["bits_per_event"] = row["total_bits"] / row["total_events"]
        row["glyph_bits_per_glyph"] = (
            row["glyph_bits"] / row["glyph_events"]
            if row["glyph_events"]
            else None
        )
        leaf_rows.append(row)

    aggregate_acc: Dict[int, Counter] = defaultdict(Counter)
    for row in leaf_rows:
        acc = aggregate_acc[row["order"]]
        for metric in (
            "total_bits",
            "total_events",
            "glyph_bits",
            "glyph_events",
            "boundary_bits",
            "boundary_events",
            "oov_events",
        ):
            acc[metric] += row[metric]

    aggregate_rows: List[dict] = []
    for order in order_set:
        acc = aggregate_acc[order]
        total_bits = acc["total_bits"]
        total_events = acc["total_events"]
        glyph_bits = acc["glyph_bits"]
        glyph_events = acc["glyph_events"]
        boundary_bits = acc["boundary_bits"]
        boundary_events = acc["boundary_events"]

        aggregate_rows.append(
            {
                "view": view,
                "order": order,
                "smoothing": "interpolated_witten_bell",
                "total_bits": total_bits,
                "total_events": total_events,
                "bits_per_event": total_bits / total_events,
                "perplexity_per_event": 2 ** (total_bits / total_events),
                "glyph_bits": glyph_bits,
                "glyph_events": glyph_events,
                "glyph_bits_per_glyph": glyph_bits / glyph_events,
                "glyph_perplexity": 2 ** (glyph_bits / glyph_events),
                "boundary_bits": boundary_bits,
                "boundary_events": boundary_events,
                "boundary_bits_per_boundary": (
                    boundary_bits / boundary_events
                    if boundary_events
                    else None
                ),
                "oov_events": int(acc["oov_events"]),
            }
        )

    unigram = next(row for row in aggregate_rows if row["order"] == 1)
    for row in aggregate_rows:
        row["delta_bits_per_event_vs_unigram"] = (
            unigram["bits_per_event"] - row["bits_per_event"]
        )
        row["relative_nll_reduction_vs_unigram"] = (
            1.0 - row["total_bits"] / unigram["total_bits"]
        )

    return aggregate_rows, leaf_rows, locus_rows
