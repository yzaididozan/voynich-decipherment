"""Train-only Tier 0 descriptive structural analysis for VOYAGER.

This module is deliberately descriptive. It does not fit predictive models,
select hyperparameters, inspect validation data, or access the locked test.

Primary representation (S0)
---------------------------
- primary transcription: ZL3b;
- conventional ZL/EVA token segmentation;
- both certain spaces ``.`` and uncertain spaces ``,`` are treated as token
  boundaries in S0;
- drawing interruptions ``<->`` and ``<~>`` are boundaries;
- IVTFF comments/tags are excluded from textual units;
- alternative readings use the first reading;
- native EVA tokens are converted through the frozen official EVA->STA1
  bitrans rules so glyph statistics use STA1 symbols rather than Python
  characters;
- ``Z1`` unknown symbols remain visible in token lengths/counts but are
  excluded from core entropy, n-gram, branching, and glyph-MI estimates.

All analytical transformations are derived; raw IVTFF is never modified.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from src.data.ivtff import LocusRecord
from src.data.sta1 import (
    STA1RuleSet,
    convert_native_text_to_sta1,
    iter_sta1_symbols,
    resolve_alternatives_first,
)


UNKNOWN_STA1 = "Z1"
S0_REPRESENTATION = "S0_ZL_CONVENTIONAL_STA1_GLYPHS"

_ANGLE_RE = re.compile(r"<[^>\n]*>")
_DRAWING_RE = re.compile(r"<(?:-|~)>")
_SPLIT_RE = re.compile(r"[.,\s]+")
_ALT_RE = re.compile(r"\[[^\]]+:[^\]]+\]")
_HIGH_ASCII_RE = re.compile(r"@\d+;")
_COMMENT_RE = re.compile(r"<![^>\n]*>")
_UNKNOWN_RUN_RE = re.compile(r"\?+")


@dataclass(frozen=True)
class AnalyticalLocus:
    """One train-set locus after S0 analytical tokenization."""

    folio: str
    physical_leaf: Optional[int]
    recto_verso: Optional[str]
    quire: Optional[int]
    section: str
    currier: str
    scribe: str
    locus: str
    locus_type: str
    paragraph: Optional[int]
    line: int
    source_line_number: int
    tokens: Tuple[Tuple[str, ...], ...]


@dataclass(frozen=True)
class Tier0Config:
    entropy_max_order: int = 4
    glyph_mi_max_distance: int = 8
    token_mi_max_distance: int = 5
    edit_vocab_limit: int = 300
    edit_max_distance: int = 2
    edit_example_limit: int = 10
    top_items_in_summary: int = 50


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path) -> Tier0Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(Tier0Config.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown Tier 0 config keys: {sorted(unknown)}")
    config = Tier0Config(**raw)

    if config.entropy_max_order < 1:
        raise ValueError("entropy_max_order must be >= 1")
    if config.glyph_mi_max_distance < 1:
        raise ValueError("glyph_mi_max_distance must be >= 1")
    if config.token_mi_max_distance < 1:
        raise ValueError("token_mi_max_distance must be >= 1")
    if config.edit_vocab_limit < 2:
        raise ValueError("edit_vocab_limit must be >= 2")
    if config.edit_max_distance not in (1, 2):
        raise ValueError("edit_max_distance must be 1 or 2")
    return config


def parse_train_leaf_ids(text: str) -> Tuple[int, ...]:
    leaves: List[int] = []
    seen = set()

    for line_number, raw in enumerate(text.splitlines(), start=1):
        value = raw.strip()
        if not value:
            continue
        match = re.fullmatch(r"f(\d+)", value)
        if not match:
            raise ValueError(
                f"Invalid train leaf ID on line {line_number}: {value!r}"
            )
        leaf = int(match.group(1))
        if leaf in seen:
            raise ValueError(f"Duplicate train leaf: f{leaf}")
        seen.add(leaf)
        leaves.append(leaf)

    return tuple(sorted(leaves))


def select_train_records(
    records: Sequence[LocusRecord],
    train_leaves: Sequence[int],
) -> List[LocusRecord]:
    """Select only train records, including fRos iff f85/f86 are train.

    This function never needs validation or test memberships.
    """
    train = set(train_leaves)

    if (85 in train) != (86 in train):
        raise ValueError(
            "Frozen Rosettes leakage control violated: f85 and f86 "
            "must be assigned together."
        )

    include_ros = 85 in train and 86 in train
    selected: List[LocusRecord] = []

    for record in records:
        if record.physical_leaf is not None:
            if record.physical_leaf in train:
                selected.append(record)
            continue

        if record.folio == "fRos":
            if include_ros:
                selected.append(record)
            continue

        raise ValueError(
            "Encountered a source record without a physical leaf and "
            f"without a declared train-selection rule: {record.folio} "
            f"{record.locus}"
        )

    observed = {
        r.physical_leaf
        for r in selected
        if r.physical_leaf is not None
    }
    missing = train - observed
    if missing:
        raise ValueError(
            "Train split contains physical leaves with no selected records: "
            + ", ".join(f"f{x}" for x in sorted(missing))
        )

    if any(
        r.physical_leaf is not None and r.physical_leaf not in train
        for r in selected
    ):
        raise AssertionError("Non-train physical leaf leaked into Tier 0")

    return selected


def _prepare_native_s0(text: str) -> str:
    """Resolve IVTFF presentation/uncertainty syntax for S0 tokenization."""
    text = resolve_alternatives_first(text)
    text = _DRAWING_RE.sub(".", text)
    text = _ANGLE_RE.sub("", text)
    return text


def tokenize_s0_sta1(
    text: str,
    rules: STA1RuleSet,
) -> Tuple[Tuple[str, ...], ...]:
    """Convert one native ZL locus to S0 tokens of STA1 glyph symbols."""
    prepared = _prepare_native_s0(text)
    native_tokens = [x for x in _SPLIT_RE.split(prepared) if x]

    tokens: List[Tuple[str, ...]] = []
    for native_token in native_tokens:
        converted = convert_native_text_to_sta1(
            native_token,
            rules,
            strict=True,
        )
        symbols = tuple(
            symbol
            for symbol in iter_sta1_symbols(
                converted,
                first_alternative=False,
            )
            if symbol is not None
        )
        if symbols:
            tokens.append(symbols)

    return tuple(tokens)


def make_analytical_loci(
    records: Sequence[LocusRecord],
    rules: STA1RuleSet,
) -> List[AnalyticalLocus]:
    result = []
    for record in records:
        result.append(
            AnalyticalLocus(
                folio=record.folio,
                physical_leaf=record.physical_leaf,
                recto_verso=record.recto_verso,
                quire=record.quire,
                section=record.section or "UNSET",
                currier=record.currier or "UNSET",
                scribe=record.scribe or "UNSET",
                locus=record.locus,
                locus_type=record.locus_type,
                paragraph=record.paragraph,
                line=record.line,
                source_line_number=record.source_line_number,
                tokens=tokenize_s0_sta1(record.text_normalized, rules),
            )
        )
    return result


def token_key(token: Sequence[str]) -> str:
    return "".join(token)


def entropy_from_counter(counter: Mapping[object, int]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
        if count > 0
    )


def empirical_conditional_entropy(
    sequences: Iterable[Sequence[str]],
    order: int,
    *,
    exclude_symbol: Optional[str] = UNKNOWN_STA1,
) -> Tuple[float, int, int]:
    """Weighted empirical H(X_t | X_{t-order:t}) within sequences."""
    if order < 1:
        raise ValueError("order must be >= 1")

    context_next: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    observations = 0

    for sequence in sequences:
        seq = list(sequence)
        for i in range(order, len(seq)):
            context = tuple(seq[i - order : i])
            nxt = seq[i]
            if exclude_symbol is not None and (
                nxt == exclude_symbol or exclude_symbol in context
            ):
                continue
            context_next[context][nxt] += 1
            observations += 1

    if observations == 0:
        return 0.0, 0, 0

    weighted = 0.0
    for next_counts in context_next.values():
        context_total = sum(next_counts.values())
        weighted += (
            context_total / observations
        ) * entropy_from_counter(next_counts)

    return weighted, observations, len(context_next)


def branching_entropy(
    sequences: Iterable[Sequence[str]],
    context_order: int,
) -> dict:
    h, observations, contexts = empirical_conditional_entropy(
        sequences,
        context_order,
        exclude_symbol=UNKNOWN_STA1,
    )

    context_next: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for sequence in sequences:
        seq = list(sequence)
        for i in range(context_order, len(seq)):
            context = tuple(seq[i - context_order : i])
            nxt = seq[i]
            if UNKNOWN_STA1 in context or nxt == UNKNOWN_STA1:
                continue
            context_next[context][nxt] += 1

    entropies = [
        entropy_from_counter(counter)
        for counter in context_next.values()
    ]

    return {
        "context_order": context_order,
        "weighted_next_symbol_entropy_bits": h,
        "unweighted_mean_context_entropy_bits": mean(entropies)
        if entropies
        else 0.0,
        "contexts": contexts,
        "transitions": observations,
    }


def distribution_stats(values: Sequence[int]) -> dict:
    if not values:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }

    ordered = sorted(values)

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = q * (len(ordered) - 1)
        lo = math.floor(position)
        hi = math.ceil(position)
        if lo == hi:
            return float(ordered[lo])
        weight = position - lo
        return ordered[lo] * (1 - weight) + ordered[hi] * weight

    return {
        "n": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean(ordered),
        "median": median(ordered),
        "p25": quantile(0.25),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
    }


def ngram_counts(
    sequences: Iterable[Sequence[str]],
    n: int,
    *,
    exclude_symbol: Optional[str] = UNKNOWN_STA1,
) -> Counter:
    counts = Counter()
    for sequence in sequences:
        seq = list(sequence)
        for i in range(0, len(seq) - n + 1):
            gram = tuple(seq[i : i + n])
            if exclude_symbol is not None and exclude_symbol in gram:
                continue
            counts[gram] += 1
    return counts


def mutual_information_by_distance(
    sequences: Iterable[Sequence[str]],
    max_distance: int,
    *,
    exclude_symbol: Optional[str] = UNKNOWN_STA1,
) -> List[dict]:
    """Empirical MI I(X_i; X_{i+d}) in bits within sequence boundaries."""
    materialized = [tuple(seq) for seq in sequences]
    rows = []

    for distance in range(1, max_distance + 1):
        joint = Counter()
        left = Counter()
        right = Counter()
        total = 0

        for seq in materialized:
            for i in range(len(seq) - distance):
                a = seq[i]
                b = seq[i + distance]
                if exclude_symbol is not None and (
                    a == exclude_symbol or b == exclude_symbol
                ):
                    continue
                joint[(a, b)] += 1
                left[a] += 1
                right[b] += 1
                total += 1

        mi = 0.0
        if total:
            for (a, b), count in joint.items():
                p_ab = count / total
                p_a = left[a] / total
                p_b = right[b] / total
                mi += p_ab * math.log2(p_ab / (p_a * p_b))

        rows.append(
            {
                "distance": distance,
                "mutual_information_bits": mi,
                "pairs": total,
                "left_states": len(left),
                "right_states": len(right),
            }
        )

    return rows


def jensen_shannon_divergence(
    a: Mapping[str, int],
    b: Mapping[str, int],
) -> float:
    """Jensen-Shannon divergence in bits, range [0, 1]."""
    keys = set(a) | set(b)
    total_a = sum(a.values())
    total_b = sum(b.values())

    if total_a == 0 or total_b == 0:
        return 0.0

    js = 0.0
    for key in keys:
        p = a.get(key, 0) / total_a
        q = b.get(key, 0) / total_b
        m = 0.5 * (p + q)

        if p > 0:
            js += 0.5 * p * math.log2(p / m)
        if q > 0:
            js += 0.5 * q * math.log2(q / m)

    return js


def pairwise_js_rows(
    grouped: Mapping[str, Counter],
    dimension: str,
) -> List[dict]:
    labels = sorted(grouped)
    rows = []
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            rows.append(
                {
                    "dimension": dimension,
                    "group_a": left,
                    "group_b": right,
                    "js_divergence_bits": jensen_shannon_divergence(
                        grouped[left],
                        grouped[right],
                    ),
                    "glyphs_a": sum(grouped[left].values()),
                    "glyphs_b": sum(grouped[right].values()),
                }
            )
    return rows


def _levenshtein_bounded(
    a: Sequence[str],
    b: Sequence[str],
    max_distance: int,
) -> int:
    """Levenshtein distance with cutoff; returns max_distance+1 if larger."""
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        row_min = current[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            value = min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current

    return previous[-1]


def edit_neighborhood_rows(
    token_counts: Counter,
    token_sequences: Mapping[str, Sequence[str]],
    *,
    vocab_limit: int,
    max_distance: int,
    example_limit: int,
) -> List[dict]:
    ranked = sorted(
        token_counts,
        key=lambda token: (-token_counts[token], token),
    )[:vocab_limit]

    neighbor_map: Dict[str, Dict[int, List[str]]] = {
        token: {d: [] for d in range(1, max_distance + 1)}
        for token in ranked
    }

    for i, token_a in enumerate(ranked):
        seq_a = token_sequences[token_a]
        for token_b in ranked[i + 1 :]:
            seq_b = token_sequences[token_b]
            distance = _levenshtein_bounded(
                seq_a,
                seq_b,
                max_distance,
            )
            if 1 <= distance <= max_distance:
                neighbor_map[token_a][distance].append(token_b)
                neighbor_map[token_b][distance].append(token_a)

    rows = []
    for token in ranked:
        row = {
            "token": token,
            "frequency": token_counts[token],
            "length_glyphs": len(token_sequences[token]),
        }
        cumulative = 0
        for distance in range(1, max_distance + 1):
            neighbors = sorted(
                neighbor_map[token][distance],
                key=lambda x: (-token_counts[x], x),
            )
            cumulative += len(neighbors)
            row[f"neighbors_distance_{distance}"] = len(neighbors)
            row[f"examples_distance_{distance}"] = "|".join(
                neighbors[:example_limit]
            )
        row[f"neighbors_within_{max_distance}"] = cumulative
        rows.append(row)

    return rows


def markup_counts(records: Sequence[LocusRecord]) -> dict:
    counts = Counter()
    for record in records:
        text = record.text_normalized
        counts["uncertain_space_markers"] += text.count(",")
        counts["alternative_readings"] += len(_ALT_RE.findall(text))
        counts["drawing_interruptions"] += len(_DRAWING_RE.findall(text))
        counts["inline_comments"] += len(_COMMENT_RE.findall(text))
        counts["high_ascii_codes"] += len(_HIGH_ASCII_RE.findall(text))
        counts["brace_ligature_spans"] += text.count("{")
        runs = _UNKNOWN_RUN_RE.findall(text)
        counts["unknown_runs"] += len(runs)
        counts["unknown_question_marks"] += sum(len(run) for run in runs)
        counts["triple_unknown_runs"] += sum(run == "???" for run in runs)
    return dict(counts)


def _glyph_sequences(loci: Sequence[AnalyticalLocus]) -> List[Tuple[str, ...]]:
    return [token for locus in loci for token in locus.tokens]


def _token_sequences_by_locus(
    loci: Sequence[AnalyticalLocus],
) -> List[Tuple[str, ...]]:
    return [
        tuple(token_key(token) for token in locus.tokens)
        for locus in loci
        if locus.tokens
    ]


def _glyph_group_counts(
    loci: Sequence[AnalyticalLocus],
    attribute: str,
) -> Dict[str, Counter]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for locus in loci:
        label = str(getattr(locus, attribute))
        for token in locus.tokens:
            for glyph in token:
                if glyph != UNKNOWN_STA1:
                    grouped[label][glyph] += 1
    return dict(grouped)


def token_internal_position_counts(
    tokens: Sequence[Sequence[str]],
) -> Dict[str, Counter]:
    result = {
        "initial": Counter(),
        "medial": Counter(),
        "final": Counter(),
    }
    for token in tokens:
        known = [x for x in token if x != UNKNOWN_STA1]
        if not known:
            continue
        result["initial"][known[0]] += 1
        result["final"][known[-1]] += 1
        if len(known) > 2:
            result["medial"].update(known[1:-1])
    return result


def _three_way_token_positions(
    token_sequences: Iterable[Sequence[Sequence[str]]],
) -> Dict[str, Counter]:
    result = {
        "initial": Counter(),
        "medial": Counter(),
        "final": Counter(),
    }
    for tokens in token_sequences:
        tokens = list(tokens)
        if not tokens:
            continue
        result["initial"][token_key(tokens[0])] += 1
        result["final"][token_key(tokens[-1])] += 1
        if len(tokens) > 2:
            result["medial"].update(token_key(t) for t in tokens[1:-1])
    return result


def _paragraph_token_sequences(
    loci: Sequence[AnalyticalLocus],
) -> List[Tuple[Tuple[str, ...], ...]]:
    grouped: Dict[Tuple[str, int], List[AnalyticalLocus]] = defaultdict(list)
    for locus in loci:
        if locus.paragraph is not None:
            grouped[(locus.folio, locus.paragraph)].append(locus)

    sequences = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda x: (x.line, x.source_line_number),
        )
        tokens = tuple(
            token
            for locus in ordered
            for token in locus.tokens
        )
        if tokens:
            sequences.append(tokens)
    return sequences


def _position_summary_rows(
    scope: str,
    counts: Mapping[str, Counter],
) -> List[dict]:
    rows = []
    for position in ("initial", "medial", "final"):
        counter = counts[position]
        total = sum(counter.values())
        rows.append(
            {
                "scope": scope,
                "position": position,
                "observations": total,
                "unique_types": len(counter),
                "entropy_bits": entropy_from_counter(counter),
            }
        )

    for left, right in (
        ("initial", "medial"),
        ("initial", "final"),
        ("medial", "final"),
    ):
        rows.append(
            {
                "scope": scope,
                "position": f"{left}_vs_{right}",
                "observations": "",
                "unique_types": "",
                "entropy_bits": "",
                "js_divergence_bits": jensen_shannon_divergence(
                    counts[left],
                    counts[right],
                ),
            }
        )
    return rows


def _counter_top(counter: Counter, limit: int) -> List[dict]:
    return [
        {"item": item, "count": count}
        for item, count in sorted(
            counter.items(),
            key=lambda kv: (-kv[1], str(kv[0])),
        )[:limit]
    ]


def build_tier0(
    records: Sequence[LocusRecord],
    loci: Sequence[AnalyticalLocus],
    *,
    config: Tier0Config,
    train_leaves: Sequence[int],
) -> Tuple[dict, Dict[str, List[dict]]]:
    tokens = _glyph_sequences(loci)
    token_keys = [token_key(token) for token in tokens]
    token_counter = Counter(token_keys)
    token_sequences = {}
    for token in tokens:
        token_sequences.setdefault(token_key(token), tuple(token))

    glyph_counter_all = Counter(
        glyph
        for token in tokens
        for glyph in token
    )
    glyph_counter_known = Counter(
        {
            glyph: count
            for glyph, count in glyph_counter_all.items()
            if glyph != UNKNOWN_STA1
        }
    )

    token_lengths = [len(token) for token in tokens]
    known_glyph_sequences = [
        tuple(g for g in token if g != UNKNOWN_STA1)
        for token in tokens
    ]

    conditional = {}
    for order in range(1, config.entropy_max_order + 1):
        h, observations, contexts = empirical_conditional_entropy(
            tokens,
            order,
        )
        conditional[str(order)] = {
            "entropy_bits": h,
            "transitions": observations,
            "contexts": contexts,
        }

    unigram_entropy = entropy_from_counter(glyph_counter_known)

    glyph_ngrams = {
        n: ngram_counts(tokens, n)
        for n in (1, 2, 3)
    }

    token_position_glyph = token_internal_position_counts(tokens)
    locus_position_token = _three_way_token_positions(
        locus.tokens for locus in loci
    )
    paragraph_sequences = _paragraph_token_sequences(loci)
    paragraph_position_token = _three_way_token_positions(
        paragraph_sequences
    )

    line_token_lengths = [
        len(locus.tokens)
        for locus in loci
        if locus.tokens
    ]
    line_glyph_lengths = [
        sum(len(token) for token in locus.tokens)
        for locus in loci
        if locus.tokens
    ]
    paragraph_token_lengths = [
        len(tokens_)
        for tokens_ in paragraph_sequences
    ]

    group_rows = {}
    for dimension, attr in (
        ("section", "section"),
        ("currier", "currier"),
        ("scribe", "scribe"),
    ):
        grouped = _glyph_group_counts(loci, attr)
        group_rows[dimension] = pairwise_js_rows(grouped, dimension)

    glyph_mi = mutual_information_by_distance(
        tokens,
        config.glyph_mi_max_distance,
    )
    token_mi = mutual_information_by_distance(
        _token_sequences_by_locus(loci),
        config.token_mi_max_distance,
        exclude_symbol=None,
    )

    edit_rows = edit_neighborhood_rows(
        token_counter,
        token_sequences,
        vocab_limit=config.edit_vocab_limit,
        max_distance=config.edit_max_distance,
        example_limit=config.edit_example_limit,
    )

    positional_rows = []
    positional_rows.extend(
        _position_summary_rows(
            "token_internal_glyph",
            token_position_glyph,
        )
    )
    positional_rows.extend(
        _position_summary_rows(
            "locus_token",
            locus_position_token,
        )
    )
    positional_rows.extend(
        _position_summary_rows(
            "paragraph_token",
            paragraph_position_token,
        )
    )

    metadata_dimensions = {
        "section": Counter(l.section for l in loci),
        "currier": Counter(l.currier for l in loci),
        "scribe": Counter(l.scribe for l in loci),
        "locus_type": Counter(l.locus_type for l in loci),
        "quire": Counter(
            "UNSET" if l.quire is None else str(l.quire)
            for l in loci
        ),
    }

    summary = {
        "schema_version": "1.0",
        "analysis_tier": "Tier 0 descriptive structural analysis",
        "scope": "TRAIN ONLY",
        "representation": S0_REPRESENTATION,
        "representation_notes": {
            "certain_period_space_is_boundary": True,
            "uncertain_comma_space_is_boundary": True,
            "drawing_interruptions_are_boundaries": True,
            "alternative_reading_policy": "first reading",
            "glyph_unit": "STA1 symbol via frozen EVA->STA1 bitrans rules",
            "unknown_Z1_policy": (
                "retained in token lengths/frequency diagnostics; excluded "
                "from core glyph entropy/ngram/branching/MI estimates"
            ),
        },
        "counts": {
            "physical_leaves": len(set(train_leaves)),
            "folios": len({l.folio for l in loci}),
            "loci": len(loci),
            "paragraphs": len(
                {
                    (l.folio, l.paragraph)
                    for l in loci
                    if l.paragraph is not None
                }
            ),
            "tokens": len(tokens),
            "unique_tokens": len(token_counter),
            "glyphs_all_including_unknown": sum(glyph_counter_all.values()),
            "glyphs_known": sum(glyph_counter_known.values()),
            "unknown_Z1_symbols": glyph_counter_all.get(UNKNOWN_STA1, 0),
            "unique_known_glyphs": len(glyph_counter_known),
            "hapax_token_types": sum(
                count == 1 for count in token_counter.values()
            ),
        },
        "markup": markup_counts(records),
        "metadata_locus_counts": {
            key: dict(sorted(counter.items()))
            for key, counter in metadata_dimensions.items()
        },
        "token_length_glyphs": distribution_stats(token_lengths),
        "locus_length_tokens": distribution_stats(line_token_lengths),
        "locus_length_glyphs": distribution_stats(line_glyph_lengths),
        "paragraph_length_tokens": distribution_stats(
            paragraph_token_lengths
        ),
        "entropy": {
            "unigram_entropy_bits_known_glyphs": unigram_entropy,
            "unigram_perplexity_known_glyphs": 2 ** unigram_entropy,
            "empirical_unigram_bits_per_glyph": unigram_entropy,
            "conditional_entropy_by_order": conditional,
        },
        "branching_entropy": {
            "order_1": branching_entropy(tokens, 1),
            "order_2": branching_entropy(tokens, 2),
        },
        "top_known_glyphs": _counter_top(
            glyph_counter_known,
            config.top_items_in_summary,
        ),
        "top_tokens": _counter_top(
            token_counter,
            config.top_items_in_summary,
        ),
        "notes": [
            "Descriptive estimates are train-only and are not confirmatory model performance.",
            "Empirical entropy estimates are plug-in estimates and are not bias-corrected.",
            "Token MI can be upward-biased in sparse vocabularies; pair counts are reported.",
            "Quire is described but is not used here as a model-selection signal.",
        ],
    }

    tables: Dict[str, List[dict]] = {}

    tables["glyph_frequency"] = [
        {
            "glyph": glyph,
            "count": count,
            "frequency_known": count / sum(glyph_counter_known.values())
            if glyph != UNKNOWN_STA1 and sum(glyph_counter_known.values())
            else "",
            "is_unknown": glyph == UNKNOWN_STA1,
        }
        for glyph, count in sorted(
            glyph_counter_all.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]

    total_tokens = sum(token_counter.values())
    tables["token_frequency"] = [
        {
            "token": token,
            "count": count,
            "frequency": count / total_tokens if total_tokens else 0.0,
            "length_glyphs": len(token_sequences[token]),
            "contains_unknown": UNKNOWN_STA1 in token_sequences[token],
        }
        for token, count in sorted(
            token_counter.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]

    length_counter = Counter(token_lengths)
    tables["token_length_distribution"] = [
        {
            "length_glyphs": length,
            "tokens": count,
            "frequency": count / len(token_lengths)
            if token_lengths
            else 0.0,
        }
        for length, count in sorted(length_counter.items())
    ]

    for n, counter in glyph_ngrams.items():
        total = sum(counter.values())
        tables[f"glyph_{n}gram_frequency"] = [
            {
                "ngram": " ".join(gram),
                "count": count,
                "frequency": count / total if total else 0.0,
            }
            for gram, count in sorted(
                counter.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        ]

    tables["glyph_mi_by_distance"] = glyph_mi
    tables["token_mi_by_distance"] = token_mi
    tables["edit_neighborhoods"] = edit_rows
    tables["positional_summary"] = positional_rows

    # Full positional frequency tables behind the summary divergences.
    for scope_name, position_counts in (
        ("token_internal_glyph", token_position_glyph),
        ("locus_token", locus_position_token),
        ("paragraph_token", paragraph_position_token),
    ):
        rows = []
        for position, counter in position_counts.items():
            total = sum(counter.values())
            for item, count in sorted(
                counter.items(),
                key=lambda kv: (-kv[1], kv[0]),
            ):
                rows.append(
                    {
                        "scope": scope_name,
                        "position": position,
                        "item": item,
                        "count": count,
                        "frequency_within_position": (
                            count / total if total else 0.0
                        ),
                    }
                )
        tables[f"{scope_name}_frequency"] = rows

    tables["section_glyph_js"] = group_rows["section"]
    tables["currier_glyph_js"] = group_rows["currier"]
    tables["scribe_glyph_js"] = group_rows["scribe"]

    # Group-conditioned glyph frequencies used by the JSD matrices.
    group_frequency_rows = []
    for dimension, attr in (
        ("section", "section"),
        ("currier", "currier"),
        ("scribe", "scribe"),
    ):
        grouped = _glyph_group_counts(loci, attr)
        for group, counter in sorted(grouped.items()):
            total = sum(counter.values())
            for glyph, count in sorted(
                counter.items(),
                key=lambda kv: (-kv[1], kv[0]),
            ):
                group_frequency_rows.append(
                    {
                        "dimension": dimension,
                        "group": group,
                        "glyph": glyph,
                        "count": count,
                        "frequency_within_group": (
                            count / total if total else 0.0
                        ),
                    }
                )
    tables["group_glyph_frequency"] = group_frequency_rows

    metadata_rows = []
    for dimension, counter in metadata_dimensions.items():
        for category, count in sorted(counter.items()):
            metadata_rows.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "loci": count,
                }
            )
    tables["metadata_locus_counts"] = metadata_rows

    return summary, tables


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tier0_outputs(
    output_dir: Path,
    *,
    summary: Mapping[str, object],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    provenance: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = dict(summary)
    summary_payload["provenance"] = dict(provenance)

    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name, rows in sorted(tables.items()):
        _write_csv(output_dir / f"{name}.csv", rows)

    manifest = {
        "schema_version": "1.0",
        "analysis_tier": "Tier 0",
        "scope": "TRAIN ONLY",
        "representation": S0_REPRESENTATION,
        "provenance": dict(provenance),
        "artifacts": sorted(
            ["summary.json"]
            + [f"{name}.csv" for name in tables]
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Checksums are calculated last and exclude SHA256SUMS itself.
    artifact_paths = sorted(
        p for p in output_dir.iterdir()
        if p.is_file() and p.name != "SHA256SUMS"
    )
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in artifact_paths
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )
