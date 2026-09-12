"""Tier-0-compatible structural benchmark for historical cipher controls.

The implementation intentionally reuses VOYAGER Tier-0 metric functions for:
- unigram entropy;
- conditional entropy;
- branching entropy;
- token-length distribution;
- glyph mutual information by distance;
- token mutual information by distance;
- token-internal positional counts/JSD;
- edit-distance neighborhoods.

Controls do not possess meaningful Voynich metadata such as Currier, scribe,
section, quire, or manuscript-page position. Those constraints are therefore
reported as unavailable rather than fabricated.
"""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.analysis.tier0 import (
    Tier0Config,
    branching_entropy,
    distribution_stats,
    edit_neighborhood_rows,
    empirical_conditional_entropy,
    entropy_from_counter,
    jensen_shannon_divergence,
    mutual_information_by_distance,
    token_internal_position_counts,
)


Token = Tuple[str, ...]
Line = Tuple[Token, ...]


def read_control_jsonl(path: Path) -> Tuple[Line, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing control corpus: {path}")

    lines: List[Line] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(
            handle, start=1
        ):
            raw = raw.strip()
            if not raw:
                continue

            payload = json.loads(raw)
            tokens = payload.get("tokens")

            if not isinstance(tokens, list):
                raise ValueError(
                    f"{path}:{line_number}: missing tokens list"
                )

            parsed: List[Token] = []
            for token in tokens:
                if (
                    not isinstance(token, list)
                    or not token
                    or not all(
                        isinstance(glyph, str) and glyph
                        for glyph in token
                    )
                ):
                    raise ValueError(
                        f"{path}:{line_number}: invalid token"
                    )
                parsed.append(tuple(token))

            if parsed:
                lines.append(tuple(parsed))

    if not lines:
        raise ValueError(f"No tokens in control corpus: {path}")

    return tuple(lines)


def _token_id(token: Sequence[str]) -> str:
    # Synthetic control glyph labels contain no ASCII unit separator.
    return "\x1f".join(token)


def _line_token_ids(
    lines: Sequence[Line],
) -> List[Tuple[str, ...]]:
    return [
        tuple(_token_id(token) for token in line)
        for line in lines
        if line
    ]


def _three_way_token_positions(
    lines: Sequence[Line],
) -> Dict[str, Counter]:
    result = {
        "initial": Counter(),
        "medial": Counter(),
        "final": Counter(),
    }

    for line in lines:
        if not line:
            continue

        ids = [_token_id(token) for token in line]
        result["initial"][ids[0]] += 1
        result["final"][ids[-1]] += 1

        if len(ids) > 2:
            result["medial"].update(ids[1:-1])

    return result


def _position_js(
    counts: Mapping[str, Counter],
) -> dict:
    return {
        "initial_vs_medial": jensen_shannon_divergence(
            counts["initial"],
            counts["medial"],
        ),
        "initial_vs_final": jensen_shannon_divergence(
            counts["initial"],
            counts["final"],
        ),
        "medial_vs_final": jensen_shannon_divergence(
            counts["medial"],
            counts["final"],
        ),
    }


def _edit_summary(
    token_counter: Counter,
    token_sequences: Mapping[str, Sequence[str]],
    *,
    config: Tier0Config,
) -> Tuple[dict, List[dict]]:
    rows = edit_neighborhood_rows(
        token_counter,
        token_sequences,
        vocab_limit=config.edit_vocab_limit,
        max_distance=config.edit_max_distance,
        example_limit=config.edit_example_limit,
    )

    if not rows:
        return {
            "vocab_evaluated": 0,
            "mean_neighbors_distance_1": 0.0,
            "mean_neighbors_within_2": 0.0,
            "frequency_weighted_mean_neighbors_distance_1": 0.0,
            "frequency_weighted_mean_neighbors_within_2": 0.0,
        }, rows

    total_frequency = sum(
        int(row["frequency"])
        for row in rows
    )

    def weighted(field: str) -> float:
        if total_frequency <= 0:
            return 0.0
        return sum(
            int(row["frequency"]) * int(row[field])
            for row in rows
        ) / total_frequency

    return {
        "vocab_evaluated": len(rows),
        "mean_neighbors_distance_1": mean(
            int(row["neighbors_distance_1"])
            for row in rows
        ),
        "mean_neighbors_within_2": mean(
            int(
                row[
                    f"neighbors_within_{config.edit_max_distance}"
                ]
            )
            for row in rows
        ),
        "frequency_weighted_mean_neighbors_distance_1": weighted(
            "neighbors_distance_1"
        ),
        "frequency_weighted_mean_neighbors_within_2": weighted(
            f"neighbors_within_{config.edit_max_distance}"
        ),
    }, rows


def benchmark_control(
    lines: Sequence[Line],
    *,
    config: Tier0Config,
) -> Tuple[dict, Dict[str, List[dict]]]:
    tokens = [
        token
        for line in lines
        for token in line
    ]
    if not tokens:
        raise ValueError("Cannot benchmark an empty control")

    glyph_counter = Counter(
        glyph
        for token in tokens
        for glyph in token
    )
    token_counter = Counter(
        _token_id(token)
        for token in tokens
    )

    token_sequences: Dict[str, Token] = {}
    for token in tokens:
        token_sequences.setdefault(
            _token_id(token),
            tuple(token),
        )

    token_lengths = [
        len(token)
        for token in tokens
    ]
    line_token_lengths = [
        len(line)
        for line in lines
    ]
    line_glyph_lengths = [
        sum(len(token) for token in line)
        for line in lines
    ]

    conditional = {}
    for order in range(
        1,
        config.entropy_max_order + 1,
    ):
        h, observations, contexts = (
            empirical_conditional_entropy(
                tokens,
                order,
                exclude_symbol=None,
            )
        )
        conditional[str(order)] = {
            "entropy_bits": h,
            "transitions": observations,
            "contexts": contexts,
        }

    unigram_entropy = entropy_from_counter(
        glyph_counter
    )

    glyph_mi = mutual_information_by_distance(
        tokens,
        config.glyph_mi_max_distance,
        exclude_symbol=None,
    )
    token_mi = mutual_information_by_distance(
        _line_token_ids(lines),
        config.token_mi_max_distance,
        exclude_symbol=None,
    )

    token_position = token_internal_position_counts(
        tokens
    )
    line_token_position = _three_way_token_positions(
        lines
    )

    token_position_js = _position_js(
        token_position
    )
    line_position_js = _position_js(
        line_token_position
    )

    edit_summary, edit_rows = _edit_summary(
        token_counter,
        token_sequences,
        config=config,
    )

    top_token_count = max(token_counter.values())
    hapax_types = sum(
        count == 1
        for count in token_counter.values()
    )

    summary = {
        "schema_version": "1.0",
        "analysis_tier": (
            "Tier-0-compatible historical cipher control benchmark"
        ),
        "counts": {
            "lines": len(lines),
            "tokens": len(tokens),
            "unique_tokens": len(token_counter),
            "glyphs": sum(glyph_counter.values()),
            "unique_glyphs": len(glyph_counter),
            "hapax_token_types": hapax_types,
            "unique_token_ratio": (
                len(token_counter) / len(tokens)
            ),
            "hapax_type_ratio": (
                hapax_types / len(token_counter)
                if token_counter
                else 0.0
            ),
            "top_token_frequency": (
                top_token_count / len(tokens)
            ),
        },
        "token_length_glyphs": distribution_stats(
            token_lengths
        ),
        "line_length_tokens": distribution_stats(
            line_token_lengths
        ),
        "line_length_glyphs": distribution_stats(
            line_glyph_lengths
        ),
        "entropy": {
            "unigram_entropy_bits": unigram_entropy,
            "conditional_entropy_by_order": conditional,
        },
        "branching_entropy": {
            "order_1": branching_entropy(tokens, 1),
            "order_2": branching_entropy(tokens, 2),
        },
        "token_internal_glyph_position_js": (
            token_position_js
        ),
        "line_token_position_js": (
            line_position_js
        ),
        "edit_neighborhood_summary": edit_summary,
        "notes": [
            (
                "Metric implementations reuse VOYAGER Tier-0 functions "
                "where the underlying structure is applicable."
            ),
            (
                "Control line boundaries come from source-edition text "
                "and are not assumed equivalent to Voynich manuscript loci."
            ),
            (
                "Raw token MI is descriptive only because sparse token "
                "vocabularies can strongly bias plug-in MI upward."
            ),
            (
                "Currier/scribe/section/quire and manuscript-layout "
                "constraints are unavailable for generic cipher controls."
            ),
        ],
    }

    tables = {
        "glyph_mi_by_distance": glyph_mi,
        "token_mi_by_distance": token_mi,
        "edit_neighborhoods": edit_rows,
    }

    return summary, tables


def scalar_metrics(
    summary: Mapping[str, object],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict:
    counts = summary["counts"]
    entropy = summary["entropy"]
    conditional = entropy[
        "conditional_entropy_by_order"
    ]
    token_length = summary[
        "token_length_glyphs"
    ]
    pos = summary[
        "token_internal_glyph_position_js"
    ]
    edit = summary[
        "edit_neighborhood_summary"
    ]

    glyph_mi = {
        int(row["distance"]): float(
            row["mutual_information_bits"]
        )
        for row in tables["glyph_mi_by_distance"]
    }

    metrics = {
        "unigram_entropy_bits": float(
            entropy["unigram_entropy_bits"]
        ),
        "conditional_entropy_order_1": float(
            conditional["1"]["entropy_bits"]
        ),
        "conditional_entropy_order_2": float(
            conditional["2"]["entropy_bits"]
        ),
        "conditional_entropy_order_3": float(
            conditional["3"]["entropy_bits"]
        ),
        "conditional_entropy_order_4": float(
            conditional["4"]["entropy_bits"]
        ),
        "token_length_mean": float(
            token_length["mean"]
        ),
        "token_length_median": float(
            token_length["median"]
        ),
        "unique_token_ratio": float(
            counts["unique_token_ratio"]
        ),
        "hapax_type_ratio": float(
            counts["hapax_type_ratio"]
        ),
        "top_token_frequency": float(
            counts["top_token_frequency"]
        ),
        "position_js_initial_medial": float(
            pos["initial_vs_medial"]
        ),
        "position_js_initial_final": float(
            pos["initial_vs_final"]
        ),
        "position_js_medial_final": float(
            pos["medial_vs_final"]
        ),
        "edit_mean_neighbors_distance_1": float(
            edit["mean_neighbors_distance_1"]
        ),
        "edit_mean_neighbors_within_2": float(
            edit["mean_neighbors_within_2"]
        ),
    }

    for distance in range(1, 9):
        metrics[
            f"glyph_mi_distance_{distance}"
        ] = float(
            glyph_mi.get(distance, 0.0)
        )

    return metrics


def voynich_scalar_metrics(
    tier0_dir: Path,
) -> dict:
    summary_path = tier0_dir / "summary.json"
    mi_path = (
        tier0_dir / "glyph_mi_by_distance.csv"
    )
    positional_path = (
        tier0_dir / "positional_summary.csv"
    )
    edit_path = tier0_dir / "edit_neighborhoods.csv"

    for path in (
        summary_path,
        mi_path,
        positional_path,
        edit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Voynich Tier-0 artifact: {path}"
            )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    counts = summary["counts"]
    entropy = summary["entropy"]
    conditional = entropy[
        "conditional_entropy_by_order"
    ]
    token_length = summary[
        "token_length_glyphs"
    ]

    with mi_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        mi_rows = list(csv.DictReader(handle))

    glyph_mi = {
        int(row["distance"]): float(
            row["mutual_information_bits"]
        )
        for row in mi_rows
    }

    with positional_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        positional_rows = list(csv.DictReader(handle))

    pos = {}
    for row in positional_rows:
        if (
            row["scope"]
            == "token_internal_glyph"
            and "_vs_" in row["position"]
        ):
            pos[row["position"]] = float(
                row["js_divergence_bits"]
            )

    with edit_path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        edit_rows = list(csv.DictReader(handle))

    def mean_int(field: str) -> float:
        if not edit_rows:
            return 0.0
        return mean(
            int(float(row[field]))
            for row in edit_rows
        )

    top_token_count = max(
        int(item["count"])
        for item in summary["top_tokens"]
    )

    metrics = {
        "unigram_entropy_bits": float(
            entropy[
                "unigram_entropy_bits_known_glyphs"
            ]
        ),
        "conditional_entropy_order_1": float(
            conditional["1"]["entropy_bits"]
        ),
        "conditional_entropy_order_2": float(
            conditional["2"]["entropy_bits"]
        ),
        "conditional_entropy_order_3": float(
            conditional["3"]["entropy_bits"]
        ),
        "conditional_entropy_order_4": float(
            conditional["4"]["entropy_bits"]
        ),
        "token_length_mean": float(
            token_length["mean"]
        ),
        "token_length_median": float(
            token_length["median"]
        ),
        "unique_token_ratio": (
            int(counts["unique_tokens"])
            / int(counts["tokens"])
        ),
        "hapax_type_ratio": (
            int(counts["hapax_token_types"])
            / int(counts["unique_tokens"])
        ),
        "top_token_frequency": (
            top_token_count
            / int(counts["tokens"])
        ),
        "position_js_initial_medial": pos[
            "initial_vs_medial"
        ],
        "position_js_initial_final": pos[
            "initial_vs_final"
        ],
        "position_js_medial_final": pos[
            "medial_vs_final"
        ],
        "edit_mean_neighbors_distance_1": mean_int(
            "neighbors_distance_1"
        ),
        "edit_mean_neighbors_within_2": mean_int(
            "neighbors_within_2"
        ),
    }

    for distance in range(1, 9):
        metrics[
            f"glyph_mi_distance_{distance}"
        ] = float(
            glyph_mi.get(distance, 0.0)
        )

    return metrics


def aggregate_rows_by_family(
    rows: Sequence[Mapping[str, object]],
    *,
    metric_names: Sequence[str],
) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}

    for row in rows:
        key = (
            str(row["source_id"]),
            str(row["family"]),
        )
        grouped.setdefault(key, []).append(row)

    output = []

    for (source_id, family), group in sorted(
        grouped.items()
    ):
        result = {
            "source_id": source_id,
            "language": group[0]["language"],
            "family": family,
            "seeds": len(group),
        }

        for metric in metric_names:
            values = [
                float(row[metric])
                for row in group
            ]
            result[
                f"{metric}_mean"
            ] = mean(values)
            result[
                f"{metric}_sd"
            ] = (
                pstdev(values)
                if len(values) > 1
                else 0.0
            )

        output.append(result)

    return output
