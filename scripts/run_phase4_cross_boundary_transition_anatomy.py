#!/usr/bin/env python3
"""VOYAGER Phase 4.D3 — Cross-Boundary Transition Anatomy.

Exploratory post-hoc decomposition of the pre-identified D2 association between
cross-boundary transition-pair entropy and the token-aware boundary boost.

This script:
- verifies the frozen D2 parent state;
- reads the already-used locked TEST leaves;
- reconstructs the exact D2 cross-boundary pair population;
- requires per-leaf H(F,I) parity with D2 before interpreting anything;
- decomposes P(final glyph, initial glyph) into marginal diversity, joint
  diversity, conditional dependence, support, and concentration;
- reports all-15 and pre-specified f57/f49 sensitivity associations.

It DOES NOT fit or rescore any predictive model and cannot alter Phase 4.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import make_analytical_loci
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules
from src.models.ngram import atomic_leaf_group, parse_leaf_ids, select_records_by_leaves


DEFAULT_CONFIG = Path(
    "configs/diagnostics/phase4_cross_boundary_transition_anatomy_v1.json"
)
Z1 = "Z1"
D2_PAIR_ENTROPY_PARITY_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in CSV: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch\n"
            f"  expected: {expected}\n"
            f"  observed: {observed}"
        )
    return observed


def finite_float(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    result = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        p = count / total
        result -= p * math.log2(p)
    return result


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def population_variance(values: Sequence[float]) -> float | None:
    if not values:
        return None
    m = mean(values)
    assert m is not None
    return sum((value - m) ** 2 for value in values) / len(values)


def population_sd(values: Sequence[float]) -> float | None:
    variance = population_variance(values)
    return None if variance is None else math.sqrt(variance)


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = mean(x)
    my = mean(y)
    assert mx is not None and my is not None

    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        # Average 1-based rank for ties.
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def percentile_rank(values: Sequence[float], target: float) -> float:
    if not values:
        raise ValueError("Cannot rank against empty values")
    below = sum(value < target for value in values)
    equal = sum(value == target for value in values)
    return (below + 0.5 * equal) / len(values)


def load_parent_d2(config: Mapping[str, object]) -> dict:
    parent = config["parent_phase4d2"]

    require_hash(
        Path(parent["config_path"]),
        parent["expected_config_sha256"],
        "Phase 4.D2 config",
    )

    summary = read_json(Path(parent["summary_path"]))
    manifest = read_json(Path(parent["run_manifest_path"]))
    per_leaf = read_csv(Path(parent["per_leaf_path"]))

    if summary.get("status") != parent["required_status"]:
        raise ValueError(f"Unexpected D2 status: {summary.get('status')!r}")

    if (
        summary.get("parent_phase4_decision_retained")
        != parent["required_phase4_decision_retained"]
    ):
        raise ValueError("D2 does not retain the required Phase-4 decision")

    if bool(summary.get("predictive_models_fit_or_rescored")):
        raise ValueError("D2 summary says predictive models were fit/rescored")

    if bool(manifest.get("predictive_models_fit_or_rescored_by_d2")):
        raise ValueError("D2 manifest says predictive models were fit/rescored")

    expected_n = int(parent["required_atomic_leaf_groups"])
    if int(summary.get("atomic_leaf_groups", -1)) != expected_n:
        raise ValueError("Unexpected D2 atomic leaf-group count")
    if len(per_leaf) != expected_n:
        raise ValueError(
            f"Expected {expected_n} D2 per-leaf rows; found {len(per_leaf)}"
        )

    observed_signs = summary.get("boundary_boost_sign_counts")
    expected_signs = parent["required_boundary_boost_sign_counts"]
    if observed_signs != expected_signs:
        raise ValueError(
            "D2 boundary-boost sign pattern changed\n"
            f"  expected={expected_signs}\n"
            f"  observed={observed_signs}"
        )

    required_columns = {
        "leaf_group",
        "boundary_boost_bits_per_event",
        "space_free_delta_bits_per_event",
        "token_aware_delta_bits_per_event",
        "across_transition_count",
        "across_transition_pair_entropy_bits",
    }
    missing = required_columns - set(per_leaf[0])
    if missing:
        raise ValueError(
            f"D2 per-leaf table missing required columns: {sorted(missing)}"
        )

    anomaly_set = set(parent["preidentified_cross_view_negative_leaf_groups"])
    observed_cross_negative = {
        row["leaf_group"]
        for row in per_leaf
        if finite_float(
            row["space_free_delta_bits_per_event"],
            label=f"{row['leaf_group']} space-free delta",
        ) < 0.0
        and finite_float(
            row["token_aware_delta_bits_per_event"],
            label=f"{row['leaf_group']} token-aware delta",
        ) < 0.0
    }
    if observed_cross_negative != anomaly_set:
        raise ValueError(
            "D2 cross-view-negative identity changed\n"
            f"  expected={sorted(anomaly_set)}\n"
            f"  observed={sorted(observed_cross_negative)}"
        )

    leaves = [row["leaf_group"] for row in per_leaf]
    if len(set(leaves)) != len(leaves):
        raise ValueError("Duplicate D2 leaf_group")

    return {
        "summary": summary,
        "manifest": manifest,
        "per_leaf": per_leaf,
        "summary_sha256": sha256_file(Path(parent["summary_path"])),
        "manifest_sha256": sha256_file(Path(parent["run_manifest_path"])),
        "per_leaf_sha256": sha256_file(Path(parent["per_leaf_path"])),
    }


def build_readable_segments(
    loci,
) -> Dict[str, List[List[Tuple[str, ...]]]]:
    """Match the D2 readable-token/Z1 reset policy exactly."""
    output: Dict[str, List[List[Tuple[str, ...]]]] = defaultdict(list)

    for locus in loci:
        leaf = atomic_leaf_group(locus)
        current: List[Tuple[str, ...]] = []

        def flush() -> None:
            nonlocal current
            if current:
                output[leaf].append(current)
                current = []

        for token in locus.tokens:
            if not token:
                continue
            glyphs = tuple(token)
            if Z1 in glyphs:
                flush()
            else:
                current.append(glyphs)
        flush()

    return dict(output)


def anatomy_for_leaf(
    leaf: str,
    segments: Sequence[Sequence[Tuple[str, ...]]],
) -> Tuple[dict, Counter]:
    pair_counts: Counter = Counter()

    for segment in segments:
        for previous, current in zip(segment, segment[1:]):
            if not previous or not current:
                raise ValueError(f"{leaf}: empty readable token")
            pair_counts[(previous[-1], current[0])] += 1

    pair_count = sum(pair_counts.values())
    if pair_count <= 0:
        raise ValueError(f"{leaf}: no cross-boundary pairs")

    final_counts: Counter = Counter()
    initial_counts: Counter = Counter()
    for (final_glyph, initial_glyph), count in pair_counts.items():
        final_counts[final_glyph] += count
        initial_counts[initial_glyph] += count

    h_final = entropy(final_counts)
    h_initial = entropy(initial_counts)
    h_joint = entropy(pair_counts)

    h_initial_given_final = max(0.0, h_joint - h_final)
    h_final_given_initial = max(0.0, h_joint - h_initial)
    mutual_information = max(0.0, h_final + h_initial - h_joint)

    distinct_final = len(final_counts)
    distinct_initial = len(initial_counts)
    distinct_pairs = len(pair_counts)
    possible_pairs = distinct_final * distinct_initial
    support_fraction = (
        distinct_pairs / possible_pairs if possible_pairs > 0 else 0.0
    )

    probabilities = sorted(
        (count / pair_count for count in pair_counts.values()),
        reverse=True,
    )
    max_pair_probability = probabilities[0]
    top5_mass = sum(probabilities[:5])
    simpson = sum(probability ** 2 for probability in probabilities)

    row = {
        "leaf_group": leaf,
        "boundary_pair_count": pair_count,
        "final_glyph_entropy_bits": h_final,
        "initial_glyph_entropy_bits": h_initial,
        "boundary_pair_entropy_bits": h_joint,
        "initial_given_final_conditional_entropy_bits": h_initial_given_final,
        "final_given_initial_conditional_entropy_bits": h_final_given_initial,
        "final_initial_mutual_information_bits": mutual_information,
        "normalized_mutual_information_joint_fraction": (
            mutual_information / h_joint if h_joint > 0.0 else 0.0
        ),
        "distinct_final_glyphs": distinct_final,
        "distinct_initial_glyphs": distinct_initial,
        "distinct_boundary_pairs": distinct_pairs,
        "effective_final_glyphs": 2.0 ** h_final,
        "effective_initial_glyphs": 2.0 ** h_initial,
        "effective_boundary_pairs": 2.0 ** h_joint,
        "possible_boundary_pairs_from_observed_marginals": possible_pairs,
        "observed_pair_support_fraction": support_fraction,
        "max_boundary_pair_probability": max_pair_probability,
        "top5_boundary_pair_probability_mass": top5_mass,
        "boundary_pair_simpson_concentration": simpson,
        "boundary_pair_effective_simpson_pairs": (
            1.0 / simpson if simpson > 0.0 else 0.0
        ),
    }

    return row, pair_counts


def correlation_row(
    rows: Sequence[Mapping[str, object]],
    *,
    feature: str,
    outcome: str,
    condition: str = "all15",
    omitted_leaf_groups: str = "",
) -> dict:
    x: List[float] = []
    y: List[float] = []
    missing = 0

    for row in rows:
        try:
            feature_value = float(row[feature])
            outcome_value = float(row[outcome])
            if not (
                math.isfinite(feature_value)
                and math.isfinite(outcome_value)
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue

        x.append(feature_value)
        y.append(outcome_value)

    return {
        "condition": condition,
        "omitted_leaf_groups": omitted_leaf_groups,
        "feature": feature,
        "outcome": outcome,
        "n_used": len(x),
        "n_missing": missing,
        "pearson_r": pearson(x, y),
        "spearman_rho": spearman(x, y),
        "p_values_computed": False,
        "confirmatory_interpretation_permitted": False,
    }


def anomaly_comparison(
    rows: Sequence[Mapping[str, object]],
    *,
    anomalies: Sequence[str],
    features: Sequence[str],
) -> List[dict]:
    by_leaf = {str(row["leaf_group"]): row for row in rows}
    leaves = sorted(by_leaf)
    output: List[dict] = []

    for anomaly in anomalies:
        if anomaly not in by_leaf:
            raise ValueError(f"Missing pre-identified anomaly leaf: {anomaly}")

        for feature in features:
            try:
                target = float(by_leaf[anomaly][feature])
                if not math.isfinite(target):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                output.append(
                    {
                        "leaf_group": anomaly,
                        "feature": feature,
                        "feature_value": None,
                        "comparison_group": "other_14",
                        "other_mean": None,
                        "other_median": None,
                        "other_population_sd": None,
                        "descriptive_z_vs_others": None,
                        "percentile_rank_among_all15": None,
                    }
                )
                continue

            others: List[float] = []
            all_values: List[float] = []

            for leaf in leaves:
                try:
                    value = float(by_leaf[leaf][feature])
                    if not math.isfinite(value):
                        continue
                except (KeyError, TypeError, ValueError):
                    continue

                all_values.append(value)
                if leaf != anomaly:
                    others.append(value)

            other_mean = mean(others)
            other_sd = population_sd(others)
            z = (
                (target - other_mean) / other_sd
                if (
                    other_mean is not None
                    and other_sd not in (None, 0.0)
                )
                else None
            )

            output.append(
                {
                    "leaf_group": anomaly,
                    "feature": feature,
                    "feature_value": target,
                    "comparison_group": f"other_{len(others)}",
                    "other_mean": other_mean,
                    "other_median": (
                        statistics.median(others) if others else None
                    ),
                    "other_population_sd": other_sd,
                    "descriptive_z_vs_others": z,
                    "percentile_rank_among_all15": (
                        percentile_rank(all_values, target)
                        if all_values
                        else None
                    ),
                }
            )

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "VOYAGER Phase 4.D3 cross-boundary transition anatomy"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing deterministic D3 output directory.",
    )
    args = parser.parse_args()

    try:
        config = read_json(args.config)

        if (
            config.get("experiment_id")
            != "phase4-cross-boundary-transition-anatomy-v1"
        ):
            raise ValueError("Unexpected D3 experiment_id")

        if (
            config.get("status")
            != "exploratory_posthoc_locked_test_diagnostic"
        ):
            raise ValueError("D3 must remain explicitly exploratory")

        parent = load_parent_d2(config)
        inputs = config["voynich_inputs"]

        source_hash = require_hash(
            Path(inputs["source_path"]),
            inputs["expected_source_sha256"],
            "ZL3b source",
        )
        rules_hash = require_hash(
            Path(inputs["sta1_rules_path"]),
            inputs["expected_sta1_rules_sha256"],
            "STA1 rules",
        )

        expected_test_hash = str(
            parent["manifest"].get("test_split_sha256", "")
        )
        if len(expected_test_hash) != 64:
            raise ValueError(
                "D2 run manifest lacks a valid test_split_sha256"
            )

        test_hash = require_hash(
            Path(inputs["locked_test_path"]),
            expected_test_hash,
            "already-used locked TEST split",
        )

        test_leaves = parse_leaf_ids(
            Path(inputs["locked_test_path"]).read_text(encoding="utf-8"),
            label="phase4d3_test_already_used",
        )
        expected_test_leaves = int(inputs["expected_test_leaves"])
        if len(test_leaves) != expected_test_leaves:
            raise ValueError(
                f"Expected {expected_test_leaves} TEST leaves; "
                f"found {len(test_leaves)}"
            )

        records = load_ivtff(
            Path(inputs["source_path"]),
            transcription_id=inputs["transcription_id"],
            strict=True,
        )
        test_records = select_records_by_leaves(
            records,
            test_leaves,
            split_name="phase4d3_test_already_used",
        )

        rules = load_bitrans_rules(
            Path(inputs["sta1_rules_path"]),
            expected_native_alphabet="Eva-",
        )
        if rules.sha256 != rules_hash:
            raise ValueError("Loaded STA1 rule hash changed")

        loci = make_analytical_loci(test_records, rules)
        segments_by_leaf = build_readable_segments(loci)

        d2_by_leaf = {
            row["leaf_group"]: row
            for row in parent["per_leaf"]
        }

        if set(segments_by_leaf) != set(d2_by_leaf):
            raise ValueError(
                "Parsed D3 leaf groups differ from D2\n"
                f"  parsed={sorted(segments_by_leaf)}\n"
                f"  D2={sorted(d2_by_leaf)}"
            )

        per_leaf: List[dict] = []
        concentration_rows: List[dict] = []
        parity_rows: List[dict] = []

        for leaf in sorted(segments_by_leaf):
            anatomy, pair_counts = anatomy_for_leaf(
                leaf,
                segments_by_leaf[leaf],
            )
            d2 = d2_by_leaf[leaf]

            d2_pair_entropy = finite_float(
                d2["across_transition_pair_entropy_bits"],
                label=f"{leaf} D2 across-pair entropy",
            )
            d2_pair_count = int(float(d2["across_transition_count"]))

            entropy_difference = (
                anatomy["boundary_pair_entropy_bits"]
                - d2_pair_entropy
            )
            entropy_parity = (
                abs(entropy_difference)
                <= D2_PAIR_ENTROPY_PARITY_TOLERANCE
            )
            count_parity = (
                anatomy["boundary_pair_count"] == d2_pair_count
            )

            parity_rows.append(
                {
                    "leaf_group": leaf,
                    "d2_boundary_pair_count": d2_pair_count,
                    "d3_boundary_pair_count": anatomy[
                        "boundary_pair_count"
                    ],
                    "count_parity": count_parity,
                    "d2_across_transition_pair_entropy_bits": d2_pair_entropy,
                    "d3_boundary_pair_entropy_bits": anatomy[
                        "boundary_pair_entropy_bits"
                    ],
                    "entropy_difference": entropy_difference,
                    "entropy_tolerance": D2_PAIR_ENTROPY_PARITY_TOLERANCE,
                    "entropy_parity": entropy_parity,
                }
            )

            if not count_parity or not entropy_parity:
                raise ValueError(
                    f"{leaf}: D3 boundary-pair reconstruction does not "
                    "match D2; refusing decomposition"
                )

            row = {
                "leaf_group": leaf,
                "section_label": d2.get("section_label", ""),
                "currier_label": d2.get("currier_label", ""),
                "scribe_label": d2.get("scribe_label", ""),
                "quire_label": d2.get("quire_label", ""),
                "locus_type_label": d2.get("locus_type_label", ""),
                "space_free_delta_bits_per_event": finite_float(
                    d2["space_free_delta_bits_per_event"],
                    label=f"{leaf} space-free delta",
                ),
                "token_aware_delta_bits_per_event": finite_float(
                    d2["token_aware_delta_bits_per_event"],
                    label=f"{leaf} token-aware delta",
                ),
                "boundary_boost_bits_per_event": finite_float(
                    d2["boundary_boost_bits_per_event"],
                    label=f"{leaf} boundary boost",
                ),
                **anatomy,
            }
            per_leaf.append(row)

            top_pairs = sorted(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            top5 = [
                {
                    "pair": f"{final}->{initial}",
                    "count": count,
                    "probability": count / anatomy["boundary_pair_count"],
                }
                for (final, initial), count in top_pairs[:5]
            ]
            concentration_rows.append(
                {
                    "leaf_group": leaf,
                    "boundary_pair_count": anatomy["boundary_pair_count"],
                    "distinct_final_glyphs": anatomy[
                        "distinct_final_glyphs"
                    ],
                    "distinct_initial_glyphs": anatomy[
                        "distinct_initial_glyphs"
                    ],
                    "distinct_boundary_pairs": anatomy[
                        "distinct_boundary_pairs"
                    ],
                    "possible_boundary_pairs_from_observed_marginals": (
                        anatomy[
                            "possible_boundary_pairs_from_observed_marginals"
                        ]
                    ),
                    "observed_pair_support_fraction": anatomy[
                        "observed_pair_support_fraction"
                    ],
                    "max_boundary_pair_probability": anatomy[
                        "max_boundary_pair_probability"
                    ],
                    "top5_boundary_pair_probability_mass": anatomy[
                        "top5_boundary_pair_probability_mass"
                    ],
                    "boundary_pair_simpson_concentration": anatomy[
                        "boundary_pair_simpson_concentration"
                    ],
                    "boundary_pair_effective_simpson_pairs": anatomy[
                        "boundary_pair_effective_simpson_pairs"
                    ],
                    "top1_pair": top5[0]["pair"] if top5 else "",
                    "top1_probability": (
                        top5[0]["probability"] if top5 else None
                    ),
                    "top2_pair": top5[1]["pair"] if len(top5) > 1 else "",
                    "top2_probability": (
                        top5[1]["probability"] if len(top5) > 1 else None
                    ),
                    "top3_pair": top5[2]["pair"] if len(top5) > 2 else "",
                    "top3_probability": (
                        top5[2]["probability"] if len(top5) > 2 else None
                    ),
                    "top4_pair": top5[3]["pair"] if len(top5) > 3 else "",
                    "top4_probability": (
                        top5[3]["probability"] if len(top5) > 3 else None
                    ),
                    "top5_pair": top5[4]["pair"] if len(top5) > 4 else "",
                    "top5_probability": (
                        top5[4]["probability"] if len(top5) > 4 else None
                    ),
                }
            )

        analysis = config["association_analysis"]
        features = list(analysis["features"])
        outcome = analysis["outcome"]

        correlations = [
            correlation_row(
                per_leaf,
                feature=feature,
                outcome=outcome,
                condition="all15",
                omitted_leaf_groups="",
            )
            for feature in features
        ]

        sensitivity_rows: List[dict] = []
        for condition, omitted in config["sensitivity_analysis"][
            "conditions"
        ].items():
            omitted_set = set(omitted)
            subset = [
                row
                for row in per_leaf
                if row["leaf_group"] not in omitted_set
            ]
            for feature in features:
                sensitivity_rows.append(
                    correlation_row(
                        subset,
                        feature=feature,
                        outcome=outcome,
                        condition=condition,
                        omitted_leaf_groups="|".join(sorted(omitted_set)),
                    )
                )

        anomalies = config["preidentified_anomaly_analysis"][
            "leaf_groups"
        ]
        comparison_features = features + [
            "boundary_boost_bits_per_event",
            "space_free_delta_bits_per_event",
            "token_aware_delta_bits_per_event",
        ]
        anomaly_rows = anomaly_comparison(
            per_leaf,
            anomalies=anomalies,
            features=comparison_features,
        )

        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else Path(config["outputs"]["root"])
        )

        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite only to reproduce the same D3 analysis."
                )
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=False)

        write_csv(
            output_dir / "per_leaf_boundary_anatomy.csv",
            per_leaf,
        )
        write_csv(
            output_dir / "anatomy_correlations.csv",
            correlations,
        )
        write_csv(
            output_dir / "anatomy_sensitivity.csv",
            sensitivity_rows,
        )
        write_csv(
            output_dir / "boundary_pair_concentration.csv",
            concentration_rows,
        )
        write_csv(
            output_dir / "f57_f49_anatomy.csv",
            anomaly_rows,
        )

        # Summarize the pre-identified D2 H(F,I) feature explicitly without
        # automatically ranking the remaining D3 decomposition features.
        pair_entropy_sensitivity = {
            row["condition"]: {
                "n_used": row["n_used"],
                "pearson_r": row["pearson_r"],
                "spearman_rho": row["spearman_rho"],
            }
            for row in sensitivity_rows
            if row["feature"] == "boundary_pair_entropy_bits"
        }

        summary = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "status": config["status"],
            "parent_phase4_decision_retained": parent["summary"][
                "parent_phase4_decision_retained"
            ],
            "phase4_reclassification_permitted": False,
            "predictive_models_fit_or_rescored": False,
            "atomic_leaf_groups": len(per_leaf),
            "d2_boundary_pair_reconstruction_parity_all_passed": all(
                bool(row["count_parity"])
                and bool(row["entropy_parity"])
                for row in parity_rows
            ),
            "preidentified_d2_feature": config["parent_phase4d2"][
                "preidentified_d2_feature"
            ],
            "preidentified_pair_entropy_sensitivity": (
                pair_entropy_sensitivity
            ),
            "preidentified_cross_view_negative_leaf_groups": anomalies,
            "interpretation_boundary": config["interpretation_boundary"],
        }
        write_json(output_dir / "summary.json", summary)

        manifest = {
            "config_path": str(args.config),
            "config_sha256": sha256_file(args.config),
            "parent_d2_config_sha256": config["parent_phase4d2"][
                "expected_config_sha256"
            ],
            "parent_d2_summary_sha256": parent["summary_sha256"],
            "parent_d2_run_manifest_sha256": parent["manifest_sha256"],
            "parent_d2_per_leaf_sha256": parent["per_leaf_sha256"],
            "source_sha256": source_hash,
            "sta1_rules_sha256": rules_hash,
            "test_split_sha256": test_hash,
            "test_was_already_accessed_before_d3": True,
            "development_validation_read_by_d3": False,
            "predictive_models_fit_or_rescored_by_d3": False,
            "d2_pair_entropy_parity_tolerance": (
                D2_PAIR_ENTROPY_PARITY_TOLERANCE
            ),
            "d2_pair_entropy_parity_all_passed": True,
            "association_feature_count": len(features),
            "sensitivity_conditions": config["sensitivity_analysis"][
                "conditions"
            ],
            "p_values_computed": False,
            "automatic_feature_selection": False,
            "automatic_winner_ranking": False,
        }
        write_json(output_dir / "run_manifest.json", manifest)

        # Keep reconstruction parity in the manifest provenance without adding
        # another user-facing required CSV beyond the frozen output contract.
        manifest["d2_reconstruction_parity_rows"] = parity_rows
        write_json(output_dir / "run_manifest.json", manifest)

        checksum_targets = sorted(
            path
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        (output_dir / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
                for path in checksum_targets
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 96)
        print("PHASE 4.D3 — CROSS-BOUNDARY TRANSITION ANATOMY")
        print("=" * 96)
        print("Status:                         EXPLORATORY / POST-HOC")
        print("Phase 4 decision:               RETAINED UNCHANGED")
        print("Predictive model refits:        NONE")
        print(f"Atomic leaf groups:             {len(per_leaf)}")
        print("D2 H(F,I) reconstruction:       PARITY PASS")
        print()
        print(
            "Pre-identified H(F,I) association with boundary boost"
        )
        print("-" * 82)
        for condition in (
            "all15",
            "without_f57",
            "without_f49",
            "without_f57_f49",
        ):
            row = pair_entropy_sensitivity[condition]
            p = (
                "NA"
                if row["pearson_r"] is None
                else f"{row['pearson_r']:+.4f}"
            )
            s = (
                "NA"
                if row["spearman_rho"] is None
                else f"{row['spearman_rho']:+.4f}"
            )
            print(
                f"{condition:<20s} "
                f"Pearson={p:>8s} Spearman={s:>8s} "
                f"n={row['n_used']}"
            )

        print()
        print("Frozen anatomy decomposition — all 15 leaves")
        print("-" * 96)
        for row in correlations:
            p = (
                "NA"
                if row["pearson_r"] is None
                else f"{row['pearson_r']:+.4f}"
            )
            s = (
                "NA"
                if row["spearman_rho"] is None
                else f"{row['spearman_rho']:+.4f}"
            )
            print(
                f"{row['feature']:<55s} "
                f"Pearson={p:>8s} Spearman={s:>8s} "
                f"n={row['n_used']}"
            )

        print()
        print(
            "Interpretation boundary: D3 decomposes a post-hoc D2 clue; "
            "it cannot alter Phase 4."
        )
        print(f"Saved: {output_dir}")
        return 0

    except Exception as exc:
        print(
            f"PHASE 4.D3 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
