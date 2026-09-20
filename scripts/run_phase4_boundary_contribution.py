#!/usr/bin/env python3
"""VOYAGER Phase 4.D2 — Boundary Contribution Localization.

Exploratory post-hoc analysis on the already-accessed Phase-4 Voynich TEST set.

This script never fits or scores a predictive model. It joins D1 HMM deltas to
pre-specified token/boundary features from the frozen S0/STA1 representation,
reports descriptive Pearson/Spearman associations, and compares the
pre-identified D1 cross-view-negative leaves f57 and f49 with the other leaves.

It cannot change the Phase-4 confirmatory result.
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

DEFAULT_CONFIG = Path("configs/diagnostics/phase4_boundary_contribution_v1.json")
Z1 = "Z1"


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    fields, seen = [], set()
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


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA mismatch\n  expected={expected}\n  observed={observed}"
        )
    return observed


def finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pop_var(values: Sequence[float]) -> float | None:
    if not values:
        return None
    m = mean(values)
    assert m is not None
    return sum((x - m) ** 2 for x in values) / len(values)


def pop_sd(values: Sequence[float]) -> float | None:
    v = pop_var(values)
    return None if v is None else math.sqrt(v)


def entropy(counter: Counter) -> float | None:
    total = sum(counter.values())
    if total <= 0:
        return None
    result = 0.0
    for count in counter.values():
        p = count / total
        result -= p * math.log2(p)
    return result


def conditional_entropy(counter: Counter) -> float | None:
    total = sum(counter.values())
    if total <= 0:
        return None
    by_prev: Dict[str, Counter] = defaultdict(Counter)
    prev_totals: Counter = Counter()
    for (prev, nxt), count in counter.items():
        by_prev[prev][nxt] += count
        prev_totals[prev] += count
    result = 0.0
    for prev, nxt_counts in by_prev.items():
        n = prev_totals[prev]
        inner = 0.0
        for count in nxt_counts.values():
            p = count / n
            inner -= p * math.log2(p)
        result += (n / total) * inner
    return result


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    assert mx is not None and my is not None
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
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
    below = sum(v < target for v in values)
    equal = sum(v == target for v in values)
    return (below + 0.5 * equal) / len(values)


def load_parent(config: Mapping[str, object]) -> dict:
    parent = config["parent_phase4d1"]
    require_hash(
        Path(parent["config_path"]),
        parent["expected_config_sha256"],
        "Phase 4.D1 config",
    )
    summary = read_json(Path(parent["summary_path"]))
    manifest = read_json(Path(parent["run_manifest_path"]))
    per_leaf = read_csv(Path(parent["per_leaf_path"]))

    if summary.get("status") != parent["required_status"]:
        raise ValueError("Unexpected D1 status")
    if summary.get("parent_phase4_decision_retained") != parent["required_phase4_decision_retained"]:
        raise ValueError("D1 does not retain required Phase-4 decision")
    if bool(summary.get("phase4_parity_all_passed")) is not True:
        raise ValueError("D1 parity was not all-pass")
    if int(summary.get("atomic_leaf_groups", -1)) != int(parent["required_atomic_leaf_groups"]):
        raise ValueError("Unexpected D1 leaf count")
    if len(per_leaf) != int(parent["required_atomic_leaf_groups"]):
        raise ValueError("Unexpected D1 per-leaf row count")

    required = {
        "leaf_group",
        "space_free_delta_bits_per_event",
        "token_aware_delta_bits_per_event",
        "token_aware_minus_space_free_delta_bits_per_event",
    }
    missing = required - set(per_leaf[0])
    if missing:
        raise ValueError(f"D1 per-leaf table missing: {sorted(missing)}")

    anomalies = set(parent["preidentified_cross_view_negative_leaf_groups"])
    cross_negative = {
        row["leaf_group"]
        for row in per_leaf
        if finite_float(row["space_free_delta_bits_per_event"], "space-free") < 0
        and finite_float(row["token_aware_delta_bits_per_event"], "token-aware") < 0
    }
    if cross_negative != anomalies:
        raise ValueError(
            f"D1 cross-view-negative identity changed: "
            f"expected={sorted(anomalies)} observed={sorted(cross_negative)}"
        )

    for row in per_leaf:
        s = finite_float(row["space_free_delta_bits_per_event"], "space-free")
        t = finite_float(row["token_aware_delta_bits_per_event"], "token-aware")
        b = finite_float(
            row["token_aware_minus_space_free_delta_bits_per_event"],
            "boundary boost",
        )
        if abs((t - s) - b) > 1e-12:
            raise ValueError(f"{row['leaf_group']}: boundary-boost arithmetic mismatch")

    return {
        "summary": summary,
        "manifest": manifest,
        "per_leaf": per_leaf,
        "summary_sha256": sha256_file(Path(parent["summary_path"])),
        "manifest_sha256": sha256_file(Path(parent["run_manifest_path"])),
        "per_leaf_sha256": sha256_file(Path(parent["per_leaf_path"])),
    }


def build_segments(loci) -> Dict[str, List[List[Tuple[str, ...]]]]:
    result: Dict[str, List[List[Tuple[str, ...]]]] = defaultdict(list)
    for locus in loci:
        leaf = atomic_leaf_group(locus)
        current: List[Tuple[str, ...]] = []

        def flush() -> None:
            nonlocal current
            if current:
                result[leaf].append(current)
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
    return dict(result)


def transition_summary(leaf: str, name: str, counts: Counter) -> dict:
    return {
        "leaf_group": leaf,
        "transition_population": name,
        "transition_count": sum(counts.values()),
        "unique_transition_pairs": len(counts),
        "transition_pair_entropy_bits": entropy(counts),
        "transition_conditional_entropy_bits": conditional_entropy(counts),
    }


def leaf_features(leaf: str, segments: Sequence[Sequence[Tuple[str, ...]]]):
    tokens = [tuple(tok) for seg in segments for tok in seg]
    if not tokens:
        raise ValueError(f"{leaf}: no readable tokens")

    lengths = [len(tok) for tok in tokens]
    counts = Counter(tokens)
    boundaries = [
        (seg[i], seg[i + 1])
        for seg in segments
        for i in range(len(seg) - 1)
    ]

    within = Counter()
    for token in tokens:
        for a, b in zip(token, token[1:]):
            within[(a, b)] += 1

    across = Counter()
    adjacent_repeats = 0
    for prev, cur in boundaries:
        across[(prev[-1], cur[0])] += 1
        adjacent_repeats += int(prev == cur)

    n_tokens = len(tokens)
    n_glyphs = sum(lengths)
    vocab = len(counts)
    n_boundaries = len(boundaries)
    repeated_occ = sum(c for c in counts.values() if c >= 2)
    singleton_occ = sum(c for c in counts.values() if c == 1)
    singleton_types = sum(1 for c in counts.values() if c == 1)

    within_pair_h = entropy(within)
    across_pair_h = entropy(across)
    within_cond_h = conditional_entropy(within)
    across_cond_h = conditional_entropy(across)

    row = {
        "leaf_group": leaf,
        "readable_segment_count": len(segments),
        "readable_token_count": n_tokens,
        "readable_glyph_count": n_glyphs,
        "mean_token_length_glyphs": mean(lengths),
        "token_length_variance_population": pop_var(lengths),
        "boundary_count": n_boundaries,
        "boundary_density_per_glyph": n_boundaries / n_glyphs,
        "vocabulary_size_exact_tokens": vocab,
        "type_token_ratio": vocab / n_tokens,
        "repeated_token_occurrence_fraction": repeated_occ / n_tokens,
        "singleton_token_occurrence_fraction": singleton_occ / n_tokens,
        "singleton_type_fraction": singleton_types / vocab,
        "adjacent_exact_repeat_fraction": (
            adjacent_repeats / n_boundaries if n_boundaries else None
        ),
        "within_transition_count": sum(within.values()),
        "within_unique_transition_pairs": len(within),
        "within_transition_pair_entropy_bits": within_pair_h,
        "within_transition_conditional_entropy_bits": within_cond_h,
        "across_transition_count": sum(across.values()),
        "across_unique_transition_pairs": len(across),
        "across_transition_pair_entropy_bits": across_pair_h,
        "across_transition_conditional_entropy_bits": across_cond_h,
        "conditional_entropy_gap_across_minus_within_bits": (
            across_cond_h - within_cond_h
            if across_cond_h is not None and within_cond_h is not None
            else None
        ),
        "pair_entropy_gap_across_minus_within_bits": (
            across_pair_h - within_pair_h
            if across_pair_h is not None and within_pair_h is not None
            else None
        ),
    }
    return (
        row,
        transition_summary(leaf, "within_token", within),
        transition_summary(leaf, "across_boundary", across),
    )


def correlation_row(rows, feature: str, outcome: str) -> dict:
    x, y = [], []
    missing = 0
    for row in rows:
        try:
            a = float(row[feature])
            b = float(row[outcome])
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError
        except (TypeError, ValueError, KeyError):
            missing += 1
            continue
        x.append(a)
        y.append(b)

    return {
        "feature": feature,
        "outcome": outcome,
        "n_used": len(x),
        "n_missing": missing,
        "pearson_r": pearson(x, y),
        "spearman_rho": spearman(x, y),
        "p_values_computed": False,
        "confirmatory_interpretation_permitted": False,
    }


def anomaly_rows(rows, anomalies, features):
    by_leaf = {row["leaf_group"]: row for row in rows}
    leaves = sorted(by_leaf)
    output = []

    for leaf in anomalies:
        for feature in features:
            try:
                target = float(by_leaf[leaf][feature])
                if not math.isfinite(target):
                    raise ValueError
            except (TypeError, ValueError, KeyError):
                output.append({
                    "leaf_group": leaf,
                    "feature": feature,
                    "feature_value": None,
                    "comparison_group": "other_13",
                    "other_mean": None,
                    "other_median": None,
                    "other_population_sd": None,
                    "descriptive_z_vs_others": None,
                    "percentile_rank_among_all15": None,
                })
                continue

            others, all_values = [], []
            for candidate in leaves:
                try:
                    value = float(by_leaf[candidate][feature])
                    if not math.isfinite(value):
                        continue
                except (TypeError, ValueError, KeyError):
                    continue
                all_values.append(value)
                if candidate != leaf:
                    others.append(value)

            m = mean(others)
            sd = pop_sd(others)
            z = None if m is None or sd in (None, 0.0) else (target - m) / sd

            output.append({
                "leaf_group": leaf,
                "feature": feature,
                "feature_value": target,
                "comparison_group": f"other_{len(others)}",
                "other_mean": m,
                "other_median": statistics.median(others) if others else None,
                "other_population_sd": sd,
                "descriptive_z_vs_others": z,
                "percentile_rank_among_all15": percentile_rank(all_values, target),
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VOYAGER Phase 4.D2 boundary contribution localization"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        config = read_json(args.config)
        if config.get("experiment_id") != "phase4-boundary-contribution-v1":
            raise ValueError("Unexpected D2 experiment_id")
        if config.get("status") != "exploratory_posthoc_locked_test_diagnostic":
            raise ValueError("D2 must remain explicitly exploratory")

        parent = load_parent(config)
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
        expected_test_hash = str(parent["manifest"].get("test_split_sha256", ""))
        if len(expected_test_hash) != 64:
            raise ValueError("D1 manifest lacks valid test_split_sha256")
        test_hash = require_hash(
            Path(inputs["locked_test_path"]),
            expected_test_hash,
            "already-used TEST split",
        )

        test_leaves = parse_leaf_ids(
            Path(inputs["locked_test_path"]).read_text(encoding="utf-8"),
            label="phase4d2_test_already_used",
        )
        if len(test_leaves) != int(inputs["expected_test_leaves"]):
            raise ValueError("Unexpected TEST leaf count")

        records = load_ivtff(
            Path(inputs["source_path"]),
            transcription_id=inputs["transcription_id"],
            strict=True,
        )
        test_records = select_records_by_leaves(
            records,
            test_leaves,
            split_name="phase4d2_test_already_used",
        )
        rules = load_bitrans_rules(
            Path(inputs["sta1_rules_path"]),
            expected_native_alphabet="Eva-",
        )
        if rules.sha256 != rules_hash:
            raise ValueError("Loaded STA1 rules differ from verified hash")
        loci = make_analytical_loci(test_records, rules)
        segments = build_segments(loci)

        d1 = {row["leaf_group"]: row for row in parent["per_leaf"]}
        if set(segments) != set(d1):
            raise ValueError(
                f"Parsed/D1 leaf mismatch: parsed={sorted(segments)} "
                f"D1={sorted(d1)}"
            )

        per_leaf, within_rows, across_rows = [], [], []
        for leaf in sorted(segments):
            feature, within, across = leaf_features(leaf, segments[leaf])
            old = d1[leaf]
            space = finite_float(old["space_free_delta_bits_per_event"], "space")
            token = finite_float(old["token_aware_delta_bits_per_event"], "token")
            boost = finite_float(
                old["token_aware_minus_space_free_delta_bits_per_event"],
                "boost",
            )
            if abs((token - space) - boost) > 1e-12:
                raise ValueError(f"{leaf}: D1 boundary boost changed")

            per_leaf.append({
                "leaf_group": leaf,
                "section_label": old.get("section_label", ""),
                "currier_label": old.get("currier_label", ""),
                "scribe_label": old.get("scribe_label", ""),
                "quire_label": old.get("quire_label", ""),
                "locus_type_label": old.get("locus_type_label", ""),
                "space_free_delta_bits_per_event": space,
                "token_aware_delta_bits_per_event": token,
                "boundary_boost_bits_per_event": boost,
                **feature,
            })
            within_rows.append(within)
            across_rows.append(across)

        corr_cfg = config["correlation_analysis"]
        correlations = [
            correlation_row(per_leaf, feature, corr_cfg["outcome"])
            for feature in corr_cfg["features"]
        ]

        anomalies = config["preidentified_anomaly_analysis"]["leaf_groups"]
        compare_features = list(corr_cfg["features"]) + [
            "boundary_boost_bits_per_event",
            "space_free_delta_bits_per_event",
            "token_aware_delta_bits_per_event",
        ]
        comparisons = anomaly_rows(per_leaf, anomalies, compare_features)

        output_dir = args.output_dir or Path(config["outputs"]["root"])
        if output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{output_dir} exists; use --overwrite only for exact reproduction"
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        write_csv(output_dir / "per_leaf_boundary_features.csv", per_leaf)
        write_csv(output_dir / "feature_correlations.csv", correlations)
        write_csv(output_dir / "within_token_transition_summary.csv", within_rows)
        write_csv(output_dir / "cross_boundary_transition_summary.csv", across_rows)
        write_csv(output_dir / "f57_f49_comparison.csv", comparisons)

        positive = sum(row["boundary_boost_bits_per_event"] > 0 for row in per_leaf)
        negative = sum(row["boundary_boost_bits_per_event"] < 0 for row in per_leaf)
        zero = len(per_leaf) - positive - negative

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
            "boundary_boost_sign_counts": {
                "positive": positive,
                "negative": negative,
                "zero": zero,
            },
            "preidentified_cross_view_negative_leaf_groups": anomalies,
            "correlations": correlations,
            "interpretation_boundary": config["interpretation_boundary"],
        }
        write_json(output_dir / "summary.json", summary)

        manifest = {
            "config_path": str(args.config),
            "config_sha256": sha256_file(args.config),
            "parent_d1_config_sha256": config["parent_phase4d1"][
                "expected_config_sha256"
            ],
            "parent_d1_summary_sha256": parent["summary_sha256"],
            "parent_d1_run_manifest_sha256": parent["manifest_sha256"],
            "parent_d1_per_leaf_sha256": parent["per_leaf_sha256"],
            "source_sha256": source_hash,
            "sta1_rules_sha256": rules_hash,
            "test_split_sha256": test_hash,
            "test_was_already_accessed_before_d2": True,
            "development_validation_read_by_d2": False,
            "predictive_models_fit_or_rescored_by_d2": False,
            "feature_count": len(corr_cfg["features"]),
            "correlation_statistics": corr_cfg["statistics"],
            "p_values_computed": False,
            "automatic_feature_selection": False,
        }
        write_json(output_dir / "run_manifest.json", manifest)

        artifacts = sorted(
            p for p in output_dir.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS"
        )
        (output_dir / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(p)}  {p.relative_to(output_dir)}"
                for p in artifacts
            ) + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 96)
        print("PHASE 4.D2 — BOUNDARY CONTRIBUTION LOCALIZATION")
        print("=" * 96)
        print("Status:                    EXPLORATORY / POST-HOC")
        print("Phase 4 decision:          RETAINED UNCHANGED")
        print("Predictive model refits:   NONE")
        print(f"Atomic leaf groups:        {len(per_leaf)}")
        print(f"Boundary-boost signs:      +{positive} / -{negative} / 0={zero}")
        print()
        print("Pre-specified feature associations with boundary boost")
        print("-" * 88)
        for row in correlations:
            pr = "NA" if row["pearson_r"] is None else f"{row['pearson_r']:+.4f}"
            sr = "NA" if row["spearman_rho"] is None else f"{row['spearman_rho']:+.4f}"
            print(
                f"{row['feature']:<52s} "
                f"Pearson={pr:>8s} Spearman={sr:>8s} n={row['n_used']}"
            )
        print()
        print("Preidentified cross-view negative leaves: f57, f49")
        print(
            "Interpretation boundary: descriptive associations only; "
            "Phase 4 remains unchanged."
        )
        print(f"Saved: {output_dir}")
        return 0

    except Exception as exc:
        print(
            f"PHASE 4.D2 ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
