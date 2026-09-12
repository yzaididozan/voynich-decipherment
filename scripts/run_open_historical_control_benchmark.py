#!/usr/bin/env python3
"""Benchmark all open-language historical cipher controls against Voynich TRAIN.

Expected input:
  4 frozen open language sources
  x 7 cipher-control families
  x 5 frozen seeds
  = 140 control corpora

This script reads Voynich TRAIN Tier-0 artifacts only. It does not read
validation.txt or test_LOCKED.txt.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import load_config
from src.evaluation.control_benchmark import (
    aggregate_rows_by_family,
    benchmark_control,
    read_control_jsonl,
    scalar_metrics,
    voynich_scalar_metrics,
)


DEFAULT_CONFIG = Path(
    "configs/evaluation/historical_control_benchmark_open_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
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

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the 140 frozen open-language historical cipher "
            "controls against Voynich TRAIN Tier-0 structure."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    try:
        if not args.config.is_file():
            raise ValueError(f"Benchmark config not found: {args.config}")

        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("schema_version") != "1.0":
            raise ValueError("Expected benchmark schema_version=1.0")

        expected_sources = int(config["expected_sources"])
        expected_families = int(config["expected_families_per_source"])
        expected_seeds = int(config["expected_seeds_per_family"])
        expected_runs = int(config["expected_total_control_runs"])

        if (
            expected_sources != 4
            or expected_families != 7
            or expected_seeds != 5
            or expected_runs != 140
        ):
            raise ValueError("Frozen open-control cardinalities changed")

        tier0_config_path = Path(config["tier0_config"])
        tier0_config = load_config(tier0_config_path)

        metric_cfg = config["metrics"]
        checks = (
            (
                "glyph_mi_max_distance",
                tier0_config.glyph_mi_max_distance,
            ),
            (
                "token_mi_max_distance",
                tier0_config.token_mi_max_distance,
            ),
            (
                "edit_vocab_limit",
                tier0_config.edit_vocab_limit,
            ),
            (
                "edit_max_distance",
                tier0_config.edit_max_distance,
            ),
            (
                "edit_example_limit",
                tier0_config.edit_example_limit,
            ),
        )
        for key, observed in checks:
            if int(metric_cfg[key]) != int(observed):
                raise ValueError(
                    f"{key} differs from frozen Tier-0 config: "
                    f"{metric_cfg[key]} != {observed}"
                )

        voynich_dir = Path(config["voynich_reference"])
        voynich_metrics = voynich_scalar_metrics(voynich_dir)
        metric_names = list(voynich_metrics)

        language_root = Path(config["language_root"])
        master_path = Path(config["language_master_manifest"])
        if not master_path.is_file():
            raise ValueError(
                "Open-language master manifest missing: "
                f"{master_path}"
            )

        master = json.loads(master_path.read_text(encoding="utf-8"))
        if master.get("freeze_id") != "open-language-controls-v1":
            raise ValueError(
                "Expected freeze_id=open-language-controls-v1"
            )

        source_entries = master.get("sources", [])
        if len(source_entries) != expected_sources:
            raise ValueError(
                f"Expected {expected_sources} open sources; "
                f"found {len(source_entries)}"
            )

        control_root = Path(config["control_root"])
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        run_output_root = output_dir / "runs"
        run_output_root.mkdir(parents=True, exist_ok=True)

        rows = []

        print("=" * 96)
        print(
            "OPEN-LANGUAGE HISTORICAL CIPHER STRUCTURAL BENCHMARK — "
            "140 RUNS"
        )
        print("=" * 96)
        print(f"Language root:     {language_root}")
        print(f"Control root:      {control_root}")
        print(f"Voynich reference: {voynich_dir}")
        print(f"Tier-0 config:     {tier0_config_path}")
        print("Voynich scope:     TRAIN ONLY")
        print("Validation/test:   NOT ACCESSED")
        print()

        for source_entry in source_entries:
            source_id = str(source_entry["source_id"])
            language = str(source_entry["language"])
            frozen_sha = str(source_entry["source_sha256"])

            source_manifest_path = language_root / source_id / "manifest.json"
            source_path = language_root / source_id / "source.txt"

            if not source_manifest_path.is_file() or not source_path.is_file():
                raise ValueError(
                    f"{source_id}: frozen source files are missing"
                )

            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )

            if source_manifest.get("source_sha256") != frozen_sha:
                raise ValueError(
                    f"{source_id}: source manifest hash differs from "
                    "master manifest"
                )
            if sha256_file(source_path) != frozen_sha:
                raise ValueError(
                    f"{source_id}: source.txt hash differs from frozen hash"
                )
            if int(source_manifest["frozen_normalized_tokens"]) != 20000:
                raise ValueError(
                    f"{source_id}: expected exactly 20,000 frozen tokens"
                )

            cipher_manifest_path = control_root / source_id / "manifest.json"
            if not cipher_manifest_path.is_file():
                raise ValueError(
                    f"Missing cipher manifest for {source_id}: "
                    f"{cipher_manifest_path}"
                )

            cipher_manifest = json.loads(
                cipher_manifest_path.read_text(encoding="utf-8")
            )
            if cipher_manifest.get("source_sha256") != frozen_sha:
                raise ValueError(
                    f"{source_id}: cipher ensemble was generated from "
                    "the wrong source bytes"
                )

            runs = cipher_manifest.get("runs", [])
            if len(runs) != expected_families * expected_seeds:
                raise ValueError(
                    f"{source_id}: expected 35 cipher runs; "
                    f"found {len(runs)}"
                )

            family_counts = {}
            for run in runs:
                family_counts[run["family"]] = (
                    family_counts.get(run["family"], 0) + 1
                )
            if len(family_counts) != expected_families:
                raise ValueError(
                    f"{source_id}: expected 7 cipher families; "
                    f"found {len(family_counts)}"
                )
            if any(count != expected_seeds for count in family_counts.values()):
                raise ValueError(
                    f"{source_id}: every family must have 5 seeds"
                )

            print(f"[{language}] {source_id}: 35 runs")

            for run in runs:
                family = str(run["family"])
                seed = int(run["seed"])
                run_dir = Path(run["path"])
                corpus_path = run_dir / "corpus.jsonl"
                cipher_summary_path = run_dir / "summary.json"

                if not corpus_path.is_file() or not cipher_summary_path.is_file():
                    raise ValueError(
                        f"Missing control artifacts: "
                        f"{source_id}/{family}/seed_{seed}"
                    )

                cipher_summary = json.loads(
                    cipher_summary_path.read_text(encoding="utf-8")
                )
                if cipher_summary.get("round_trip_verified") is not True:
                    raise ValueError(
                        f"{source_id}/{family}/{seed}: "
                        "round-trip verification is not PASS"
                    )

                lines = read_control_jsonl(corpus_path)
                summary, tables = benchmark_control(
                    lines,
                    config=tier0_config,
                )
                metrics = scalar_metrics(summary, tables)

                run_out = (
                    run_output_root
                    / source_id
                    / family
                    / f"seed_{seed}"
                )
                run_out.mkdir(parents=True, exist_ok=True)

                payload = {
                    "schema_version": "1.0",
                    "source_id": source_id,
                    "language": language,
                    "source_sha256": frozen_sha,
                    "family": family,
                    "seed": seed,
                    "control_corpus_path": str(corpus_path),
                    "control_corpus_sha256": sha256_file(corpus_path),
                    "voynich_reference": str(voynich_dir),
                    "voynich_scope": "TRAIN ONLY",
                    "validation_accessed": False,
                    "locked_test_accessed": False,
                    "benchmark": summary,
                    "scalar_metrics": metrics,
                }

                (run_out / "summary.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                write_csv(
                    run_out / "glyph_mi_by_distance.csv",
                    tables["glyph_mi_by_distance"],
                )
                write_csv(
                    run_out / "token_mi_by_distance.csv",
                    tables["token_mi_by_distance"],
                )
                write_csv(
                    run_out / "edit_neighborhoods.csv",
                    tables["edit_neighborhoods"],
                )

                row = {
                    "source_id": source_id,
                    "language": language,
                    "family": family,
                    "seed": seed,
                    "source_sha256": frozen_sha,
                    "control_glyphs": summary["counts"]["glyphs"],
                    "control_tokens": summary["counts"]["tokens"],
                    "control_unique_glyphs": summary["counts"]["unique_glyphs"],
                    "control_unique_tokens": summary["counts"]["unique_tokens"],
                }

                for metric in metric_names:
                    value = float(metrics[metric])
                    reference = float(voynich_metrics[metric])
                    row[metric] = value
                    row[f"voynich_{metric}"] = reference
                    row[f"delta_vs_voynich_{metric}"] = value - reference
                    row[f"abs_delta_vs_voynich_{metric}"] = abs(
                        value - reference
                    )

                rows.append(row)

            print("  benchmarked 35/35")

        if len(rows) != expected_runs:
            raise ValueError(
                f"Expected {expected_runs} benchmark rows; "
                f"found {len(rows)}"
            )

        matrix_path = output_dir / "mechanism_matrix.csv"
        write_csv(matrix_path, rows)

        family_summary = aggregate_rows_by_family(
            rows,
            metric_names=metric_names,
        )
        family_summary_path = output_dir / "family_summary.csv"
        write_csv(family_summary_path, family_summary)

        reference_path = output_dir / "voynich_train_reference_metrics.json"
        reference_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "scope": "VOYNICH TRAIN ONLY",
                    "source": str(voynich_dir),
                    "metrics": voynich_metrics,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "analysis": (
                        "open historical-language cipher controls versus "
                        "Voynich Tier-0 structural benchmark"
                    ),
                    "language_freeze_id": "open-language-controls-v1",
                    "voynich_scope": "TRAIN ONLY",
                    "language_sources": expected_sources,
                    "cipher_families_per_source": expected_families,
                    "seeds_per_family": expected_seeds,
                    "total_control_runs": len(rows),
                    "single_similarity_score": False,
                    "metric_names": metric_names,
                    "interpretation_guardrails": config[
                        "interpretation_guardrails"
                    ],
                    "validation_accessed": False,
                    "locked_test_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        checksum_paths = (
            matrix_path,
            family_summary_path,
            reference_path,
            summary_path,
        )
        (output_dir / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in checksum_paths
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(f"Control runs:      {len(rows)}")
        print(f"Mechanism matrix:  {matrix_path}")
        print(f"Family summary:    {family_summary_path}")
        print("Single score:      NOT COMPUTED")
        print("Voynich validation/test: NOT ACCESSED")
        return 0

    except Exception as exc:
        print(
            f"OPEN-CONTROL BENCHMARK ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
