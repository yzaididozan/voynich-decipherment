#!/usr/bin/env python3
"""Run the Tier-0-compatible benchmark over all 140 cipher controls."""

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
    "configs/evaluation/historical_control_benchmark_v1.json"
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

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark every frozen historical cipher control with the "
            "same applicable Tier-0 structural metrics used for Voynich."
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
            raise ValueError(
                f"Benchmark config not found: {args.config}"
            )

        config = json.loads(
            args.config.read_text(encoding="utf-8")
        )
        if config.get("schema_version") != "1.0":
            raise ValueError(
                "Expected benchmark schema_version=1.0"
            )

        tier0_config_path = Path(
            config["tier0_config"]
        )
        tier0_config = load_config(
            tier0_config_path
        )

        # Guard that the benchmark's requested metrics remain aligned with
        # the frozen Tier-0 configuration.
        expected = config["metrics"]
        if (
            int(expected["glyph_mi_max_distance"])
            != tier0_config.glyph_mi_max_distance
        ):
            raise ValueError(
                "glyph_mi_max_distance differs from frozen Tier-0 config"
            )
        if (
            int(expected["token_mi_max_distance"])
            != tier0_config.token_mi_max_distance
        ):
            raise ValueError(
                "token_mi_max_distance differs from frozen Tier-0 config"
            )
        if (
            int(expected["edit_vocab_limit"])
            != tier0_config.edit_vocab_limit
        ):
            raise ValueError(
                "edit_vocab_limit differs from frozen Tier-0 config"
            )
        if (
            int(expected["edit_max_distance"])
            != tier0_config.edit_max_distance
        ):
            raise ValueError(
                "edit_max_distance differs from frozen Tier-0 config"
            )

        voynich_dir = Path(
            config["voynich_reference"]
        )
        voynich_metrics = voynich_scalar_metrics(
            voynich_dir
        )
        metric_names = list(
            voynich_metrics.keys()
        )

        language_root = Path(
            "data/controls/languages"
        )
        master_manifest_path = (
            language_root
            / "language_controls_v1_manifest.json"
        )
        if not master_manifest_path.is_file():
            raise ValueError(
                "Frozen language-control master manifest missing: "
                f"{master_manifest_path}"
            )

        master_manifest = json.loads(
            master_manifest_path.read_text(
                encoding="utf-8"
            )
        )
        source_entries = master_manifest[
            "source_manifests"
        ]

        if len(source_entries) != 4:
            raise ValueError(
                "Expected exactly four frozen language sources"
            )

        control_root = Path(
            config["control_root"]
        )
        output_dir = Path(
            config["output_dir"]
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        run_output_root = output_dir / "runs"
        run_output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        print("=" * 96)
        print(
            "HISTORICAL CIPHER CONTROL STRUCTURAL BENCHMARK — "
            "140 RUNS"
        )
        print("=" * 96)
        print(f"Voynich reference: {voynich_dir}")
        print(f"Control root:      {control_root}")
        print(f"Tier-0 config:     {tier0_config_path}")
        print("Voynich scope:     TRAIN ONLY")
        print("Validation/test:   NOT ACCESSED")
        print()

        for source_entry in source_entries:
            source_id = str(
                source_entry["source_id"]
            )
            language = str(
                source_entry["language"]
            )

            source_manifest_path = (
                language_root
                / source_id
                / "manifest.json"
            )
            source_manifest = json.loads(
                source_manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            source_sha = source_manifest[
                "source_sha256"
            ]

            source_control_root = (
                control_root / source_id
            )
            source_cipher_manifest_path = (
                source_control_root
                / "manifest.json"
            )
            if not source_cipher_manifest_path.is_file():
                raise ValueError(
                    f"Missing cipher manifest for {source_id}: "
                    f"{source_cipher_manifest_path}"
                )

            cipher_manifest = json.loads(
                source_cipher_manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            if (
                cipher_manifest["source_sha256"]
                != source_sha
            ):
                raise ValueError(
                    f"{source_id}: cipher-control source hash does "
                    "not match frozen language source"
                )

            runs = cipher_manifest["runs"]
            if len(runs) != 35:
                raise ValueError(
                    f"{source_id}: expected 35 cipher runs; "
                    f"found {len(runs)}"
                )

            print(
                f"[{language}] {source_id}: "
                f"{len(runs)} runs"
            )

            for run in runs:
                family = str(run["family"])
                seed = int(run["seed"])
                run_dir = Path(run["path"])

                corpus_path = run_dir / "corpus.jsonl"
                cipher_summary_path = (
                    run_dir / "summary.json"
                )
                cipher_summary = json.loads(
                    cipher_summary_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    cipher_summary.get(
                        "round_trip_verified"
                    )
                    is not True
                ):
                    raise ValueError(
                        f"{source_id}/{family}/{seed}: "
                        "round-trip verification is not PASS"
                    )

                lines = read_control_jsonl(
                    corpus_path
                )
                summary, tables = benchmark_control(
                    lines,
                    config=tier0_config,
                )
                metrics = scalar_metrics(
                    summary,
                    tables,
                )

                run_out = (
                    run_output_root
                    / source_id
                    / family
                    / f"seed_{seed}"
                )
                run_out.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                payload = {
                    "schema_version": "1.0",
                    "source_id": source_id,
                    "language": language,
                    "source_sha256": source_sha,
                    "family": family,
                    "seed": seed,
                    "control_corpus_path": str(
                        corpus_path
                    ),
                    "control_corpus_sha256": (
                        sha256_file(corpus_path)
                    ),
                    "voynich_reference": str(
                        voynich_dir
                    ),
                    "voynich_scope": "TRAIN ONLY",
                    "validation_accessed": False,
                    "locked_test_accessed": False,
                    "benchmark": summary,
                    "scalar_metrics": metrics,
                }

                (run_out / "summary.json").write_text(
                    json.dumps(
                        payload,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
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
                    "source_sha256": source_sha,
                    "control_glyphs": summary[
                        "counts"
                    ]["glyphs"],
                    "control_tokens": summary[
                        "counts"
                    ]["tokens"],
                    "control_unique_glyphs": summary[
                        "counts"
                    ]["unique_glyphs"],
                    "control_unique_tokens": summary[
                        "counts"
                    ]["unique_tokens"],
                }

                for metric in metric_names:
                    value = float(metrics[metric])
                    reference = float(
                        voynich_metrics[metric]
                    )
                    row[metric] = value
                    row[
                        f"voynich_{metric}"
                    ] = reference
                    row[
                        f"delta_vs_voynich_{metric}"
                    ] = value - reference
                    row[
                        f"abs_delta_vs_voynich_{metric}"
                    ] = abs(value - reference)

                rows.append(row)

            print("  benchmarked 35/35")

        if len(rows) != 140:
            raise ValueError(
                f"Expected 140 benchmark rows; found {len(rows)}"
            )

        matrix_path = (
            output_dir / "mechanism_matrix.csv"
        )
        write_csv(matrix_path, rows)

        family_summary = (
            aggregate_rows_by_family(
                rows,
                metric_names=metric_names,
            )
        )
        family_summary_path = (
            output_dir / "family_summary.csv"
        )
        write_csv(
            family_summary_path,
            family_summary,
        )

        voynich_metrics_path = (
            output_dir
            / "voynich_train_reference_metrics.json"
        )
        voynich_metrics_path.write_text(
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

        top_summary = {
            "schema_version": "1.0",
            "analysis": (
                "historical cipher controls versus Voynich "
                "Tier-0 structural benchmark"
            ),
            "voynich_reference": str(voynich_dir),
            "voynich_scope": "TRAIN ONLY",
            "language_sources": len(
                source_entries
            ),
            "cipher_families_per_source": 7,
            "seeds_per_family": 5,
            "total_control_runs": len(rows),
            "single_similarity_score": False,
            "metric_names": metric_names,
            "interpretation_guardrails": config[
                "interpretation_guardrails"
            ],
            "validation_accessed": False,
            "locked_test_accessed": False,
            "artifacts": [
                "mechanism_matrix.csv",
                "family_summary.csv",
                "voynich_train_reference_metrics.json",
                "runs/<source>/<family>/seed_<seed>/...",
            ],
        }

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                top_summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        checksums = [
            matrix_path,
            family_summary_path,
            voynich_metrics_path,
            summary_path,
        ]
        (output_dir / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in checksums
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(f"Control runs:      {len(rows)}")
        print(
            f"Mechanism matrix:  {matrix_path}"
        )
        print(
            f"Family summary:    {family_summary_path}"
        )
        print(
            "No aggregate similarity score was computed."
        )
        print(
            "Leakage guard: Voynich validation and locked test "
            "were not read or required."
        )
        return 0

    except Exception as exc:
        print(
            f"CONTROL-BENCHMARK ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
