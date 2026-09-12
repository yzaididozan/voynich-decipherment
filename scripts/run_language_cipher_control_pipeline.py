#!/usr/bin/env python3
"""Run the frozen historical-language -> cipher-control -> benchmark pipeline.

Stages:
1. Fetch/freeze four 20k-token historical-language sources if v1 does not
   already exist. If it exists, verify hashes and do not re-fetch.
2. Generate 7 cipher families x 5 seeds = 35 controls per source.
3. Run the Tier-0-compatible structural benchmark over 140 controls against
   Voynich TRAIN Tier-0 only.

The pipeline never reads Voynich validation.txt or test_LOCKED.txt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REGISTRY = Path(
    "configs/corpus/language_controls_v1.json"
)
DEFAULT_LANGUAGE_ROOT = Path(
    "data/controls/languages"
)
DEFAULT_CIPHER_CONFIG = Path(
    "configs/cipher/historical_controls_v1.json"
)
DEFAULT_CIPHER_ROOT = Path(
    "data/controls/ciphers/historical_v1"
)
DEFAULT_BENCHMARK_CONFIG = Path(
    "configs/evaluation/historical_control_benchmark_v1.json"
)


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(
        command,
        check=True,
        cwd=REPO_ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze four historical-language controls, generate all "
            "historical cipher ensembles, and benchmark them."
        )
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help=(
            "Require an already-frozen language-controls-v1 manifest "
            "instead of invoking the fetch/verifier stage."
        ),
    )
    args = parser.parse_args()

    try:
        freeze_script = (
            REPO_ROOT
            / "scripts/fetch_freeze_language_controls.py"
        )
        generator_script = (
            REPO_ROOT
            / "scripts/generate_historical_cipher_controls.py"
        )
        benchmark_script = (
            REPO_ROOT
            / "scripts/run_historical_control_benchmark.py"
        )

        for path in (
            freeze_script,
            generator_script,
            benchmark_script,
            DEFAULT_REGISTRY,
            DEFAULT_CIPHER_CONFIG,
            DEFAULT_BENCHMARK_CONFIG,
        ):
            if not path.is_file():
                raise ValueError(
                    f"Required pipeline file missing: {path}"
                )

        master_manifest_path = (
            DEFAULT_LANGUAGE_ROOT
            / "language_controls_v1_manifest.json"
        )

        if args.skip_fetch:
            if not master_manifest_path.is_file():
                raise ValueError(
                    "--skip-fetch requires an existing frozen "
                    "language-controls-v1 master manifest"
                )
        else:
            run(
                [
                    sys.executable,
                    str(freeze_script),
                    "--registry",
                    str(DEFAULT_REGISTRY),
                    "--output-root",
                    str(DEFAULT_LANGUAGE_ROOT),
                ]
            )

        master = json.loads(
            master_manifest_path.read_text(
                encoding="utf-8"
            )
        )

        source_entries = master[
            "source_manifests"
        ]
        if len(source_entries) != 4:
            raise ValueError(
                "Expected exactly four frozen source corpora"
            )

        print()
        print("=" * 92)
        print(
            "GENERATE HISTORICAL CIPHER ENSEMBLES "
            "(35 PER SOURCE; 140 TOTAL)"
        )
        print("=" * 92)

        for entry in source_entries:
            source_id = str(
                entry["source_id"]
            )
            source_path = (
                DEFAULT_LANGUAGE_ROOT
                / source_id
                / "source.txt"
            )

            run(
                [
                    sys.executable,
                    str(generator_script),
                    "--source",
                    str(source_path),
                    "--source-id",
                    source_id,
                    "--config",
                    str(DEFAULT_CIPHER_CONFIG),
                    "--output-dir",
                    str(DEFAULT_CIPHER_ROOT),
                ]
            )

        run(
            [
                sys.executable,
                str(benchmark_script),
                "--config",
                str(DEFAULT_BENCHMARK_CONFIG),
            ]
        )

        print()
        print("=" * 92)
        print("HISTORICAL CONTROL PIPELINE COMPLETE")
        print("=" * 92)
        print("Frozen sources:   4")
        print("Cipher families:  7")
        print("Seeds/family:     5")
        print("Control corpora:  140")
        print(
            "Benchmark output: results/controls/"
            "historical_cipher_benchmark/v1"
        )
        print(
            "Leakage guard: Voynich validation and locked test "
            "were not read or required."
        )
        return 0

    except subprocess.CalledProcessError as exc:
        print(
            f"PIPELINE ERROR: command failed with exit "
            f"{exc.returncode}",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except Exception as exc:
        print(
            f"PIPELINE ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
