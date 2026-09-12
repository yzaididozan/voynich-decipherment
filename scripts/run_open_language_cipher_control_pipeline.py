#!/usr/bin/env python3
"""Generate 140 controls from the frozen open-language v1 sources and benchmark.

Requires:
- data/controls/languages_open_v1/language_controls_open_v1_manifest.json
- scripts/generate_historical_cipher_controls.py
- configs/cipher/historical_controls_v1.json
- scripts/run_open_historical_control_benchmark.py

No network access is needed once the four open-language sources are frozen.
No Voynich validation or locked-test file is read.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

LANGUAGE_ROOT = Path("data/controls/languages_open_v1")
MASTER = (
    LANGUAGE_ROOT
    / "language_controls_open_v1_manifest.json"
)
GENERATOR = Path("scripts/generate_historical_cipher_controls.py")
CIPHER_CONFIG = Path("configs/cipher/historical_controls_v1.json")
CONTROL_ROOT = Path("data/controls/ciphers/historical_open_v1")
BENCHMARK = Path("scripts/run_open_historical_control_benchmark.py")
BENCHMARK_CONFIG = Path(
    "configs/evaluation/historical_control_benchmark_open_v1.json"
)


def run(command: list[str]) -> None:
    print()
    print("$ " + " ".join(command))
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> int:
    try:
        for path in (
            MASTER,
            GENERATOR,
            CIPHER_CONFIG,
            BENCHMARK,
            BENCHMARK_CONFIG,
        ):
            if not path.is_file():
                raise ValueError(f"Required file missing: {path}")

        master = json.loads(MASTER.read_text(encoding="utf-8"))
        if master.get("freeze_id") != "open-language-controls-v1":
            raise ValueError(
                "Expected open-language-controls-v1 source freeze"
            )

        sources = master.get("sources", [])
        if len(sources) != 4:
            raise ValueError(
                f"Expected 4 frozen sources; found {len(sources)}"
            )

        print("=" * 92)
        print("GENERATE OPEN-LANGUAGE HISTORICAL CIPHER ENSEMBLES")
        print("=" * 92)
        print("Sources:          4")
        print("Families/source:  7")
        print("Seeds/family:     5")
        print("Expected controls: 140")
        print("Network:          NOT REQUIRED")
        print("Voynich test:     NOT ACCESSED")

        for entry in sources:
            source_id = str(entry["source_id"])
            source_path = LANGUAGE_ROOT / source_id / "source.txt"

            if not source_path.is_file():
                raise ValueError(
                    f"Frozen source missing: {source_path}"
                )

            run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--source",
                    str(source_path),
                    "--source-id",
                    source_id,
                    "--config",
                    str(CIPHER_CONFIG),
                    "--output-dir",
                    str(CONTROL_ROOT),
                ]
            )

        run(
            [
                sys.executable,
                str(BENCHMARK),
                "--config",
                str(BENCHMARK_CONFIG),
            ]
        )

        print()
        print("=" * 92)
        print("OPEN-LANGUAGE CIPHER CONTROL PIPELINE COMPLETE")
        print("=" * 92)
        print("Frozen language sources: 4")
        print("Generated controls:       140")
        print(
            "Controls:                 "
            "data/controls/ciphers/historical_open_v1"
        )
        print(
            "Benchmark:                "
            "results/controls/historical_cipher_benchmark/open_v1"
        )
        print("Voynich validation/test:  NOT ACCESSED")
        return 0

    except subprocess.CalledProcessError as exc:
        print(
            f"PIPELINE ERROR: command exited {exc.returncode}",
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
