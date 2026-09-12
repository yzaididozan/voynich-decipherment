#!/usr/bin/env python3
"""Freeze deterministic source-line splits for Level-2 predictive controls.

This script inspects only the already-frozen open-language source files.
It does not inspect any Level-2 model result.

Membership is generated independently of text content by SHA-256 ranking of
(source_id, line_index, frozen split seed). The same membership is reused for
all seven families and all five cipher keys derived from one language.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.level2_predictive_controls import (
    deterministic_unit_split,
    sha256_file,
)


DEFAULT_CONFIG = Path(
    "configs/evaluation/level2_predictive_controls_v1.json"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    args = parser.parse_args()

    try:
        config_bytes = args.config.read_bytes()
        config_sha = sha256(config_bytes).hexdigest()
        config = json.loads(config_bytes.decode("utf-8"))

        if config.get("freeze_id") != "level2-predictive-controls-v1":
            raise ValueError("Unexpected Level-2 freeze_id")

        split_cfg = config["split"]
        split_root = Path(config["split_root"])
        split_root.mkdir(parents=True, exist_ok=True)

        manifest_path = split_root / "manifest.json"

        if manifest_path.is_file():
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if manifest.get("config_sha256") != config_sha:
                raise ValueError(
                    "Level-2 splits already exist under a different "
                    "config hash. Create a new version rather than overwrite."
                )

            for source_id, entry in manifest["sources"].items():
                path = split_root / f"{source_id}.json"
                if not path.is_file():
                    raise ValueError(
                        f"Frozen split file missing: {path}"
                    )
                observed = sha256_file(path)
                if observed != entry["split_sha256"]:
                    raise ValueError(
                        f"Frozen split hash mismatch: {source_id}"
                    )

            print("=" * 88)
            print("LEVEL-2 CONTROL SPLITS — ALREADY FROZEN")
            print("=" * 88)
            print(f"Config SHA: {config_sha}")
            print("Sources:    4/4 verified")
            print("No split membership was regenerated.")
            return 0

        sources_out = {}
        language_root = Path(config["language_root"])
        weights = split_cfg["weights"]
        seed = int(split_cfg["seed"])

        print("=" * 88)
        print("FREEZE LEVEL-2 PREDICTIVE CONTROL SPLITS")
        print("=" * 88)
        print(f"Config SHA: {config_sha}")
        print(f"Split seed: {seed}")
        print("Unit:       source line")
        print("Weights:    train=72 validation=15 reserved_test=15")
        print()

        for source_id, expected in config["source_units"].items():
            source_path = (
                language_root
                / source_id
                / "source.txt"
            )
            if not source_path.is_file():
                raise ValueError(
                    f"Frozen source missing: {source_path}"
                )

            observed_sha = sha256_file(source_path)
            if observed_sha != expected["source_sha256"]:
                raise ValueError(
                    f"{source_id}: source SHA mismatch"
                )

            with source_path.open(encoding="utf-8") as handle:
                line_count = sum(1 for line in handle if line.strip())

            if line_count != int(expected["expected_lines"]):
                raise ValueError(
                    f"{source_id}: expected "
                    f"{expected['expected_lines']} nonempty lines; "
                    f"found {line_count}"
                )

            payload = deterministic_unit_split(
                source_id,
                line_count,
                seed=seed,
                weights=weights,
            )
            payload.update(
                {
                    "schema_version": "1.0",
                    "freeze_id": config["freeze_id"],
                    "config_sha256": config_sha,
                    "source_sha256": observed_sha,
                    "language": expected["language"],
                    "reserved_test_parsed_or_scored_by_level2": False,
                }
            )

            path = split_root / f"{source_id}.json"
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            split_sha = sha256_file(path)

            sources_out[source_id] = {
                "language": expected["language"],
                "source_sha256": observed_sha,
                "split_path": str(path),
                "split_sha256": split_sha,
                "counts": payload["counts"],
            }

            counts = payload["counts"]
            print(
                f"[{expected['language']}] {source_id}: "
                f"train={counts['train']} "
                f"validation={counts['validation']} "
                f"reserved_test={counts['reserved_test']}"
            )

        manifest = {
            "schema_version": "1.0",
            "freeze_id": config["freeze_id"],
            "config_path": str(args.config),
            "config_sha256": config_sha,
            "assignment": split_cfg["assignment"],
            "seed": seed,
            "weights": weights,
            "sources": sources_out,
            "voynich_locked_test_accessed": False,
            "level2_results_inspected_before_split_freeze": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        checksum_paths = [
            split_root / f"{source_id}.json"
            for source_id in config["source_units"]
        ] + [manifest_path]

        (split_root / "SHA256SUMS").write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in checksum_paths
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(f"Saved: {manifest_path}")
        print("Reserved test indices are frozen but not used by Level 2.")
        return 0

    except Exception as exc:
        print(
            f"LEVEL-2 SPLIT ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
