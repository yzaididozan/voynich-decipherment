#!/usr/bin/env python3
"""Run the frozen train-only Tier 0 descriptive pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import (
    build_tier0,
    load_config,
    make_analytical_loci,
    parse_train_leaf_ids,
    select_train_records,
    sha256_file,
    write_tier0_outputs,
)
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules


EXPECTED_ZL3B_SHA256 = (
    "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
)
EXPECTED_TRAIN_SPLIT_SHA256 = (
    "5bb5232d5e211a4bb90800f259294c554c1c9e2005548055ebaf94e782e3cc00"
)
EXPECTED_STA_EVA_RULES_SHA256 = (
    "7f37853510144fb3e2dc3ee9458d634f41e6d95bc1fbf1c4b8f479a53a021f81"
)

DEFAULT_SOURCE = Path("data/raw/zl/ZL3b-n.txt")
DEFAULT_TRAIN_SPLIT = Path("data/splits/v1/train.txt")
DEFAULT_RULES = Path("data/reference/sta1/STA-Eva_def.bit")
DEFAULT_CONFIG = Path("configs/tier0/train_v1.json")
DEFAULT_OUTPUT = Path("results/tier0/train/v1")


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch.\n"
            f"  observed: {observed}\n"
            f"  expected: {expected}\n"
            "Refusing to run Tier 0 on a non-frozen input."
        )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run train-only Tier 0 structural description. This command "
            "does not read validation.txt or test_LOCKED.txt."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--train-split",
        type=Path,
        default=DEFAULT_TRAIN_SPLIT,
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        source_hash = _require_hash(
            args.source,
            EXPECTED_ZL3B_SHA256,
            "ZL3b source",
        )
        split_hash = _require_hash(
            args.train_split,
            EXPECTED_TRAIN_SPLIT_SHA256,
            "Frozen train split",
        )
        rules_hash = _require_hash(
            args.rules,
            EXPECTED_STA_EVA_RULES_SHA256,
            "STA-Eva rules",
        )

        if not args.config.is_file():
            raise ValueError(f"Tier 0 config not found: {args.config}")
        config_hash = sha256_file(args.config)
        config = load_config(args.config)

        train_leaves = parse_train_leaf_ids(
            args.train_split.read_text(encoding="utf-8")
        )
        if len(train_leaves) != 72:
            raise ValueError(
                f"Frozen train split must contain 72 leaves; "
                f"found {len(train_leaves)}"
            )

        records = load_ivtff(
            args.source,
            transcription_id="ZL3b",
            strict=True,
        )
        train_records = select_train_records(records, train_leaves)

        rules = load_bitrans_rules(
            args.rules,
            expected_native_alphabet="Eva-",
        )
        if rules.sha256 != rules_hash:
            raise ValueError("Loaded rules hash does not match verified rules")

        analytical = make_analytical_loci(train_records, rules)

        summary, tables = build_tier0(
            train_records,
            analytical,
            config=config,
            train_leaves=train_leaves,
        )

        provenance = {
            "primary_transcription": "ZL3b",
            "source_path": str(args.source),
            "source_sha256": source_hash,
            "train_split_path": str(args.train_split),
            "train_split_sha256": split_hash,
            "sta1_rules_path": str(args.rules),
            "sta1_rules_sha256": rules_hash,
            "config_path": str(args.config),
            "config_sha256": config_hash,
            "validation_accessed": False,
            "locked_test_accessed": False,
        }

        write_tier0_outputs(
            args.output_dir,
            summary=summary,
            tables=tables,
            provenance=provenance,
        )

        counts = summary["counts"]
        entropy = summary["entropy"]

        print("=" * 72)
        print("TIER 0 DESCRIPTIVE STRUCTURAL ANALYSIS — TRAIN ONLY")
        print("=" * 72)
        print(f"Representation:      {summary['representation']}")
        print(f"Physical leaves:     {counts['physical_leaves']}")
        print(f"Loci:                {counts['loci']:,}")
        print(f"Tokens:              {counts['tokens']:,}")
        print(f"Unique tokens:       {counts['unique_tokens']:,}")
        print(f"Known STA1 glyphs:   {counts['glyphs_known']:,}")
        print(f"Unknown Z1 symbols:  {counts['unknown_Z1_symbols']:,}")
        print(
            "Unigram entropy:     "
            f"{entropy['unigram_entropy_bits_known_glyphs']:.6f} bits"
        )
        print()
        for order, payload in sorted(
            entropy["conditional_entropy_by_order"].items(),
            key=lambda item: int(item[0]),
        ):
            print(
                f"H(X|{order}-glyph context): "
                f"{payload['entropy_bits']:.6f} bits "
                f"(n={payload['transitions']:,})"
            )
        print()
        print(f"Saved Tier 0 artifacts: {args.output_dir}")
        print(
            "Leakage guard: validation and test memberships were not read."
        )
        return 0

    except Exception as exc:
        print(f"TIER 0 ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
