#!/usr/bin/env python3
"""Fit categorical HMM baselines on train and evaluate validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.tier0 import make_analytical_loci, sha256_file
from src.data.ivtff import load_ivtff
from src.data.sta1 import load_bitrans_rules
from src.models.hmm import CategoricalHMM, evaluate_hmm
from src.models.ngram import (
    build_baseline_sequences,
    parse_leaf_ids,
    select_records_by_leaves,
)


EXPECTED_ZL3B_SHA256 = (
    "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
)
EXPECTED_TRAIN_SHA256 = (
    "5bb5232d5e211a4bb90800f259294c554c1c9e2005548055ebaf94e782e3cc00"
)
EXPECTED_VALIDATION_SHA256 = (
    "bdae856ac8c52849a0abd0fd6361f42fd57579e76f7f844309a7ddc9e0fc02f1"
)
EXPECTED_RULES_SHA256 = (
    "7f37853510144fb3e2dc3ee9458d634f41e6d95bc1fbf1c4b8f479a53a021f81"
)

DEFAULT_SOURCE = Path("data/raw/zl/ZL3b-n.txt")
DEFAULT_TRAIN = Path("data/splits/v1/train.txt")
DEFAULT_VALIDATION = Path("data/splits/v1/validation.txt")
DEFAULT_RULES = Path("data/reference/sta1/STA-Eva_def.bit")
DEFAULT_CONFIG = Path("configs/baselines/hmm_v1.json")
DEFAULT_OUTPUT = Path("results/baselines/hmm/v1")


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch.\n"
            f"  observed: {observed}\n"
            f"  expected: {expected}\n"
            "Refusing to run against a non-frozen input."
        )
    return observed


def load_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("schema_version") != "1.0":
        raise ValueError("Unsupported HMM config schema")

    if data.get("hidden_states") != [2, 4, 8, 16]:
        raise ValueError(
            "HMM hidden-state grid must be exactly [2, 4, 8, 16]"
        )

    if data.get("views") != ["space_free", "token_aware"]:
        raise ValueError(
            "HMM views must be exactly space_free and token_aware"
        )

    if data.get("restarts") != 5:
        raise ValueError("HMM restarts must be exactly 5")

    if data.get("restart_selection") != "highest_training_log_likelihood":
        raise ValueError(
            "HMM restart selection must use training likelihood only"
        )

    if data.get("validation_selection") != "lowest_bits_per_event":
        raise ValueError(
            "HMM validation selection must use lowest bits/event"
        )

    if float(data.get("pseudocount", -1)) != 0.5:
        raise ValueError("HMM pseudocount must be exactly 0.5")

    return data


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_model(
    path: Path,
    model: CategoricalHMM,
    *,
    view: str,
    hidden_states: int,
    restart: int,
    seed: int,
) -> None:
    params = model.parameters()
    np.savez_compressed(
        path,
        vocabulary=params["vocabulary"],
        initial=params["initial"],
        transition=params["transition"],
        emission=params["emission"],
        view=np.asarray(view),
        hidden_states=np.asarray(hidden_states),
        restart=np.asarray(restart),
        seed=np.asarray(seed),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit categorical HMM baselines on frozen train leaves and "
            "evaluate frozen validation leaves only. The locked test is "
            "not read or required."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION,
    )
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        source_hash = require_hash(
            args.source,
            EXPECTED_ZL3B_SHA256,
            "ZL3b source",
        )
        train_hash = require_hash(
            args.train,
            EXPECTED_TRAIN_SHA256,
            "Frozen train split",
        )
        validation_hash = require_hash(
            args.validation,
            EXPECTED_VALIDATION_SHA256,
            "Frozen validation split",
        )
        rules_hash = require_hash(
            args.rules,
            EXPECTED_RULES_SHA256,
            "STA-Eva rules",
        )

        if not args.config.is_file():
            raise ValueError(f"Config not found: {args.config}")
        config_hash = sha256_file(args.config)
        config = load_config(args.config)

        train_leaves = parse_leaf_ids(
            args.train.read_text(encoding="utf-8"),
            label="train",
        )
        validation_leaves = parse_leaf_ids(
            args.validation.read_text(encoding="utf-8"),
            label="validation",
        )

        if len(train_leaves) != 72:
            raise ValueError(
                f"Expected 72 train leaves; found {len(train_leaves)}"
            )
        if len(validation_leaves) != 15:
            raise ValueError(
                "Expected 15 validation leaves; "
                f"found {len(validation_leaves)}"
            )

        overlap = set(train_leaves) & set(validation_leaves)
        if overlap:
            raise ValueError(
                "Train/validation leaf overlap: "
                + ", ".join(f"f{x}" for x in sorted(overlap))
            )

        records = load_ivtff(
            args.source,
            transcription_id="ZL3b",
            strict=True,
        )
        train_records = select_records_by_leaves(
            records,
            train_leaves,
            split_name="train",
        )
        validation_records = select_records_by_leaves(
            records,
            validation_leaves,
            split_name="validation",
        )

        rules = load_bitrans_rules(
            args.rules,
            expected_native_alphabet="Eva-",
        )
        if rules.sha256 != rules_hash:
            raise ValueError("Loaded rules hash differs from verified hash")

        train_loci = make_analytical_loci(train_records, rules)
        validation_loci = make_analytical_loci(
            validation_records,
            rules,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = args.output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        restart_rows = []
        validation_rows = []
        leaf_rows = []
        locus_rows = []

        base_seed = int(config["seed"])
        hidden_state_grid = list(config["hidden_states"])

        for view_index, view in enumerate(config["views"]):
            print()
            print(f"[{view}] preparing frozen event streams...")

            train_examples = build_baseline_sequences(
                train_loci,
                view=view,
            )
            validation_examples = build_baseline_sequences(
                validation_loci,
                view=view,
            )
            train_sequences = [
                example.symbols
                for example in train_examples
                if example.symbols
            ]

            for hidden_states in hidden_state_grid:
                fitted = []

                for restart in range(config["restarts"]):
                    seed = (
                        base_seed
                        + view_index * 1_000_000
                        + hidden_states * 10_000
                        + restart
                    )

                    print(
                        f"[{view}] K={hidden_states:2d} "
                        f"restart={restart + 1}/{config['restarts']} "
                        f"seed={seed}"
                    )

                    model = CategoricalHMM(
                        hidden_states,
                        pseudocount=config["pseudocount"],
                        max_iterations=config["max_iterations"],
                        min_iterations=config["min_iterations"],
                        tolerance_bits_per_event=(
                            config["tolerance_bits_per_event"]
                        ),
                        batch_size=config["batch_size"],
                    )

                    fit = model.fit(
                        train_sequences,
                        seed=seed,
                    )

                    restart_rows.append(
                        {
                            "view": view,
                            "hidden_states": hidden_states,
                            "restart": restart,
                            "seed": seed,
                            "iterations": fit.iterations,
                            "converged": fit.converged,
                            "training_log_likelihood_nats": (
                                fit.training_log_likelihood_nats
                            ),
                            "training_events": fit.training_events,
                            "training_bits_per_event": (
                                fit.training_bits_per_event
                            ),
                        }
                    )
                    fitted.append((fit, restart, seed, model))

                # Critical leakage safeguard: random restart selection uses
                # TRAINING likelihood only.
                best_fit, best_restart, best_seed, best_model = max(
                    fitted,
                    key=lambda item: (
                        item[0].training_log_likelihood_nats,
                        -item[1],
                    ),
                )

                aggregate, leaves, loci = evaluate_hmm(
                    best_model,
                    validation_examples,
                    view=view,
                    hidden_states=hidden_states,
                    restart=best_restart,
                    seed=best_seed,
                )
                aggregate["train_selected_restart"] = True
                aggregate["training_bits_per_event"] = (
                    best_fit.training_bits_per_event
                )
                aggregate["training_iterations"] = best_fit.iterations
                aggregate["training_converged"] = best_fit.converged

                validation_rows.append(aggregate)
                leaf_rows.extend(leaves)
                locus_rows.extend(loci)

                model_path = (
                    model_dir
                    / f"{view}_K{hidden_states}_train_best.npz"
                )
                save_model(
                    model_path,
                    best_model,
                    view=view,
                    hidden_states=hidden_states,
                    restart=best_restart,
                    seed=best_seed,
                )

                print(
                    f"  train-best restart={best_restart + 1}; "
                    f"train={best_fit.training_bits_per_event:.6f} "
                    f"bits/event; "
                    f"validation={aggregate['bits_per_event']:.6f} "
                    f"bits/event"
                )

        # Validation selects K only after train-only restart selection.
        selected_by_view = {}
        for view in config["views"]:
            candidates = [
                row
                for row in validation_rows
                if row["view"] == view
            ]
            selected = min(
                candidates,
                key=lambda row: (
                    row["bits_per_event"],
                    row["hidden_states"],
                ),
            )
            selected_by_view[view] = {
                "hidden_states": selected["hidden_states"],
                "bits_per_event": selected["bits_per_event"],
                "perplexity_per_event": selected["perplexity_per_event"],
                "restart": selected["restart"],
                "seed": selected["seed"],
                "tie_breaker": "lower hidden-state count",
            }

        write_csv(
            args.output_dir / "training_restarts.csv",
            restart_rows,
        )
        write_csv(
            args.output_dir / "validation_results.csv",
            validation_rows,
        )
        write_csv(
            args.output_dir / "per_leaf_validation.csv",
            leaf_rows,
        )
        write_csv(
            args.output_dir / "per_locus_validation.csv",
            locus_rows,
        )

        provenance = {
            "primary_transcription": "ZL3b",
            "source_path": str(args.source),
            "source_sha256": source_hash,
            "train_split_path": str(args.train),
            "train_split_sha256": train_hash,
            "validation_split_path": str(args.validation),
            "validation_split_sha256": validation_hash,
            "sta1_rules_path": str(args.rules),
            "sta1_rules_sha256": rules_hash,
            "config_path": str(args.config),
            "config_sha256": config_hash,
            "numpy_version": np.__version__,
            "train_physical_leaves": len(train_leaves),
            "validation_physical_leaves": len(validation_leaves),
            "locked_test_accessed": False,
        }

        summary = {
            "schema_version": "1.0",
            "analysis_tier": "Tier 1 HMM baseline",
            "fit_scope": "TRAIN ONLY",
            "evaluation_scope": "VALIDATION ONLY",
            "locked_test_accessed": False,
            "model": "categorical_hidden_markov_model",
            "training_algorithm": "Baum-Welch EM",
            "restart_selection": (
                "highest training log likelihood only"
            ),
            "validation_selection": "lowest bits/event over K",
            "selected_by_view": selected_by_view,
            "validation_results": validation_rows,
            "provenance": provenance,
            "guardrails": [
                "Random restarts are selected using training likelihood only.",
                "Validation selects hidden-state count but never the random restart.",
                "HMM and n-gram comparisons must use the same representation/view.",
                "Locked test remains untouched.",
            ],
        }

        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "schema_version": "1.0",
            "config": config,
            "provenance": provenance,
            "artifacts": [
                "training_restarts.csv",
                "validation_results.csv",
                "per_leaf_validation.csv",
                "per_locus_validation.csv",
                "summary.json",
                "models/*.npz",
            ],
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifacts = sorted(
            path
            for path in args.output_dir.rglob("*")
            if path.is_file()
            and path.name != "SHA256SUMS"
        )
        checksum_lines = [
            f"{sha256_file(path)}  {path.relative_to(args.output_dir)}"
            for path in artifacts
        ]
        (args.output_dir / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

        print()
        print("=" * 78)
        print("HMM BASELINE — TRAIN FIT / VALIDATION SELECTION")
        print("=" * 78)
        print(
            "Hidden states: "
            + ", ".join(str(x) for x in config["hidden_states"])
        )
        print(f"Restarts/K:   {config['restarts']}")
        print(
            "Restart rule: highest TRAINING likelihood only"
        )
        print("Test:         NOT ACCESSED")
        print()

        for view in config["views"]:
            print(view)
            print("-" * len(view))
            rows = sorted(
                [
                    row
                    for row in validation_rows
                    if row["view"] == view
                ],
                key=lambda row: row["hidden_states"],
            )
            for row in rows:
                print(
                    f"K={row['hidden_states']:2d}  "
                    f"bits/event={row['bits_per_event']:.6f}  "
                    f"ppl={row['perplexity_per_event']:.3f}  "
                    f"restart={row['restart'] + 1}  "
                    f"OOV={row['oov_events']}"
                )

            selected = selected_by_view[view]
            print(
                f"best validation K: {selected['hidden_states']} "
                f"({selected['bits_per_event']:.6f} bits/event)"
            )
            print()

        print(f"Saved HMM artifacts: {args.output_dir}")
        print(
            "Leakage guard: test_LOCKED.txt was not read or required."
        )
        return 0

    except Exception as exc:
        print(f"HMM BASELINE ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
