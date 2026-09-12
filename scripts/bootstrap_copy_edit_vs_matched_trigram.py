#!/usr/bin/env python3
"""Paired physical-leaf bootstrap: selected copy-edit model vs matched trigram.

Compares the validation-selected local copy-edit/self-copy model against the
matched token-reset 3-gram + EOT produced by run_copy_edit_validation.py.

Delta definition:

    delta = copy-edit bits/event - matched 3-gram bits/event

Positive delta means the matched 3-gram has lower held-out NLL and is better.

Bootstrap resampling is performed over physical-leaf groups with replacement,
preserving model pairing. Corpus bits/event is computed as a ratio of summed
bits to summed events for every bootstrap replicate.

The locked test split is not read or required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DEFAULT_COPY_INPUT = Path(
    "results/baselines/copy_edit/v1/per_leaf_validation.csv"
)
DEFAULT_TRIGRAM_INPUT = Path(
    "results/baselines/copy_edit/v1/matched_trigram_per_leaf.csv"
)
DEFAULT_COPY_SUMMARY = Path(
    "results/baselines/copy_edit/v1/summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/baselines/copy_edit_vs_matched_trigram/v1"
)

SEED = 40814041438
N_BOOTSTRAP = 10_000

EXPECTED_SELECTED_WINDOW = 16
EXPECTED_TRIGRAM_ORDER = 3
EXPECTED_TRIGRAM_VIEW = "token_reset_eot"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"No rows in input: {path}")

    return rows


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = position - lo

    return (
        ordered[lo] * (1.0 - fraction)
        + ordered[hi] * fraction
    )


def aggregate_bits_per_event(
    rows: Iterable[Mapping[str, object]],
    *,
    event_field: str,
) -> float:
    materialized = list(rows)
    total_bits = sum(float(row["total_bits"]) for row in materialized)
    total_events = sum(
        int(float(row[event_field]))
        for row in materialized
    )

    if total_events <= 0:
        raise ValueError("Aggregate event count must be positive")

    return total_bits / total_events


def index_unique_by_leaf(
    rows: Sequence[dict],
    *,
    label: str,
) -> Dict[str, dict]:
    indexed: Dict[str, dict] = {}

    for row in rows:
        leaf = str(row.get("leaf_group", "")).strip()
        if not leaf:
            raise ValueError(f"{label}: row missing leaf_group")

        if leaf in indexed:
            raise ValueError(
                f"Duplicate {label} row for leaf_group={leaf!r}"
            )

        indexed[leaf] = row

    if not indexed:
        raise ValueError(f"No rows found for {label}")

    return indexed


def selected_source_window(summary: Mapping[str, object]) -> int:
    if summary.get("locked_test_accessed") is not False:
        raise ValueError(
            "Copy-edit summary does not explicitly state "
            "locked_test_accessed=false"
        )

    try:
        selected = int(summary["selected_source_window"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Copy-edit summary lacks a valid selected_source_window"
        ) from exc

    if selected != EXPECTED_SELECTED_WINDOW:
        raise ValueError(
            "Unexpected selected source window. "
            f"Observed W={selected}; expected frozen validation winner "
            f"W={EXPECTED_SELECTED_WINDOW}."
        )

    return selected


def select_copy_rows(
    rows: Sequence[dict],
    *,
    source_window: int,
) -> Dict[str, dict]:
    selected = []

    for row in rows:
        try:
            row_window = int(row["source_window"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Copy-edit per-leaf file lacks a valid source_window column"
            ) from exc

        if row_window == source_window:
            selected.append(row)

    return index_unique_by_leaf(
        selected,
        label=f"copy-edit W={source_window}",
    )


def select_trigram_rows(rows: Sequence[dict]) -> Dict[str, dict]:
    selected = []

    for row in rows:
        try:
            order = int(row["order"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Matched-trigram per-leaf file lacks a valid order column"
            ) from exc

        view = str(row.get("view", ""))

        if (
            order == EXPECTED_TRIGRAM_ORDER
            and view == EXPECTED_TRIGRAM_VIEW
        ):
            selected.append(row)

    return index_unique_by_leaf(
        selected,
        label=(
            f"matched {EXPECTED_TRIGRAM_VIEW} "
            f"{EXPECTED_TRIGRAM_ORDER}-gram"
        ),
    )


def verify_paired_rows(
    copy_edit: Mapping[str, dict],
    trigram: Mapping[str, dict],
) -> List[str]:
    copy_leaves = set(copy_edit)
    trigram_leaves = set(trigram)

    if copy_leaves != trigram_leaves:
        only_copy = sorted(copy_leaves - trigram_leaves)
        only_trigram = sorted(trigram_leaves - copy_leaves)

        raise ValueError(
            "Model outputs do not contain identical leaf groups.\n"
            f"  only in copy-edit: {only_copy}\n"
            f"  only in matched trigram: {only_trigram}"
        )

    leaves = sorted(copy_leaves)

    if not leaves:
        raise ValueError("No paired physical-leaf groups found")

    for leaf in leaves:
        try:
            copy_events = int(float(copy_edit[leaf]["events"]))
            trigram_events = int(
                float(trigram[leaf]["total_events"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{leaf}: invalid event-count field"
            ) from exc

        if copy_events != trigram_events:
            raise ValueError(
                f"{leaf}: event-count mismatch between models: "
                f"copy-edit={copy_events}, trigram={trigram_events}. "
                "Refusing an unpaired comparison."
            )

        if copy_events <= 0:
            raise ValueError(
                f"{leaf}: non-positive validation event count"
            )

        copy_bits = float(copy_edit[leaf]["total_bits"])
        trigram_bits = float(trigram[leaf]["total_bits"])

        if not math.isfinite(copy_bits) or not math.isfinite(trigram_bits):
            raise ValueError(
                f"{leaf}: non-finite held-out NLL"
            )

    return leaves


def check_aggregate_against_summary(
    summary: Mapping[str, object],
    *,
    copy_bpe: float,
    trigram_bpe: float,
) -> None:
    try:
        summary_copy = float(
            summary["selected_validation_bits_per_event"]
        )
        matched = summary["matched_trigram"]
        if not isinstance(matched, Mapping):
            raise TypeError
        summary_trigram = float(matched["bits_per_event"])
        summary_order = int(matched["order"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Copy-edit summary lacks expected aggregate comparison fields"
        ) from exc

    if summary_order != EXPECTED_TRIGRAM_ORDER:
        raise ValueError(
            "Summary matched-trigram order differs from frozen order 3"
        )

    tolerance = 1e-9

    if abs(copy_bpe - summary_copy) > tolerance:
        raise ValueError(
            "Per-leaf copy-edit aggregate does not reproduce summary value: "
            f"per-leaf={copy_bpe:.12f}, summary={summary_copy:.12f}"
        )

    if abs(trigram_bpe - summary_trigram) > tolerance:
        raise ValueError(
            "Per-leaf matched-trigram aggregate does not reproduce "
            "summary value: "
            f"per-leaf={trigram_bpe:.12f}, "
            f"summary={summary_trigram:.12f}"
        )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_comparison(
    copy_rows: Sequence[dict],
    trigram_rows: Sequence[dict],
    summary: Mapping[str, object],
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[dict, List[dict]]:
    source_window = selected_source_window(summary)

    copy_edit = select_copy_rows(
        copy_rows,
        source_window=source_window,
    )
    trigram = select_trigram_rows(trigram_rows)

    leaves = verify_paired_rows(copy_edit, trigram)

    observed_copy = aggregate_bits_per_event(
        (copy_edit[leaf] for leaf in leaves),
        event_field="events",
    )
    observed_trigram = aggregate_bits_per_event(
        (trigram[leaf] for leaf in leaves),
        event_field="total_events",
    )

    check_aggregate_against_summary(
        summary,
        copy_bpe=observed_copy,
        trigram_bpe=observed_trigram,
    )

    # Positive delta means copy-edit has higher NLL, so trigram is better.
    observed_delta = observed_copy - observed_trigram
    relative_reduction_vs_copy = observed_delta / observed_copy

    rng = random.Random(seed)
    bootstrap_deltas: List[float] = []
    bootstrap_rows: List[dict] = []

    for replicate in range(1, n_bootstrap + 1):
        sampled_leaves = [
            rng.choice(leaves)
            for _ in range(len(leaves))
        ]

        boot_copy = aggregate_bits_per_event(
            (copy_edit[leaf] for leaf in sampled_leaves),
            event_field="events",
        )
        boot_trigram = aggregate_bits_per_event(
            (trigram[leaf] for leaf in sampled_leaves),
            event_field="total_events",
        )
        delta = boot_copy - boot_trigram

        bootstrap_deltas.append(delta)
        bootstrap_rows.append(
            {
                "replicate": replicate,
                "copy_edit_bits_per_event": boot_copy,
                "matched_trigram_bits_per_event": boot_trigram,
                "delta_copy_minus_trigram_bits_per_event": delta,
            }
        )

    lower = percentile(bootstrap_deltas, 0.025)
    upper = percentile(bootstrap_deltas, 0.975)

    probability_trigram_better = (
        sum(delta > 0.0 for delta in bootstrap_deltas)
        / n_bootstrap
    )
    probability_copy_better = (
        sum(delta < 0.0 for delta in bootstrap_deltas)
        / n_bootstrap
    )
    probability_tie = (
        sum(delta == 0.0 for delta in bootstrap_deltas)
        / n_bootstrap
    )

    if lower > 0.0:
        conclusion = "MATCHED 3-GRAM BETTER; CI EXCLUDES ZERO"
    elif upper < 0.0:
        conclusion = "COPY-EDIT BETTER; CI EXCLUDES ZERO"
    else:
        conclusion = "CI CROSSES ZERO"

    result = {
        "selected_source_window": source_window,
        "matched_trigram_order": EXPECTED_TRIGRAM_ORDER,
        "matched_trigram_view": EXPECTED_TRIGRAM_VIEW,
        "bootstrap_units": len(leaves),
        "bootstrap_replicates": n_bootstrap,
        "seed": seed,
        "copy_edit_bits_per_event": observed_copy,
        "matched_trigram_bits_per_event": observed_trigram,
        "delta_copy_minus_trigram_bits_per_event": observed_delta,
        "relative_nll_reduction_trigram_vs_copy_edit": (
            relative_reduction_vs_copy
        ),
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "bootstrap_probability_trigram_better": (
            probability_trigram_better
        ),
        "bootstrap_probability_copy_edit_better": (
            probability_copy_better
        ),
        "bootstrap_probability_tie": probability_tie,
        "conclusion": conclusion,
        "locked_test_accessed": False,
    }

    return result, bootstrap_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Paired physical-leaf bootstrap: validation-selected "
            "local copy-edit/self-copy model vs matched token-reset trigram."
        )
    )
    parser.add_argument(
        "--copy-input",
        type=Path,
        default=DEFAULT_COPY_INPUT,
    )
    parser.add_argument(
        "--trigram-input",
        type=Path,
        default=DEFAULT_TRIGRAM_INPUT,
    )
    parser.add_argument(
        "--copy-summary",
        type=Path,
        default=DEFAULT_COPY_SUMMARY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=N_BOOTSTRAP,
    )
    args = parser.parse_args()

    try:
        if args.replicates < 1:
            raise ValueError("--replicates must be >= 1")

        copy_rows = read_csv(args.copy_input)
        trigram_rows = read_csv(args.trigram_input)
        summary = read_json(args.copy_summary)

        result, bootstrap_rows = bootstrap_comparison(
            copy_rows,
            trigram_rows,
            summary,
            seed=args.seed,
            n_bootstrap=args.replicates,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = args.output_dir / "comparison.csv"
        bootstrap_path = args.output_dir / "bootstrap_deltas.csv"
        summary_path = args.output_dir / "summary.json"

        write_csv(comparison_path, [result])
        write_csv(bootstrap_path, bootstrap_rows)

        output_summary = {
            "schema_version": "1.0",
            "analysis": (
                "paired physical-leaf bootstrap: validation-selected "
                "local copy-edit/self-copy model vs matched token-reset "
                "3-gram + EOT"
            ),
            "copy_edit_input": str(args.copy_input),
            "copy_edit_input_sha256": sha256_file(args.copy_input),
            "matched_trigram_input": str(args.trigram_input),
            "matched_trigram_input_sha256": sha256_file(
                args.trigram_input
            ),
            "copy_edit_summary_input": str(args.copy_summary),
            "copy_edit_summary_input_sha256": sha256_file(
                args.copy_summary
            ),
            "selected_source_window": result[
                "selected_source_window"
            ],
            "matched_trigram_order": EXPECTED_TRIGRAM_ORDER,
            "matched_trigram_view": EXPECTED_TRIGRAM_VIEW,
            "bootstrap_unit": "physical_leaf_group",
            "bootstrap_method": (
                "paired resampling of leaf groups with replacement; "
                "ratio-of-sums corpus bits/event computed separately "
                "for each model in every replicate"
            ),
            "seed": args.seed,
            "bootstrap_replicates": args.replicates,
            "delta_definition": (
                "copy-edit bits/event minus matched-trigram bits/event; "
                "positive means matched trigram is better"
            ),
            "locked_test_accessed": False,
            "result": result,
        }

        summary_path.write_text(
            json.dumps(
                output_summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        checksum_targets = [
            comparison_path,
            bootstrap_path,
            summary_path,
        ]
        checksum_path = args.output_dir / "SHA256SUMS"
        checksum_path.write_text(
            "\n".join(
                f"{sha256_file(path)}  {path.name}"
                for path in checksum_targets
            )
            + "\n",
            encoding="utf-8",
        )

        print("=" * 88)
        print(
            "PAIRED PHYSICAL-LEAF BOOTSTRAP — "
            "COPY-EDIT VS MATCHED 3-GRAM"
        )
        print("=" * 88)
        print(f"Copy-edit input:  {args.copy_input}")
        print(f"Trigram input:    {args.trigram_input}")
        print(f"Summary input:    {args.copy_summary}")
        print(f"Seed:             {args.seed}")
        print(f"Replicates:       {args.replicates:,}")
        print("Locked test:      NOT ACCESSED")
        print()
        print(
            "Delta = copy-edit bits/event - matched 3-gram bits/event "
            "(positive => matched 3-gram better)"
        )
        print()
        print(
            f"  selected copy W:       "
            f"{result['selected_source_window']}"
        )
        print(
            f"  bootstrap units:       "
            f"{result['bootstrap_units']}"
        )
        print(
            f"  copy-edit bits/event:  "
            f"{result['copy_edit_bits_per_event']:.6f}"
        )
        print(
            f"  matched 3gram b/e:     "
            f"{result['matched_trigram_bits_per_event']:.6f}"
        )
        print(
            f"  copy - 3gram delta:    "
            f"{result['delta_copy_minus_trigram_bits_per_event']:.6f} "
            "bits/event"
        )
        print(
            f"  3gram NLL reduction:   "
            f"{100 * result['relative_nll_reduction_trigram_vs_copy_edit']:.3f}% "
            "vs copy-edit"
        )
        print(
            f"  paired bootstrap 95% CI: "
            f"[{result['ci_95_lower']:.6f}, "
            f"{result['ci_95_upper']:.6f}]"
        )
        print(
            f"  bootstrap P(3gram better): "
            f"{result['bootstrap_probability_trigram_better']:.4f}"
        )
        print(
            f"  bootstrap P(copy better):  "
            f"{result['bootstrap_probability_copy_edit_better']:.4f}"
        )
        print(
            f"  result:                "
            f"{result['conclusion']}"
        )
        print()
        print(f"Saved: {comparison_path}")
        print(f"Saved: {bootstrap_path}")
        print(f"Saved: {summary_path}")
        print(f"Saved: {checksum_path}")
        print("Locked test was not accessed.")

        return 0

    except Exception as exc:
        print(
            f"COPY-VS-TRIGRAM BOOTSTRAP ERROR: "
            f"{type(exc).__name__}: {exc}",
            file=__import__("sys").stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
