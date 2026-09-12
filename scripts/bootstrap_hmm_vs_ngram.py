#!/usr/bin/env python3
"""Paired physical-leaf bootstrap: selected HMM vs frozen 3-gram baseline.

Compares, within each frozen representation:

    delta = HMM(K=16) bits/event - 3-gram bits/event

Positive delta means the 3-gram has lower held-out NLL and is therefore
better. Resampling is performed over physical-leaf groups, preserving the
pairing between the two models.

This script reads validation outputs only. It does not read or require the
locked test split.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


NGRAM_INPUT = Path(
    "results/baselines/ngram/v1/per_leaf_validation.csv"
)
HMM_INPUT = Path(
    "results/baselines/hmm/v1/per_leaf_validation.csv"
)
OUTPUT_DIR = Path(
    "results/baselines/hmm_vs_ngram/v1"
)

SEED = 40814041438
N_BOOTSTRAP = 10_000

NGRAM_ORDER = 3
HMM_HIDDEN_STATES = 16
VIEWS = ("space_free", "token_aware")


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
) -> float:
    rows = list(rows)
    total_bits = sum(float(row["total_bits"]) for row in rows)
    total_events = sum(int(float(row["total_events"])) for row in rows)

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
        leaf = row["leaf_group"]

        if leaf in indexed:
            raise ValueError(
                f"Duplicate {label} row for leaf_group={leaf!r}"
            )

        indexed[leaf] = row

    if not indexed:
        raise ValueError(f"No rows found for {label}")

    return indexed


def select_ngram_rows(
    rows: Sequence[dict],
    *,
    view: str,
) -> Dict[str, dict]:
    selected = [
        row
        for row in rows
        if row["view"] == view
        and int(row["order"]) == NGRAM_ORDER
    ]

    return index_unique_by_leaf(
        selected,
        label=f"{view} {NGRAM_ORDER}-gram",
    )


def select_hmm_rows(
    rows: Sequence[dict],
    *,
    view: str,
) -> Dict[str, dict]:
    selected = [
        row
        for row in rows
        if row["view"] == view
        and int(row["hidden_states"]) == HMM_HIDDEN_STATES
    ]

    return index_unique_by_leaf(
        selected,
        label=f"{view} HMM K={HMM_HIDDEN_STATES}",
    )


def verify_paired_rows(
    ngram: Mapping[str, dict],
    hmm: Mapping[str, dict],
    *,
    view: str,
) -> List[str]:
    ngram_leaves = set(ngram)
    hmm_leaves = set(hmm)

    if ngram_leaves != hmm_leaves:
        only_ngram = sorted(ngram_leaves - hmm_leaves)
        only_hmm = sorted(hmm_leaves - ngram_leaves)

        raise ValueError(
            f"{view}: model outputs do not contain identical leaf groups.\n"
            f"  only in n-gram: {only_ngram}\n"
            f"  only in HMM:    {only_hmm}"
        )

    leaves = sorted(ngram_leaves)

    for leaf in leaves:
        ngram_events = int(float(ngram[leaf]["total_events"]))
        hmm_events = int(float(hmm[leaf]["total_events"]))

        if ngram_events != hmm_events:
            raise ValueError(
                f"{view} {leaf}: event-count mismatch between models: "
                f"3-gram={ngram_events}, HMM={hmm_events}. "
                "Refusing an unpaired comparison."
            )

        if ngram_events <= 0:
            raise ValueError(
                f"{view} {leaf}: non-positive validation event count"
            )

    return leaves


def bootstrap_view(
    ngram_rows: Sequence[dict],
    hmm_rows: Sequence[dict],
    *,
    view: str,
) -> dict:
    ngram = select_ngram_rows(ngram_rows, view=view)
    hmm = select_hmm_rows(hmm_rows, view=view)

    leaves = verify_paired_rows(
        ngram,
        hmm,
        view=view,
    )

    observed_ngram = aggregate_bits_per_event(
        ngram[leaf] for leaf in leaves
    )
    observed_hmm = aggregate_bits_per_event(
        hmm[leaf] for leaf in leaves
    )

    # Positive delta means HMM NLL is higher, so trigram is better.
    observed_delta = observed_hmm - observed_ngram

    # Fractional NLL reduction obtained by using the trigram instead of HMM.
    relative_reduction_vs_hmm = (
        observed_delta / observed_hmm
    )

    # Give each view a deterministic non-overlapping stream while keeping
    # the project-wide frozen seed.
    view_offset = 0 if view == "space_free" else 1_000_000
    rng = random.Random(SEED + view_offset)

    bootstrap_deltas: List[float] = []

    for _ in range(N_BOOTSTRAP):
        sampled_leaves = [
            rng.choice(leaves)
            for _ in range(len(leaves))
        ]

        boot_ngram = aggregate_bits_per_event(
            ngram[leaf] for leaf in sampled_leaves
        )
        boot_hmm = aggregate_bits_per_event(
            hmm[leaf] for leaf in sampled_leaves
        )

        bootstrap_deltas.append(
            boot_hmm - boot_ngram
        )

    lower = percentile(bootstrap_deltas, 0.025)
    upper = percentile(bootstrap_deltas, 0.975)

    probability_trigram_better = (
        sum(delta > 0.0 for delta in bootstrap_deltas)
        / N_BOOTSTRAP
    )
    probability_hmm_better = (
        sum(delta < 0.0 for delta in bootstrap_deltas)
        / N_BOOTSTRAP
    )

    if lower > 0.0:
        conclusion = "3-GRAM BETTER; CI EXCLUDES ZERO"
    elif upper < 0.0:
        conclusion = "HMM BETTER; CI EXCLUDES ZERO"
    else:
        conclusion = "CI CROSSES ZERO"

    return {
        "view": view,
        "ngram_order": NGRAM_ORDER,
        "hmm_hidden_states": HMM_HIDDEN_STATES,
        "bootstrap_units": len(leaves),
        "bootstrap_replicates": N_BOOTSTRAP,
        "seed": SEED,
        "ngram_bits_per_event": observed_ngram,
        "hmm_bits_per_event": observed_hmm,
        "delta_hmm_minus_ngram_bits_per_event": observed_delta,
        "relative_nll_reduction_ngram_vs_hmm": (
            relative_reduction_vs_hmm
        ),
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "bootstrap_probability_trigram_better": (
            probability_trigram_better
        ),
        "bootstrap_probability_hmm_better": (
            probability_hmm_better
        ),
        "conclusion": conclusion,
        "locked_test_accessed": False,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ngram_rows = read_csv(NGRAM_INPUT)
    hmm_rows = read_csv(HMM_INPUT)

    results = [
        bootstrap_view(
            ngram_rows,
            hmm_rows,
            view=view,
        )
        for view in VIEWS
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        OUTPUT_DIR / "comparison.csv",
        results,
    )

    summary = {
        "schema_version": "1.0",
        "analysis": (
            "paired physical-leaf bootstrap: "
            "HMM K=16 vs 3-gram Witten-Bell"
        ),
        "ngram_input": str(NGRAM_INPUT),
        "hmm_input": str(HMM_INPUT),
        "ngram_order": NGRAM_ORDER,
        "hmm_hidden_states": HMM_HIDDEN_STATES,
        "seed": SEED,
        "bootstrap_replicates": N_BOOTSTRAP,
        "delta_definition": (
            "HMM bits/event minus 3-gram bits/event; "
            "positive means 3-gram is better"
        ),
        "locked_test_accessed": False,
        "results": results,
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("PAIRED PHYSICAL-LEAF BOOTSTRAP — HMM K=16 VS 3-GRAM")
    print("=" * 88)
    print(f"N-gram input: {NGRAM_INPUT}")
    print(f"HMM input:    {HMM_INPUT}")
    print(f"Seed:         {SEED}")
    print(f"Replicates:   {N_BOOTSTRAP:,}")
    print()
    print(
        "Delta = HMM bits/event - 3-gram bits/event "
        "(positive => 3-gram better)"
    )
    print()

    for result in results:
        print(result["view"])
        print("-" * len(result["view"]))
        print(
            f"  bootstrap units:       "
            f"{result['bootstrap_units']}"
        )
        print(
            f"  3-gram bits/event:     "
            f"{result['ngram_bits_per_event']:.6f}"
        )
        print(
            f"  HMM K=16 bits/event:   "
            f"{result['hmm_bits_per_event']:.6f}"
        )
        print(
            f"  HMM - 3gram delta:     "
            f"{result['delta_hmm_minus_ngram_bits_per_event']:.6f} "
            "bits/event"
        )
        print(
            f"  3gram NLL reduction:   "
            f"{100 * result['relative_nll_reduction_ngram_vs_hmm']:.3f}% "
            "vs HMM"
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
            f"  bootstrap P(HMM better):   "
            f"{result['bootstrap_probability_hmm_better']:.4f}"
        )
        print(
            f"  result:                "
            f"{result['conclusion']}"
        )
        print()

    print(f"Saved: {OUTPUT_DIR / 'comparison.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'summary.json'}")
    print("Locked test was not accessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
