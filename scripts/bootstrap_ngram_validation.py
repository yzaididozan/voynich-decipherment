#!/usr/bin/env python3

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


INPUT = Path("results/baselines/ngram/v1/per_leaf_validation.csv")
SEED = 40814041438
N_BOOTSTRAP = 10_000


def percentile(values, q):
    values = sorted(values)

    if len(values) == 1:
        return values[0]

    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo

    return values[lo] * (1 - frac) + values[hi] * frac


def bits_per_event(rows):
    total_bits = sum(float(row["total_bits"]) for row in rows)
    total_events = sum(int(float(row["total_events"])) for row in rows)

    return total_bits / total_events


def bootstrap_comparison(rows, view, competitor_order, target_order=3):
    relevant = [
        row
        for row in rows
        if row["view"] == view
        and int(row["order"]) in {competitor_order, target_order}
    ]

    by_leaf = defaultdict(dict)

    for row in relevant:
        leaf = row["leaf_group"]
        order = int(row["order"])

        if order in by_leaf[leaf]:
            raise ValueError(
                f"Duplicate row for {view}, {leaf}, order {order}"
            )

        by_leaf[leaf][order] = row

    incomplete = {
        leaf: sorted(data)
        for leaf, data in by_leaf.items()
        if set(data) != {competitor_order, target_order}
    }

    if incomplete:
        raise ValueError(
            f"Incomplete paired observations: {incomplete}"
        )

    leaves = sorted(by_leaf)

    if not leaves:
        raise ValueError(
            f"No data for {view}: {competitor_order} vs {target_order}"
        )

    competitor_rows = [
        by_leaf[leaf][competitor_order]
        for leaf in leaves
    ]
    target_rows = [
        by_leaf[leaf][target_order]
        for leaf in leaves
    ]

    competitor_bpe = bits_per_event(competitor_rows)
    target_bpe = bits_per_event(target_rows)

    observed_delta = competitor_bpe - target_bpe

    rng = random.Random(
        SEED
        + competitor_order * 100
        + target_order
        + (0 if view == "space_free" else 10_000)
    )

    bootstrap_deltas = []

    for _ in range(N_BOOTSTRAP):
        sampled_leaves = [
            rng.choice(leaves)
            for _ in range(len(leaves))
        ]

        sampled_competitor = [
            by_leaf[leaf][competitor_order]
            for leaf in sampled_leaves
        ]
        sampled_target = [
            by_leaf[leaf][target_order]
            for leaf in sampled_leaves
        ]

        competitor_boot = bits_per_event(sampled_competitor)
        target_boot = bits_per_event(sampled_target)

        bootstrap_deltas.append(
            competitor_boot - target_boot
        )

    lower = percentile(bootstrap_deltas, 0.025)
    upper = percentile(bootstrap_deltas, 0.975)

    probability_target_better = sum(
        delta > 0
        for delta in bootstrap_deltas
    ) / N_BOOTSTRAP

    relative_reduction = observed_delta / competitor_bpe

    return {
        "view": view,
        "comparison": f"{target_order}-gram vs {competitor_order}-gram",
        "bootstrap_units": len(leaves),
        "competitor_bits_per_event": competitor_bpe,
        "target_bits_per_event": target_bpe,
        "delta_bits_per_event": observed_delta,
        "relative_reduction": relative_reduction,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "bootstrap_probability_target_better": probability_target_better,
    }


def main():
    if not INPUT.is_file():
        raise SystemExit(f"Missing input: {INPUT}")

    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print("=" * 88)
    print("PAIRED PHYSICAL-LEAF BOOTSTRAP — VALIDATION N-GRAM COMPARISONS")
    print("=" * 88)
    print(f"Input:      {INPUT}")
    print(f"Seed:       {SEED}")
    print(f"Replicates: {N_BOOTSTRAP:,}")
    print()

    for view in ("space_free", "token_aware"):
        print(view)
        print("-" * len(view))

        for competitor in (2, 4):
            result = bootstrap_comparison(
                rows,
                view=view,
                competitor_order=competitor,
                target_order=3,
            )

            print(result["comparison"])
            print(
                f"  bootstrap units:       "
                f"{result['bootstrap_units']}"
            )
            print(
                f"  competitor bits/event: "
                f"{result['competitor_bits_per_event']:.6f}"
            )
            print(
                f"  3-gram bits/event:     "
                f"{result['target_bits_per_event']:.6f}"
            )
            print(
                f"  improvement:           "
                f"{result['delta_bits_per_event']:.6f} bits/event"
            )
            print(
                f"  relative reduction:    "
                f"{100 * result['relative_reduction']:.3f}%"
            )
            print(
                f"  paired bootstrap 95% CI: "
                f"[{result['ci_95_lower']:.6f}, "
                f"{result['ci_95_upper']:.6f}]"
            )
            print(
                f"  bootstrap P(3-gram better): "
                f"{result['bootstrap_probability_target_better']:.4f}"
            )

            if result["ci_95_lower"] > 0:
                interpretation = "3-GRAM BETTER; CI EXCLUDES ZERO"
            elif result["ci_95_upper"] < 0:
                interpretation = "COMPETITOR BETTER; CI EXCLUDES ZERO"
            else:
                interpretation = "CI CROSSES ZERO"

            print(f"  result:                {interpretation}")
            print()

    print(
        "Positive delta means lower validation NLL for the 3-gram."
    )
    print("Locked test was not accessed.")


if __name__ == "__main__":
    main()
