#!/usr/bin/env python3
from pathlib import Path

path = Path("src/models/ngram.py")
if not path.is_file():
    raise SystemExit(f"Not found: {path}")

text = path.read_text(encoding="utf-8")

old = '''    unigram = next(row for row in aggregate_rows if row["order"] == 1)
    for row in aggregate_rows:
        row["delta_bits_per_event_vs_unigram"] = (
            unigram["bits_per_event"] - row["bits_per_event"]
        )
        row["relative_nll_reduction_vs_unigram"] = (
            1.0 - row["total_bits"] / unigram["total_bits"]
        )

    return aggregate_rows, leaf_rows, locus_rows
'''

new = '''    unigram = next(
        (row for row in aggregate_rows if row["order"] == 1),
        None,
    )
    for row in aggregate_rows:
        if unigram is None:
            row["delta_bits_per_event_vs_unigram"] = None
            row["relative_nll_reduction_vs_unigram"] = None
        else:
            row["delta_bits_per_event_vs_unigram"] = (
                unigram["bits_per_event"] - row["bits_per_event"]
            )
            row["relative_nll_reduction_vs_unigram"] = (
                1.0 - row["total_bits"] / unigram["total_bits"]
            )

    return aggregate_rows, leaf_rows, locus_rows
'''

if new in text:
    print("ngram.py already patched.")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("Patched src/models/ngram.py")
else:
    raise SystemExit(
        "Expected evaluate_model reporting block was not found. No changes made."
    )

test_path = Path("tests/test_ngram_baseline.py")
if not test_path.is_file():
    raise SystemExit(f"Not found: {test_path}")

test_text = test_path.read_text(encoding="utf-8")
test_name = "test_evaluation_supports_single_nonunigram_order"

if test_name not in test_text:
    addition = '''

def test_evaluation_supports_single_nonunigram_order():
    from src.models.ngram import BaselineSequence

    model = WittenBellNGram(max_order=3).fit(
        [
            ("A", "B", "C"),
            ("A", "B", "D"),
            ("A", "B", "C"),
        ]
    )

    examples = [
        BaselineSequence(
            leaf_group="f1",
            folio="f1r",
            locus="f1r.1,+P0",
            symbols=("A", "B", "C"),
            is_boundary=(False, False, False),
        )
    ]

    aggregate, leaves, loci = evaluate_model(
        model,
        examples,
        orders=(3,),
        view="single_order_regression",
    )

    assert len(aggregate) == 1
    assert aggregate[0]["order"] == 3
    assert aggregate[0]["delta_bits_per_event_vs_unigram"] is None
    assert aggregate[0]["relative_nll_reduction_vs_unigram"] is None
    assert len(leaves) == 1
    assert len(loci) == 1
'''
    test_path.write_text(test_text.rstrip() + addition + "\n", encoding="utf-8")
    print("Added regression test to tests/test_ngram_baseline.py")
else:
    print("Regression test already present.")

print("Done.")
