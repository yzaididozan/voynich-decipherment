from __future__ import annotations

import json

from src.analysis.tier0 import Tier0Config
from src.evaluation.control_benchmark import (
    benchmark_control,
    read_control_jsonl,
    scalar_metrics,
)


def test_read_control_jsonl_and_benchmark(tmp_path):
    path = tmp_path / "corpus.jsonl"
    rows = [
        {
            "line_index": 1,
            "tokens": [
                ["G0001", "G0002"],
                ["G0001", "G0003"],
                ["G0001", "G0002"],
            ],
        },
        {
            "line_index": 2,
            "tokens": [
                ["G0002"],
                ["G0001", "G0002"],
            ],
        },
    ]
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    lines = read_control_jsonl(path)

    config = Tier0Config(
        entropy_max_order=4,
        glyph_mi_max_distance=8,
        token_mi_max_distance=5,
        edit_vocab_limit=300,
        edit_max_distance=2,
        edit_example_limit=10,
        top_items_in_summary=50,
    )

    summary, tables = benchmark_control(
        lines,
        config=config,
    )

    assert summary["counts"]["tokens"] == 5
    assert summary["counts"]["glyphs"] == 9
    assert summary["counts"]["unique_glyphs"] == 3
    assert (
        summary["entropy"][
            "conditional_entropy_by_order"
        ]["1"]["transitions"]
        > 0
    )

    metrics = scalar_metrics(
        summary,
        tables,
    )

    assert "unigram_entropy_bits" in metrics
    assert "conditional_entropy_order_4" in metrics
    assert "glyph_mi_distance_8" in metrics
    assert "edit_mean_neighbors_within_2" in metrics


def test_benchmark_has_no_fake_voynich_metadata():
    lines = (
        (
            ("G0001",),
            ("G0002",),
        ),
    )
    config = Tier0Config()

    summary, _ = benchmark_control(
        lines,
        config=config,
    )

    # These concepts may be named in the limitation note, but the control
    # must not fabricate corresponding data/metric fields.
    forbidden_keys = {
        "currier",
        "scribe",
        "section",
        "quire",
        "metadata_locus_counts",
        "currier_glyph_js",
        "scribe_glyph_js",
        "section_glyph_js",
    }

    def all_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from all_keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from all_keys(item)

    assert forbidden_keys.isdisjoint(set(all_keys(summary)))
