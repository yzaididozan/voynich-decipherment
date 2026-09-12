from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from src.data.ivtff import parse_ivtff_lines


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_script_module(
    Path(__file__).resolve().parents[1]
    / "scripts/generate_image_validation_sample.py",
    "generate_image_validation_sample",
)


def synthetic_records():
    lines = ["#=IVTFF Eva- 2.0 M 5\n"]
    # 18 train-like pages across 18 leaves with varying metadata/markup.
    for leaf in range(1, 19):
        rv = "r" if leaf % 2 else "v"
        folio = f"f{leaf}{rv}"
        currier = "A" if leaf % 2 else "B"
        hand = str((leaf % 5) + 1)
        section = "H" if leaf % 3 == 0 else ("B" if leaf % 3 == 1 else "A")
        lines.append(
            f"<{folio}> <! $Q=A $I={section} $L={currier} $H={hand}>\n"
        )
        text = "o.a"
        if leaf == 3:
            text = "o,a"
        if leaf == 4:
            text = "[o:a].a"
        if leaf == 5:
            text = "o.?"
        if leaf == 6:
            text = "o<->a"
        lines.append(f"<{folio}.1,+P0> <%>{text}<$>\n")
        lines.append(f"<{folio}.2,+P0> o.a\n")
        lines.append(f"<{folio}.3,+P0> a.o\n")

    return list(
        parse_ivtff_lines(
            lines,
            transcription_id="ZL3b",
        )
    )


def test_build_candidates_and_select_deterministically():
    records = synthetic_records()
    candidates, by_page = GENERATOR.build_page_candidates(records)

    assert len(candidates) == 18
    assert len(by_page) == 18

    selected_a = GENERATOR.select_representative_pages(
        candidates,
        sample_size=15,
        seed=40814041438,
        max_pages_per_leaf_group=1,
    )
    selected_b = GENERATOR.select_representative_pages(
        candidates,
        sample_size=15,
        seed=40814041438,
        max_pages_per_leaf_group=1,
    )

    assert [x.folio for x in selected_a] == [x.folio for x in selected_b]
    assert len({x.leaf_group for x in selected_a}) == 15


def test_sample_covers_markup_and_ordinary_pages():
    records = synthetic_records()
    candidates, _ = GENERATOR.build_page_candidates(records)

    selected = GENERATOR.select_representative_pages(
        candidates,
        sample_size=15,
        seed=40814041438,
        max_pages_per_leaf_group=1,
    )

    features = set().union(*(x.feature_set() for x in selected))
    assert "profile:ordinary_no_markup" in features
    assert "markup:uncertain_space" in features
    assert "markup:alternative_reading" in features
    assert "markup:unknown_glyph" in features
    assert "markup:drawing_interruption" in features


def test_representative_loci_include_richest():
    records = synthetic_records()
    _, by_page = GENERATOR.build_page_candidates(records)
    page = by_page["f3r"]

    chosen = GENERATOR.representative_loci(page, limit=2)
    assert len(chosen) == 2
    assert any("," in row.text_normalized for row in chosen)


def test_summary_script_detects_complete_pass(tmp_path: Path):
    input_dir = tmp_path / "review"
    input_dir.mkdir()

    with (input_dir / "pages.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["folio", "review_status"],
        )
        writer.writeheader()
        writer.writerow({"folio": "f1r", "review_status": "PASS"})

    with (input_dir / "loci.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["locus", "check_status"],
        )
        writer.writeheader()
        writer.writerow({"locus": "f1r.1,+P0", "check_status": "PASS"})

    output = input_dir / "review_summary.json"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/summarize_image_validation.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-dir",
            str(input_dir),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "PASS_WITH_DOCUMENTED_TRANSCRIPTION_VARIATION" in result.stdout
    assert output.is_file()
