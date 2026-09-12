from __future__ import annotations

from collections import Counter
from pathlib import Path
import importlib.util
import sys

import pytest

from src.analysis.tier0 import (
    Tier0Config,
    _levenshtein_bounded,
    build_tier0,
    empirical_conditional_entropy,
    entropy_from_counter,
    jensen_shannon_divergence,
    make_analytical_loci,
    mutual_information_by_distance,
    parse_train_leaf_ids,
    select_train_records,
    tokenize_s0_sta1,
    write_tier0_outputs,
)
from src.data.ivtff import parse_ivtff_lines
from src.data.sta1 import load_bitrans_rules


RULES = """##BIT STA1 Eva-
A1  o
A3  a
B1  d
C2  s
Ch  e'
E1  i
F1  in
G1  iin
J1  e
Ja  c
Jb  h
K1  ch
K2  ee
L1  sh
U1  ckh
U5  {cto}
Aa  @221;
Z1  ?
"""


def make_rules(tmp_path: Path):
    path = tmp_path / "STA-Eva_def.bit"
    path.write_text(RULES, encoding="ascii")
    return load_bitrans_rules(path, expected_native_alphabet="Eva-")


def test_parse_train_leaf_ids_is_strict():
    assert parse_train_leaf_ids("f1\nf2\nf85\n") == (1, 2, 85)
    with pytest.raises(ValueError):
        parse_train_leaf_ids("f1r\n")
    with pytest.raises(ValueError):
        parse_train_leaf_ids("f1\nf1\n")


def test_s0_tokenization_boundaries_comments_alternatives_and_drawing(tmp_path: Path):
    rules = make_rules(tmp_path)

    tokens = tokenize_s0_sta1(
        "o,a.[o:a].sh<!comment>ee<->in",
        rules,
    )

    assert tokens == (
        ("A1",),
        ("A3",),
        ("A1",),
        ("L1", "K2"),
        ("F1",),
    )


def test_entropy_and_conditional_entropy_simple_case():
    assert entropy_from_counter(Counter({"A": 2, "B": 2})) == pytest.approx(1.0)

    h, transitions, contexts = empirical_conditional_entropy(
        [("A", "B", "A", "B")],
        1,
        exclude_symbol=None,
    )
    assert h == pytest.approx(0.0)
    assert transitions == 3
    assert contexts == 2


def test_mutual_information_detects_perfect_alternation():
    rows = mutual_information_by_distance(
        [
            ("A", "B", "A", "B"),
            ("B", "A", "B", "A"),
        ],
        1,
        exclude_symbol=None,
    )
    assert rows[0]["mutual_information_bits"] == pytest.approx(1.0)


def test_js_identical_is_zero_and_disjoint_is_one():
    assert jensen_shannon_divergence(
        {"A": 10},
        {"A": 7},
    ) == pytest.approx(0.0)

    assert jensen_shannon_divergence(
        {"A": 10},
        {"B": 10},
    ) == pytest.approx(1.0)


def test_bounded_levenshtein_uses_symbol_units():
    assert _levenshtein_bounded(
        ("A1", "B1"),
        ("A1", "C2"),
        2,
    ) == 1
    assert _levenshtein_bounded(
        ("A1",),
        ("A1", "B1", "C2", "A3"),
        2,
    ) == 3


def test_train_selection_excludes_nontrain_and_handles_fros():
    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=H $L=A $H=1>
<f1r.1,+P0> <%>o.a<$>
<f2r> <! $Q=A $I=H $L=A $H=1>
<f2r.1,+P0> <%>a.o<$>
<f85r1> <! $Q=S $I=C $L=B $H=2>
<f85r1.1,+P0> <%>o<$>
<f86r1> <! $Q=S $I=C $L=B $H=2>
<f86r1.1,+P0> <%>a<$>
<fRos> <! $Q=S $I=C $L=B $H=2>
<fRos.1,+P0> <%>o.a<$>
"""
    records = list(
        parse_ivtff_lines(
            sample.splitlines(True),
            transcription_id="ZL3b",
        )
    )

    selected = select_train_records(records, (1,))
    assert {r.folio for r in selected} == {"f1r"}

    selected_ros = select_train_records(records, (1, 85, 86))
    assert "fRos" in {r.folio for r in selected_ros}
    assert "f2r" not in {r.folio for r in selected_ros}

    with pytest.raises(ValueError):
        select_train_records(records, (1, 85))


def test_build_and_write_tier0_integration(tmp_path: Path):
    rules = make_rules(tmp_path)

    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=H $L=A $H=1>
<f1r.1,+P0> <%>o.a.sh.ee<$>
<f1r.2,+P0> in.iin.ch.ee
<f1v> <! $Q=A $I=H $L=A $H=1>
<f1v.1,+P0> <%>a,o.?.sh<$>
"""
    records = list(
        parse_ivtff_lines(
            sample.splitlines(True),
            transcription_id="ZL3b",
        )
    )
    selected = select_train_records(records, (1,))
    analytical = make_analytical_loci(selected, rules)

    summary, tables = build_tier0(
        selected,
        analytical,
        config=Tier0Config(
            entropy_max_order=2,
            glyph_mi_max_distance=2,
            token_mi_max_distance=1,
            edit_vocab_limit=5,
            edit_max_distance=1,
            edit_example_limit=3,
            top_items_in_summary=5,
        ),
        train_leaves=(1,),
    )

    assert summary["scope"] == "TRAIN ONLY"
    assert summary["counts"]["physical_leaves"] == 1
    assert summary["counts"]["tokens"] > 0
    assert "glyph_frequency" in tables
    assert "group_glyph_frequency" in tables
    assert "token_internal_glyph_frequency" in tables

    output_dir = tmp_path / "out"
    write_tier0_outputs(
        output_dir,
        summary=summary,
        tables=tables,
        provenance={
            "validation_accessed": False,
            "locked_test_accessed": False,
        },
    )

    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "run_manifest.json").is_file()
    assert (output_dir / "SHA256SUMS").is_file()
    assert (output_dir / "glyph_frequency.csv").is_file()
    assert (output_dir / "section_glyph_js.csv").is_file()

    written = __import__("json").loads(
        (output_dir / "summary.json").read_text()
    )
    assert written["provenance"]["locked_test_accessed"] is False
