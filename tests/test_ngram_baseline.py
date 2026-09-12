from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.tier0 import make_analytical_loci
from src.data.ivtff import parse_ivtff_lines
from src.data.sta1 import load_bitrans_rules
from src.models.ngram import (
    BOS,
    UNK,
    WB,
    WittenBellNGram,
    build_baseline_sequences,
    evaluate_model,
    parse_leaf_ids,
    select_records_by_leaves,
)


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
    return load_bitrans_rules(
        path,
        expected_native_alphabet="Eva-",
    )


def test_leaf_parser_and_split_selection():
    assert parse_leaf_ids("f1\nf2\n", label="train") == (1, 2)
    with pytest.raises(ValueError):
        parse_leaf_ids("f1r\n", label="train")
    with pytest.raises(ValueError):
        parse_leaf_ids("f1\nf1\n", label="train")


def test_select_records_never_needs_test_membership():
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
    assert {
        r.folio
        for r in select_records_by_leaves(
            records,
            (1,),
            split_name="train",
        )
    } == {"f1r"}

    assert "fRos" in {
        r.folio
        for r in select_records_by_leaves(
            records,
            (85, 86),
            split_name="validation",
        )
    }

    with pytest.raises(ValueError):
        select_records_by_leaves(
            records,
            (85,),
            split_name="validation",
        )


def test_sequence_views_and_unknown_reset(tmp_path: Path):
    rules = make_rules(tmp_path)
    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=H $L=A $H=1>
<f1r.1,+P0> <%>o.a?.sh<$>
"""
    records = list(
        parse_ivtff_lines(
            sample.splitlines(True),
            transcription_id="ZL3b",
        )
    )
    loci = make_analytical_loci(records, rules)

    space_free = build_baseline_sequences(
        loci,
        view="space_free",
    )
    token_aware = build_baseline_sequences(
        loci,
        view="token_aware",
    )

    # Unknown is removed and splits context.
    assert all("Z1" not in seq.symbols for seq in space_free)
    assert len(space_free) == 2

    # Boundary events are explicit only in token-aware view.
    assert any(WB in seq.symbols for seq in token_aware)
    assert all(
        len(seq.symbols) == len(seq.is_boundary)
        for seq in token_aware
    )


def test_witten_bell_probabilities_normalize_and_are_nonzero_for_unk():
    model = WittenBellNGram(max_order=3).fit(
        [
            ("A", "B", "A"),
            ("A", "C"),
        ]
    )

    support = list(model.seen_vocabulary) + ["never-seen"]
    for order, history in (
        (1, ()),
        (2, ("A",)),
        (3, (BOS, "A")),
        (3, ("never", "seen")),
    ):
        total = sum(
            model.probability(symbol, history, order=order)
            for symbol in support
        )
        assert total == pytest.approx(1.0, abs=1e-12)

    assert model.probability(
        "never-seen",
        ("A",),
        order=2,
    ) > 0.0


def test_higher_order_improves_deterministic_pattern():
    train = [("A", "B") for _ in range(50)]
    validation = [("A", "B") for _ in range(10)]

    model = WittenBellNGram(max_order=2).fit(train)

    from src.models.ngram import BaselineSequence

    examples = [
        BaselineSequence(
            leaf_group="f1",
            folio="f1r",
            locus=f"f1r.{i},+P0",
            symbols=seq,
            is_boundary=(False, False),
        )
        for i, seq in enumerate(validation)
    ]

    aggregate, _, _ = evaluate_model(
        model,
        examples,
        orders=(1, 2),
        view="space_free",
    )

    by_order = {row["order"]: row for row in aggregate}
    assert by_order[2]["bits_per_event"] < by_order[1]["bits_per_event"]


def test_evaluation_preserves_per_leaf_and_per_locus_rows():
    model = WittenBellNGram(max_order=2).fit(
        [("A", "B"), ("A", "B")]
    )

    from src.models.ngram import BaselineSequence

    examples = [
        BaselineSequence(
            leaf_group="f10",
            folio="f10r",
            locus="f10r.1,+P0",
            symbols=("A", "B"),
            is_boundary=(False, False),
        ),
        BaselineSequence(
            leaf_group="f11",
            folio="f11r",
            locus="f11r.1,+P0",
            symbols=("A", "B"),
            is_boundary=(False, False),
        ),
    ]

    aggregate, leaves, loci = evaluate_model(
        model,
        examples,
        orders=(1, 2),
        view="space_free",
    )

    assert len(aggregate) == 2
    assert len(leaves) == 4
    assert len(loci) == 4
    assert {row["leaf_group"] for row in leaves} == {"f10", "f11"}
