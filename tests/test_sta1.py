from __future__ import annotations

from pathlib import Path

import pytest

from src.data.ivtff import parse_ivtff_lines

from src.data.sta1 import (
    STA1ConversionError,
    STA1CountError,
    STA1RuleError,
    convert_native_text_to_sta1,
    count_native_text_as_sta1,
    count_native_text_as_sta1_ivtff_semantic,
    count_records_as_sta1,
    count_sta1_characters,
    load_bitrans_rules,
    resolve_alternatives_first,
    rule_file_for_alphabet,
)


RULES = """##BIT  STA1 Eva-
#=~
<(comment)>
#(comment)
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


def test_rule_file_for_supported_alphabets():
    root = Path("rules")
    assert rule_file_for_alphabet("Eva-", root) == root / "STA-Eva_def.bit"
    assert rule_file_for_alphabet("EvaT", root) == root / "STA-EvaT_def.bit"
    assert rule_file_for_alphabet("v101", root) == root / "STA-v101_def.bit"


def test_rule_file_for_unknown_alphabet_fails():
    with pytest.raises(STA1RuleError):
        rule_file_for_alphabet("Unknown", Path("rules"))


def test_load_rules_reverses_native_to_sta1(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert rules.sta_alphabet == "STA1"
    assert rules.native_alphabet == "Eva-"

    mapping = {rule.source: rule.target for rule in rules.rules}
    assert mapping["o"] == "A1"
    assert mapping["ckh"] == "U1"
    assert mapping["@221;"] == "Aa"


def test_greedy_longest_match(tmp_path: Path):
    rules = make_rules(tmp_path)

    # If conversion were not greedy this could be segmented as i+i+n.
    assert convert_native_text_to_sta1("iin", rules) == "G1"
    assert convert_native_text_to_sta1("in", rules) == "F1"
    assert convert_native_text_to_sta1("i", rules) == "E1"


def test_greedy_conversion_across_words(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert (
        convert_native_text_to_sta1("iin.in.i", rules)
        == "G1.F1.E1"
    )


def test_ligature_rule_is_a_single_sta1_symbol(tmp_path: Path):
    rules = make_rules(tmp_path)
    converted = convert_native_text_to_sta1("{cto}", rules)
    assert converted == "U5"
    assert count_sta1_characters(converted) == 1


def test_high_ascii_rule_is_a_single_sta1_symbol(tmp_path: Path):
    rules = make_rules(tmp_path)
    converted = convert_native_text_to_sta1("@221;", rules)
    assert converted == "Aa"
    assert count_sta1_characters(converted) == 1


def test_ivtff_angle_markers_are_preserved(tmp_path: Path):
    rules = make_rules(tmp_path)
    source = "<%>dai<->o<$>"
    converted = convert_native_text_to_sta1(source, rules)
    assert converted == "<%>B1A3E1<->A1<$>"
    assert count_sta1_characters(converted) == 4


def test_alternatives_are_converted_but_preserved(tmp_path: Path):
    rules = make_rules(tmp_path)
    converted = convert_native_text_to_sta1("[ckh:ch]", rules)
    assert converted == "[U1:K1]"


def test_character_count_uses_first_alternative():
    assert resolve_alternatives_first("[U1:K1A1]") == "U1"
    assert count_sta1_characters("[U1:K1A1]") == 1


def test_unknown_single_counts_as_one():
    assert count_sta1_characters("Z1") == 1


def test_unknown_run_counts_as_one():
    assert count_sta1_characters("Z1Z1Z1") == 1


def test_separated_unknowns_count_independently():
    assert count_sta1_characters("Z1.Z1") == 2
    assert count_sta1_characters("Z1,Z1") == 2


def test_native_unknown_run_converts_then_collapses(tmp_path: Path):
    rules = make_rules(tmp_path)
    converted = convert_native_text_to_sta1("???", rules)
    assert converted == "Z1Z1Z1"
    assert count_native_text_as_sta1("???", rules) == 1


def test_structural_separators_do_not_count():
    assert count_sta1_characters("A1.A3,G1") == 3


def test_angle_metadata_does_not_count():
    assert count_sta1_characters("<%>A1<->A3<$>") == 2


def test_strict_conversion_rejects_unmapped_native_text(tmp_path: Path):
    rules = make_rules(tmp_path)
    with pytest.raises(STA1ConversionError):
        convert_native_text_to_sta1("w", rules, strict=True)


def test_non_strict_conversion_leaves_unmapped_text(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert convert_native_text_to_sta1("w", rules, strict=False) == "w"


def test_malformed_sta1_text_fails():
    with pytest.raises(STA1CountError):
        count_sta1_characters("A1x")


def test_wrong_native_alphabet_header_fails(tmp_path: Path):
    path = tmp_path / "bad.bit"
    path.write_text(RULES, encoding="ascii")
    with pytest.raises(STA1RuleError):
        load_bitrans_rules(
            path,
            expected_native_alphabet="v101",
        )


def test_rule_file_hash_is_recorded(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert len(rules.sha256) == 64
    int(rules.sha256, 16)


def test_bitrans_rule_priority_handles_real_eva_apostrophe_case(tmp_path: Path):
    rules = make_rules(tmp_path)

    # Real ZL3b exposed this distinction.  A left-to-right tokenizer would
    # consume "ee" first and strand the apostrophe.  Bitrans instead applies
    # equal-length rules in file order, so e' is handled before ee.
    converted = convert_native_text_to_sta1("shee'", rules)

    assert converted == "L1J1Ch"
    assert count_sta1_characters(converted) == 3


def test_generated_sta1_output_is_not_reprocessed(tmp_path: Path):
    path = tmp_path / "STA-Eva_def.bit"
    path.write_text(
        """##BIT  STA1 Eva-
#=~
<(comment)>
#(comment)
A1  o
B1  A
""",
        encoding="ascii",
    )
    rules = load_bitrans_rules(path, expected_native_alphabet="Eva-")

    # o -> A1.  The generated 'A' inside A1 must never be treated as native
    # input for the later A -> B1 rule.
    assert convert_native_text_to_sta1("o", rules) == "A1"


def test_table_reproduction_and_documented_unknown_counts_are_distinct():
    text = "A1Z1Z1Z1A3"

    table_count = count_sta1_characters(
        text,
        collapse_unknown_runs=False,
    )
    documented_count = count_sta1_characters(
        text,
        collapse_unknown_runs=True,
    )

    assert table_count == 5
    assert documented_count == 3
    assert table_count - documented_count == 2


def test_default_direct_count_keeps_documented_unknown_collapse_behavior():
    # Direct counting remains conservative/documentation-oriented. Corpus QC
    # explicitly requests collapse_unknown_runs=False for table reproduction.
    assert count_sta1_characters("Z1Z1Z1") == 1


def test_ivtff_semantic_single_unknown_is_one(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert count_native_text_as_sta1_ivtff_semantic("?", rules) == 1


def test_ivtff_semantic_double_unknown_is_two_single_unknowns(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert count_native_text_as_sta1_ivtff_semantic("??", rules) == 2


def test_ivtff_semantic_triple_unknown_is_one_unknown_sequence(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert count_native_text_as_sta1_ivtff_semantic("???", rules) == 1


def test_ivtff_semantic_unknown_sequence_inside_text(tmp_path: Path):
    rules = make_rules(tmp_path)
    assert count_native_text_as_sta1_ivtff_semantic("o.???.a", rules) == 3


def test_ivtff_semantic_rejects_ambiguous_long_unknown_run(tmp_path: Path):
    rules = make_rules(tmp_path)
    with pytest.raises(STA1CountError):
        count_native_text_as_sta1_ivtff_semantic("????", rules)


def test_corpus_count_marks_semantic_measure_unavailable_for_four_questions(
    tmp_path: Path,
):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "STA-Eva_def.bit").write_text(RULES, encoding="ascii")

    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=T $L=A $H=1>
<f1r.1,@P0> <%>o.????.a<$>
"""
    records = list(
        parse_ivtff_lines(
            sample.splitlines(True),
            transcription_id="synthetic",
        )
    )

    result = count_records_as_sta1(
        records,
        rules_dir=rules_dir,
    )

    # Codewise conversion remains valid: o + ???? + a = 6 STA1 codes.
    assert result.characters == 6
    assert result.characters_ivtff_semantic is None
    assert result.ivtff_semantic_valid is False
    assert result.ivtff_semantic_error is not None
    assert "????" in result.ivtff_semantic_error


def test_corpus_count_keeps_semantic_measure_when_ivtff_unknowns_are_valid(
    tmp_path: Path,
):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "STA-Eva_def.bit").write_text(RULES, encoding="ascii")

    sample = """#=IVTFF Eva- 2.0 M 5
<f1r> <! $Q=A $I=T $L=A $H=1>
<f1r.1,@P0> <%>o.???.a<$>
"""
    records = list(
        parse_ivtff_lines(
            sample.splitlines(True),
            transcription_id="synthetic",
        )
    )

    result = count_records_as_sta1(
        records,
        rules_dir=rules_dir,
    )

    assert result.characters == 5
    assert result.characters_ivtff_semantic == 3
    assert result.ivtff_semantic_valid is True
    assert result.ivtff_semantic_error is None
