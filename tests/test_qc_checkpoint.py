from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_report(
    corpus: str,
    *,
    loci: int,
    paragraphs: int,
    words: int,
    sta_codewise,
    sta_semantic,
    sta_expected,
):
    sta_status = (
        "not_applicable_lossy_representation"
        if sta_expected is None
        else "PASS"
    )

    return {
        "transcription_id": corpus,
        "source_path": f"data/raw/{corpus}.txt",
        "loci": loci,
        "paragraphs": paragraphs,
        "long_words": words,
        "ivtff_alphabet": "Eva-",
        "ivtff_version": "2.0",
        "sta1_characters": sta_codewise,
        "sta1_characters_ivtff_semantic": sta_semantic,
        "sta1_rules_file": "rules.bit",
        "sta1_rules_sha256": "a" * 64,
        "comparison": {
            "loci": {
                "observed": loci,
                "expected": loci,
                "status": "PASS",
            },
            "paragraphs": {
                "observed": paragraphs,
                "expected": None if corpus.startswith("RF1b") else paragraphs,
                "status": "not_published" if corpus.startswith("RF1b") else "PASS",
            },
            "long_words": {
                "observed": words,
                "expected": words,
                "status": "PASS",
            },
            "sta1_characters": {
                "observed": sta_codewise,
                "observed_ivtff_semantic": sta_semantic,
                "expected": sta_expected,
                "status": sta_status,
            },
        },
    }


def test_summary_classifies_codewise_semantic_and_lossy(tmp_path: Path):
    module = load_script(
        Path(__file__).resolve().parents[1] / "scripts" / "summarize_corpus_qc.py",
        "summarize_corpus_qc",
    )

    values = {
        "ZL3b": (5385, 740, 36278, 157304, 157247, 157304),
        "GC2a": (5367, 775, 38242, 157096, 157083, 157096),
        "IT2a": (5215, 772, 37919, 155352, 155292, 155292),
        "RF1b-full": (5385, 0, 37848, 157254, 157202, 157254),
        "RF1b-basic": (5385, 0, 37848, 157580, None, None),
    }

    paths = {}

    for corpus, args in values.items():
        path = tmp_path / f"{corpus}_corpus_report.json"
        path.write_text(
            json.dumps(make_report(corpus, loci=args[0], paragraphs=args[1],
                                   words=args[2], sta_codewise=args[3],
                                   sta_semantic=args[4], sta_expected=args[5])),
            encoding="utf-8",
        )
        paths[corpus] = path

    summary = module.build_summary(paths)
    rows = {row["corpus"]: row for row in summary["rows"]}

    assert summary["overall_qc"] == "PASS"
    assert rows["ZL3b"]["sta1_validation"] == "exact codewise"
    assert rows["IT2a"]["sta1_validation"] == "exact IVTFF-semantic"
    assert rows["RF1b-basic"]["qc"] == "PASS WITH DOCUMENTED LIMITATION"
    assert summary["freeze_readiness"]["ready_to_generate_locked_split"] is False


def test_summary_rejects_true_mismatch(tmp_path: Path):
    module = load_script(
        Path(__file__).resolve().parents[1] / "scripts" / "summarize_corpus_qc.py",
        "summarize_corpus_qc_2",
    )

    report = make_report(
        "ZL3b",
        loci=5385,
        paragraphs=740,
        words=36278,
        sta_codewise=157000,
        sta_semantic=156999,
        sta_expected=157304,
    )

    try:
        module.summarize_report("ZL3b", report)
    except ValueError as exc:
        assert "STA1 count mismatch" in str(exc)
    else:
        raise AssertionError("Expected a real STA1 mismatch to fail QC")


def test_diverse_selection_is_deterministic():
    module = load_script(
        Path(__file__).resolve().parents[1] / "scripts" / "spotcheck_transcriptions.py",
        "spotcheck_transcriptions",
    )

    class Record:
        def __init__(self, section, scribe, currier, locus_type):
            self.section = section
            self.scribe = scribe
            self.currier = currier
            self.locus_type = locus_type

    index = {
        "a": Record("herbal", "1", "A", "P"),
        "b": Record("biological", "2", "B", "P"),
        "c": Record("zodiac", "3", None, "L"),
        "d": Record("herbal", "4", "A", "C"),
        "e": Record("text_only", "5", "B", "R"),
    }

    first = module.select_diverse_loci(
        index,
        list(index),
        sample_size=4,
        seed=40814041438,
    )
    second = module.select_diverse_loci(
        index,
        list(index),
        sample_size=4,
        seed=40814041438,
    )

    assert first == second
    assert len(first) == 4
    assert len(set(first)) == 4
