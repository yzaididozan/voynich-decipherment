from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_leaf_split.py"
    name = "generate_leaf_split_testmodule"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic(module):
    sections = ["herbal", "biological", "cosmological", "text_only"]
    scribes = ["1", "2", "3", "4", "5"]
    curriers = ["A", "B", "UNSET"]
    types = ["P", "L", "C", "R"]

    profiles = {}
    units = []

    for leaf in range(1, 103):
        sec = sections[(leaf - 1) % len(sections)]
        scr = scribes[(leaf - 1) % len(scribes)]
        cur = curriers[(leaf - 1) % len(curriers)]
        typ = types[(leaf - 1) % len(types)]
        loci = 6 + leaf % 7

        p = module.LeafProfile(
            leaf=leaf,
            locus_count=loci,
            currier_presence=module.Counter({cur: 1}),
            scribe_presence=module.Counter({scr: 1}),
            section_presence=module.Counter({sec: 1}),
            locus_type_counts=module.Counter({typ: loci}),
            quire_presence=module.Counter({str((leaf - 1) // 8 + 1): 1}),
            folios={f"f{leaf}r", f"f{leaf}v"},
        )
        profiles[leaf] = p

        if leaf not in (85, 86):
            units.append(
                module.AtomicUnit(
                    unit_id=f"f{leaf}",
                    leaves=(leaf,),
                    leaf_count=1,
                    locus_count=loci,
                    currier_presence=p.currier_presence.copy(),
                    scribe_presence=p.scribe_presence.copy(),
                    section_presence=p.section_presence.copy(),
                    locus_type_counts=p.locus_type_counts.copy(),
                    quire_presence=p.quire_presence.copy(),
                )
            )

    a, b = profiles[85], profiles[86]
    units.append(
        module.AtomicUnit(
            unit_id="f85+f86",
            leaves=(85, 86),
            leaf_count=2,
            locus_count=a.locus_count + b.locus_count + 10,
            currier_presence=a.currier_presence + b.currier_presence,
            scribe_presence=a.scribe_presence + b.scribe_presence,
            section_presence=a.section_presence + b.section_presence,
            locus_type_counts=a.locus_type_counts + b.locus_type_counts,
            quire_presence=a.quire_presence + b.quire_presence,
            special_pages=("fRos",),
        )
    )
    return units, profiles


def test_rounding_102_leaves():
    m = load_module()
    assert m.largest_remainder_counts(102, m.SPLIT_RATIOS) == {
        "train": 72,
        "validation": 15,
        "test": 15,
    }


def test_deterministic_exact_atomic_split():
    m = load_module()
    units, profiles = synthetic(m)
    targets = {"train": 72, "validation": 15, "test": 15}

    a1, s1 = m.optimize_assignment(
        units,
        profiles=profiles,
        target_leaf_counts=targets,
        seed=m.SEED,
        restarts=3,
        swap_proposals=80,
    )
    a2, s2 = m.optimize_assignment(
        units,
        profiles=profiles,
        target_leaf_counts=targets,
        seed=m.SEED,
        restarts=3,
        swap_proposals=80,
    )

    assert m.split_leaves(a1) == m.split_leaves(a2)
    assert s1 == pytest.approx(s2)

    m.validate_assignment(
        a1,
        observed_leaves=range(1, 103),
        target_leaf_counts=targets,
    )


def test_f85_f86_linked():
    m = load_module()
    units, profiles = synthetic(m)
    targets = {"train": 72, "validation": 15, "test": 15}

    assignment, _ = m.optimize_assignment(
        units,
        profiles=profiles,
        target_leaf_counts=targets,
        seed=m.SEED,
        restarts=2,
        swap_proposals=50,
    )
    leaves = m.split_leaves(assignment)

    owners85 = [s for s in m.SPLIT_ORDER if 85 in leaves[s]]
    owners86 = [s for s in m.SPLIT_ORDER if 86 in leaves[s]]
    assert owners85 == owners86
    assert len(owners85) == 1


def test_outputs_have_expected_leaf_counts(tmp_path: Path):
    m = load_module()
    units, profiles = synthetic(m)
    targets = {"train": 72, "validation": 15, "test": 15}

    assignment, score = m.optimize_assignment(
        units,
        profiles=profiles,
        target_leaf_counts=targets,
        seed=m.SEED,
        restarts=2,
        swap_proposals=50,
    )

    m.write_outputs(
        output_dir=tmp_path,
        source=Path("data/raw/zl/ZL3b-n.txt"),
        source_hash=m.EXPECTED_ZL3B_SHA256,
        assignment=assignment,
        profiles=profiles,
        targets=targets,
        score=score,
        seed=m.SEED,
        restarts=2,
        swap_proposals=50,
        freeze_gate={"status": "PASS"},
    )

    assert len((tmp_path / "train.txt").read_text().splitlines()) == 72
    assert len((tmp_path / "validation.txt").read_text().splitlines()) == 15
    assert len((tmp_path / "test_LOCKED.txt").read_text().splitlines()) == 15
