#!/usr/bin/env python3
"""Generate the preregistered ZL3b physical-leaf 70/15/15 split.

Design frozen by the project charter:
- atomic split unit: physical leaf (recto + verso together);
- 70% train, 15% validation, 15% locked test;
- group-stratify as feasible by Currier, scribe, section, locus composition;
- frozen seed: 40814041438.

For 102 observed leaves, largest-remainder rounding is 72 / 15 / 15.

The IVTFF pseudo-page fRos belongs to the f85/f86 Rosettes composite.  Leaves
85 and 86 are therefore linked into one indivisible assignment unit, and fRos
is assigned to that same split to prevent leakage.

Outputs:
    data/splits/v1/train.txt
    data/splits/v1/validation.txt
    data/splits/v1/test_LOCKED.txt
    data/splits/v1/split_manifest.json
    data/splits/v1/split_qc.json

Only manuscript metadata and locus-type counts are used for optimization.
Voynich glyph/token content is not used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import stat
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ivtff import LocusRecord, load_ivtff


SEED = 40814041438
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
EXPECTED_ZL3B_SHA256 = (
    "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"
)
DEFAULT_SOURCE = Path("data/raw/zl/ZL3b-n.txt")
DEFAULT_OUTPUT_DIR = Path("data/splits/v1")
DEFAULT_QC_SUMMARY = Path("results/qc/transcription_summary.json")
ALGORITHM_VERSION = "physical-leaf-stratified-v1"

LINKED_LEAF_GROUPS = ((85, 86),)
SPECIAL_PAGE_TO_LINKED_GROUP = {"fRos": (85, 86)}

FAMILY_WEIGHTS = {
    "currier_presence": 1.0,
    "scribe_presence": 1.0,
    "section_presence": 1.0,
    "locus_type_counts": 1.0,
    "locus_total": 0.75,
}
COVERAGE_GAP_PENALTY = 100.0
DEFAULT_RESTARTS = 48
DEFAULT_SWAP_PROPOSALS = 500


@dataclass
class LeafProfile:
    leaf: int
    locus_count: int = 0
    currier_presence: Counter = field(default_factory=Counter)
    scribe_presence: Counter = field(default_factory=Counter)
    section_presence: Counter = field(default_factory=Counter)
    locus_type_counts: Counter = field(default_factory=Counter)
    quire_presence: Counter = field(default_factory=Counter)
    folios: set = field(default_factory=set)


@dataclass
class AtomicUnit:
    unit_id: str
    leaves: Tuple[int, ...]
    leaf_count: int
    locus_count: int
    currier_presence: Counter
    scribe_presence: Counter
    section_presence: Counter
    locus_type_counts: Counter
    quire_presence: Counter
    special_pages: Tuple[str, ...] = ()


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def largest_remainder_counts(total: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to 1.0")

    names = list(ratios)
    exact = {name: total * ratios[name] for name in names}
    result = {name: math.floor(exact[name]) for name in names}
    remainder = total - sum(result.values())

    ranking = sorted(
        names,
        key=lambda name: (
            -(exact[name] - result[name]),
            names.index(name),
        ),
    )
    for name in ranking[:remainder]:
        result[name] += 1
    return result


def _cat(value: Optional[object]) -> str:
    return "UNSET" if value is None else str(value)


def build_leaf_profiles(records: Sequence[LocusRecord]) -> Dict[int, LeafProfile]:
    grouped: Dict[int, List[LocusRecord]] = defaultdict(list)
    for record in records:
        if record.physical_leaf is not None:
            grouped[record.physical_leaf].append(record)

    profiles = {}
    for leaf, group in grouped.items():
        profile = LeafProfile(leaf=leaf, locus_count=len(group))
        profile.folios.update(r.folio for r in group)

        for category in {_cat(r.currier) for r in group}:
            profile.currier_presence[category] = 1
        for category in {_cat(r.scribe) for r in group}:
            profile.scribe_presence[category] = 1
        for category in {_cat(r.section) for r in group}:
            profile.section_presence[category] = 1
        for category in {_cat(r.quire) for r in group}:
            profile.quire_presence[category] = 1
        for record in group:
            profile.locus_type_counts[_cat(record.locus_type)] += 1

        profiles[leaf] = profile
    return profiles


def _merge(items: Iterable[Counter]) -> Counter:
    out = Counter()
    for item in items:
        out.update(item)
    return out


def build_atomic_units(
    records: Sequence[LocusRecord],
    profiles: Mapping[int, LeafProfile],
) -> List[AtomicUnit]:
    observed = set(profiles)

    for group in LINKED_LEAF_GROUPS:
        missing = set(group) - observed
        if missing:
            raise ValueError(
                f"linked group {group} references missing leaves {sorted(missing)}"
            )

    special_records: Dict[str, List[LocusRecord]] = defaultdict(list)
    unassigned = []
    for record in records:
        if record.physical_leaf is None:
            if record.folio in SPECIAL_PAGE_TO_LINKED_GROUP:
                special_records[record.folio].append(record)
            else:
                unassigned.append(record.folio)

    if unassigned:
        raise ValueError(
            "records without physical-leaf mapping: "
            + ", ".join(sorted(set(unassigned)))
        )

    linked_lookup = {}
    for group in LINKED_LEAF_GROUPS:
        for leaf in group:
            if leaf in linked_lookup:
                raise ValueError(f"leaf f{leaf} appears in multiple linked groups")
            linked_lookup[leaf] = tuple(sorted(group))

    units = []
    consumed = set()

    for leaf in sorted(observed):
        if leaf in consumed:
            continue

        leaves = linked_lookup.get(leaf, (leaf,))
        consumed.update(leaves)
        base = [profiles[x] for x in leaves]

        pages = tuple(
            page
            for page, group in SPECIAL_PAGE_TO_LINKED_GROUP.items()
            if tuple(sorted(group)) == leaves
        )
        extras = [r for page in pages for r in special_records.get(page, [])]

        currier = _merge(p.currier_presence for p in base)
        scribe = _merge(p.scribe_presence for p in base)
        section = _merge(p.section_presence for p in base)
        quire = _merge(p.quire_presence for p in base)
        locus_types = _merge(p.locus_type_counts for p in base)

        # fRos is not counted as another physical leaf. Its metadata/loci are
        # nevertheless attached to the linked f85/f86 unit.
        for category in {_cat(r.currier) for r in extras}:
            currier[category] += 1
        for category in {_cat(r.scribe) for r in extras}:
            scribe[category] += 1
        for category in {_cat(r.section) for r in extras}:
            section[category] += 1
        for category in {_cat(r.quire) for r in extras}:
            quire[category] += 1
        for record in extras:
            locus_types[_cat(record.locus_type)] += 1

        units.append(
            AtomicUnit(
                unit_id="+".join(f"f{x}" for x in leaves),
                leaves=tuple(leaves),
                leaf_count=len(leaves),
                locus_count=sum(p.locus_count for p in base) + len(extras),
                currier_presence=currier,
                scribe_presence=scribe,
                section_presence=section,
                locus_type_counts=locus_types,
                quire_presence=quire,
                special_pages=pages,
            )
        )

    return units


def _sum_field(units: Iterable[AtomicUnit], field: str) -> Counter:
    return _merge(getattr(unit, field) for unit in units)


def _stats(units: Sequence[AtomicUnit]) -> dict:
    return {
        "leaf_count": sum(u.leaf_count for u in units),
        "locus_total": sum(u.locus_count for u in units),
        "currier_presence": _sum_field(units, "currier_presence"),
        "scribe_presence": _sum_field(units, "scribe_presence"),
        "section_presence": _sum_field(units, "section_presence"),
        "locus_type_counts": _sum_field(units, "locus_type_counts"),
        "quire_presence": _sum_field(units, "quire_presence"),
    }


def _assignment_stats(assignment: Mapping[str, Sequence[AtomicUnit]]) -> Dict[str, dict]:
    return {split: _stats(list(assignment[split])) for split in SPLIT_ORDER}


def _eligible_coverage(profiles: Mapping[int, LeafProfile]) -> Dict[str, set]:
    result = {}
    for family in ("currier_presence", "scribe_presence", "section_presence"):
        support = Counter()
        for profile in profiles.values():
            for category in getattr(profile, family):
                support[category] += 1
        result[family] = {category for category, n in support.items() if n >= 3}
    return result


def objective(
    assignment: Mapping[str, Sequence[AtomicUnit]],
    *,
    global_stats: Mapping[str, object],
    target_leaf_counts: Mapping[str, int],
    eligible_coverage: Mapping[str, set],
) -> float:
    split_stats = _assignment_stats(assignment)
    total_leaves = global_stats["leaf_count"]
    total_loci = global_stats["locus_total"]
    score = 0.0

    # Keep text volume broadly proportional without using text content.
    for split in SPLIT_ORDER:
        target_fraction = target_leaf_counts[split] / total_leaves
        observed_fraction = split_stats[split]["locus_total"] / total_loci
        score += FAMILY_WEIGHTS["locus_total"] * (
            observed_fraction - target_fraction
        ) ** 2

    for family in (
        "currier_presence",
        "scribe_presence",
        "section_presence",
        "locus_type_counts",
    ):
        global_counter = global_stats[family]
        global_total = sum(global_counter.values())
        categories = sorted(global_counter)
        family_score = 0.0

        for split in SPLIT_ORDER:
            local = split_stats[split][family]
            local_total = sum(local.values())
            for category in categories:
                global_share = global_counter[category] / global_total
                local_share = local[category] / local_total if local_total else 0.0
                family_score += math.sqrt(global_share) * (
                    local_share - global_share
                ) ** 2

        score += FAMILY_WEIGHTS[family] * family_score / max(1, len(categories))

    for family, categories in eligible_coverage.items():
        for category in categories:
            for split in SPLIT_ORDER:
                if split_stats[split][family].get(category, 0) == 0:
                    score += COVERAGE_GAP_PENALTY

    return score


def _random_feasible(
    units: Sequence[AtomicUnit],
    targets: Mapping[str, int],
    rng: random.Random,
) -> Optional[Dict[str, List[AtomicUnit]]]:
    # General enough for the one 2-leaf linked unit plus singleton leaves.
    for _ in range(100):
        assignment = {split: [] for split in SPLIT_ORDER}
        remaining = dict(targets)
        shuffled = list(units)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda u: -u.leaf_count)

        failed = False
        for unit in shuffled:
            candidates = [
                split for split in SPLIT_ORDER
                if remaining[split] >= unit.leaf_count
            ]
            if not candidates:
                failed = True
                break

            weights = [remaining[s] for s in candidates]
            chosen = rng.choices(candidates, weights=weights, k=1)[0]
            assignment[chosen].append(unit)
            remaining[chosen] -= unit.leaf_count

        if not failed and all(v == 0 for v in remaining.values()):
            return assignment
    return None


def _copy_assignment(a: Mapping[str, Sequence[AtomicUnit]]) -> Dict[str, List[AtomicUnit]]:
    return {split: list(a[split]) for split in SPLIT_ORDER}


def _random_local_search(
    assignment: Dict[str, List[AtomicUnit]],
    *,
    global_stats: Mapping[str, object],
    targets: Mapping[str, int],
    eligible: Mapping[str, set],
    rng: random.Random,
    proposals: int,
) -> Tuple[Dict[str, List[AtomicUnit]], float]:
    current = objective(
        assignment,
        global_stats=global_stats,
        target_leaf_counts=targets,
        eligible_coverage=eligible,
    )

    for _ in range(proposals):
        split_a, split_b = rng.sample(SPLIT_ORDER, 2)

        candidates_a = [
            i for i, u in enumerate(assignment[split_a])
            if any(
                v.leaf_count == u.leaf_count
                for v in assignment[split_b]
            )
        ]
        if not candidates_a:
            continue

        ia = rng.choice(candidates_a)
        unit_a = assignment[split_a][ia]
        candidates_b = [
            i for i, u in enumerate(assignment[split_b])
            if u.leaf_count == unit_a.leaf_count
        ]
        ib = rng.choice(candidates_b)

        unit_b = assignment[split_b][ib]
        assignment[split_a][ia] = unit_b
        assignment[split_b][ib] = unit_a

        candidate = objective(
            assignment,
            global_stats=global_stats,
            target_leaf_counts=targets,
            eligible_coverage=eligible,
        )

        if candidate < current - 1e-15:
            current = candidate
        else:
            assignment[split_a][ia] = unit_a
            assignment[split_b][ib] = unit_b

    return assignment, current


def _signature(assignment: Mapping[str, Sequence[AtomicUnit]]) -> tuple:
    return tuple(
        tuple(
            sorted(
                leaf
                for unit in assignment[split]
                for leaf in unit.leaves
            )
        )
        for split in SPLIT_ORDER
    )


def optimize_assignment(
    units: Sequence[AtomicUnit],
    *,
    profiles: Mapping[int, LeafProfile],
    target_leaf_counts: Mapping[str, int],
    seed: int,
    restarts: int,
    swap_proposals: int,
) -> Tuple[Dict[str, List[AtomicUnit]], float]:
    global_stats = _stats(list(units))
    eligible = _eligible_coverage(profiles)
    master = random.Random(seed)

    best = None
    best_score = math.inf
    best_sig = None

    for _ in range(restarts):
        rng = random.Random(master.randrange(2**63))
        candidate = _random_feasible(units, target_leaf_counts, rng)
        if candidate is None:
            continue

        candidate, score = _random_local_search(
            candidate,
            global_stats=global_stats,
            targets=target_leaf_counts,
            eligible=eligible,
            rng=rng,
            proposals=swap_proposals,
        )
        sig = _signature(candidate)

        if (
            score < best_score - 1e-15
            or (
                math.isclose(score, best_score, abs_tol=1e-15)
                and (best_sig is None or sig < best_sig)
            )
        ):
            best = _copy_assignment(candidate)
            best_score = score
            best_sig = sig

    if best is None:
        raise RuntimeError("no feasible assignment found")
    return best, best_score


def split_leaves(assignment: Mapping[str, Sequence[AtomicUnit]]) -> Dict[str, List[int]]:
    return {
        split: sorted(
            leaf for unit in assignment[split] for leaf in unit.leaves
        )
        for split in SPLIT_ORDER
    }


def validate_assignment(
    assignment: Mapping[str, Sequence[AtomicUnit]],
    *,
    observed_leaves: Sequence[int],
    target_leaf_counts: Mapping[str, int],
) -> None:
    leaves = split_leaves(assignment)
    sets = {split: set(leaves[split]) for split in SPLIT_ORDER}

    for split in SPLIT_ORDER:
        if len(leaves[split]) != target_leaf_counts[split]:
            raise ValueError(
                f"{split}: {len(leaves[split])} leaves; "
                f"expected {target_leaf_counts[split]}"
            )

    for i, a in enumerate(SPLIT_ORDER):
        for b in SPLIT_ORDER[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(f"{a}/{b} overlap: {sorted(overlap)}")

    observed = set(observed_leaves)
    union = set().union(*(sets[s] for s in SPLIT_ORDER))
    if union != observed:
        raise ValueError(
            f"coverage mismatch; missing={sorted(observed-union)}, "
            f"extra={sorted(union-observed)}"
        )

    for group in LINKED_LEAF_GROUPS:
        owners = [
            split for split in SPLIT_ORDER
            if set(group) & sets[split]
        ]
        if len(set(owners)) != 1:
            raise ValueError(f"linked group {group} crosses splits")


def _counter(counter: Counter) -> dict:
    return dict(sorted(counter.items(), key=lambda x: str(x[0])))


def build_qc(
    assignment: Mapping[str, Sequence[AtomicUnit]],
    *,
    profiles: Mapping[int, LeafProfile],
    targets: Mapping[str, int],
    score: float,
) -> dict:
    all_units = [u for split in SPLIT_ORDER for u in assignment[split]]
    global_stats = _stats(all_units)
    local_stats = _assignment_stats(assignment)
    eligible = _eligible_coverage(profiles)

    gaps = []
    for family, categories in eligible.items():
        for category in sorted(categories):
            for split in SPLIT_ORDER:
                if local_stats[split][family].get(category, 0) == 0:
                    gaps.append(
                        {"family": family, "category": category, "split": split}
                    )

    splits = {}
    for split in SPLIT_ORDER:
        s = local_stats[split]
        splits[split] = {
            "leaf_count": s["leaf_count"],
            "leaf_fraction": s["leaf_count"] / global_stats["leaf_count"],
            "locus_total": s["locus_total"],
            "locus_fraction": s["locus_total"] / global_stats["locus_total"],
            "currier_presence": _counter(s["currier_presence"]),
            "scribe_presence": _counter(s["scribe_presence"]),
            "section_presence": _counter(s["section_presence"]),
            "locus_type_counts": _counter(s["locus_type_counts"]),
            "quire_presence_report_only": _counter(s["quire_presence"]),
        }

    return {
        "objective_score": score,
        "target_leaf_counts": dict(targets),
        "coverage_rule": "category occurs on >=3 physical leaves",
        "coverage_gaps": gaps,
        "global": {
            "leaf_count": global_stats["leaf_count"],
            "locus_total": global_stats["locus_total"],
            "currier_presence": _counter(global_stats["currier_presence"]),
            "scribe_presence": _counter(global_stats["scribe_presence"]),
            "section_presence": _counter(global_stats["section_presence"]),
            "locus_type_counts": _counter(global_stats["locus_type_counts"]),
            "quire_presence_report_only": _counter(global_stats["quire_presence"]),
        },
        "splits": splits,
    }


def _freeze_gate(
    path: Path,
    *,
    confirm_manual_spotcheck: bool,
) -> dict:
    if not path.is_file():
        raise ValueError(f"missing corpus QC summary: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    readiness = data.get("freeze_readiness", {})

    if data.get("overall_qc") != "PASS":
        raise ValueError("overall corpus QC is not PASS")
    if readiness.get("published_count_validation_complete") is not True:
        raise ValueError("published-count validation is not complete")

    if (
        readiness.get("manual_cross_transcription_spotcheck_complete") is True
        and readiness.get("ready_to_generate_locked_split") is True
    ):
        return {"status": "PASS", "summary_path": str(path)}

    if confirm_manual_spotcheck:
        return {
            "status": "CONFIRMED_AT_GENERATION",
            "summary_path": str(path),
            "note": (
                "Manual 15-locus review was explicitly confirmed when the "
                "split was generated; summary readiness flags were not yet set."
            ),
        }

    raise ValueError(
        "manual spot-check is not recorded complete. If it has actually been "
        "reviewed, rerun with --confirm-manual-spotcheck."
    )


def write_outputs(
    *,
    output_dir: Path,
    source: Path,
    source_hash: str,
    assignment: Mapping[str, Sequence[AtomicUnit]],
    profiles: Mapping[int, LeafProfile],
    targets: Mapping[str, int],
    score: float,
    seed: int,
    restarts: int,
    swap_proposals: int,
    freeze_gate: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    leaves = split_leaves(assignment)

    files = {
        "train": output_dir / "train.txt",
        "validation": output_dir / "validation.txt",
        "test": output_dir / "test_LOCKED.txt",
    }
    for split, path in files.items():
        path.write_text(
            "".join(f"f{x}\n" for x in leaves[split]),
            encoding="utf-8",
        )

    special = {}
    for split in SPLIT_ORDER:
        for unit in assignment[split]:
            for page in unit.special_pages:
                special[page] = {
                    "split": split,
                    "linked_leaves": [f"f{x}" for x in unit.leaves],
                }

    manifest = {
        "schema_version": "1.0",
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "primary_transcription": "ZL3b",
        "source_path": str(source),
        "source_sha256": source_hash,
        "target_ratios": dict(SPLIT_RATIOS),
        "target_leaf_counts": dict(targets),
        "atomic_unit": "physical_leaf",
        "recto_verso_together": True,
        "stratified_as_feasible_by": [
            "Currier class",
            "scribal hand",
            "manuscript section",
            "locus composition",
        ],
        "optimization_uses_text_content": False,
        "optimization": {
            "restarts": restarts,
            "swap_proposals_per_restart": swap_proposals,
            "objective_score": score,
            "family_weights": dict(FAMILY_WEIGHTS),
            "coverage_gap_penalty": COVERAGE_GAP_PENALTY,
        },
        "linked_leaf_groups": [
            [f"f{x}" for x in group] for group in LINKED_LEAF_GROUPS
        ],
        "special_page_assignments": special,
        "freeze_gate": dict(freeze_gate),
        "splits": {
            split: [f"f{x}" for x in leaves[split]]
            for split in SPLIT_ORDER
        },
        "locked_test_policy": (
            "Do not inspect or use the locked test for segmentation, model "
            "selection, hyperparameter tuning, threshold setting, or mechanism "
            "selection. Evaluate it only after those decisions are frozen."
        ),
    }

    qc = build_qc(
        assignment,
        profiles=profiles,
        targets=targets,
        score=score,
    )

    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "split_qc.json").write_text(
        json.dumps(qc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Local guardrail only; Git permissions are not a portable lock.
    files["test"].chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qc-summary", type=Path, default=DEFAULT_QC_SUMMARY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument(
        "--swap-proposals",
        type=int,
        default=DEFAULT_SWAP_PROPOSALS,
    )
    parser.add_argument(
        "--confirm-manual-spotcheck",
        action="store_true",
        help=(
            "Use only when the 15-locus review has actually been completed "
            "but transcription_summary.json has not yet been marked ready."
        ),
    )
    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
        help="Not recommended; bypass frozen ZL3b hash enforcement.",
    )
    args = parser.parse_args()

    try:
        gate = _freeze_gate(
            args.qc_summary,
            confirm_manual_spotcheck=args.confirm_manual_spotcheck,
        )

        if not args.source.is_file():
            raise ValueError(f"missing primary corpus: {args.source}")

        source_hash = sha256_file(args.source)
        if (
            source_hash != EXPECTED_ZL3B_SHA256
            and not args.allow_source_hash_mismatch
        ):
            raise ValueError(
                f"ZL3b SHA-256 mismatch: {source_hash}; "
                f"expected {EXPECTED_ZL3B_SHA256}"
            )

        records = load_ivtff(
            args.source,
            transcription_id="ZL3b",
            strict=True,
        )
        profiles = build_leaf_profiles(records)
        if len(profiles) != 102:
            raise ValueError(
                f"expected 102 observed physical leaves; found {len(profiles)}"
            )

        units = build_atomic_units(records, profiles)
        targets = largest_remainder_counts(len(profiles), SPLIT_RATIOS)
        expected_targets = {"train": 72, "validation": 15, "test": 15}
        if targets != expected_targets:
            raise ValueError(f"unexpected target counts: {targets}")

        assignment, score = optimize_assignment(
            units,
            profiles=profiles,
            target_leaf_counts=targets,
            seed=args.seed,
            restarts=args.restarts,
            swap_proposals=args.swap_proposals,
        )
        validate_assignment(
            assignment,
            observed_leaves=sorted(profiles),
            target_leaf_counts=targets,
        )
        write_outputs(
            output_dir=args.output_dir,
            source=args.source,
            source_hash=source_hash,
            assignment=assignment,
            profiles=profiles,
            targets=targets,
            score=score,
            seed=args.seed,
            restarts=args.restarts,
            swap_proposals=args.swap_proposals,
            freeze_gate=gate,
        )

        leaves = split_leaves(assignment)
        stats = _assignment_stats(assignment)
        print("=" * 72)
        print("PREREGISTERED PHYSICAL-LEAF SPLIT")
        print("=" * 72)
        print(f"Seed: {args.seed}")
        print("Atomic unit: physical leaf (recto/verso together)")
        print("Linked leakage-control group: f85 + f86; fRos follows that split")
        print(f"Objective score: {score:.12g}")
        print()
        for split in SPLIT_ORDER:
            print(
                f"{split:10s} leaves={len(leaves[split]):3d} "
                f"loci={stats[split]['locus_total']:5d}"
            )
        print()
        print(f"Saved: {args.output_dir / 'train.txt'}")
        print(f"Saved: {args.output_dir / 'validation.txt'}")
        print(f"Saved: {args.output_dir / 'test_LOCKED.txt'}")
        print(f"Saved: {args.output_dir / 'split_manifest.json'}")
        print(f"Saved: {args.output_dir / 'split_qc.json'}")
        print()
        print("LOCKED TEST: do not inspect or tune against test_LOCKED.txt.")
        return 0

    except Exception as exc:
        print(f"SPLIT GENERATION ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
