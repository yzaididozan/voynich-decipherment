#!/usr/bin/env python3
from pathlib import Path

TARGET = Path("scripts/compare_sampled_transcriptions.py")

if not TARGET.is_file():
    raise SystemExit(f"Missing {TARGET}")

text = TARGET.read_text(encoding="utf-8")

old_missing = '''        if missing:
            output.append(
                {
                    **base,
                    "comparison_class": "MISSING_LOCUS",
                    "mandatory_manual_review": True,
                    "priority_score": 1200.0,
                    "review_reason": (
                        "Exact locus ID missing from: "
                        + ", ".join(missing)
                    ),
'''

new_missing = '''        if missing:
            # Special coverage difference: ZL3b includes certain marginal
            # @Lx loci that GC2a and IT2a do not encode under a corresponding
            # locus. This is a corpus-scope difference, not evidence that the
            # ZL3b reading is wrong.
            coverage_difference = (
                locus in indexes["ZL3b"]
                and set(missing) == {"GC2a", "IT2a"}
                and ",@Lx" in locus
            )

            output.append(
                {
                    **base,
                    "comparison_class": (
                        "COVERAGE_DIFFERENCE_SPECIAL_MARGINALIA"
                        if coverage_difference
                        else "MISSING_LOCUS"
                    ),
                    "mandatory_manual_review": (
                        False if coverage_difference else True
                    ),
                    "exclude_from_manual_review": coverage_difference,
                    "priority_score": (
                        -1.0 if coverage_difference else 1200.0
                    ),
                    "review_reason": (
                        (
                            "ZL3b-only @Lx marginal locus absent from both "
                            "GC2a and IT2a; treated as a transcription-scope "
                            "coverage difference, not a disagreement."
                        )
                        if coverage_difference
                        else (
                            "Exact locus ID missing from: "
                            + ", ".join(missing)
                        )
                    ),
'''

if old_missing not in text:
    if "COVERAGE_DIFFERENCE_SPECIAL_MARGINALIA" in text:
        print("Coverage-difference patch already present.")
    else:
        raise SystemExit(
            "Could not find the expected MISSING_LOCUS block. "
            "Refusing to patch an unexpected script version."
        )
else:
    text = text.replace(old_missing, new_missing, 1)

old_candidates = '''            if row["locus"] not in selected_loci
            and not bool(row["all_three_token_exact"])
'''

new_candidates = '''            if row["locus"] not in selected_loci
            and not bool(row["all_three_token_exact"])
            and not bool(row.get("exclude_from_manual_review", False))
'''

if old_candidates not in text:
    if 'row.get("exclude_from_manual_review", False)' in text:
        print("Manual-review exclusion patch already present.")
    else:
        raise SystemExit(
            "Could not find the expected candidate-selection block. "
            "Refusing to patch an unexpected script version."
        )
else:
    text = text.replace(old_candidates, new_candidates, 1)

TARGET.write_text(text, encoding="utf-8")

print(f"Patched: {TARGET}")
print()
print("Now rerun:")
print("  python scripts/compare_sampled_transcriptions.py")
