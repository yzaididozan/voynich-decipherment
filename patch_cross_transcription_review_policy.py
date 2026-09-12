#!/usr/bin/env python3
from pathlib import Path
import json

SCRIPT = Path("scripts/compare_sampled_transcriptions.py")
CONFIG = Path("configs/qc/cross_transcription_sample_v1.json")

if not SCRIPT.is_file():
    raise SystemExit(f"Missing {SCRIPT}")
if not CONFIG.is_file():
    raise SystemExit(f"Missing {CONFIG}")

text = SCRIPT.read_text(encoding="utf-8")

old = '''    if min_sim < low_threshold:
        return (
            "LOW_GLYPH_AGREEMENT",
            True,
            900.0 + 100.0 * (1.0 - min_sim) + uncertainty,
            (
                f"At least one pair has glyph similarity below "
                f"{low_threshold:.2f}."
            ),
        )
'''

new = '''    if min_sim < low_threshold:
        return (
            "LOW_GLYPH_AGREEMENT",
            False,
            900.0 + 100.0 * (1.0 - min_sim) + uncertainty,
            (
                f"At least one pair has glyph similarity below "
                f"{low_threshold:.2f}; retained as a targeted candidate "
                f"rather than mandatory manual review."
            ),
        )
'''

if old in text:
    text = text.replace(old, new, 1)
elif '"LOW_GLYPH_AGREEMENT",\n            False,' in text:
    print("LOW_GLYPH_AGREEMENT policy patch already present.")
else:
    raise SystemExit(
        "Could not find the expected LOW_GLYPH_AGREEMENT block. "
        "Refusing to patch an unexpected script version."
    )

SCRIPT.write_text(text, encoding="utf-8")

cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
cfg["target_disagreement_reviews"] = 8
cfg["consensus_control_reviews"] = 2
cfg["max_initial_reviews_per_folio"] = 1

policy = cfg.setdefault("policy", {})
policy["manual_review_design"] = (
    "Boundary-only disagreements remain mandatory because segmentation is "
    "directly relevant to the slot-grammar model. Low-glyph-agreement loci "
    "are triaged rather than exhaustively adjudicated by a non-specialist "
    "reviewer. After mandatory cases, select highest-priority disagreement "
    "cases to reach eight disagreement checks total, favoring distinct "
    "folios, then add two all-three-exact consensus controls."
)
policy["target_total_when_no_other_mandatory_cases"] = 10

CONFIG.write_text(
    json.dumps(cfg, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(f"Patched: {SCRIPT}")
print(f"Updated: {CONFIG}")
print()
print("Now rerun:")
print("  python scripts/compare_sampled_transcriptions.py")
