# VOYAGER Phase 4 — Locked Voynich Evaluation

**Experiment ID:** `phase-4-voynich-locked-evaluation-v1`  
**Status:** Confirmatory locked-test evaluation  
**Repository destination:** `docs/phase_4_voynich_preregistration.md`

## 1. Purpose

Phase 3A.6 ended mechanism development with one fixed candidate: a
base-preserving K=16 surface system whose variants are allocated without
replacement within each lexical base's 16-occurrence cycle. Phase 3B then
confirmed the complete five-relation Level-2 hierarchy on untouched reserved
historical-language controls.

Phase 4 asks a different question:

> Does the actual **locked Voynich test partition** exhibit the same five
> preregistered predictive relations?

Phase 4 is not a decipherment attempt. It does not map Voynich tokens back to
French, German, Italian, Latin, or any other plaintext. It does not fit the
Phase 3A.6 cipher key to Voynich. It tests a predictive fingerprint.

## 2. What Phase 3B established

The entry condition is a completed `PHASE_3B_PASS` with:

- 20/20 reserved-control runs completed;
- 100 relation outcomes;
- the full five-relation hierarchy reproduced;
- Token-Markov replicated 5/5 seeds in French, German, Italian, and Latin;
- development validation excluded from the Phase 3B confirmatory scoring;
- Voynich `test_LOCKED` not accessed.

The exact Phase 3B protocol SHA-256 is:

```text
b712bbdf13e3bf121e5a045ad0d3b4a76e496a8246b4afdc63f6477a24564ce8
```

The inherited Phase 3A.6 mechanism is fixed at K=16, one suffix glyph, balanced
without-replacement allocation per lexical base.

## 3. Primary Voynich corpus

Phase 4 uses only the already-frozen primary representation:

```text
transcription       Zandbergen–Landini ZL v3b
path                data/raw/zl/ZL3b-n.txt
SHA-256             bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc
format              IVTFF 2.0
native alphabet     Eva-
analytical view     S0_ZL_CONVENTIONAL_STA1_GLYPHS
```

S0 means:

- conventional ZL/EVA token segmentation;
- both certain `.` and uncertain `,` spaces are token boundaries;
- `<->` and `<~>` drawing interruptions are boundaries;
- first alternative reading is used;
- the existing EVA→STA1 conversion is used;
- `Z1` follows each frozen relation's existing unknown-symbol policy.

The STA1 rules are:

```text
data/reference/sta1/STA-Eva_def.bit
SHA-256 7f37853510144fb3e2dc3ee9458d634f41e6d95bc1fbf1c4b8f479a53a021f81
```

No alternative transcription or preprocessing variant may replace the primary
test after the locked result is seen. Cross-transcription robustness, if later
performed, is a separate analysis.

## 4. Frozen manuscript split

The atomic split unit is the **physical leaf**, with recto/verso kept together.

```text
TRAIN       72 leaves
VALIDATION  15 leaves
TEST        15 leaves
TOTAL      102 leaves
```

Known pre-test hashes:

```text
data/splits/v1/train.txt
5bb5232d5e211a4bb90800f259294c554c1c9e2005548055ebaf94e782e3cc00

data/splits/v1/validation.txt
bdae856ac8c52849a0abd0fd6361f42fd57579e76f7f844309a7ddc9e0fc02f1
```

`data/splits/v1/test_LOCKED.txt` is deliberately **not opened during Phase 4
preflight**. Its observed hash is recorded only after the explicit locked-test
execution begins.

The Rosettes leakage rule remains in force: f85 and f86 form a linked group and
the fRos pseudo-page follows that assignment.

## 5. Critical no-selection rule

The locked test may evaluate already-selected models. It may not select models.

Therefore Phase 4 does **not** call the old development/control evaluator in a
way that permits a grid search on `test_LOCKED`.

The already-selected settings are:

```text
HMM space-free      16 hidden states
HMM token-aware     16 hidden states
slot grammar        16 templates
copy-edit           source window 16
Token-Markov        context order from the already-existing
                    results/baselines/token_markov/v1/summary.json
                    (must be 1 or 2 and locked_test_accessed=false)
matched trigram     order 3
```

The Token-Markov context value is intentionally read from the preexisting
validation summary rather than guessed or selected on the test. The preflight
records its value and summary hash before locked-test access.

## 6. Five frozen relations

The operational relations are unchanged:

```text
hmm_space_free
hmm_token_aware
slot_grammar
token_markov
copy_edit
```

For every relation:

```text
delta = competitor bits/event - matched trigram bits/event
```

The preregistered Voynich-direction match is:

```text
delta > 0
AND
paired physical-leaf bootstrap 95% CI lower bound > 0
```

Bootstrap settings:

```text
unit        physical leaf
replicates  10,000
seed        40814041438
```

## 7. Primary Phase 4 decision

The primary conjunction is the complete five-relation pattern.

If all five relations satisfy the frozen strong-direction rule:

```text
CONSISTENT_NONDISCRIMINATING
```

Meaning:

> The untouched Voynich test is statistically consistent with the complete
> predictive fingerprint reproduced by the Phase 3A.6 mechanism and confirmed
> in Phase 3B controls.

The word **non-discriminating** is mandatory. The five-relation match does not
show that this is the unique mechanism capable of producing those relations.

If one or more relations does not satisfy the frozen rule:

```text
INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN
```

This means the full preregistered fingerprint did not survive the locked Voynich
test. Individual matching relations remain reportable, but the conjunction
fails.

This Phase 4 runner is never allowed to emit `DISCRIMINATING_SUPPORT`.
Discrimination among competing mechanisms requires a separate frozen tournament
or likelihood-based comparison.

## 8. What a positive result would *not* establish

Even `CONSISTENT_NONDISCRIMINATING` does not establish:

- that the Voynich Manuscript was generated by this exact mechanism;
- that K=16 was historically used;
- that the suffix variants correspond to literal written suffixes;
- that any source language has been identified;
- that a plaintext has been recovered;
- that the manuscript has been deciphered.

It establishes an out-of-sample structural compatibility result.

## 9. Preflight procedure

Before locked-test access:

```bash
python -m pytest -q
python scripts/run_phase_4_voynich_locked.py
```

The second command is **preflight only**. It must not open, hash, stat, parse, or
list `data/splits/v1/test_LOCKED.txt`.

Inspect the output. It records:

- Phase 4 config hash;
- Phase 3B protocol/result status;
- Phase 3A.6 and Level-2 hashes;
- ZL3b and STA1 rule hashes;
- TRAIN and VALIDATION split hashes;
- validation-frozen model settings;
- the preexisting Token-Markov validation selection and summary hash.

Then freeze the protocol:

```bash
git add \
  configs/phase_4_voynich_locked.yaml \
  docs/phase_4_voynich_preregistration.md \
  scripts/run_phase_4_voynich_locked.py

git commit -m "Freeze Phase 4 locked Voynich evaluation"

git rev-parse HEAD

sha256sum \
  configs/phase_4_voynich_locked.yaml \
  docs/phase_4_voynich_preregistration.md \
  scripts/run_phase_4_voynich_locked.py
```

Record the commit and hashes before proceeding.

## 10. Locked execution

Only after the protocol is frozen:

```bash
python scripts/run_phase_4_voynich_locked.py --execute-locked-test
```

At first test access the runner immediately creates:

```text
results/locked/phase4_voynich/v1/LOCKED_TEST_ACCESSED.json
```

This marker records protocol/input hashes and makes the one-shot boundary
explicit. Once this marker exists, the locked test has been touched even if a
later software error interrupts scoring.

## 11. Software recovery versus scientific rerun

If execution fails after the access marker because of a software/environment
error, preserve the marker and traceback. Do not change scientific choices.

A software-only repair may resume only if it cannot change:

- corpus membership;
- representation;
- model hyperparameters;
- relation definitions;
- bootstrap settings;
- thresholds;
- or interpretation.

Any scientifically meaningful modification creates a new **exploratory**
analysis; it cannot replace this locked Phase 4 result.

## 12. Required outputs

The intended output directory is:

```text
results/locked/phase4_voynich/v1/
  LOCKED_TEST_ACCESSED.json
  relation_results.csv
  bootstrap/
    hmm_space_free.csv
    hmm_token_aware.csv
    slot_grammar.csv
    token_markov.csv
    copy_edit.csv
  summary.json
  run_manifest.json
  SHA256SUMS
```

`summary.json` records every relation independently and one primary decision.

## 13. Stop rule

After test access, do not:

- tune failed relations;
- expand a grid;
- change S0 tokenization;
- swap transcription;
- change uncertainty treatment;
- remove inconvenient leaves;
- alter bootstrap settings;
- alter thresholds;
- rerun an edited confirmatory protocol against the same locked split.

The locked result is retained as observed.

## 14. Next scientific stage

If Phase 4 is `CONSISTENT_NONDISCRIMINATING`, the next step is not to announce a
decipherment. The next step is a separately frozen **mechanism-discrimination
stage**: compare the confirmed candidate against plausible competing generators
or channels using predictions that distinguish them, ideally on information not
used to construct the current fingerprint.

If Phase 4 is inconsistent, retain the failure and return to exploratory
mechanism work without relabeling a modified rerun as confirmatory.
