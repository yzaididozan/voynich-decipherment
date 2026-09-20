# VOYAGER Phase 3B — Locked Reserved-Control Confirmation

**Experiment ID:** `phase-3b-reserved-control-confirmation-v1`  
**Status:** Confirmatory  
**Repository destination:** `docs/phase_3b_preregistration.md`

## 1. Purpose

Phase 3A.6 completed exploratory mechanism development. It retained the frozen
Phase-3A.4 K=16 representation and changed only the allocation of the existing
16 suffix variants: each encoded lexical base receives a deterministic,
seed-specific permutation, and repeated occurrences traverse the 16 variants
without replacement before cycling.

On development validation, Phase 3A.6 reproduced the inherited five-relation
Level-2 hierarchy. The confirmatory question is therefore fixed **before any
reserved-control encoding or scoring**:

> Does the exact frozen Phase 3A.6 recurrence-avoiding K=16 mechanism reproduce
> the inherited five-relation Level-2 hierarchy on the untouched reserved
> historical-language controls?

Phase 3B is not a mechanism search. It contains no K grid, no allocation grid,
no language-specific repair, no new predictive model, and no threshold chosen
from reserved results.

## 2. Frozen candidate

The candidate is exactly:

```text
K                         = 16
suffix length             = 1 glyph
base encoding             = Phase-3A.4 TRAIN-fitted monoalphabetic base
suffix inventory/order    = exact Phase-3A.4 K16 inventory inherited by 3A.6
allocation unit           = encoded lexical base
allocation                = balanced without replacement within 16-occurrence cycles
per-base permutation      = deterministic from the frozen Phase-3A.6 namespace/seed/base rule
```

No representation or allocation choice may change after the Phase 3B protocol
is frozen.

The required Phase 3A.6 lineage is:

```text
configs/mechanisms/recurrence_avoiding_surface_allocation_v1.json
SHA-256 f26fef20aad928ee98a3e4824ca0911c8855de2ff1c0908d5388fae0ffbd2535

src/ciphers/recurrence_avoiding_surface_allocation.py
SHA-256 2a6b27af36b0b896b40409450dffbd443d7a378d4deffaa390e8a989d25e33c9
```

The runner also requires the completed Phase 3A.6 summary to report 20 runs,
100 relation rows, complete parent lineage, zero avoidable validation
collisions, the full inherited Level-2 hierarchy, and no prior reserved or
Voynich test access.

## 3. Frozen evaluator

Phase 3B reuses the existing Level-2 evaluator without changing model families,
selection rules, bootstrap rules, or operational relations:

```text
configs/evaluation/level2_predictive_controls_v1.json
SHA-256 6f6214286c9c8e47e90c27cf4c41da60f21c12fd8c31c07a4a98e3aada8b236c
```

The five relations remain:

```text
hmm_space_free
hmm_token_aware
slot_grammar
token_markov
copy_edit
```

The evaluator API still names its held-out argument `validation_units`; during
Phase 3B that argument is populated **only with RESERVED units**. Development
VALIDATION units are never supplied to the evaluator.

## 4. Confirmatory data flow

For each source and seed:

```text
CONTROL TRAIN
    |
    | fit exact frozen key
    v
encode TRAIN with Phase-3A.6 schedule
    |
    | freeze final TRAIN per-base occurrence counters
    v
CONTROL RESERVED TEST
    |
    | encode directly from frozen TRAIN state
    v
Level-2 held-out evaluation
```

Development VALIDATION is bypassed entirely for confirmatory encoding and
scoring. It does not advance recurrence counters.

This choice is fixed because allowing development-validation occurrences to
change the reserved ciphertext would permit development data to influence the
confirmatory representation. The reserved encoding therefore begins from the
final TRAIN state exactly as a new held-out continuation of the frozen
TRAIN-fitted mechanism.

Reserved rows are processed in ascending frozen line index, and tokens within a
row are processed left-to-right. No reserved look-ahead is permitted.

## 5. Frozen sources, splits, and seeds

The four historical-language controls are unchanged:

| Source | Language | TRAIN | DEV VALIDATION | RESERVED |
|---|---:|---:|---:|---:|
| `french_profiterole_ud218` | French | 1858 | 387 | 387 |
| `german_rem_v21` | German | 3155 | 657 | 658 |
| `italian_old_ud218` | Italian | 522 | 109 | 109 |
| `latin_udante_ud218` | Latin | 512 | 106 | 107 |

The five seeds remain:

```text
40814041438
40814041439
40814041440
40814041441
40814041442
```

Thus Phase 3B contains:

```text
4 languages x 5 seeds = 20 confirmatory runs
20 runs x 5 relations = 100 relation outcomes
```

## 6. Frozen confirmatory decision rule

The inherited Level-2 rule is carried forward unchanged.

A run is a **strong Voynich-direction match** when:

```text
delta > 0
and
95% paired-bootstrap CI lower > 0
```

A language/relation replicates when:

```text
>= 4/5 seeds are strong
```

A relation is robust when:

```text
>= 3/4 languages replicate
```

**Phase 3B PASS** requires:

```text
all five relations are robust
```

Otherwise the result is **Phase 3B FAIL**.

There is no special Token-Markov threshold, no requirement to reproduce the
exact Phase-3A.6 effect magnitudes, and no relaxation if one language or one
relation is close to the boundary.

## 7. Interpretation of PASS

A PASS means that the frozen recurrence-avoiding mechanism reproduced the
pre-existing five-relation Level-2 hierarchy on untouched reserved
historical-language controls under the predeclared replication rule.

A PASS does **not** establish that the mechanism generated the Voynich
Manuscript, identify a plaintext language, or authorize retrospective changes
to Phase 3B. Voynich `test_LOCKED` remains untouched. A separate Voynich-test
protocol must be written and frozen before that dataset is accessed.

## 8. Interpretation of FAIL

A FAIL is retained as the confirmatory outcome. The reserved set may not be
reused as a fresh confirmatory set after changing K, allocation strength,
frequency thresholds, language-specific behavior, the evaluator, seeds, or the
decision threshold.

Post-failure diagnostics may describe the failure, but any revised mechanism
returns to exploratory status and requires a genuinely new confirmatory test
resource before another confirmation claim.

## 9. Prohibited actions after reserved access

The following are prohibited within Phase 3B:

```text
K > 16
K < 16 selected from reserved behavior
alternate balancing strengths
randomized balancing schedules
frequency thresholds
special treatment for French or Latin
transition-conditioned allocation
changing suffix length
changing the base encoding
changing source membership
changing seeds
changing Level-2 model grids or bootstrap rules
changing the >=4/5 language rule
changing the >=3/4 relation rule
redefining PASS after seeing reserved results
accessing Voynich test_LOCKED from the Phase 3B runner
```

## 10. One-shot execution procedure

Before reserved encoding or scoring:

```bash
python -m pytest -q

git add \
  configs/phase_3b_confirmatory.yaml \
  docs/phase_3b_preregistration.md \
  scripts/run_phase_3b_confirmatory_control.py

git commit -m "Freeze Phase 3B reserved-control confirmation"

git rev-parse HEAD
sha256sum \
  configs/phase_3b_confirmatory.yaml \
  docs/phase_3b_preregistration.md \
  scripts/run_phase_3b_confirmatory_control.py
```

Record the commit and hashes externally or in the project log **before** the
next command.

Then, and only then:

```bash
python scripts/run_phase_3b_confirmatory_control.py --execute-reserved-test
```

The explicit flag is intentional: running the script without it performs only
preflight verification and does not encode or score RESERVED.

## 11. Expected outputs

```text
data/controls/ciphers/phase3b_reserved_confirmation_v1/
  <source>/
    balanced16/
      seed_<seed>/
        corpus.jsonl
        key.json
        summary.json
        SHA256SUMS

results/controls/phase3b_reserved_confirmation/v1/
  run_relation_matrix.csv
  language_relation_summary.csv
  relation_replication_summary.csv
  global_relation_summary.csv
  lineage_verification.csv
  summary.json
  SHA256SUMS
  runs/
    <source>/
      balanced16/
        seed_<seed>/
          relations.csv
          summary.json
          SHA256SUMS
```

The top-level `summary.json` records a single confirmatory decision:

```text
PHASE_3B_PASS
```

or

```text
PHASE_3B_FAIL
```

A scientific FAIL is not a software error and therefore does not require a
nonzero process exit status.

## 12. Voynich boundary

The Phase 3B script contains no Voynich corpus path and must not import,
materialize, tokenize, inspect, or score `test_LOCKED`.

After Phase 3B, regardless of outcome, the Phase 3B files and result artifacts
remain immutable. Only a separately frozen protocol may govern any later
Voynich test.
