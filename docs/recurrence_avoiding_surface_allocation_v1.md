# Phase 3A.6 — Recurrence-Avoiding Base-Preserving Surface Allocation

**Experiment ID:** `recurrence-avoiding-surface-allocation-exploration-v1`  
**Status:** Exploratory  
**Repository destination:** `docs/recurrence_avoiding_surface_allocation_v1.md`

## 1. Motivation

The mechanism search has narrowed to one remaining predictive failure.

Phase 3A.4 showed that increasing base-preserving surface multiplicity strongly
moved the exact-token Markov relation toward the Voynich direction:

```text
K=4    -0.128436 bits/event
K=8    -0.038280 bits/event
K=16   +0.028097 bits/event
```

while HMM space-free, HMM token-aware, slot grammar, and local copy-edit
remained strongly in the Voynich direction.

However, the K=16 token relation replicated strongly in only:

```text
German     5/5
Italian    5/5
French     0/5
Latin      0/5
```

which is 2/4 languages rather than the frozen >=3/4 robustness requirement.

Phase 3A.5 held the same 16 abstract identities fixed and changed only their
two-slot representation:

```text
1x16   +0.024881
2x8    +0.024605
4x4    +0.024283
```

The full hierarchy still failed only at token Markov, again at 2/4 languages.
Simple Cartesian factorization therefore did not solve the residual problem.

Phase 3A.D1 then decomposed the exact frozen token-Markov likelihood difference
token by token.

All 60 aggregate reconstructions passed:

```text
60 / 60 PASS
maximum parity error approximately 1.2e-14
```

The decisive localization was not repeated exact transitions. It was recurrence
of the current exact ciphertext token itself.

For validation tokens whose exact current ciphertext form occurred once in
TRAIN but whose exact previous->current ciphertext bigram occurred zero times
in TRAIN:

```text
French    delta = -0.4942
German    delta = -0.2270
Italian   delta = -0.3942
Latin     delta = -0.7184
```

In Latin, 93% of all token-Markov advantage bits occurred on previously seen
exact ciphertext tokens whose exact transition was unseen in TRAIN.

Conversely, when the current exact ciphertext token itself was unseen:

```text
French    +0.2723
German    +0.2805
Italian   +0.2570
Latin     +0.1843
```

the matched trigram strongly won in every language.

This motivates a direct intervention on **avoidable exact-surface collisions**.

## 2. Scientific question

> If the Phase-3A.4 K=16 base form, suffix inventory, token length, coverage,
> sources, seeds, and evaluator are held fixed, does allocating the existing 16
> suffix variants without replacement within each lexical base's 16-occurrence
> cycle reduce exact-surface recurrence enough to reproduce the full frozen
> predictive hierarchy?

The intervention does not increase K.

It does not add context-conditioned encoding.

It does not change suffix length.

It does not add a new model.

## 3. Files

Install:

```text
configs/mechanisms/recurrence_avoiding_surface_allocation_v1.json
src/ciphers/recurrence_avoiding_surface_allocation.py
scripts/run_recurrence_avoiding_surface_allocation_exploration.py
tests/test_recurrence_avoiding_surface_allocation.py
docs/recurrence_avoiding_surface_allocation_v1.md
```

## 4. Frozen parent

The exact representation parent is Phase 3A.4 K=16.

Required hashes:

```text
configs/mechanisms/base_form_suffix_extended_v1.json
514f440de8f086f0a5d9292f85289930de98fbc0893251e345ed134ef278b1f8

src/ciphers/base_form_suffix_extended.py
3969000c5ca678a68f259f2967d4164f25dc1360b7bba0e945d619704d12b345
```

The new mechanism must inherit, source by source and seed by seed:

```text
exact TRAIN-fitted monoalphabetic base mapping
exact K=16 master suffix inventory
exact master suffix order
one appended suffix glyph per token
100% TRAIN/VALIDATION coverage
same physical source/split memberships
```

The runner verifies these requirements against the existing generated
Phase-3A.4 K16 controls.

## 5. Diagnostic lineage

Phase 3A.6 was chosen after the frozen Phase 3A.D1 diagnostic.

Required diagnostic hashes:

```text
configs/diagnostics/token_markov_failure_localization_v1.json
296363defa5bf65ef3f82d2c514f9918fb14a46296de916c579e9eb88dbd104c

src/analysis/token_markov_failure_localization.py
b729ee11b1909ac2972a150c051671ad2ca7dad8bbe3456c80747324be19e4e5
```

The runner also requires the Phase-3A.D1 summary to report:

```text
parity runs          = 60
parity runs passed   = 60
```

and to point to the same Phase-3A.5 relation matrix observed during the
diagnostic:

```text
6a3bddb21785d2e1d082ab9c9ca0ab123a21c98930585c0249b9b3983d655c8d
```

This records that Phase 3A.6 is a diagnostic-driven follow-up rather than an
unmotivated post-hoc mechanism.

## 6. Parent allocation versus Phase 3A.6 allocation

### Phase 3A.4 K=16

The suffix identity for an occurrence was effectively selected independently
from the 16 available variants using occurrence-level deterministic hashing.

Repeated occurrences of one lexical base could therefore collide before all
16 variants had been used.

Illustrative pattern:

```text
BASE+V03
BASE+V08
BASE+V03   <- avoidable recurrence
BASE+V12
BASE+V08   <- avoidable recurrence
```

### Phase 3A.6

Every encoded lexical base receives one deterministic seed-specific
permutation of the same 16 inherited suffix indices.

For example:

```text
[7, 12, 2, 15, 8, 1, 13, 6, 10, 3, 16, 5, 9, 14, 4, 11]
```

The first 16 occurrences of that encoded base traverse the permutation once:

```text
occurrence  1 -> V07
occurrence  2 -> V12
occurrence  3 -> V02
...
occurrence 16 -> V11
```

Only occurrence 17 is allowed to reuse the first suffix:

```text
occurrence 17 -> V07
```

Thus every consecutive 16-occurrence cycle contains every suffix exactly once.

## 7. Allocation unit

Balancing is keyed by the **encoded lexical base**, not by a global occurrence
counter.

The encoded base is the complete Phase-3A.4 monoalphabetic base form before
the suffix is appended.

Therefore:

```text
same underlying word/base
    -> same 16-state cycle

different underlying word/base
    -> independent deterministic permutation
```

The permutation hash uses:

```text
namespace
seed
serialized encoded base
suffix index
```

The full permutation is fixed before VALIDATION is processed.

## 8. TRAIN allocation

TRAIN is encoded first.

TRAIN lines are processed in:

```text
ascending line index
```

and tokens within each line are processed:

```text
left to right
```

For each encoded base, maintain a zero-based occurrence counter:

```text
n(base)
```

The assigned suffix index is:

```text
permutation(base, seed)[n(base) mod 16]
```

After encoding the token:

```text
n(base) += 1
```

No validation information is involved.

## 9. VALIDATION continuation

After TRAIN is fully encoded, the final per-base TRAIN occurrence counts are
frozen.

VALIDATION then begins from those counts.

Suppose one base appears six times in TRAIN. Its first six permutation states
have been used.

The first VALIDATION occurrence receives state seven, not a restarted state
one:

```text
TRAIN:
1 2 3 4 5 6

VALIDATION:
7 8 9 ...
```

VALIDATION is processed online in ascending line/token order.

The encoder may not inspect future validation occurrences to optimize or
rearrange assignments.

This preserves the causal intervention:

> continue a predetermined lexical-base cycle from TRAIN state.

## 10. Avoidable versus unavoidable recurrence

If a base appeared fewer than 16 times in TRAIN, some suffix variants remain
unused for that base.

For example:

```text
TRAIN count = 14
```

The first two validation occurrences can receive the two unused suffixes.

Those are **avoidable-collision opportunities**.

Phase 3A.6 requires:

```text
avoidable validation collisions = 0
```

After those two assignments, all 16 variants have been represented in TRAIN +
the validation continuation. Further validation occurrences necessarily cycle
and may reuse an exact TRAIN surface form.

That is **unavoidable under fixed K=16** and is retained.

The experiment therefore does not claim to eliminate recurrence entirely.

It eliminates recurrence only when an unused variant remains available under
the frozen 16-form inventory.

## 11. Why K remains 16

Phase 3A.4 already imposed a hard stop against expanding K beyond 16 after
seeing the K=16 result.

Phase 3A.6 respects that stop rule.

It tests:

```text
same K = 16
same 16 suffixes
different allocation policy
```

It does not test:

```text
K = 32
K = 64
```

and those conditions may not be added after seeing the Phase-3A.6 result.

## 12. Why token length returns to base + 1

Phase 3A.5 used two-glyph affixes only to test compositional factorization.

Phase 3A.D1 showed that factorization did not solve the residual problem.

Phase 3A.6 therefore returns to the exact Phase-3A.4 K=16 representation:

```text
BASE + one inherited suffix glyph
```

This allows the new controls to differ from the Phase-3A.4 K16 parent only in
allocation policy.

The runner verifies that every TRAIN and VALIDATION token has the same encoded
base and the same total token length as its Phase-3A.4 parent counterpart.

## 13. Parent collision comparison

For every source/seed, the runner reads both:

```text
frozen Phase-3A.4 K16 TRAIN/VALIDATION
new balanced Phase-3A.6 TRAIN/VALIDATION
```

and computes:

```text
fraction of validation exact ciphertext tokens already seen in TRAIN
```

for each.

Output:

```text
collision_comparison.csv
collision_language_summary.csv
```

The directional mechanism prediction is:

```text
balanced fraction < parent K16 fraction
```

especially in French and Latin.

This is a mechanism diagnostic, not the predictive success criterion.

## 14. Parent lineage verification

For all 20 new controls, the runner requires:

```text
mono base map identical
master suffix inventory/order identical
parent master-key fingerprint identical
encoded base identical token by token on TRAIN and VALIDATION
token length identical token by token
```

Output:

```text
parent_lineage_verification.csv
```

Any mismatch aborts the experiment.

## 15. Experimental grid

Only one new intervention is tested:

```text
allocation = balanced16
```

Grid:

```text
4 languages
x 5 frozen seeds
= 20 controls
```

Each control receives the five frozen Level-2 comparisons:

```text
20 x 5
= 100 relation outcomes
```

No additional balancing strengths or allocation schedules are allowed after
results are observed.

## 16. Frozen evaluator

Phase 3A.6 reuses:

```text
configs/evaluation/level2_predictive_controls_v1.json
```

with frozen SHA:

```text
6f6214286c9c8e47e90c27cf4c41da60f21c12fd8c31c07a4a98e3aada8b236c
```

The five operational relations remain:

```text
hmm_space_free
hmm_token_aware
slot_grammar
token_markov
copy_edit
```

No hyperparameter grid is added.

No aggregate similarity score is introduced.

## 17. Primary prediction

The main prediction is about the token relation.

Because Phase 3A.D1 localized the largest residual lexical-recurrence problem
to Latin, followed by French, the predicted movement is:

```text
largest upward change:    Latin
next most important:      French
```

German and Italian already reproduce the token relation and should remain in
the positive Voynich direction.

The other four relations should remain robust.

This is an exploratory mechanistic prediction, not a guaranteed result.

## 18. Frozen replication criterion

The inherited Level-2 criterion remains unchanged.

A run is a strong Voynich-direction match when:

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

The full hierarchy is reproduced only when:

```text
all five relations are robust
```

No weaker Phase-3A.6-specific success threshold is introduced.

## 19. Stop rule

After Phase 3A.6 results are observed, do not add:

```text
K > 16
partial balancing strengths
randomized balancing schedules
frequency thresholds
special treatment for Latin
special treatment for French
transition-conditioned allocation
```

to this experiment.

If Phase 3A.6 fails the full hierarchy, retain the result and formulate a
distinct mechanism hypothesis.

## 20. Data protection

Allowed:

```text
CONTROL TRAIN
CONTROL VALIDATION
```

Not allowed:

```text
CONTROL RESERVED TEST
Voynich test_LOCKED
```

Generated reserved rows remain:

```json
{"line_index": 123, "reserved_test_opaque": true}
```

The frozen Level-2 reader streams these rows past without scoring them.

## 21. Tests

Run:

```bash
source .venv/bin/activate

python -m pytest -q tests/test_recurrence_avoiding_surface_allocation.py
python -m pytest -q
```

The dedicated tests cover:

- K fixed at 16;
- exact inheritance of the Phase-3A.4 base map;
- exact inheritance of the Phase-3A.4 suffix inventory/order;
- deterministic per-base permutation;
- all 16 variants used once before reuse;
- cycle repetition only after state 16;
- base preservation;
- one-glyph suffix length;
- round-trip decoding;
- deterministic generation;
- reserved-row opacity;
- validation continuation from frozen TRAIN counts;
- zero avoidable validation collisions;
- correct unavoidable reuse after all 16 states are exhausted;
- `min(TRAIN count, 16)` unique TRAIN variants per base;
- reversible unseen validation characters;
- required diagnostic invariants;
- output serialization.

## 22. Freeze before execution

After the dedicated and full repository tests pass:

```bash
git add \
  configs/mechanisms/recurrence_avoiding_surface_allocation_v1.json \
  src/ciphers/recurrence_avoiding_surface_allocation.py \
  scripts/run_recurrence_avoiding_surface_allocation_exploration.py \
  tests/test_recurrence_avoiding_surface_allocation.py \
  docs/recurrence_avoiding_surface_allocation_v1.md

git commit -m "Freeze Phase 3A.6 recurrence-avoiding surface allocation"
```

Then run:

```bash
python scripts/run_recurrence_avoiding_surface_allocation_exploration.py
```

## 23. Expected generated controls

```text
data/controls/ciphers/recurrence_avoiding_surface_allocation_v1/
  <source>/
    balanced16/
      seed_<seed>/
        corpus.jsonl
        key.json
        summary.json
        SHA256SUMS
```

## 24. Expected result outputs

```text
results/controls/recurrence_avoiding_surface_allocation_exploration/v1/
├── run_relation_matrix.csv
├── parent_lineage_verification.csv
├── collision_comparison.csv
├── collision_language_summary.csv
├── mechanism_diagnostics.csv
├── language_relation_summary.csv
├── relation_replication_summary.csv
├── global_relation_summary.csv
├── summary.json
├── SHA256SUMS
└── runs/
```

Inspect in this order:

```text
1. parent_lineage_verification.csv
2. collision_language_summary.csv
3. language_relation_summary.csv
4. relation_replication_summary.csv
5. global_relation_summary.csv
6. summary.json
```

## 25. Interpretation outcomes

### If exact-surface recurrence falls and the token relation becomes robust

Then the Phase-3A.D1 localization has identified a causally useful intervention
in the synthetic controls.

If all other four relations also remain robust, stop mechanism development.

Do not access reserved data.

Write and freeze Phase 3B first.

### If exact-surface recurrence falls but the token relation still fails

Then exact-token recurrence explains substantial local likelihood advantage but
is not sufficient to recreate the full Voynich predictive fingerprint.

Do not intensify balancing post hoc.

A different mechanism hypothesis is required.

### If exact-surface recurrence does not fall

Then the intervention or parent comparison has failed mechanically and the
predictive result should not be interpreted.

### If German/Italian regress

Then collision avoidance may be overcorrecting or disturbing the useful local
structure even if Latin/French improve.

The frozen full-hierarchy rule decides whether the overall mechanism survives.

## 26. Confirmatory boundary

Phase 3A.6 remains exploratory even if it reproduces all five relations.

A successful development result does **not** authorize immediate reserved-test
access.

The next step would be:

```text
write Phase 3B protocol
freeze candidate mechanism
freeze confirmatory decision rule
freeze reserved-test procedure
commit/tag
only then access reserved controls
```

Voynich `test_LOCKED` remains untouched beyond that as well.
