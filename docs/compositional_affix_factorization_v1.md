# Phase 3A.5 — Compositional Affix Factorization

**Experiment ID:** `compositional-affix-factorization-exploration-v1`  
**Status:** Exploratory  
**Repository destination:** `docs/compositional_affix_factorization_v1.md`

## 1. Why this experiment exists

Phase 3A.4 extended the base-form-preserving suffix mechanism to:

```text
K = 4, 8, 16
```

while keeping:

```text
coverage       = 100%
base form      = preserved
token length   = base + one suffix glyph
suffix choice  = occurrence-level and independent of plaintext word identity
```

The global token-Markov-minus-matched-trigram trajectory was:

```text
K=4   -0.128436 bits/event
K=8   -0.038280 bits/event
K=16  +0.028097 bits/event
```

At K=16, the other four Level-2 relations remained strong in 20/20 runs:

```text
HMM space-free    +0.3631
HMM token-aware   +0.3680
slot grammar      +0.5042
copy-edit         +1.4480
```

The token-Markov relation, however, reproduced the frozen strong-direction rule
in only two of four languages:

```text
French     0/5 strong
German     5/5 strong
Italian    5/5 strong
Latin      0/5 strong
```

The frozen Level-2 robustness rule requires a relation to replicate in at
least 3/4 languages, where a language-level relation replicates when at least
4/5 seeds are strong.

Therefore Phase 3A.4 was a strong development near-miss rather than a
confirmatory candidate.

Phase 3A.5 asks a narrower mechanism question:

> Holding exact whole-token multiplicity fixed at 16, can the same 16 variant
> identities be represented in a more compositionally structured two-glyph
> affix system such that local character prediction improves without restoring
> excessive exact-token predictability?

No reserved data are used.

## 2. Files

Install these five new files:

```text
configs/mechanisms/compositional_affix_factorization_v1.json
src/ciphers/compositional_affix_factorization.py
scripts/run_compositional_affix_factorization_exploration.py
tests/test_compositional_affix_factorization.py
docs/compositional_affix_factorization_v1.md
```

Do not edit the frozen Phase-3A.4 implementation.

## 3. Frozen parent

Phase 3A.5 is explicitly descended from the completed Phase-3A.4 mechanism.

The expected frozen parent hashes are:

```text
configs/mechanisms/base_form_suffix_extended_v1.json
514f440de8f086f0a5d9292f85289930de98fbc0893251e345ed134ef278b1f8

src/ciphers/base_form_suffix_extended.py
3969000c5ca678a68f259f2967d4164f25dc1360b7bba0e945d619704d12b345
```

The runner refuses to execute if either parent hash differs.

## 4. What is held fixed

Every Phase-3A.5 condition has:

```text
whole-token variants     = 16
suffix length             = 2 glyphs
coverage                  = 100% TRAIN/VALIDATION
base encoding             = exact Phase-3A.4 inherited monoalphabetic mapping
occurrence variant index  = exact Phase-3A.4 K=16 schedule
plaintext identity        = excluded from variant choice
```

Thus Phase 3A.5 does **not** increase K beyond 16.

It changes the internal representation of the same 16 abstract surface
variants.

## 5. Exact Phase-3A.4 variant schedule

For every TRAIN or VALIDATION token occurrence, Phase 3A.5 uses the same
16-way abstract variant index as Phase 3A.4 K=16:

```text
stable_uint64(
    "base-form-suffix-choice-v1",
    seed,
    line_index,
    token_index,
) % 16
```

This means that for a given source, seed, line, and token position:

```text
Phase 3A.4 K=16 abstract variant index
==
Phase 3A.5 abstract variant index
```

Only the glyph-level representation of that index changes.

The runner verifies this lineage against the existing frozen Phase-3A.4 K=16
generated controls.

## 6. The three factorization conditions

Every token receives two affix slots:

```text
BASE + A + B
```

The same 16 abstract indices are mapped bijectively to pairs in three ways.

### 1 × 16

```text
A1 B01
A1 B02
...
A1 B16
```

There is one possible state in slot A and sixteen in slot B.

### 2 × 8

```text
A1 B01 ... A1 B08
A2 B01 ... A2 B08
```

There are two possible states in slot A and eight in slot B.

### 4 × 4

```text
A1 B01 A1 B02 A1 B03 A1 B04
A2 B01 A2 B02 A2 B03 A2 B04
A3 B01 A3 B02 A3 B03 A3 B04
A4 B01 A4 B02 A4 B03 A4 B04
```

There are four possible states in each slot.

All conditions therefore have:

```text
number of unique affix pairs = 16
```

and every abstract variant index has exactly one affix-pair representation.

## 7. Mapping rule

For a factorization:

```text
R × C
```

where:

```text
R * C = 16
```

and abstract variant index:

```text
i ∈ [0, 15]
```

the pair is:

```text
slot_a_index = floor(i / C)
slot_b_index = i mod C
```

The map is invertible.

Therefore exact surface-token equality remains a one-to-one function of:

```text
encoded base
+
abstract 16-way variant identity
```

in every condition.

This is an important experimental control.

## 8. Exact-token identity partition is held fixed

Because every condition maps the same abstract variant identity bijectively to
one unique affix pair:

```text
same plaintext base + same variant index
    -> same exact ciphertext token

same plaintext base + different variant index
    -> different exact ciphertext token
```

for all three factorizations.

Therefore these quantities should be exactly invariant across factorization for
a given source/seed/split:

```text
abstract token-identity fingerprint
unique exact ciphertext-token count
exact ciphertext-token recurrence fraction
same-surface relation induced by the 16-way variant schedule
```

The runner verifies this.

This means Phase 3A.5 is not simply changing how many exact word forms exist.

It is changing how those same exact identities are decomposed into glyph-level
structure.

## 9. Base-form preservation

The base portion remains the exact Phase-3A.4 TRAIN-fitted monoalphabetic
encoding.

Example:

```text
plaintext:
dominus

base:
M04 M11 M03 M08 M06 M14 M09
```

A Phase-3A.5 token is:

```text
M04 M11 M03 M08 M06 M14 M09 A2 B3
```

Removing the final two glyphs must recover the same encoded base for every
occurrence.

Required:

```text
base-form recovery accuracy      = 1.0
base-form preservation fraction  = 1.0
```

## 10. Token-length control

All three conditions append exactly two glyphs:

```text
1x16  BASE + A + B
2x8   BASE + A + B
4x4   BASE + A + B
```

Therefore:

```text
ciphertext token length = encoded base length + 2
```

for every condition.

This avoids confounding factorization with token length.

## 11. Information-theoretic caution

The three factorizations do **not** reduce the nominal amount of random variant
identity.

Under ideal uniform use of all 16 variant indices:

```text
H(variant index) = log2(16) = 4 bits
```

for every condition.

For example:

```text
1x16:
H(A) + H(B|A) = 0 + 4 = 4 bits

2x8:
H(A) + H(B|A) = 1 + 3 = 4 bits

4x4:
H(A) + H(B|A) = 2 + 2 = 4 bits
```

Thus a favorable result should not be described as "factorization reduced
variant entropy."

Instead, Phase 3A.5 tests whether the **geometry of the representation** changes
the behavior of finite-order/local predictive models while exact whole-token
identity is held fixed.

This distinction must be retained in interpretation.

## 12. Experimental grid

The frozen exploratory grid is:

```text
factorization = [1x16, 2x8, 4x4]
```

with:

```text
4 languages
x 3 factorizations
x 5 frozen seeds
= 60 controls
```

Each control receives all five frozen Level-2 predictive comparisons:

```text
60 x 5 = 300 relation outcomes
```

No additional factorization may be added after seeing the results.

## 13. Primary exploratory prediction

The development prediction is:

```text
token-Markov minus matched-trigram delta

1x16   lowest
2x8    intermediate
4x4    highest
```

while:

```text
hmm_space_free
hmm_token_aware
slot_grammar
copy_edit
```

remain in the positive Voynich direction.

This prediction is explicitly exploratory. It is not an information-theoretic
necessity because all conditions retain four nominal bits of variant identity.

## 14. Frozen Level-2 replication rule

Phase 3A.5 reports the already-frozen Level-2 directional replication rule.

A language/relation replicates when:

```text
>= 4/5 seeds
```

are strong direction matches.

A relation is robust when it replicates in:

```text
>= 3/4 languages
```

A factorization reproduces the full directional hierarchy only when:

```text
all five relations are robust
```

This reporting does not automatically promote a factorization to confirmatory
status.

Even if a development condition reaches the rule, Phase 3B must be written and
frozen before reserved-test use.

## 15. Diagnostics

For every source/factorization/seed, the generator records:

- total tokens;
- factorization dimensions;
- 16 nominal whole-token variants;
- suffix length of two;
- 100% mechanism coverage;
- unique plaintext word types;
- unique ciphertext tokens;
- exact ciphertext-token recurrence fraction;
- glyph-alphabet size;
- round-trip accuracy;
- base-form recovery accuracy;
- base-form preservation fraction;
- token-length-plus-two accuracy;
- exact Phase-3A.4 K=16 variant-schedule accuracy;
- realized surface forms per plaintext word;
- same-surface probability conditional on the same plaintext word;
- realized abstract variant indices;
- realized affix pairs;
- slot-A entropy;
- slot-B entropy;
- joint affix-pair entropy;
- abstract variant-index entropy;
- canonical abstract token-identity SHA-256 fingerprint.

Required invariants:

```text
round-trip accuracy                        = 1.0
base-form recovery accuracy                = 1.0
base-form preservation fraction            = 1.0
token-length-plus-two accuracy              = 1.0
Phase-3A.4 K=16 variant-schedule accuracy  = 1.0
```

## 16. Parent-lineage verification

The runner checks every one of the 60 runs against the frozen Phase-3A.4 K=16
controls.

For each source/seed/factorization it verifies:

```text
base mapping == Phase 3A.4
parent master-key fingerprint == Phase 3A.4
TRAIN abstract base+variant-index sequence == Phase 3A.4 K=16
VALIDATION abstract base+variant-index sequence == Phase 3A.4 K=16
```

The output is:

```text
parent_lineage_verification.csv
```

A lineage mismatch is a hard error.

## 17. Factorization invariance verification

Within every source/seed, the runner also requires that all three
factorizations have identical:

```text
TRAIN abstract token-identity fingerprint
VALIDATION abstract token-identity fingerprint
TRAIN unique exact-token count
VALIDATION unique exact-token count
TRAIN exact-token recurrence fraction
VALIDATION exact-token recurrence fraction
```

This protects the intended causal interpretation:

> exact whole-token identity structure is held fixed while glyph-level
> factorization changes.

## 18. Frozen evaluator

No new predictive model is introduced.

Phase 3A.5 reuses:

```text
configs/evaluation/level2_predictive_controls_v1.json
src/evaluation/level2_predictive_controls.py
```

and the five frozen relations:

```text
hmm_space_free
hmm_token_aware
slot_grammar
token_markov
copy_edit
```

with their existing matched trigram comparators, selection rules, restarts, and
paired bootstrap procedure.

## 19. Data protection

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

Reserved source lines remain opaque:

```json
{"line_index": 123, "reserved_test_opaque": true}
```

No reserved ciphertext is generated.

## 20. Installation

Place the files at:

```text
configs/mechanisms/compositional_affix_factorization_v1.json
src/ciphers/compositional_affix_factorization.py
scripts/run_compositional_affix_factorization_exploration.py
tests/test_compositional_affix_factorization.py
docs/compositional_affix_factorization_v1.md
```

Then run:

```bash
source .venv/bin/activate

python -m pytest -q tests/test_compositional_affix_factorization.py
python -m pytest -q
```

The dedicated tests verify:

- exact factorization grid;
- inherited Phase-3A.4 base mapping;
- deterministic factor-state ordering;
- 16 unique invertible pairs in every condition;
- shared factor-state prefixes;
- exact Phase-3A.4 K=16 occurrence schedule;
- complete base preservation;
- fixed two-glyph affix length;
- exclusion of plaintext identity from variant choice;
- identical abstract variant index across factorizations;
- realization of all 16 variants;
- invariant abstract exact-token partition;
- invariant same-surface probability;
- deterministic generation;
- reserved-test opacity;
- validation-unseen character reversibility;
- required diagnostics;
- compatibility with the frozen Level-2 reader;
- rejection of non-frozen post-hoc factorizations.

## 21. Freeze before execution

After tests pass, freeze the Phase-3A.5 protocol before seeing results:

```bash
git add \
  configs/mechanisms/compositional_affix_factorization_v1.json \
  src/ciphers/compositional_affix_factorization.py \
  scripts/run_compositional_affix_factorization_exploration.py \
  tests/test_compositional_affix_factorization.py \
  docs/compositional_affix_factorization_v1.md

git commit -m "Freeze Phase 3A.5 compositional affix exploration"
```

Only after that commit run:

```bash
python scripts/run_compositional_affix_factorization_exploration.py
```

## 22. Outputs

Generated controls:

```text
data/controls/ciphers/compositional_affix_factorization_v1/
  <source>/
    F1x16/
      seed_<seed>/
    F2x8/
      seed_<seed>/
    F4x4/
      seed_<seed>/
```

Results:

```text
results/controls/compositional_affix_factorization_exploration/v1/
├── run_relation_matrix.csv
├── mechanism_diagnostics.csv
├── parent_lineage_verification.csv
├── factorization_diagnostics_summary.csv
├── factorization_language_relation_summary.csv
├── factorization_global_relation_summary.csv
├── factorization_replication_summary.csv
├── summary.json
├── SHA256SUMS
└── runs/
```

Inspect first:

```text
parent_lineage_verification.csv
factorization_replication_summary.csv
factorization_global_relation_summary.csv
summary.json
```

## 23. Interpretation boundary

### If one factorization reaches the inherited full hierarchy

That would mean all five frozen directional relations are robust under the
already-defined Level-2 rule on development validation.

Stop exploratory mechanism tuning.

Do not access reserved data yet.

Write and commit a separate Phase-3B confirmatory protocol that freezes the
candidate mechanism, factorization, seeds, evaluator, decision rule, and
reserved-test procedure.

### If no factorization reaches the full hierarchy

Do not add 8x2, 16x1, longer suffixes, or another condition to Phase 3A.5 after
seeing the result.

Retain the negative or partial result and formulate a new exploratory
mechanism.

### If the token relation changes while exact-token identity diagnostics remain identical

That is evidence that the predictive difference comes from glyph-level
representation rather than a change in the number or recurrence structure of
whole-token identities.

### If the three conditions are nearly identical

That is also informative. Because all three retain four bits of nominal
variant identity, a null factorization effect would indicate that simple
Cartesian decomposition alone does not provide the missing structural
constraint.

## 24. Confirmatory boundary

Phase 3A.5 remains exploratory regardless of the result.

Neither the control reserved test nor the Voynich locked test is accessed by
this phase.
