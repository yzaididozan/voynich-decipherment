# Phase 3A.3 — Base-Form-Preserving Surface Variation

**Experiment ID:** `base-form-suffix-variation-exploration-v1`  
**Status:** Exploratory  
**Repository destination:** `docs/base_form_suffix_variation_v1.md`

## 1. Motivation

Phase 3A and Phase 3A.2 narrowed the mechanism question.

Phase 3A showed that increasing the number of related surface variants
available to a fixed set of plaintext words consistently weakened the
predictive usefulness of exact ciphertext-token identity:

```text
mean token-Markov minus matched-trigram delta

K=1  -0.408859
K=2  -0.371439
K=4  -0.329034
```

The effect was directionally consistent across all four historical-language
controls and five frozen seeds.

However, the Phase-3A mechanism applied only to the top 64 TRAIN word types.
A follow-up coverage audit showed that those words represented only about
36%–48% of validation tokens.

Phase 3A.2 therefore held K=4 fixed and increased the number of words receiving
arbitrary lexical code families:

```text
N = 64, 256, 1024
```

That coverage intervention did not produce the predicted monotone movement:

```text
N=64    -0.384154
N=256   -0.427142
N=1024  -0.412085
```

Thus merely applying arbitrary multi-code lexical identifiers to more of the
corpus is insufficient.

Phase 3A.3 tests the next mechanistic possibility:

> Can exact whole-token identity be weakened while preserving the complete
> internal character structure of each encoded word?

This is a post-hoc exploratory follow-up informed by the development results.
It does not alter Level 2 and it does not use either reserved test set.

## 2. Files

Install these five new files:

```text
configs/mechanisms/base_form_suffix_variation_v1.json
src/ciphers/base_form_suffix_variation.py
scripts/run_base_form_suffix_variation_exploration.py
tests/test_base_form_suffix_variation.py
docs/base_form_suffix_variation_v1.md
```

Do not edit the frozen Phase-3A or Phase-3A.2 implementations to perform this
experiment.

## 3. Mechanism

Every plaintext token receives two components:

```text
TRAIN-fitted monoalphabetic encoded base
+
exactly one suffix glyph
```

Example:

```text
plaintext:
dominus

encoded base:
M04 M11 M03 M08 M06 M14 M09

K=1 possible surface:
M04 M11 M03 M08 M06 M14 M09 V1

K=2 possible surfaces:
M04 M11 M03 M08 M06 M14 M09 V1
M04 M11 M03 M08 M06 M14 M09 V2

K=4 possible surfaces:
M04 M11 M03 M08 M06 M14 M09 V1
M04 M11 M03 M08 M06 M14 M09 V2
M04 M11 M03 M08 M06 M14 M09 V3
M04 M11 M03 M08 M06 M14 M09 V4
```

The complete encoded base remains unchanged across occurrences and across K.

Only the identity of the final suffix glyph varies.

## 4. Experimental grid

The exploratory grid is:

```text
K = 1, 2, 4
```

The complete experiment is:

```text
4 historical-language controls
x 3 K conditions
x 5 frozen seeds
= 60 synthetic controls
```

Each receives the five frozen Level-2 predictive comparisons:

```text
60 x 5 = 300 relation-level outcomes
```

## 5. Full token coverage

Unlike Phase 3A and Phase 3A.2, Phase 3A.3 has no lexical target-count
parameter.

Every TRAIN and VALIDATION token receives:

```text
encoded base + one suffix
```

Therefore mechanism coverage is:

```text
TRAIN       100%
VALIDATION  100%
```

A validation word type need not have appeared in TRAIN. As long as its
characters were observed in TRAIN, its base is encoded using the frozen
TRAIN-fitted character map.

If VALIDATION contains a character absent from TRAIN, the character is encoded
as a deterministic reversible Unicode-codepoint glyph. No mapping is fit from
validation.

## 6. Token-length control

Token length must not confound K.

Therefore K=1 does **not** mean "base with no suffix."

Every K condition appends exactly one glyph:

```text
K=1  base + one of one allowed suffix
K=2  base + one of two allowed suffixes
K=4  base + one of four allowed suffixes
```

For every ciphertext token in every condition:

```text
ciphertext length = encoded-base length + 1
```

The generator verifies this mechanically on TRAIN and VALIDATION.

## 7. Shared master key across K

For each source language and seed, the generator fits one TRAIN-only master key
containing:

```text
one monoalphabetic character substitution
one ordered set of four suffix glyphs
```

The K conditions activate nested prefixes:

```text
K=1  suffixes[0:1]
K=2  suffixes[0:2]
K=4  suffixes[0:4]
```

The base mapping and four-suffix master order therefore remain identical across
K for a fixed source/seed.

The key contains a deterministic:

```text
master_key_sha256
```

The runner verifies that this fingerprint is invariant across K before writing
the aggregate result.

## 8. Suffix choice deliberately excludes word identity

The suffix is intended to create occurrence-level surface variation rather than
a second lexical code.

The suffix choice is therefore derived from:

```text
seed
line_index
token_index
```

and **not** from:

```text
plaintext token identity
```

Conceptually:

```text
choice = stable_hash(
    seed,
    line_index,
    token_index
) % K
```

Thus two different plaintext words occupying the same occurrence coordinates
under the same key would receive the same suffix choice.

The suffix itself does not encode which word is present.

## 9. Structural property being tested

Consider two occurrences of one plaintext word.

At K=4 they may become:

```text
M2 M8 M4 V1
M2 M8 M4 V4
```

An exact-token model treats these as different whole-token identities.

A character model still observes the repeated internal base:

```text
M2 M8 M4
```

This creates the structural asymmetry Phase 3A.3 is designed to test:

```text
weaken exact surface-token identity
while
preserving complete local base-form structure
```

That differs fundamentally from Phase 3A.2, where increasing coverage replaced
more words with arbitrary lexical identifiers.

## 10. Primary exploratory prediction

The main outcome remains:

```text
token-Markov bits/event - matched-trigram bits/event
```

The predicted K trajectory is:

```text
K=1   most negative
K=2   less negative
K=4   least negative / potentially zero or positive
```

At the same time, the other four Voynich-direction comparisons should ideally
remain positive:

```text
trigram > HMM space-free
trigram > HMM token-aware
trigram > finite-template slot grammar
trigram > local copy-edit
```

The scientific question is not merely whether the token-Markov relation moves.
It is whether it moves **without sacrificing the local predictive fingerprint**.

No scalar "Voynich similarity score" is calculated.

## 11. Mechanism diagnostics

For every source/K/seed run, the generator records:

- total tokens;
- mechanism coverage fraction;
- unique plaintext word types;
- unique ciphertext tokens;
- exact ciphertext-token recurrence fraction;
- ciphertext glyph-alphabet size;
- round-trip decoding accuracy;
- base-form recovery accuracy;
- base-form preservation fraction;
- number of plaintext types with multiple observed base forms;
- token-length-plus-one accuracy;
- mean realized surface forms per plaintext word;
- maximum realized surface forms per plaintext word;
- same-surface probability conditional on the same plaintext word;
- active suffix count;
- suffix counts;
- suffix entropy;
- normalized suffix entropy.

Three invariants are required:

```text
base-form recovery accuracy        = 1.0
base-form preservation fraction    = 1.0
token-length-plus-one accuracy      = 1.0
```

For frequently repeated words, the expected same-surface probabilities are
approximately:

```text
K=1  1.00
K=2  0.50
K=4  0.25
```

Sampling variation is expected for sparse words.

## 12. Frozen predictive evaluator

Phase 3A.3 defines no new predictive competitor.

It reuses the frozen Level-2 evaluator for:

```text
hmm_space_free
hmm_token_aware
slot_grammar
token_markov
copy_edit
```

including the existing:

- matched trigram comparators;
- model-selection rules;
- HMM restart rules;
- slot-grammar candidate grid;
- token-Markov selection;
- copy-edit selection;
- paired bootstrap procedure.

Required pre-existing project files include:

```text
configs/evaluation/level2_predictive_controls_v1.json
src/evaluation/level2_predictive_controls.py
data/controls/splits/level2_v1/
data/controls/languages_open_v1/
```

## 13. Test protection

Allowed:

```text
CONTROL TRAIN
CONTROL VALIDATION
```

Prohibited:

```text
CONTROL RESERVED TEST
Voynich test_LOCKED
```

Reserved source lines are never encoded.

The generated JSONL preserves one physical row per original source line, but a
reserved row contains only:

```json
{"line_index": 123, "reserved_test_opaque": true}
```

No reserved ciphertext token sequence is generated.

## 14. Installation and freeze-before-results rule

Place the five files at their intended paths.

Run:

```bash
source .venv/bin/activate

python -m pytest -q tests/test_base_form_suffix_variation.py
python -m pytest -q
```

The dedicated tests verify, among other things:

- deterministic TRAIN-only master-key construction;
- seed sensitivity;
- nested K suffix sets;
- base preservation at K=1,2,4;
- exactly one appended suffix for all K;
- suffix choice independence from plaintext token identity;
- K=1 fixed surface identity;
- K=4 realization of multiple surfaces;
- identical base mapping and master suffix order across K;
- reserved-test opacity;
- validation-unseen character reversibility;
- 100% TRAIN/VALIDATION mechanism coverage;
- decreasing same-surface probability as K increases on repeated synthetic
  words;
- compatibility with the frozen Level-2 JSONL reader;
- well-formed suffix-entropy diagnostics.

If all tests pass, commit the protocol **before running the experiment**:

```bash
git add \
  configs/mechanisms/base_form_suffix_variation_v1.json \
  src/ciphers/base_form_suffix_variation.py \
  scripts/run_base_form_suffix_variation_exploration.py \
  tests/test_base_form_suffix_variation.py \
  docs/base_form_suffix_variation_v1.md

git commit -m "Freeze Phase 3A.3 base-form suffix exploration"
```

Only after that commit run:

```bash
python scripts/run_base_form_suffix_variation_exploration.py
```

## 15. Outputs

Synthetic controls:

```text
data/controls/ciphers/base_form_suffix_variation_v1/
  <source>/
    K1/
      seed_<seed>/
    K2/
      seed_<seed>/
    K4/
      seed_<seed>/
```

Each generated control contains:

```text
corpus.jsonl
key.json
summary.json
SHA256SUMS
```

Exploratory results:

```text
results/controls/base_form_suffix_variation_exploration/v1/
├── run_relation_matrix.csv
├── mechanism_diagnostics.csv
├── suffix_diagnostics_summary.csv
├── k_language_relation_summary.csv
├── k_global_relation_summary.csv
├── summary.json
├── SHA256SUMS
└── runs/
```

Inspect first:

```text
suffix_diagnostics_summary.csv
k_global_relation_summary.csv
summary.json
```

## 16. Interpretation

### Outcome A — token relation improves while the other four remain stable

This would support the hypothesis that **surface multiplicity plus preserved
sublexical structure** can reproduce an important part of the Voynich
predictive fingerprint.

Do not immediately use a reserved test. A separate Phase-3B protocol must
first freeze the candidate K, mechanism, parameters, and decision rule.

### Outcome B — token relation improves but remains substantially negative

Then base preservation is useful but still insufficient. Retain the result and
identify the next single mechanism dimension rather than layering arbitrary
operations.

### Outcome C — token relation improves but another predictive relation fails

Then the experiment exposes a new tradeoff. The mechanism suppresses exact
token identity at the cost of some other structural property required by the
Voynich fingerprint.

### Outcome D — little or no K trajectory

Then appending independent suffix-like variation to a preserved base is not
sufficient to explain the Phase-3A effect. That would argue against simple
occurrence-level affix noise as the missing mechanism.

## 17. Confirmatory boundary

Phase 3A.3 remains exploratory regardless of how favorable its validation
results appear.

No K becomes confirmatory simply because it resembles the Voynich validation
fingerprint.

A confirmatory stage requires a separately written and committed Phase-3B
protocol before any reserved control-test data are encoded, parsed, or scored.
The Voynich locked test remains protected by the broader project freeze.
