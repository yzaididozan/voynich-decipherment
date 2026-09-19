# Phase 3A.4 — Extended Base-Form Surface Multiplicity

**Experiment ID:** `base-form-suffix-extended-exploration-v1`  
**Status:** Exploratory  
**Repository destination:** `docs/base_form_suffix_extended_v1.md`

## 1. Why this experiment exists

Phase 3A.3 was the first intervention in the VOYAGER mechanism-discovery
sequence to selectively improve the previously failing exact-token relation
without disrupting the other four predictive relations.

Its global mean validation deltas were:

```text
                         K=1       K=2       K=4

HMM space-free          +0.4380    +0.4321    +0.4182
HMM token-aware         +0.3869    +0.3818    +0.3790
slot grammar            +0.5102    +0.5116    +0.5086
token Markov            -0.3812    -0.2462    -0.1284
copy-edit               +1.3797    +1.4054    +1.4272
```

For every K, the four non-token relations were strong in 20/20 runs.

At the same time, the token-Markov relation moved monotonically toward the
Voynich direction:

```text
K=1  -0.381229 bits/event
K=2  -0.246202 bits/event
K=4  -0.128436 bits/event
```

The Voynich validation reference for this relation is positive:

```text
Voynich token-Markov minus matched-trigram delta
≈ +0.127077 bits/event
```

Phase 3A.4 therefore extends the **same mechanism** rather than introducing a
new one.

The single scientific question is:

> Does additional base-form-preserving surface multiplicity continue the
> Phase-3A.3 trajectory far enough to enter the positive Voynich direction
> while leaving the other four predictive relations intact?

This remains exploratory. No reserved test is used.

## 2. New files

Install these five new files:

```text
configs/mechanisms/base_form_suffix_extended_v1.json
src/ciphers/base_form_suffix_extended.py
scripts/run_base_form_suffix_extended_exploration.py
tests/test_base_form_suffix_extended.py
docs/base_form_suffix_extended_v1.md
```

Do not edit the frozen Phase-3A.3 files to implement this experiment.

## 3. Frozen parent requirement

Phase 3A.4 is explicitly bridged to the completed Phase-3A.3 mechanism.

The expected parent files and hashes are:

```text
configs/mechanisms/base_form_suffix_variation_v1.json
SHA256:
948fb11eb86903ca6b7661674c27180ad3a7cc4a1ae7ac60b46de19fd7574fa5

src/ciphers/base_form_suffix_variation.py
SHA256:
197a6bfaa67339b57173908ee3d5767537d532f06ad72ab00b4a36f4faa671f8
```

The Phase-3A.4 runner refuses to proceed if either parent hash differs.

This protects the meaning of the K=4 bridge.

## 4. Experimental grid

The exploratory grid is:

```text
K = 4, 8, 16
```

The complete design is:

```text
4 historical-language controls
x 3 K conditions
x 5 frozen seeds
= 60 synthetic controls
```

Each synthetic control receives the same five frozen Level-2 predictive
comparisons:

```text
60 x 5 = 300 relation-level outcomes
```

There is no K value above 16 in this experiment.

## 5. Mechanism held fixed

Every plaintext token is represented as:

```text
complete TRAIN-fitted monoalphabetic encoded base
+
exactly one suffix glyph
```

Example:

```text
plaintext:
dominus

encoded base:
M04 M11 M03 M08 M06 M14 M09

possible surface:
M04 M11 M03 M08 M06 M14 M09 V....
```

The entire encoded base remains unchanged.

The only quantity being extended is the number of available final suffix
identities.

Coverage remains:

```text
TRAIN       100%
VALIDATION  100%
```

Token length remains:

```text
ciphertext token length = encoded base length + 1
```

for K=4, K=8, and K=16.

## 6. The K=4 bridge is mandatory

The K=4 condition in Phase 3A.4 is not merely intended to be statistically
similar to Phase 3A.3.

It must be **byte-for-byte identical** at the generated `corpus.jsonl` level.

For the same:

```text
source
TRAIN/VALIDATION split
seed
line index
token index
```

the Phase-3A.4 K=4 ciphertext must be exactly the same as the frozen Phase-3A.3
K=4 ciphertext.

The runner verifies this for all:

```text
4 sources x 5 seeds = 20 bridge comparisons
```

before writing final aggregate results.

A bridge failure is a hard error.

The final output contains:

```text
k4_bridge_verification.csv
```

with the parent and Phase-3A.4 corpus hashes.

## 7. How the master suffix set is extended

Phase 3A.3 constructed one TRAIN-only base-character map and one ordered
four-suffix set per source/seed.

Phase 3A.4 obtains that exact parent key using the frozen Phase-3A.3
constructor.

The first four suffixes are therefore inherited unchanged:

```text
Phase-3A.4 suffixes[0:4]
==
Phase-3A.3 master_suffixes
```

Twelve new suffix labels are then added:

```text
V0005 ... V0016
```

Their order is determined by a separate deterministic extension hash:

```text
stable_hash(
    "base-form-suffix-extension-v1",
    seed,
    suffix_label
)
```

This extension operation cannot alter:

- the Phase-3A.3 monoalphabetic base map;
- the first four suffix identities;
- the first four suffix order;
- the Phase-3A.3 occurrence-level suffix-choice hash.

The resulting 16-suffix master inventory is shared across K:

```text
K=4   first 4
K=8   first 8
K=16  first 16
```

## 8. Exact occurrence-choice continuity

Phase 3A.4 retains the exact Phase-3A.3 occurrence-level choice rule:

```text
stable_uint64(
    "base-form-suffix-choice-v1",
    seed,
    line_index,
    token_index,
) % K
```

The hash namespace is unchanged.

The input tuple is unchanged.

Plaintext token identity remains excluded.

Therefore at K=4 the selected suffix is exactly the Phase-3A.3 selected suffix.

At K=8 and K=16, only the modulo range expands.

## 9. Why K=4, 8, and 16

Phase 3A.3 produced:

```text
K=1  -0.381229
K=2  -0.246202
K=4  -0.128436
```

The movement from K=1 to K=2 was approximately:

```text
+0.135 bits/event
```

and from K=2 to K=4 approximately:

```text
+0.118 bits/event
```

Those development results justify testing whether the same mechanism continues
toward or through zero.

They do **not** justify extrapolating that it must do so.

K=8 tests the next doubling.

K=16 provides one additional predeclared doubling to identify either continued
movement or a plateau/tradeoff.

The experiment stops there.

## 10. Primary exploratory prediction

The main response remains:

```text
token-Markov bits/event - matched-trigram bits/event
```

The predicted trajectory is:

```text
K=4    negative baseline inherited from Phase 3A.3
K=8    less negative / possibly around zero
K=16   less negative still / possibly positive
```

The mechanism is only structurally promising if the other four
Voynich-direction relations continue to remain positive:

```text
trigram > HMM space-free
trigram > HMM token-aware
trigram > finite-template slot grammar
trigram > local copy-edit
```

The key event of interest is therefore not merely a positive token-Markov
delta.

It is the conjunction:

```text
token-Markov delta enters the Voynich direction
AND
the other four directional relations remain intact
```

This is still an exploratory result, not a confirmatory success criterion.

## 11. No magnitude-based winner score

The frozen Level-2 hierarchy criterion was based on relation direction and
bootstrap confidence, not a preregistered scalar distance from Voynich.

Phase 3A.4 therefore does not invent a post-hoc aggregate similarity score.

It reports:

- each predictive delta;
- language-level behavior;
- seed-level behavior;
- strong-direction counts;
- mechanism diagnostics;
- K=4 bridge verification;
- whether the token-Markov global mean trajectory is monotone;
- whether its global mean becomes positive.

Magnitude comparisons to Voynich may be discussed descriptively but do not
replace the frozen relation-level evidence.

## 12. Diagnostics

For each source/K/seed run, Phase 3A.4 records:

- 100% mechanism coverage;
- exact round-trip accuracy;
- base-form recovery accuracy;
- base-form preservation fraction;
- token-length-plus-one accuracy;
- exact ciphertext-token recurrence fraction;
- unique ciphertext-token count;
- mean realized surface forms per plaintext word;
- same-surface probability conditional on the same plaintext word;
- active suffix count;
- suffix counts;
- suffix entropy;
- normalized suffix entropy;
- extended master-key fingerprint;
- parent Phase-3A.3 master-key fingerprint.

Required invariants are:

```text
round-trip accuracy                  = 1.0
base-form recovery accuracy          = 1.0
base-form preservation fraction      = 1.0
token-length-plus-one accuracy        = 1.0
K=4 Phase-3A.3 corpus byte identity   = TRUE
```

For frequently repeated words, same-surface probability should approximately
decline with K:

```text
K=4    ≈ 1/4
K=8    ≈ 1/8
K=16   ≈ 1/16
```

Sampling variation is expected for sparse words.

## 13. Frozen evaluator

Phase 3A.4 introduces no new predictive model.

It reuses:

```text
configs/evaluation/level2_predictive_controls_v1.json
src/evaluation/level2_predictive_controls.py
```

and the five frozen Level-2 relations:

```text
hmm_space_free
hmm_token_aware
slot_grammar
token_markov
copy_edit
```

with the existing matched trigram comparators, model selection, deterministic
restart rules, and paired bootstrap procedure.

## 14. Test protection

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

Reserved source lines are never encoded.

Their generated rows remain only opaque placeholders such as:

```json
{"line_index": 123, "reserved_test_opaque": true}
```

No reserved ciphertext token sequence is produced.

## 15. Stop rule

Phase 3A.4 predeclares a hard exploratory ceiling:

```text
K <= 16
```

Do not respond to the result by adding K=32, K=64, or another larger value
inside this experiment.

If K=8/16:

- plateaus below zero;
- reverses;
- or fixes token Markov while damaging another relation;

retain that result and change the mechanism hypothesis.

Do not keep increasing K until the desired sign appears.

## 16. Installation

Place the files at:

```text
configs/mechanisms/base_form_suffix_extended_v1.json
src/ciphers/base_form_suffix_extended.py
scripts/run_base_form_suffix_extended_exploration.py
tests/test_base_form_suffix_extended.py
docs/base_form_suffix_extended_v1.md
```

Run:

```bash
source .venv/bin/activate

python -m pytest -q tests/test_base_form_suffix_extended.py
python -m pytest -q
```

The dedicated tests include explicit checks that:

- the Phase-3A.4 base map equals Phase 3A.3;
- the first four suffixes equal Phase 3A.3 in the same order;
- K=4 token generation equals Phase 3A.3;
- the complete generated K=4 row sequence equals Phase 3A.3;
- separately written K=4 `corpus.jsonl` files are byte-identical;
- the Phase-3A.3 suffix-choice hash namespace is preserved;
- K=4/8/16 suffix sets are nested;
- all K conditions preserve the encoded base;
- every token receives exactly one suffix;
- plaintext identity is excluded from suffix choice;
- K=4/8/16 can realize their intended surface multiplicities;
- the master key is invariant across K;
- reserved lines remain opaque;
- validation-unseen characters remain reversible without validation fitting;
- the frozen Level-2 reader can consume the generated corpus;
- K above 16 is rejected.

## 17. Freeze before execution

After the tests pass, freeze the Phase-3A.4 protocol **before seeing its
results**:

```bash
git add \
  configs/mechanisms/base_form_suffix_extended_v1.json \
  src/ciphers/base_form_suffix_extended.py \
  scripts/run_base_form_suffix_extended_exploration.py \
  tests/test_base_form_suffix_extended.py \
  docs/base_form_suffix_extended_v1.md

git commit -m "Freeze Phase 3A.4 extended base-form suffix exploration"
```

Only then run:

```bash
python scripts/run_base_form_suffix_extended_exploration.py
```

## 18. Outputs

Generated controls:

```text
data/controls/ciphers/base_form_suffix_extended_v1/
  <source>/
    K4/
      seed_<seed>/
    K8/
      seed_<seed>/
    K16/
      seed_<seed>/
```

Aggregate results:

```text
results/controls/base_form_suffix_extended_exploration/v1/
├── run_relation_matrix.csv
├── mechanism_diagnostics.csv
├── k4_bridge_verification.csv
├── suffix_diagnostics_summary.csv
├── k_language_relation_summary.csv
├── k_global_relation_summary.csv
├── summary.json
├── SHA256SUMS
└── runs/
```

Inspect first:

```text
k4_bridge_verification.csv
k_global_relation_summary.csv
summary.json
```

The bridge file must show 20/20 exact corpus matches before interpreting the
new K=8/K=16 results.

## 19. Interpretation boundary

### If K=8 or K=16 produces a positive token relation and preserves the other four

This would be the first tested mechanism family to enter the complete
Voynich-direction predictive hierarchy on development validation data.

Stop exploratory K extension.

Do not access the reserved test yet.

Write a separate Phase-3B protocol that freezes:

- the exact mechanism;
- chosen K;
- source controls;
- seeds;
- evaluator;
- replication rule;
- confirmatory decision rule;
- reserved-test access procedure.

Commit that protocol before reserved-test encoding or scoring.

### If the token relation remains negative but continues improving

Phase 3A.4 still stops at K=16.

A continued-but-insufficient trajectory is evidence that independent suffix
multiplicity contributes but does not fully explain the fingerprint.

Design a new structural intervention rather than increasing K again.

### If the trajectory plateaus or reverses

Treat simple independent suffix multiplicity as insufficient.

### If another relation breaks

Treat that as a mechanistic tradeoff. Do not optimize K solely for the
token-Markov sign.

## 20. Confirmatory boundary

Phase 3A.4 remains exploratory regardless of how compelling K=8 or K=16 looks.

No K becomes confirmatory from this runner.

The control reserved test and the Voynich locked test remain protected until a
separate confirmatory protocol is written and frozen.
