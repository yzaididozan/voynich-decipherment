# VOYAGER Phase 4.D1 — Space-Free HMM Failure Localization

**Experiment ID:** `phase4-space-free-hmm-localization-v1`  
**Status:** Exploratory, post-hoc locked-test diagnostic  
**Repository destination:** `docs/phase4_space_free_hmm_localization_v1.md`

## 1. Why this diagnostic exists

The locked Phase 4 Voynich evaluation ended with:

```text
PHASE 4: INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN
```

The five preregistered relations were:

```text
hmm_space_free     delta=+0.036929  CI=[-0.026250,+0.067946]  strong=NO
hmm_token_aware    delta=+0.083878  CI=[+0.030869,+0.111533]  strong=YES
slot_grammar       delta=+0.451959  CI=[+0.396709,+0.486926]  strong=YES
token_markov       delta=+0.110831  CI=[+0.087903,+0.145633]  strong=YES
copy_edit          delta=+1.120575  CI=[+0.948527,+1.214823]  strong=YES
```

The primary Phase 4 decision is permanently retained as a failure of the full
five-relation conjunction.

Phase 4.D1 asks only:

> Why did the space-free HMM relation point in the predicted positive direction
> but fail to exclude zero, while the token-aware HMM relation was strong?

This is a localization problem, not a new confirmation attempt.

## 2. Why the locked test is accessed again

The Phase 4 scorer saved the aggregate relation rows and bootstrap replicate
deltas, but it did **not** save the exact per-leaf HMM and matched-trigram score
rows.

Therefore the leaf-level discrepancy cannot be reconstructed solely from the
stored Phase 4 output files.

The Voynich test is no longer untouched: Phase 4 already accessed it. Phase
4.D1 may therefore reconstruct the **exact same frozen HMM scoring** on the
same test leaves for diagnostic purposes.

This does not create a second confirmatory test. The resulting analysis is
explicitly post-hoc and exploratory.

## 3. Frozen parent result

The diagnostic requires the original Phase 4 protocol:

```text
configs/phase_4_voynich_locked.yaml
SHA-256:
55b02cacb041872a20bc87db55d506a3d483da030c520553f77fac60ab9002db
```

and the exact Phase 4 locked scorer:

```text
src/evaluation/phase4_voynich_locked.py
SHA-256:
a3990a1e4d834b8b302cf7fe0244bd902887ece82c248f7982232686d6f50045
```

It also requires the recorded Phase 4 result to state:

```text
decision = INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN
locked_test_accessed = true
all_five_relations_strong = false
```

with relation status:

```text
hmm_space_free     false
hmm_token_aware    true
slot_grammar       true
token_markov       true
copy_edit          true
```

If those conditions are not present, the diagnostic aborts.

## 4. Models are not changed

Phase 4.D1 reconstructs exactly two relations:

```text
hmm_space_free
hmm_token_aware
```

Both use:

```text
hidden states      16
restarts           5
restart selection  highest TRAIN log likelihood only
matched comparator frozen order-3 Witten–Bell n-gram in the same view
```

The HMM model size is **not** selected on TEST.

The five random restarts are not a test-set search: as in the original
baseline, the winning restart is selected solely from TRAIN likelihood before
the TEST score is consulted.

The frozen model configs are:

```text
configs/baselines/hmm_v1.json
SHA-256:
86c0e7bc5fa29d45230a57b41f9402321e118cf7f54a3a410fa5feafc649dd20

configs/baselines/ngram_v1.json
SHA-256:
f0c00cb95364aa7b138c215ccd3befa9991dc106740172115f5c685d57f7e31c
```

## 5. Exact Phase 4 parity comes first

Before any leaf is interpreted, D1 reconstructs both HMM relations and requires
parity with `results/locked/phase4_voynich/v1/relation_results.csv`.

It compares:

```text
competitor bits/event
matched trigram bits/event
delta
95% CI lower
95% CI upper
bootstrap units
bootstrap replicates
bootstrap seed
```

using an absolute numerical tolerance of:

```text
1e-9
```

The paired bootstrap is exactly:

```text
unit        atomic physical-leaf group
replicates  10,000
seed        40814041438
statistic   ratio-of-sums bits/event
```

If parity fails, the script aborts **before** writing the localization
interpretation.

This protects against accidentally diagnosing a different implementation than
the one that generated the Phase 4 result.

## 6. Per-leaf decomposition

For each atomic leaf group and each HMM view, D1 records:

```text
HMM total bits
trigram total bits
event count
HMM bits/event
trigram bits/event
leaf delta bits/event
raw delta bits
exact contribution to the corpus delta
```

The exact contribution is:

```text
(HMM_bits_leaf - trigram_bits_leaf) / total_corpus_events
```

and therefore the contributions sum exactly to the corpus-level delta for that
view.

A leaf-level delta is descriptive. It is not an independent hypothesis test.

## 7. Leave-one-leaf-out influence

For every atomic leaf group, D1 recomputes the corpus ratio-of-sums after
removing that group.

It reports:

```text
full corpus delta
leave-one-out delta
leave-one-out minus full delta
```

Interpretation:

- a positive change after removal means that leaf was pulling the aggregate
  delta downward;
- a negative change after removal means that leaf was pulling the aggregate
  delta upward.

This analysis is an influence diagnostic only.

If removing a particular leaf makes the space-free relation look
"significant", that does **not** make Phase 4 pass and is not permission to
exclude the leaf.

## 8. Space-free versus token-aware

The cleanest internal contrast is:

```text
space-free HMM    weak / CI crosses zero
token-aware HMM   strong / CI above zero
```

D1 therefore places both per-leaf deltas on the same row and reports:

```text
token_aware_delta - space_free_delta
```

A large positive gap identifies leaves on which explicit word-boundary events
are associated with a larger HMM-versus-trigram separation.

The diagnostic also reports descriptive:

- sign concordance across leaves;
- Pearson correlation of leaf-level deltas.

No new threshold is attached to those quantities.

## 9. Metadata localization

The already-parsed analytical loci provide manuscript descriptors including:

```text
section
Currier assignment
scribe
quire
locus type
```

For each atomic leaf group, all observed values are collected.

If a leaf has one value, that value is used.

If it contains multiple values, it is labeled:

```text
MIXED[value1|value2|...]
```

D1 then reports descriptive ratio-of-sums HMM, trigram, and delta values for
each metadata label.

These are post-hoc descriptive summaries.

They must **not** be reported as confirmatory subgroup discoveries and no
subgroup significance tests are performed.

## 10. Required outputs

```text
results/diagnostics/phase4_space_free_hmm_localization/v1/
  aggregate_parity.csv
  per_leaf_hmm_deltas.csv
  leave_one_leaf_out.csv
  metadata_group_summary.csv
  bootstrap/
    hmm_space_free.csv
    hmm_token_aware.csv
  summary.json
  run_manifest.json
  SHA256SUMS
```

The first file to inspect is:

```text
aggregate_parity.csv
```

Every parity field must pass.

The second is:

```text
per_leaf_hmm_deltas.csv
```

which identifies the distribution of positive and negative leaf-level effects.

## 11. Interpretation rules

Permitted language includes:

> The Phase 4 space-free mismatch was broadly distributed across leaves.

or:

> The Phase 4 space-free uncertainty was concentrated in a small number of
> influential leaf groups.

or:

> The token-aware relation was more positive than the space-free relation in
> most locked-test leaf groups.

Those statements must be supported by the actual D1 outputs.

Prohibited conclusions include:

- "Phase 4 actually passed";
- "4/5 should count as a pass";
- "Phase 4 passes if we exclude leaf X";
- "section X confirms the mechanism";
- "the HMM should have used another K";
- "the bootstrap should use another threshold";
- "the Voynich mechanism is now identified."

## 12. Running the diagnostic

After the Phase 4 result has been archived:

```bash
python scripts/run_phase4_space_free_hmm_localization.py
```

This script does access the already-used Voynich TEST split.

It does **not** read the development VALIDATION split.

It does **not** modify any Phase 4 artifact.

## 13. What comes after D1

The next step depends on the localization result.

If the mismatch is driven by a few leaves, the scientific question becomes why
those leaves differ structurally; they are not removed from Phase 4.

If the mismatch is broad, the candidate mechanism may genuinely fail to
capture a space-free property of Voynich.

If the token-aware/space-free gap tracks manuscript structure, a future
mechanism may need to explain why token boundaries carry predictive information.

Any new mechanism inspired by D1 begins a new exploratory development phase.
It cannot be evaluated against the same Voynich TEST set and relabeled as a new
independent confirmation.
