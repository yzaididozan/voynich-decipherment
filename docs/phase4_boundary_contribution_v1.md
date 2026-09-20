# VOYAGER Phase 4.D2 — Boundary Contribution Localization

**Experiment ID:** `phase4-boundary-contribution-v1`  
**Status:** Exploratory / post-hoc locked-test diagnostic

## Purpose

Phase 4 failed its frozen five-relation conjunction because the space-free HMM relation had a positive point estimate but a 95% paired-leaf bootstrap interval that crossed zero. Phase 4.D1 reproduced the original HMM results exactly and localized the discrepancy across the 15 locked atomic leaf groups.

D1 found:

```text
space-free leaf signs:   +10 / -5
token-aware leaf signs:  +13 / -2
cross-view Pearson:      +0.9670
```

The two leaf groups that remained negative in both representations were pre-identified by D1 as `f57` and `f49`. D1 also showed that token-aware minus space-free delta was positive for all 15 locked leaf groups.

Phase 4.D2 asks:

> What measurable information associated with token boundaries covaries with the systematic token-aware boost, and what is unusual about f57 and f49?

D2 does not train or score any predictive model.

## Parent result and status

D2 requires the exact D1 configuration:

```text
configs/diagnostics/phase4_space_free_hmm_localization_v1.json
SHA-256:
f797b278e3e8d4389e8a5f3a537d527b38d06f01ecc6198b0bbce3f41e0aee17
```

It requires D1 to retain:

```text
Phase 4 decision =
INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN
Phase 4 parity all passed = true
atomic leaf groups = 15
```

D2 is not another locked confirmation. The Voynich TEST set has already been accessed, and every D2 inference is exploratory.

## No model refitting

D2 must not fit, retrain, rescore, or select any HMM, character n-gram, slot grammar, Token-Markov model, or copy-edit model.

Its outcome is copied directly from D1:

```text
boundary_boost_bits_per_event =
token_aware_delta_bits_per_event
-
space_free_delta_bits_per_event
```

The manuscript is parsed only to calculate descriptive token and character-transition properties for the same locked leaves.

## Readable-token policy

A readable token is a nonempty S0/STA1 token that does not contain `Z1`.

A token containing `Z1` is excluded and terminates the current readable segment. No across-token-boundary transition may jump through `Z1`, and readable segments never cross analytical-locus boundaries.

## Frozen features

For each atomic leaf group:

```text
readable_token_count
readable_glyph_count
mean_token_length_glyphs
token_length_variance_population
boundary_count
boundary_density_per_glyph
vocabulary_size_exact_tokens
type_token_ratio
repeated_token_occurrence_fraction
singleton_token_occurrence_fraction
singleton_type_fraction
adjacent_exact_repeat_fraction
```

Two character-transition populations are kept separate.

Within-token example:

```text
A B C
A->B
B->C
```

Across-boundary example:

```text
A B | C D
B->C
```

For each population D2 reports the transition count, number of unique ordered transition pairs, empirical pair entropy `H(previous,next)`, and empirical conditional entropy `H(next|previous)`, all in bits.

Frozen contrasts:

```text
conditional_entropy_gap_across_minus_within_bits
pair_entropy_gap_across_minus_within_bits
```

## Association analysis

For every pre-specified feature in the JSON, D2 reports its association with `boundary_boost_bits_per_event` using:

```text
Pearson r
Spearman rho
n used
n missing
```

No p-values are computed. No automatic feature selection, winner ranking, or confirmatory threshold is allowed.

`readable_token_count` and `readable_glyph_count` are included deliberately so that a large association with text quantity is visible rather than mistaken for mechanism-specific structure.

## f57 and f49 comparison

D1 identified `f57` and `f49` before D2 as the only leaves negative in both HMM views. For each frozen feature D2 reports:

```text
feature value
other-leaves mean
other-leaves median
other-leaves population SD
descriptive z-score
percentile rank among all 15
```

There is no significance test and no permission to exclude either leaf.

## Required outputs

```text
results/diagnostics/phase4_boundary_contribution/v1/
  per_leaf_boundary_features.csv
  feature_correlations.csv
  within_token_transition_summary.csv
  cross_boundary_transition_summary.csv
  f57_f49_comparison.csv
  summary.json
  run_manifest.json
  SHA256SUMS
```

## Interpretation boundary

Permitted, when supported:

> The boundary boost was descriptively larger on leaves with a larger across-minus-within conditional-entropy gap.

or:

> f57 had an unusually high cross-boundary transition entropy relative to the other locked leaves.

Not permitted:

> Therefore token boundaries explain the Voynich manuscript.

Not permitted:

> Excluding f57 makes Phase 4 pass.

Not permitted:

> The largest D2 correlation identifies the historical mechanism.

## Running D2

```bash
python scripts/run_phase4_boundary_contribution.py
```

The script accesses the already-used TEST split for descriptive features. It does not read the development validation split and does not fit any model.

The original Phase 4 result remains permanently:

```text
INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN
```
