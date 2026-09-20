# VOYAGER Phase 4.D3 — Cross-Boundary Transition Anatomy

**Experiment ID:** `phase4-cross-boundary-transition-anatomy-v1`  
**Status:** Exploratory / post-hoc locked-test diagnostic

## 1. Why D3 exists

Phase 4 retained the confirmatory decision:

```text
INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN
```

Phase 4.D1 localized the sole failed relation and showed that preserving token
boundaries increased the HMM-versus-trigram delta on all 15 locked leaf groups.

Phase 4.D2 then compared a frozen set of token/boundary features with that
per-leaf boundary boost. Its strongest pre-specified association was:

```text
across_transition_pair_entropy_bits
Pearson  ≈ +0.8046
Spearman ≈ +0.6357
n = 15
```

A post-D2 sensitivity check showed:

```text
without f57:
Pearson  ≈ +0.4627
Spearman ≈ +0.5516

without f57 and f49:
Pearson  ≈ +0.4846
Spearman ≈ +0.5549
```

Thus f57 strongly amplifies the Pearson effect, but the rank association and a
moderate positive Pearson association remain after the pre-identified anomalies
are omitted descriptively.

D3 does **not** search for another predictive model. It decomposes that one
pre-identified D2 structural clue.

The D3 question is:

> What property of the joint distribution of token-final × token-initial STA1
> glyphs accounts descriptively for the cross-boundary pair-entropy association
> with the D1/D2 boundary boost?

## 2. Parent D2 is frozen

D3 requires:

```text
configs/diagnostics/phase4_boundary_contribution_v1.json
SHA-256:
3c9328d15ac337f7765b448d2c3c59a4f9b5a9ab8decc901f7c6420b53741572
```

It also requires the D2 outputs to retain:

```text
status = exploratory_posthoc_locked_test_diagnostic
parent Phase 4 decision =
INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN
predictive models fit/rescored = false
atomic leaf groups = 15
boundary boost signs = +15 / -0 / 0=0
```

The D2 per-leaf table supplies the fixed outcome:

```text
boundary_boost_bits_per_event
```

D3 never recomputes the HMM or trigram scores.

## 3. Boundary-pair definition

A readable token is a nonempty S0/STA1 token that does not contain `Z1`.

A readable segment is a maximal sequence of readable tokens inside one
analytical locus. A token containing `Z1` terminates the segment.

For adjacent readable tokens:

```text
... A B | C D ...
```

the cross-boundary pair is:

```text
(B, C)
```

where:

```text
F = token-final glyph
I = token-initial glyph
```

No pair crosses a `Z1` token or an analytical-locus boundary.

## 4. Frozen decomposition

D3 decomposes the empirical distribution `P(F,I)`.

### Marginal diversity

```text
H(F)
H(I)
distinct final glyphs
distinct initial glyphs
2^H(F)
2^H(I)
```

`2^H` is the Shannon effective number of categories.

### Joint diversity

```text
H(F,I)
distinct observed boundary pairs
2^H(F,I)
```

`H(F,I)` is the same cross-boundary transition-pair entropy highlighted by D2.

### Conditional structure and dependence

```text
H(I|F) = H(F,I) - H(F)
H(F|I) = H(F,I) - H(I)

I(F;I) = H(F) + H(I) - H(F,I)
```

D3 also reports:

```text
I(F;I) / H(F,I)
```

as a descriptive normalized dependence measure. If `H(F,I)=0`, the normalized
value is defined as zero.

These quantities distinguish two explanations for high joint entropy:

1. the marginals themselves are diverse; versus
2. final and initial glyphs combine with relatively weak or complex dependence.

### Support

```text
possible observed-marginal pair support =
distinct finals × distinct initials

observed support fraction =
distinct observed pairs / possible support
```

This asks whether final and initial glyph inventories combine sparsely or
broadly.

### Concentration

D3 reports:

```text
largest single pair probability
top-5 pair probability mass
sum p(pair)^2
1 / sum p(pair)^2
```

The final quantity is the inverse-Simpson effective number of boundary pairs.

These features distinguish high entropy caused by a broad, flat repertoire from
high entropy caused by a larger support with substantial concentration.

## 5. D3 association analysis

Each frozen anatomy feature is compared descriptively with:

```text
boundary_boost_bits_per_event
```

using:

```text
Pearson r
Spearman rho
n used
n missing
```

There are no p-values and no automatic winner selection.

The purpose is decomposition, not another hypothesis tournament.

## 6. Built-in leverage sensitivity

For every frozen anatomy feature, D3 reports the same correlations under:

```text
all15
without_f57
without_f49
without_f57_f49
```

This is an influence diagnostic only.

The omitted-leaf conditions are **not** alternate versions of Phase 4 and
cannot be used to exclude those leaves from the confirmatory result.

A component is descriptively more robust if its direction and approximate
strength persist under the pre-specified sensitivity conditions.

## 7. f57 and f49

D1/D2 pre-identified `f57` and `f49` as the only leaves negative in both HMM
views.

D2 further showed that f57 is an extreme boundary-regime outlier, whereas f49
is much more ordinary on the measured boundary features.

D3 therefore reports, for every anatomy feature:

```text
feature value
other-leaves mean
other-leaves median
other-leaves population SD
descriptive z-score versus the others
percentile rank among all 15
```

No significance test is performed.

## 8. Required outputs

```text
results/diagnostics/phase4_cross_boundary_transition_anatomy/v1/
  per_leaf_boundary_anatomy.csv
  anatomy_correlations.csv
  anatomy_sensitivity.csv
  boundary_pair_concentration.csv
  f57_f49_anatomy.csv
  summary.json
  run_manifest.json
  SHA256SUMS
```

### `per_leaf_boundary_anatomy.csv`

One row per locked atomic leaf group containing the fixed D2 boundary boost and
all frozen decomposition features.

### `anatomy_correlations.csv`

All-15 descriptive associations between each anatomy feature and the boundary
boost.

### `anatomy_sensitivity.csv`

The same associations for all four pre-specified influence conditions.

### `boundary_pair_concentration.csv`

A compact leaf-level view of support and concentration quantities.

### `f57_f49_anatomy.csv`

Long-form descriptive comparison of the two pre-identified anomalous leaves
against the remaining leaves.

## 9. Interpretation

Possible supported conclusion:

> The cross-boundary pair-entropy association appears to arise primarily from
> broader joint support rather than stronger final–initial dependence.

or:

> The D2 association is better accounted for by final/initial marginal
> diversity than by pair-support fraction.

Only the actual D3 outputs can determine which description is appropriate.

Not permitted:

> D3 rescued Phase 4.

Not permitted:

> f57 should be removed.

Not permitted:

> The largest D3 correlation identifies the historical Voynich mechanism.

Not permitted:

> This constitutes decipherment.

## 10. Running D3

```bash
python scripts/run_phase4_cross_boundary_transition_anatomy.py
```

The script accesses the already-used locked TEST leaves for descriptive
boundary pairs. It does not read validation data and does not fit a predictive
model.

The Phase 4 confirmatory result remains unchanged.
