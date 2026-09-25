# VOYAGER Phase 6B — Preregistered Voynich Recurrence-Cycle Test

**Experiment ID:** `phase-6b-voynich-recurrence-cycle-v1`

**Status:** preregistered mechanism-derived analysis on previously accessed Voynich

**Frozen config:** `configs/phase_6b_voynich_recurrence_cycle.yaml`

**Config SHA-256:** `63cf3a365165aea24f5fc461e31bee0ccb00a9e0406cb172e28e0bd1fcd09ac9`

## 1. Purpose

Phase 6A calibrated a period-16 recurrence statistic that follows directly from
the frozen recurrence-aware K=16 surface allocator. The candidate produced the
exact theoretical `cycle16_contrast = 1.0` in 20/20 control runs and strongly
separated from both prespecified matched null allocators.

Phase 6B asks one prespecified manuscript question:

> Does the frozen S0/STA1 Voynich surface show excess same-suffix recurrence at
> occurrence lag 16 within exact `token[:-1]` families, beyond what is expected
> from those families' own observed suffix inventories and frequencies?

This is not a new untouched-manuscript test. The Voynich transcription has
already been accessed in Phases 4 and 4.D1–D3. The specific recurrence statistic
and its period-16 decision rule are being frozen **before calculating that
statistic on Voynich**.

## 2. Parent Phase 6A

Required parent:

```text
configs/phase_6a_recurrence_cycle_prediction.yaml
SHA-256
99a7c07fba77c68cdf6d9c080442ccd9b3ebbc1315e256c6c239028052cf7575
```

Phase 6A must retain:

```text
decision              PHASE_6A_RECURRENCE_CYCLE_CONTROL_CALIBRATION_PASS
successful runs       20/20
languages replicated  4/4
Voynich accessed      false
```

Phase 6B inherits without modification:

```text
base family       token[:-1]
suffix            token[-1]
period            16
eligible family   >=17 occurrences
primary statistic cycle16_contrast
```

## 3. Prior results remain permanent

Phase 6B cannot revise:

```text
Phase 4:
INCONSISTENT_WITH_FULL_FIVE_RELATION_PATTERN

Phase 5A:
PHASE_5A_PROSPECTIVE_BOUNDARY_PREDICTION_NOT_SUPPORTED
```

A positive Phase 6B result would be a new structural result, not a retroactive
Phase-4 or Phase-5A pass.

## 4. Frozen Voynich representation

Primary transcription:

```text
Zandbergen-Landini ZL v3b
data/raw/zl/ZL3b-n.txt
SHA-256
bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc
```

STA1 conversion:

```text
data/reference/sta1/STA-Eva_def.bit
SHA-256
7f37853510144fb3e2dc3ee9458d634f41e6d95bc1fbf1c4b8f479a53a021f81
```

Analytical representation:

```text
S0_ZL_CONVENTIONAL_STA1_GLYPHS
```

This retains the existing conventional ZL/EVA segmentation and EVA→STA1
conversion.

### Corpus scope

Phase 6B uses the **whole ZL3b analytical corpus**, not only the former
15-leaf Phase-4 TEST split.

Reason: the recurrence schedule is a global same-base occurrence process, and
Phase 6A calibrated it on each complete control stream (TRAIN followed by TEST).
The whole-corpus choice is frozen before inspecting the Voynich recurrence
statistic and maximizes the number of families capable of supplying lag-16
pairs.

This choice does **not** create independent evidence: the whole Voynich corpus
has been previously accessed.

## 5. Frozen token inclusion rule

A token is usable iff:

1. it is nonempty;
2. it contains no STA1 `Z1`;
3. it contains at least two STA1 glyphs.

Tokens containing `Z1` are excluded.

One-glyph tokens are excluded because:

```text
token[:-1]
```

would be the empty tuple and therefore would not correspond to the frozen
candidate's nonempty encoded base plus one suffix glyph.

Excluded tokens do not reset the occurrence sequence of any other readable
base family.

## 6. Global occurrence order

Strict ZL3b parser order is preserved.

Within that stream:

1. preserve analytical-locus order;
2. preserve token order within each locus;
3. group usable tokens by exact `tuple(token[:-1])`;
4. append each token's final glyph to that family's suffix sequence.

Family sequences continue across locus and physical-leaf boundaries.

No counter reset occurs at a locus, page, leaf, Currier-language, illustration,
or section boundary.

No ordering alternative may be selected after the result is seen.

## 7. Observable base and suffix

For one usable token:

```text
base_family = tuple(token[:-1])
suffix      = token[-1]
```

Example schematic:

```text
A B C x  -> base (A,B,C), suffix x
A B C y  -> base (A,B,C), suffix y
```

No plaintext label or cipher key is used.

## 8. Eligible families

A family enters the primary statistic iff it contains at least **17 usable
occurrences**.

Fail-closed data sufficiency additionally requires:

```text
>=10 eligible base families
>=64 pooled lag-16 pairs
```

If either condition fails, Phase 6B does not change thresholds. It records a
fail-closed `NOT_DETECTED` result.

## 9. Observed recurrence profile

For each lag `k = 1..16`, pool all eligible within-family pairs:

```text
(i, i+k)
```

and compute:

```text
match_rate(k)
  = same-suffix pairs at lag k
    --------------------------
    all eligible pairs at lag k
```

Define:

```text
short_lag_mean =
  mean(match_rate(1), ..., match_rate(15))

cycle16_contrast =
  match_rate(16) - short_lag_mean
```

Period 16 is not estimated. It is inherited from the frozen K=16 mechanism.

## 10. Why a permutation null is required

A high raw lag-16 match rate could occur simply because a family uses one final
glyph very frequently.

Therefore Phase 6B does **not** compare the observed statistic to zero alone.

For every eligible base family, the null independently permutes the observed
suffix sequence **within that family**.

This preserves exactly:

- base-family identity;
- number of occurrences in every family;
- each family's suffix multiset;
- each family's suffix frequencies;
- the total set of observed base+suffix combinations under the frozen
  decomposition.

It destroys only:

- suffix position relative to within-family occurrence number.

Thus the null asks whether occurrence order contains an excess period-16
signal beyond static family-specific suffix frequencies.

## 11. Frozen permutation procedure

```text
replicates  10,000
seed        40814041438
tail        one-sided upper
```

For every permutation replicate, recompute:

```text
lag16_match_rate
cycle16_contrast
```

Monte-Carlo p-value:

```text
p =
  (1 + number of null values >= observed value)
  ------------------------------------------------
  (10,000 + 1)
```

The two prespecified p-values are:

1. `p_lag16`
2. `p_cycle16_contrast`

Success requires **both**, so neither can rescue failure of the other.

Frozen threshold:

```text
p_lag16            <= 0.001
p_cycle16_contrast <= 0.001
```

## 12. Primary decision rule

The period-16 fingerprint is detected iff **all** of the following hold:

```text
1. data sufficiency passes
2. cycle16_contrast > 0
3. match_rate(16) > every match_rate(k), k=1..15
4. one-sided permutation p_lag16 <= 0.001
5. one-sided permutation p_cycle16_contrast <= 0.001
```

Final labels:

```text
PHASE_6B_PERIOD16_RECURRENCE_FINGERPRINT_DETECTED

PHASE_6B_PERIOD16_RECURRENCE_FINGERPRINT_NOT_DETECTED
```

There is no partial pass and no post-hoc relaxation.

The strict lag-16 peak requirement prevents a broad or monotonic recurrence
effect from being relabeled as a specific period-16 fingerprint.

## 13. Secondary lag-32 check

Families with at least 33 usable occurrences contribute to a prespecified
secondary lag-32 match rate.

The runner also reports a within-family permutation p-value for lag 32.

This is **descriptive only**. It cannot rescue a failed primary Phase 6B result.

## 14. Family-inventory description

For each eligible family, report:

- occurrence count;
- number of distinct suffix glyphs;
- lag-16 pair count;
- lag-16 same-suffix count;
- lag-16 match rate.

Also report how many eligible families contain more than sixteen distinct
suffix glyphs.

This inventory does not enter the primary decision.

## 15. No alternative search

After execution, do not search and promote:

- K/period 2–15 or 17+;
- two-glyph suffixes;
- initial-glyph variants;
- prefix stripping;
- alternative base boundaries;
- selected manuscript sections;
- selected leaves;
- family-frequency cutoffs other than 17;
- favorable Currier languages;
- favorable transcription subsets.

Such analyses, if ever performed, are exploratory and cannot alter Phase 6B.

## 16. Execution protocol

Default invocation is preflight only:

```bash
python scripts/run_phase_6b_voynich_recurrence_cycle.py
```

Preflight verifies:

- Phase 6B config contract;
- Phase 6A PASS and exact parent config SHA;
- ZL3b source hash;
- STA1 rules hash;
- no existing conflicting execution.

It does **not** calculate the recurrence statistic.

Freeze/commit the Phase 6B protocol before execution.

Execute once:

```bash
python scripts/run_phase_6b_voynich_recurrence_cycle.py --execute-analysis
```

Before calculating the recurrence outcome, the runner writes:

```text
results/voynich/phase6b_recurrence_cycle/v1/ANALYSIS_EXECUTED.json
```

A second execution is refused by default.

`--rerun-identical` permits only exact reproduction when the stored config SHA
equals the current config SHA.

## 17. Outputs

```text
results/voynich/phase6b_recurrence_cycle/v1/
  ANALYSIS_EXECUTED.json
  observed_lag_profile.csv
  eligible_families.csv
  permutation_distribution.csv
  summary.json
  run_manifest.json
  SHA256SUMS
```

## 18. Interpretation boundaries

If the fingerprint is detected, the strongest allowed conclusion is:

> Under the frozen S0/STA1 representation and the preregistered observable
> `token[:-1]` family definition, the previously accessed Voynich corpus shows
> an excess same-suffix recurrence peak specifically at occurrence lag 16 that
> is unlikely under a null preserving each family's observed suffix
> frequencies.

That result would be compatible with the fixed-cycle K=16 mechanism.

It would **not** establish that:

- the K=16 mechanism uniquely generated the manuscript;
- the final glyph is literally a cryptographic suffix;
- the observable base equals a plaintext lexical base;
- a source language has been identified;
- plaintext has been recovered;
- the manuscript has been deciphered.

If the fingerprint is not detected, retain that failure. Do not search other
periods or representations on the same corpus and present them as confirmatory.
