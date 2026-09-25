# VOYAGER Phase 6A — Recurrence-Cycle Fingerprint Control Calibration

**Experiment ID:** `phase-6a-recurrence-cycle-control-calibration-v1`

**Status:** confirmatory mechanism-control calibration

**Frozen config:** `configs/phase_6a_recurrence_cycle_prediction.yaml`

**Config SHA-256:** `99a7c07fba77c68cdf6d9c080442ccd9b3ebbc1315e256c6c239028052cf7575`

## 1. Motivation

Phase 5A prospectively tested a boundary-structure hypothesis that had first
been isolated post-hoc in the Voynich Phase-4 diagnostics. It did not replicate
under the frozen recurrence-aware K=16 control mechanism. Phase 6A therefore
does **not** search the Phase-5A rows for a replacement feature.

Instead, Phase 6A tests an observable consequence that follows directly from
the already-frozen K=16 generator:

> For each encoded lexical base, the same permutation of sixteen one-glyph
> suffix variants repeats every sixteen occurrences.

This is a substantially more mechanism-specific statement than the five
Level-2 HMM/grammar/Markov/copy relations or the D3 boundary-entropy
association.

Phase 6A is **control calibration only**. It does not access Voynich.

## 2. Data status

Phase 6A deliberately reuses the historical-language controls already frozen
for Phase 5A:

`data/controls/languages_phase5a_v1/manifest.json`

Frozen manifest SHA-256:

`52a2005974aff935821e6930b900923ab1bb3f7155ef05faa05c72304df35641`

The four sources are:

- French — `french_gutenberg_les_miserables_t1_17489`
- German — `german_gutenberg_simplicissimus_55171`
- Italian — `italian_gutenberg_i_promessi_sposi_45334`
- Latin — `latin_gutenberg_aeneidos_227`

These data have already been accessed. **They are not fresh prospective
controls for Phase 6A.** Their only role is to verify that the frozen statistic
detects the candidate's direct period-16 signature and distinguishes it from
matched alternative allocation rules.

Both TRAIN and TEST are used. The generator key is still fitted on TRAIN only.
TRAIN is encoded first, then TEST continues directly from the final TRAIN
per-base occurrence counters. The recurrence statistic is computed over the
combined encoded occurrence stream.

## 3. Frozen candidate

The candidate remains exactly:

`src/ciphers/recurrence_avoiding_surface_allocation.py`

Generator SHA-256:

`2a6b27af36b0b896b40409450dffbd443d7a378d4deffaa390e8a989d25e33c9`

Mechanism config SHA-256:

`f26fef20aad928ee98a3e4824ca0911c8855de2ff1c0908d5388fae0ffbd2535`

For a given encoded lexical base, let its fixed suffix permutation be

`p[0], p[1], ..., p[15]`.

Occurrence `n` receives:

`p[n mod 16]`.

Therefore, for the same base:

- occurrences separated by lags 1–15 must have different suffixes;
- occurrences separated by lag 16 must have the same suffix;
- the same is true again at lag 32, 48, etc.

No K value or suffix definition is estimated in Phase 6A.

## 4. Matched null 1 — cycle shuffle

File:

`src/ciphers/recurrence_cycle_shuffle_null.py`

SHA-256:

`f9f46c1fb33de54a1d60ea0cb48cb51807af84e3cf8fe282f8081d2f8e3bea96`

The cycle-shuffle null preserves:

- the exact TRAIN-fitted base representation;
- the exact sixteen suffix glyphs;
- a one-glyph suffix;
- without-replacement usage of all sixteen suffixes inside every block of
  sixteen occurrences.

It changes only one property: each 16-occurrence cycle receives a new
deterministic base × seed × cycle permutation.

Thus it preserves the local recurrence-avoidance intervention while removing
the candidate's fixed across-cycle permutation.

## 5. Matched null 2 — IID K=16

File:

`src/ciphers/recurrence_iid_null.py`

SHA-256:

`193025f23c3ad940c7c1ccc0bf3ab87894768c7771f3414565ca0bda9140e740`

The IID null preserves:

- the same TRAIN-fitted base map;
- the same sixteen suffix glyphs;
- the same one-glyph suffix length.

Occurrence allocation is instead a deterministic pseudo-random function of
base × seed × occurrence number. It does not enforce without-replacement
allocation inside a cycle.

## 6. Observable representation

The recurrence statistic is calculated **only from generated ciphertext
tokens** after encoding.

For every ciphertext token:

```text
base_family = token[:-1]
suffix      = token[-1]
```

Plaintext labels do not enter the recurrence statistic.

Tokens are grouped by identical `base_family`. Within a family, occurrence
order is the order in the combined TRAIN-then-TEST encoded stream.

This observable mapping is frozen because it is the direct surface form of the
candidate mechanism: inherited base plus exactly one suffix glyph.

## 7. Eligible families

A base family enters the primary statistic iff it occurs at least **17 times**.

This threshold is fixed because lag 16 requires at least one same-family
occurrence pair.

Each source × seed × model run must additionally contain:

- at least **10 eligible families**;
- at least **64 pooled lag-16 pairs**.

Failure to meet either threshold is a fail-closed run error, not grounds for
changing the family threshold.

## 8. Primary recurrence statistic

For each lag `k` from 1 through 16, pool every eligible same-family occurrence
pair `(i, i+k)`.

Define:

```text
match_rate(k)
    = number of pooled pairs with identical suffix glyphs
      ----------------------------------------------------
      total number of pooled same-family pairs at lag k
```

The primary statistic is:

```text
cycle16_contrast
    = match_rate(16)
      - mean(match_rate(1), ..., match_rate(15))
```

### Candidate prediction

The frozen fixed-cycle K=16 mechanism predicts exactly:

```text
match_rate(1..15) = 0
match_rate(16)    = 1
cycle16_contrast  = 1
```

up to numerical tolerance `1e-12`.

### Cycle-shuffle null

Each individual cycle still uses all sixteen variants exactly once, but the
next cycle has a newly shuffled permutation. Lag-16 equality therefore loses
its deterministic identity and is expected to be near chance relative to the
candidate.

### IID null

Suffix equality is approximately `1/16` at every lag, so the expected
cycle-16 contrast is approximately zero.

No empirical null result was inspected before freezing these rules.

## 9. Primary decision rule

For one source × seed run to succeed, all of the following must hold:

1. The fixed-cycle candidate reproduces its exact theoretical pattern:
   - lag 1–15 match rates are zero within `1e-12`;
   - lag-16 match rate is one within `1e-12`;
   - cycle-16 contrast is one within `1e-12`.
2. Candidate `cycle16_contrast` is greater than the cycle-shuffle contrast.
3. Candidate `cycle16_contrast` is greater than the IID contrast.
4. The candidate exceeds the better null by at least **0.50**.

A language replicates only if **5/5 seeds** succeed.

Phase 6A passes only if **all 4/4 languages replicate**, i.e. **20/20
source × seed runs succeed**.

Final labels:

```text
PHASE_6A_RECURRENCE_CYCLE_CONTROL_CALIBRATION_PASS
PHASE_6A_RECURRENCE_CYCLE_CONTROL_CALIBRATION_FAIL
```

There is no p-value, CI, effect-size retuning, dropped language, dropped seed,
or post-run threshold relaxation.

## 10. Secondary descriptive check

For families occurring at least 33 times, the runner also reports the pooled
lag-32 suffix match rate.

The fixed-cycle candidate predicts lag-32 equality of one.

Lag 32 is **secondary only**. It cannot rescue a failed primary calibration.

## 11. Seeds

Exactly:

```text
40814041438
40814041439
40814041440
40814041441
40814041442
```

These are inherited from the previous control phases.

## 12. Voynich prohibition

Phase 6A is not allowed to open, hash, list, parse, or otherwise access Voynich
transcription/split material.

In particular, the control runner rejects configured input paths containing
Voynich path fragments such as:

```text
data/raw/zl
data/reference/sta1
data/splits/v1
```

No Phase-4 or Voynich token stream is scored.

## 13. Outputs

```text
results/controls/phase6a_recurrence_cycle/v1/
  per_lag_match_rates.csv
  per_run_cycle_statistics.csv
  language_replication.csv
  summary.json
  run_manifest.json
  SHA256SUMS
```

`per_lag_match_rates.csv` contains one row per source × seed × model × lag.

`per_run_cycle_statistics.csv` contains the three model contrasts and the
frozen per-run decision.

## 14. Rerun policy

If `summary.json` already exists, the runner refuses by default.

`--rerun-identical` is allowed only when the existing summary records the exact
same Phase-6A config SHA-256 as the current config. This permits deterministic
reproduction but not retuning.

## 15. Interpretation

A Phase 6A PASS establishes only that:

> The frozen period-16 recurrence statistic behaves as theoretically expected
> under the fixed-cycle K=16 generator and strongly separates that generator
> from the two prespecified matched allocation alternatives on the reused
> historical-language controls.

It does **not** establish that:

- Voynich exhibits the fingerprint;
- K=16 generated Voynich;
- any source language has been identified;
- any plaintext has been recovered;
- Phase 4 passed;
- Phase 5A passed.

A Phase 6A FAIL means the proposed statistic is not adequately calibrated
under the frozen implementation and **must not be promoted to a Voynich test
without a new preregistered control design**.

## 16. Gate to Phase 6B

Only after a frozen Phase 6A PASS should a separate Phase 6B Voynich
preregistration be written.

Before any Phase 6B calculation, freeze:

- `base_family = token[:-1]`;
- `suffix = token[-1]`;
- period exactly 16;
- minimum family count exactly 17;
- lags exactly 1–16;
- the same `cycle16_contrast`;
- the exact manuscript unit/aggregation rule;
- the null/comparator strategy;
- the decision rule.

Do not inspect alternative suffix lengths, alternative periods, or alternative
prefix/suffix stripping rules on Voynich and then select the best one.
