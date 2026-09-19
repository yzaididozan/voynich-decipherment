# Phase 3A.D1 — Token-Markov Failure Localization

**Experiment ID:** `token-markov-failure-localization-v1`  
**Status:** Exploratory diagnostic  
**Repository destination:** `docs/token_markov_failure_localization_v1.md`

## 1. Why this diagnostic exists

Phase 3A.5 ended with an unusually clean residual failure pattern:

```text
                    1×16    2×8    4×4

HMM space-free       PASS    PASS    PASS
HMM token-aware      PASS    PASS    PASS
slot grammar         PASS    PASS    PASS
token Markov         FAIL    FAIL    FAIL
copy-edit            PASS    PASS    PASS
```

For the token-Markov relation, every Phase-3A.5 factorization reproduced the
frozen strong-direction rule in only two of four languages:

```text
French     0/5 strong
German     5/5 strong
Italian    5/5 strong
Latin      0/5 strong
```

At the same time, Phase 3A.4 had already shown that base-preserving surface
multiplicity is a major causal lever in the synthetic controls:

```text
K=4     -0.128436 bits/event
K=8     -0.038280 bits/event
K=16    +0.028097 bits/event
```

Phase 3A.5 then showed that refactorizing the same 16 exact surface identities
as `1×16`, `2×8`, or `4×4` had essentially no effect.

The next scientific question is therefore no longer:

> What additional suffix representation should we try?

It is:

> Where, specifically, does the remaining exact-token predictive information
> come from in French and Latin, and how does that differ from German and
> Italian?

Phase 3A.D1 is designed to answer that question before another mechanism is
invented.

## 2. This is not a new cipher experiment

Phase 3A.D1:

- generates no new cipher;
- changes no Phase-3A.5 ciphertext;
- expands no model grid;
- selects no new token context order;
- accesses no Voynich locked test;
- scores no reserved control unit.

It is a post-hoc development diagnostic over already-generated Phase-3A.5
TRAIN and VALIDATION controls.

Any Phase-3A.6 mechanism suggested by this analysis must be specified and
frozen in a separate protocol before new controls are generated.

## 3. Files

Install:

```text
configs/diagnostics/token_markov_failure_localization_v1.json
src/analysis/token_markov_failure_localization.py
scripts/run_token_markov_failure_localization.py
tests/test_token_markov_failure_localization.py
docs/token_markov_failure_localization_v1.md
```

## 4. Frozen parent inputs

The diagnostic requires the exact Phase-3A.5 protocol that generated the
observed result:

```text
configs/mechanisms/compositional_affix_factorization_v1.json
SHA256:
a14fd5af12adc0e226b23c4939de3271d8c64eae616a628a163805a8efe880c7

src/ciphers/compositional_affix_factorization.py
SHA256:
52e996b39924b9b29384e77035077e81bb748865f1206afcdcbb1fa7522c915c
```

It also requires the frozen Level-2 evaluator configuration:

```text
configs/evaluation/level2_predictive_controls_v1.json
SHA256:
6f6214286c9c8e47e90c27cf4c41da60f21c12fd8c31c07a4a98e3aada8b236c
```

The runner aborts if these hashes differ.

## 5. Primary and robustness conditions

Detailed token-level localization is frozen to:

```text
Phase-3A.5 factorization = 1x16
```

This is the Phase-3A.5 baseline and is the closest two-glyph representation to
the atomic K=16 Phase-3A.4 mechanism.

The scorer is nevertheless run against all:

```text
1x16
2x8
4x4
```

for aggregate parity verification.

Therefore:

```text
4 languages × 3 factorizations × 5 seeds
= 60 aggregate parity checks
```

but only:

```text
4 languages × 1 primary factorization × 5 seeds
= 20 detailed localization runs
```

are written into the large token-level contribution file.

This avoids tripling a diagnostic dataset after Phase 3A.5 already established
that factorization had essentially no effect.

## 6. Exact quantity being decomposed

The frozen Level-2 token model predicts a whole ciphertext token from previous
exact ciphertext-token identity using interpolated Witten–Bell smoothing.

At the unigram backoff level it uses the exact matched character trigram that
generates each token independently and terminates it with an explicit `<EOT>`.

For every validation token, Phase 3A.D1 computes:

```text
token_markov_bits
=
-log2 P_token_markov(cipher token | previous cipher tokens)
```

and:

```text
matched_trigram_bits
=
-log2 P_character_trigram(cipher token + EOT)
```

The diagnostic contribution is:

```text
delta_bits
=
token_markov_bits - matched_trigram_bits
```

Interpretation:

```text
delta_bits > 0
    character trigram predicted this token better

delta_bits < 0
    exact-token Markov predicted this token better
```

The frozen event denominator remains:

```text
ciphertext glyph count + 1 EOT event per token
```

## 7. Exact aggregate reconstruction

A central guardrail is that the token-level decomposition must reconstruct the
already-frozen Phase-3A.5 relation.

For each of the 60 runs:

```text
sum(token_markov_bits) / sum(events)
-
sum(matched_trigram_bits) / sum(events)
```

must equal the stored Phase-3A.5:

```text
delta_competitor_minus_trigram_bits_per_event
```

within:

```text
1e-9
```

The diagnostic separately checks:

- competitor bits/event;
- matched-trigram bits/event;
- their delta.

If any run fails aggregate parity, the runner aborts before interpreting the
localization.

The output is:

```text
aggregate_parity_checks.csv
```

This is the first file to inspect.

## 8. Context order is inherited, not reselected

Phase 3A.D1 does not use the diagnostic to choose context order.

For each Phase-3A.5 source/factorization/seed run, it reads:

```text
selected_hyperparameter
```

from:

```text
results/controls/compositional_affix_factorization_exploration/v1/
run_relation_matrix.csv
```

and uses that already-selected token context order.

Thus the diagnostic is conditional on the same validation-selected model used
for the frozen Phase-3A.5 relation.

This limitation should be stated in any methods section.

## 9. TRAIN-only explanatory features

Every explanatory property used for grouping is computed from TRAIN only.

Validation tokens are scored and then assigned to bins according to those
frozen TRAIN statistics.

### Plaintext-control features

Because these are synthetic controls, the known source-language token is
available for mechanism diagnosis.

For each validation occurrence the diagnostic records TRAIN-derived:

```text
current plaintext token frequency
previous plaintext token frequency
deterministic current-token frequency rank
previous → current plaintext bigram frequency
P(current | previous)
number of distinct TRAIN next tokens after previous
previous-token transition concentration
number of distinct TRAIN predecessors of current
top-N current-token membership
```

The plaintext labels belong to the historical-language controls only.

They are **not** interpreted as evidence about hidden Voynich plaintext.

### Ciphertext features

The diagnostic also records:

```text
current exact ciphertext-token TRAIN frequency
previous exact ciphertext-token TRAIN frequency
previous → current exact ciphertext bigram frequency
P(cipher_current | cipher_previous)
distinct next exact tokens after cipher_previous
cipher transition concentration
```

These are especially important because the frozen competitor is itself an
exact-token model.

### Structural features

The diagnostic records:

```text
plaintext token length
line position:
    singleton
    initial
    medial
    final

current abstract Phase-3A.5 variant index
previous abstract variant index
TRAIN variant-index bigram frequency
TRAIN P(variant_current | variant_previous)
```

The variant index is decoded from the two Phase-3A.5 affix glyphs and remains
the same 16-way abstract identity used in the parent experiment.

## 10. Frozen bins

Bins are defined in the diagnostic config before execution.

### TRAIN token frequency

```text
0
1
2–3
4–7
8–15
16–31
32–63
64–127
128+
```

### TRAIN bigram frequency

```text
0
1
2–3
4–7
8–15
16–31
32+
```

### Conditional probability / transition concentration

```text
0
(0, .01]
(.01, .05]
(.05, .10]
(.10, .25]
(.25, .50]
(.50, 1]
```

### Plaintext token length

```text
1–3
4–5
6–7
8–10
11+
```

### Top-frequency sets

```text
top 8
top 16
top 32
top 64
```

Ranks are deterministic: descending TRAIN frequency, with lexical
representation used only as a tie breaker.

## 11. Why both rates and contribution mass are reported

A frequency class can matter in two different ways.

It can have a strong per-event effect:

```text
delta_bits_per_event
```

or it can account for a large amount of the total token-Markov advantage simply
because many tokens fall into that class.

Therefore every bin summary reports both:

```text
delta_bits_per_event_ratio_of_sums
```

and:

```text
share_of_language_token_markov_advantage_bits
```

where token-Markov advantage bits are:

```text
max(-delta_bits, 0)
```

The first asks:

> How strongly does the model relationship behave inside this bin?

The second asks:

> Where does the total mass of occasions on which token Markov wins occur?

Neither statistic alone is causal because the grouping variables are
correlated.

## 12. Language contrast

The existing Phase-3A.5 development contrast is frozen as:

```text
robust:
    German
    Italian

non-robust:
    French
    Latin
```

The diagnostic does not redefine these groups.

The useful question is whether French and Latin show a concentration of
negative token contributions in a property that is absent or substantially
weaker in German and Italian.

Examples of interpretable outcomes include:

### High-frequency localization

If most residual token-Markov gain occurs for very frequent plaintext or exact
ciphertext tokens:

```text
possible Phase-3A.6 question:
Should surface realization depend on lexical recurrence pressure?
```

### Transition localization

If the residual gain is concentrated in high-frequency or high-probability
word transitions:

```text
possible Phase-3A.6 question:
Should realization vary as a function of local token context?
```

### Length localization

If short tokens dominate the residual advantage:

```text
possible Phase-3A.6 question:
Does fixed multiplicity undersuppress identity for short/highly reusable forms?
```

### Boundary localization

If line-initial or other position classes dominate:

```text
possible Phase-3A.6 question:
Does the surface-realization mechanism need positional conditioning?
```

### Diffuse failure

If no feature class sharply distinguishes French/Latin from German/Italian:

```text
conclusion:
Do not force a frequency-specific or transition-specific mechanism.
Surface multiplicity alone may be insufficient.
```

These are interpretation examples, not automatic decisions implemented by the
runner.

## 13. Important statistical limitations

Phase 3A.D1 is explicitly descriptive.

It does not produce confirmatory p-values.

The five seeds are robustness perturbations of the same source corpus and are
not treated as five independent linguistic experiments.

Features such as:

```text
frequency
bigram frequency
conditional probability
token length
recurrence
```

are correlated.

Therefore a bin containing much of the negative-bit mass does not prove that
the binning variable caused the model advantage.

The purpose is failure localization and hypothesis generation.

## 14. Reserved-data protection

The diagnostic uses only frozen TRAIN and VALIDATION memberships.

For generated Phase-3A.5 controls, the frozen Level-2 reader:

- enumerates the monolithic JSONL file;
- JSON-parses only TRAIN and VALIDATION rows;
- streams reserved rows past without JSON parsing.

For historical-language source files, the diagnostic similarly tokenizes only
TRAIN and VALIDATION line indices. Other source lines are counted for file
length but not split into tokens or used in statistics.

The diagnostic never accesses:

```text
Voynich test_LOCKED
```

## 15. Expected outputs

```text
results/diagnostics/token_markov_failure_localization/v1/
├── aggregate_parity_checks.csv
├── token_contributions.csv
├── transition_contributions.csv
├── frequency_bin_summary.csv
├── bigram_frequency_bin_summary.csv
├── cipher_recurrence_bin_summary.csv
├── cipher_bigram_frequency_bin_summary.csv
├── plain_conditional_probability_bin_summary.csv
├── cipher_conditional_probability_bin_summary.csv
├── token_length_summary.csv
├── line_position_summary.csv
├── variant_index_summary.csv
├── top_frequency_summary.csv
├── language_summary.csv
├── factorization_robustness_summary.csv
├── summary.json
└── SHA256SUMS
```

`token_contributions.csv` contains detailed rows only for `1x16`.

All three Phase-3A.5 factorizations are still reconstructed in
`aggregate_parity_checks.csv`.

## 16. Recommended inspection order

After a successful run, inspect:

```text
1. aggregate_parity_checks.csv
2. language_summary.csv
3. frequency_bin_summary.csv
4. bigram_frequency_bin_summary.csv
5. cipher_recurrence_bin_summary.csv
6. cipher_bigram_frequency_bin_summary.csv
7. plain_conditional_probability_bin_summary.csv
8. cipher_conditional_probability_bin_summary.csv
9. token_length_summary.csv
10. line_position_summary.csv
11. summary.json
```

The parity file must show all 60 runs passing before the localization is
interpreted.

## 17. Tests

Run:

```bash
source .venv/bin/activate

python -m pytest -q tests/test_token_markov_failure_localization.py
python -m pytest -q
```

The dedicated tests cover:

- frozen bin assignment;
- line-position classes;
- decoding all three Phase-3A.5 factorization geometries;
- selected-line plaintext reading;
- deterministic TRAIN frequency ranks;
- TRAIN-only lexical/transition statistics;
- exact per-token token-Markov/trigram decomposition;
- aggregate parity pass/failure behavior;
- ratio-of-sums bin summaries;
- negative-advantage-mass localization;
- top-N summaries;
- transition-row filtering;
- language status summaries.

## 18. Freeze before running the diagnostic

Phase 3A.D1 itself is post-hoc with respect to Phase 3A.5, but the diagnostic
definitions and bins should still be frozen before examining the localization.

After tests pass:

```bash
git add \
  configs/diagnostics/token_markov_failure_localization_v1.json \
  src/analysis/token_markov_failure_localization.py \
  scripts/run_token_markov_failure_localization.py \
  tests/test_token_markov_failure_localization.py \
  docs/token_markov_failure_localization_v1.md

git commit -m "Freeze Phase 3A.D1 token-Markov failure localization"
```

Then run:

```bash
python scripts/run_token_markov_failure_localization.py
```

## 19. What happens after Phase 3A.D1

Do not automatically turn the largest bin into a new cipher.

Instead:

1. verify all 60 aggregate reconstructions;
2. identify whether French/Latin residual advantage is localized or diffuse;
3. compare the same feature profiles with German/Italian;
4. formulate one mechanistic explanation;
5. write a separate Phase-3A.6 protocol;
6. freeze that protocol;
7. only then generate new synthetic controls.

Phase 3A.D1 is intended to make Phase 3A.6 hypothesis-driven rather than
trial-and-error.

## 20. Confirmatory boundary

Nothing in Phase 3A.D1 changes the confirmatory boundary.

Still untouched:

```text
control reserved test
Voynich locked test
```

Phase 3B remains unavailable until a development mechanism satisfies the
frozen hierarchy criterion and a separate confirmatory protocol is committed
before reserved-test access.
