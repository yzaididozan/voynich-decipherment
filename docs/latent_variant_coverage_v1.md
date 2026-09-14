# Phase 3A.2 — Lexical Coverage Dose–Response

**Experiment ID:** `latent-variant-coverage-exploration-v1`  
**Status:** Exploratory  
**Repository destination:** `docs/latent_variant_coverage_v1.md`

## 1. Why this experiment exists

Phase 3A manipulated the number of related ciphertext variants available to a
fixed set of the top 64 TRAIN words:

```text
K = 1, 2, 4
```

The exact-token Markov minus matched-trigram validation delta moved
consistently toward the Voynich direction as K increased:

```text
K=1  -0.408859 bits/event
K=2  -0.371439 bits/event
K=4  -0.329034 bits/event
```

The direction was consistent across French, German, Italian, and Latin and
across all five frozen cipher seeds. However, K=4 still remained negative,
whereas the Voynich reference delta is positive.

The mechanism diagnostics showed why a coverage intervention is now warranted.
At N=64, only the following fractions of validation tokens were receiving the
related-variant treatment:

```text
French   0.4695
German   0.4300
Italian  0.4758
Latin    0.3622
```

At the same time, K=4 itself was functioning as designed: the probability that
two occurrences of the same targeted plaintext word used the exact same
surface token was approximately 0.25 in every language, and variant
utilization was high.

A separate TRAIN-selected coverage audit showed:

```text
language        top64      top256      top1024
French          0.4695      0.6357       0.7908
German          0.4300      0.6173       0.7557
Italian         0.4758      0.6389       0.7614
Latin           0.3622      0.5187       0.6654
```

Phase 3A.2 therefore changes the next variable one at a time:

> Hold the amount and geometry of variation per targeted word fixed, and vary
> only how many TRAIN-selected lexical families receive that variation.

This is a post-hoc exploratory follow-up informed by Phase 3A. It is not a
retroactive modification of Level 2 and is not confirmatory evidence.

## 2. Files

Install these five new files:

```text
configs/mechanisms/latent_variant_coverage_v1.json
src/ciphers/latent_variant_coverage.py
scripts/run_latent_variant_coverage_exploration.py
tests/test_latent_variant_coverage.py
docs/latent_variant_coverage_v1.md
```

Do not edit the frozen Phase-3A files to implement Phase 3A.2.

## 3. Experimental grid

The exploratory coverage grid is:

```text
N = 64, 256, 1024
K = 4 fixed
```

where N is the number of most frequent CONTROL-TRAIN word types receiving a
related ciphertext code family.

The complete design is:

```text
4 languages
x 3 N conditions
x 5 frozen seeds
= 60 synthetic controls
```

Each control receives the same five frozen Level-2 predictive comparisons:

```text
60 x 5 = 300 relation-level outcomes
```

## 4. Fixed code geometry

Every targeted plaintext word receives exactly four related four-glyph
ciphertext forms:

```text
N_a N_b N_c N_d1
N_a N_b N_c N_d2
N_a N_b N_c N_d3
N_a N_b N_c N_d4
```

The first three positions identify the latent lexical family.

With a fixed 16-symbol code alphabet, this provides:

```text
16^3 = 4096
```

possible lexical-family identifiers, enough for the maximum N=1024 condition.

The fourth position is the only position allowed to vary, and exactly four
final glyphs are available per target word.

Therefore all four variants for one target word have pairwise Hamming distance
exactly 1.

Across all N conditions, the following remain fixed:

```text
K                         = 4
target-code alphabet      = 16 glyphs
target-code length        = 4 glyphs
family-identity positions = 1–3
variant position          = 4
variant operation         = final-position substitution
fallback mapping          = same TRAIN-fitted mapping
occurrence variant rule   = same deterministic hash rule
```

Only N changes.

## 5. Master-codebook design

A key methodological requirement is that increasing N must not silently
reassign codes to words already active at lower N.

For each source language and seed, the generator therefore fits one
**1024-word master codebook** using CONTROL TRAIN only.

TRAIN words are ranked by:

1. descending TRAIN frequency;
2. lexical order to break frequency ties.

The active target sets are nested prefixes:

```text
N=64    = ranks 1..64
N=256   = ranks 1..256
N=1024  = ranks 1..1024
```

A word shared between N=64, N=256, and N=1024 receives the exact same
four-member code family in every condition.

For the same plaintext occurrence, the deterministic occurrence-level variant
choice is also identical across N.

Consequently the intervention from N=64 to N=256 consists only of activating
ranks 65..256, and the intervention from N=256 to N=1024 consists only of
activating ranks 257..1024.

The generator stores a deterministic `master_codebook_sha256`. The runner
verifies that this fingerprint is identical across all N conditions for each
source/seed pair before producing the final aggregate result.

## 6. Fallback encoding

Words outside the active top-N set use a monoalphabetic character substitution
fit from CONTROL TRAIN only.

Target-code glyphs and fallback glyphs use distinct namespaces:

```text
target code glyphs: N....
fallback glyphs:    M....
```

A character that appears in VALIDATION but never appeared in TRAIN receives a
deterministic reversible Unicode-codepoint glyph. This prevents validation from
fitting a new character mapping.

## 7. Important comparison with Phase 3A

Phase 3A used two-glyph target codes.

Phase 3A.2 uses four-glyph target codes in **every** N condition.

Therefore:

> Phase-3A.2 N=64 is the internal baseline for the coverage experiment. It is
> not treated as numerically identical to Phase-3A K=4/N=64.

The causal comparison for lexical coverage is strictly:

```text
Phase 3A.2 N=64
vs
Phase 3A.2 N=256
vs
Phase 3A.2 N=1024
```

because code geometry is identical within that comparison.

## 8. Primary exploratory prediction

The main response remains:

```text
token-Markov bits/event - matched-trigram bits/event
```

The predicted coverage trajectory is:

```text
N=64       most negative
N=256      less negative
N=1024     least negative / potentially positive
```

The mechanism is more interesting if this occurs while the other four
Voynich-direction relationships remain positive:

```text
trigram > HMM space-free
trigram > HMM token-aware
trigram > finite-template slot grammar
trigram > local copy-edit
```

This is an exploratory trajectory prediction, not a preregistered
confirmatory success threshold.

No scalar "Voynich similarity score" is calculated.

## 9. Mechanism diagnostics

For every source/N/seed run, the experiment records:

- validation targeted-token fraction;
- active target-word count;
- unique ciphertext-token count;
- exact ciphertext-token recurrence fraction;
- glyph alphabet size;
- round-trip accuracy;
- realized variants per target word;
- variant-utilization fraction;
- probability that two occurrences of the same targeted plaintext word use
  the same exact ciphertext surface token;
- within-family pairwise Hamming distance;
- master-codebook fingerprint.

The expected same-surface probability for heavily observed target words remains
near 1/4 because K is fixed at four. The quantity intentionally changing is
the fraction of lexical material exposed to the variant mechanism.

## 10. Frozen predictive evaluator

Phase 3A.2 does not invent new predictive models.

It reuses the frozen Level-2 evaluator and its existing procedures for:

```text
hmm_space_free
hmm_token_aware
slot_grammar
token_markov
copy_edit
```

including the matched trigram comparators, model-selection rules, and paired
bootstrap procedure.

Required pre-existing project files include:

```text
configs/evaluation/level2_predictive_controls_v1.json
src/evaluation/level2_predictive_controls.py
data/controls/splits/level2_v1/
data/controls/languages_open_v1/
```

## 11. Test protection

Phase 3A.2 may use:

```text
CONTROL TRAIN        yes
CONTROL VALIDATION   yes
```

It may not use:

```text
CONTROL RESERVED TEST   no
VOYNICH test_LOCKED     no
```

Reserved source lines are not encoded.

The generated JSONL contains one physical row per original source line so the
frozen line indices remain valid, but reserved rows contain only:

```json
{"line_index": 123, "reserved_test_opaque": true}
```

No reserved ciphertext token sequence is generated.

## 12. Installation and protocol freeze

Place the five files at their intended repository paths.

Then run:

```bash
source .venv/bin/activate

python -m pytest -q tests/test_latent_variant_coverage.py
python -m pytest -q
```

The dedicated tests check, among other things:

- TRAIN-only frequency ranking;
- unique four-glyph target families;
- four variants per family;
- final-position-only variation;
- nested active target sets;
- code-family invariance for words shared across N;
- occurrence-level variant invariance across N;
- fallback behavior before a word becomes active;
- deterministic master-codebook fingerprinting;
- reserved-test opacity;
- validation-unseen character reversibility;
- monotonic targeted coverage in a synthetic check;
- compatibility with the frozen Level-2 control reader.

If all tests pass, commit the Phase-3A.2 protocol **before running the
experiment**:

```bash
git add \
  configs/mechanisms/latent_variant_coverage_v1.json \
  src/ciphers/latent_variant_coverage.py \
  scripts/run_latent_variant_coverage_exploration.py \
  tests/test_latent_variant_coverage.py \
  docs/latent_variant_coverage_v1.md

git commit -m "Freeze Phase 3A.2 lexical coverage exploration"
```

Only after that commit run:

```bash
python scripts/run_latent_variant_coverage_exploration.py
```

## 13. Outputs

Synthetic controls are written under:

```text
data/controls/ciphers/latent_variant_coverage_v1/
  <source>/
    N64/
      seed_<seed>/
    N256/
      seed_<seed>/
    N1024/
      seed_<seed>/
```

Each contains:

```text
corpus.jsonl
key.json
summary.json
SHA256SUMS
```

Exploratory predictive results are written under:

```text
results/controls/latent_variant_coverage_exploration/v1/
├── run_relation_matrix.csv
├── mechanism_diagnostics.csv
├── coverage_summary.csv
├── n_language_relation_summary.csv
├── n_global_relation_summary.csv
├── summary.json
├── SHA256SUMS
└── runs/
```

The first files to inspect after completion are:

```text
coverage_summary.csv
n_global_relation_summary.csv
summary.json
```

`summary.json` reports:

- mean token-Markov delta at each N;
- whether the mean token-Markov trajectory is monotonically non-decreasing;
- observed validation coverage by N;
- behavior of the other four predictive relations;
- successful verification that the master codebook stayed invariant across N.

## 14. Interpretation before any further experiment

Several outcomes are informative.

### Outcome A — coverage reaches or approaches the Voynich token relation

If increasing N produces a strong monotone trajectory and the token-Markov
delta reaches approximately zero or becomes positive while the other four
relations remain stable, lexical coverage is a plausible missing ingredient.

Do **not** immediately use the reserved test. First define a separate Phase-3B
confirmatory protocol with a frozen N, mechanism, parameters, and decision
rule.

### Outcome B — token delta improves but plateaus below zero

Then coverage contributes causally but is insufficient. Retain that negative
result and test a new mechanism dimension, such as variant geometry, in a new
exploratory phase.

Do not simply increase N beyond the trained vocabulary until a sign change is
forced.

### Outcome C — coverage fixes token identity but damages other relations

Then the experiment reveals a tradeoff: broad lexical variation weakens exact
token identity at the cost of the local predictive structure that also needs
to be preserved.

That is a mechanistic falsification, not a failed project.

### Outcome D — little additional movement with N

Then the Phase-3A K trajectory was driven mainly by the amount of variation per
already-targeted word, not by lexical coverage. The next experiment should
change variant structure rather than vocabulary size.

## 15. Confirmatory boundary

Phase 3A.2 remains exploratory regardless of how compelling its validation
curve looks.

No N value becomes confirmatory merely because it resembles Voynich.

A future confirmatory stage requires a separately written and committed
protocol before any reserved control-test data are encoded, parsed, or scored.
The Voynich locked test remains protected by the broader project freeze.
