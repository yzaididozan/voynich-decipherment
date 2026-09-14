# Phase 3A — Constrained Lexical Surface Variation

**Experiment ID:** `latent-variant-exploration-v1`  
**Status:** Exploratory  
**Repository destination:** `docs/latent_variant_hypothesis_v1.md`

## 1. Motivation

Level 2 exposed a specific unresolved Voynich signature rather than a generic
failure of historical cipher controls.

Several token-preserving mechanisms reproduced four of five operational
Voynich model-ordering relations but failed the exact-token Markov comparison.
In those controls, exact ciphertext word identity retained too much lexical
sequence information. Homophonic and transposition controls weakened exact
token identity, but at the cost of other local predictive properties. Null
insertion occupied an intermediate regime.

Phase 3A therefore asks a narrower question:

> Can constrained many-to-one lexical surface variation weaken exact-token
> sequential predictability while preserving strong local glyph-level
> predictability?

This is an exploratory mechanism-discovery experiment. It is not a retrofit of
Level 2, is not confirmatory evidence for a cipher family, and does not access
either reserved test set.

## 2. Files

Install the five new files at these repository paths:

```text
configs/mechanisms/latent_variant_v1.json
src/ciphers/latent_variant.py
scripts/run_latent_variant_exploration.py
tests/test_latent_variant.py
docs/latent_variant_hypothesis_v1.md
```

No existing Level-1 or Level-2 file should be modified.

## 3. Mechanism

The mechanism is a deliberately constrained related multi-code nomenclator.

The 64 most frequent **CONTROL TRAIN** plaintext word types receive special
two-glyph code families. All other words use a train-fitted monoalphabetic
character substitution.

For each targeted word, the maximum code family contains four forms:

```text
canonical:  N_a N_b0
variant 2:  N_a N_b1
variant 3:  N_a N_b2
variant 4:  N_a N_b3
```

Thus every noncanonical form differs from the canonical code by exactly one
substitution at the token-final position.

The exploratory grid is:

```text
K = 1, 2, 4
```

where K is the number of permitted surface forms for each targeted plaintext
word.

### Important control

Across K:

- code-token length remains exactly 2;
- the same 16-symbol nomenclator code alphabet is used;
- the same TRAIN-selected target vocabulary is used for a given seed;
- the same underlying code-family geometry is used;
- only the number of permitted final-position alternatives changes.

The 16-symbol alphabet is partitioned so that up to 64 lexical families can be
represented with globally collision-free two-glyph surface forms.

`K=1` is the internal fixed-code baseline for this Phase-3 mechanism. It is
**not** claimed to be byte-identical to the earlier Level-2 nomenclator
generator. This distinction is intentional: Phase 3 holds one code geometry
constant while varying only lexical surface multiplicity.

## 4. TRAIN-only mechanism fitting

For each language and seed:

1. read the already-frozen historical source;
2. load the already-frozen Level-2 source-line split;
3. use only TRAIN lines to:
   - select the top 64 target words;
   - construct the nomenclator family assignment;
   - fit the fallback character substitution;
4. encode TRAIN and VALIDATION using that frozen key;
5. do not encode reserved-test source lines.

Validation therefore cannot change:

- target-word membership;
- code-family assignment;
- fallback substitution;
- code alphabet;
- K.

If a validation character was never observed in TRAIN, it receives a
deterministic reversible Unicode-codepoint glyph. This avoids fitting a new
mapping from validation while preserving the fact that the corresponding
cipher glyph is unseen by the predictive models during training.

## 5. Reserved-test protection

Phase 3A uses:

```text
CONTROL TRAIN       yes
CONTROL VALIDATION  yes

CONTROL RESERVED TEST  no
VOYNICH test_LOCKED    no
```

The generated JSONL keeps one physical row per original source line. Reserved
rows contain only:

```json
{"line_index": 123, "reserved_test_opaque": true}
```

They contain no ciphertext tokens.

The existing Level-2 `read_control_units()` reader streams past these rows
before token parsing, so they cannot enter model fitting or validation
scoring.

## 6. Predictive evaluation

The runner reuses the already-frozen Level-2 evaluator rather than defining
new predictive models.

For every generated control it repeats the five operational comparisons:

```text
character trigram > HMM, space-free
character trigram > HMM, token-aware
character trigram > finite-template slot grammar
character trigram > exact token Markov
character trigram > local copy-edit
```

The original Level-2 HMM, slot-grammar, token-Markov, copy-edit grids,
train-only restart selection, validation selection, matched trigram
denominators, and 10,000-replicate paired bootstrap remain unchanged.

The Phase-3 runner therefore requires:

```text
configs/evaluation/level2_predictive_controls_v1.json
src/evaluation/level2_predictive_controls.py
data/controls/splits/level2_v1/
```

## 7. Primary exploratory prediction

The main response curve is:

```text
exact-token Markov delta
= token-Markov bits/event - matched-trigram bits/event
```

Voynich has a positive delta: the matched character trigram beats the exact
token Markov model.

The Phase-3 prediction is:

> As K increases from 1 to 2 to 4, controlled surface variation should move
> the exact-token Markov delta upward from the fixed-code regime toward zero
> and potentially positive values, while the other four relations remain in
> the Voynich direction.

This is a trajectory prediction, not a frozen confirmatory success threshold.

No single weighted "Voynich similarity" score is calculated.

## 8. Mechanism diagnostics

In addition to the five predictive deltas, each generated control records:

- targeted-token fraction;
- unique ciphertext-token count;
- exact ciphertext-token recurrence fraction;
- ciphertext glyph-alphabet size;
- exact round-trip accuracy;
- realized variants per target word;
- variant-utilization fraction;
- probability that two occurrences of the same targeted plaintext word use
  the same exact ciphertext surface token;
- mean within-family pairwise Hamming distance.

Because the synthetic mapping is known, exact decoding of TRAIN and VALIDATION
is verified mechanically.

## 9. Experimental size

The complete Phase-3A grid is:

```text
4 languages
x 3 K values
x 5 seeds
= 60 synthetic controls
```

Each receives five predictive comparisons:

```text
60 x 5 = 300 relation-level outcomes
```

## 10. Installation

Place the downloadable files at their intended paths, then run:

```bash
source .venv/bin/activate

python -m pytest -q tests/test_latent_variant.py
python -m pytest -q
```

If those pass, record the exploratory protocol before running it:

```bash
git add \
  configs/mechanisms/latent_variant_v1.json \
  src/ciphers/latent_variant.py \
  scripts/run_latent_variant_exploration.py \
  tests/test_latent_variant.py \
  docs/latent_variant_hypothesis_v1.md

git commit -m "Add exploratory constrained lexical variant mechanism"
```

Then execute:

```bash
python scripts/run_latent_variant_exploration.py
```

The run is resumable. Existing controls/results are reused only when their
Phase-3 config hash, generator hash, source hash, split hash, and Level-2
config hash agree.

## 11. Outputs

Generated controls:

```text
data/controls/ciphers/latent_variant_v1/
  <source>/
    K1/
      seed_<seed>/
    K2/
      seed_<seed>/
    K4/
      seed_<seed>/
```

Each run contains:

```text
corpus.jsonl
key.json
summary.json
SHA256SUMS
```

Exploratory results:

```text
results/controls/latent_variant_exploration/v1/
├── run_relation_matrix.csv
├── mechanism_diagnostics.csv
├── k_language_relation_summary.csv
├── k_global_relation_summary.csv
├── summary.json
├── SHA256SUMS
└── runs/
```

The first files to inspect are:

```text
k_global_relation_summary.csv
mechanism_diagnostics.csv
summary.json
```

`summary.json` reports the mean token-Markov delta for K=1, K=2, and K=4 and
whether that mean trajectory is monotonically non-decreasing.

## 12. Interpretation rules

Phase 3A is allowed to teach us which part of the mechanism works and which K
region is interesting.

It is **not** allowed to turn the best observed K into confirmatory evidence.

If Phase 3A reveals a stable region that:

1. weakens or reverses the exact-token Markov advantage;
2. retains the other local predictive relations;
3. behaves similarly across languages and seeds;

then the next step is a separate **Phase 3B confirmatory protocol**.

Before Phase 3B:

- freeze the exact mechanism;
- freeze K;
- freeze all parameters;
- freeze decision thresholds;
- freeze analysis code.

Only after that should the untouched control reserved test be evaluated once.

The Voynich locked test remains untouched until the project's broader
structural-model freeze permits its use.

## 13. What a failure means

If increasing K suppresses exact-token predictability but simultaneously
causes HMM/slot/copy-edit behavior to diverge from Voynich, then constrained
final-position lexical variation is insufficient.

That negative result should be retained. The next mechanism class should then
change one additional generative property at a time rather than layering
multiple adaptive operations onto this model.
