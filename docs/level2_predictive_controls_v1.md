# VOYAGER Level 2 Predictive-Hierarchy Controls v1

## Confirmatory question

Do any of the seven already-frozen historical cipher-control families cause
the same predictive model ordering observed in Voynich validation data?

The Level-2 target is deliberately narrower than "which control looks most
Voynich-like?" It tests a previously observed predictive signature:

- character trigram > categorical HMM;
- character trigram > finite-template slot grammar;
- character trigram > exact token Markov;
- character trigram > local copy-edit/self-copy.

The HMM comparison has two previously frozen event views, space-free and
token-aware, so the implementation contains five operational relations.

## Why all seven families are tested

Level 1 suggested that nomenclator-like encoding and null insertion reproduce
different subsets of the descriptive Voynich constraints. That observation
must not be used to select only those families for confirmatory testing.

Level 2 therefore evaluates all seven frozen families:

1. monoalphabetic substitution;
2. homophonic substitution;
3. verbose substitution;
4. null insertion;
5. nomenclator-like encoding;
6. columnar transposition;
7. syllabic/hybrid encoding.

All four frozen source languages and all five frozen cipher keys are retained.

## Control data split

The split unit is one source-text line, which is also one line in each
generated control JSONL file. Membership is assigned without examining text:
source-line indices are ranked by SHA-256 of the frozen split seed, source ID,
and line index.

Weights reproduce the project's 72/15/15 physical-leaf proportions:

- TRAIN weight: 72
- VALIDATION weight: 15
- RESERVED TEST weight: 15

Integer counts use largest-remainder allocation.

Expected v1 counts:

| Source | Lines | Train | Validation | Reserved |
|---|---:|---:|---:|---:|
| Latin UDante | 725 | 512 | 107 | 106 |
| Italian Old | 740 | 522 | 109 | 109 |
| German ReM | 4,470 | 3,155 | 658 | 657 |
| French PROFITEROLE | 2,632 | 1,858 | 387 | 387 |

Exactly the same membership is used for every family and key generated from a
given source language.

Reserved-test indices are frozen at this stage, but the Level-2 runner streams
past those JSONL rows without parsing, tokenizing, fitting, or scoring them.
They are not used for the present experiment.

## Frozen model procedures

The existing VOYAGER implementations are reused rather than rewritten.

### Character trigram

Interpolated Witten-Bell trigram.

For HMM comparisons, it uses the exact matching event view:

- space-free;
- token-aware with explicit word-boundary events.

For token-based competitors, the matched comparator independently resets at
every token and appends one explicit `<EOT>` event.

### Categorical HMM

- views: space-free and token-aware;
- hidden states: 2, 4, 8, 16;
- 5 deterministic restarts per state count;
- restart chosen by TRAIN likelihood only;
- VALIDATION chooses state count by lowest bits/event;
- lower state count wins an exact tie;
- pseudocount 0.5;
- maximum 50 EM iterations, minimum 5;
- tolerance 1e-5 bits/event;
- batch size 256.

No state-count expansion is permitted after Level-2 results are observed.

### Finite-template slot grammar

- K = 1, 2, 4, 8, 16 templates;
- 8 relative slots;
- maximum token length 64;
- 5 deterministic restarts;
- restart chosen by TRAIN likelihood only;
- VALIDATION chooses K by lowest bits/event;
- lower K wins a tie;
- pseudocount 0.5;
- maximum 50 iterations, minimum 5;
- tolerance 1e-5 bits/token.

### Exact token Markov

- context orders: 1 and 2 previous tokens;
- maximum context: 2;
- exact token identities;
- character-trigram + `<EOT>` open-vocabulary base;
- VALIDATION chooses lower bits/event;
- lower context order wins a tie.

The runner explicitly verifies that the internal character base has the same
validation NLL as the external matched token-reset trigram.

### Local copy-edit

- source windows: 1, 4, 16 previous tokens;
- maximum token length 64;
- pseudocount 0.5;
- training source inference and edit channel are unchanged from the frozen
  Voynich model;
- rho is fit on TRAIN only;
- VALIDATION chooses source window by lowest bits/event;
- lower window wins a tie.

## Paired uncertainty analysis

Every selected competitor is compared with its exact matched trigram by a
10,000-replicate paired bootstrap over validation source-line units.

For every bootstrap replicate, the two models are evaluated on the same
resampled units and corpus bits/event is calculated as a ratio of sums.

The reported delta is:

`competitor bits/event - trigram bits/event`

Therefore a positive delta has the same direction as every frozen Voynich
comparison: the character trigram is better.

The bootstrap is conditional on the validation-selected hyperparameter. It
does not include uncertainty induced by state-count/template/context/window
selection. This limitation must be reported.

## Frozen reproduction criterion

For one control run and one operational relation:

**Strong direction match**
- point delta > 0; and
- paired-bootstrap 95% CI lower bound > 0.

For one family × language × relation:

**Language relation replication**
- at least 4 of 5 frozen cipher keys are strong direction matches.

For one family × relation across languages:

**Family relation robust**
- language relation replication occurs in at least 3 of 4 languages.

For a family overall:

**Full predictive-hierarchy reproduction**
- all five operational relations are family-relation robust.

Cipher keys are robustness perturbations. They are not treated as independent
biological/social experimental replicates.

## Interpretation

A full Level-2 match means that a mechanism family can induce the same broad
predictive successes and failures as the Voynich validation corpus across
multiple source languages and cipher keys. It does not establish that the
Voynich Manuscript historically used that cipher, and it is not decipherment.

Failure of full hierarchy reproduction is also informative. Relation-level
matches must still be reported, because a mechanism can reproduce one aspect
of the predictive signature while failing others.

No new cipher family, combined nomenclator+null model, model grid, or threshold
may be introduced into this frozen v1 after results are observed. Such work
belongs in a separately labeled exploratory/v2 experiment.

## Leakage guard

- Voynich `test_LOCKED.txt`: never accessed.
- Level-2 reserved control test: indices frozen, units not parsed or scored.
- All model selection occurs on Level-2 validation only after TRAIN fitting.
