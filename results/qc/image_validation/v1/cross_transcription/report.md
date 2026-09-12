# Cross-Transcription Agreement — Frozen Image-QC Sample

This report compares the sampled loci across **ZL3b, GC2a, and IT2a** after converting each native transcription alphabet to STA1.

It is a triage report, not an adjudication of which transcription is paleographically correct.

## Summary

- Sampled loci: **57**
- Exact three-way token agreement: **7/57**
- GC2a + IT2a consensus against primary ZL3b: **0**
- Recommended manuscript-image checks: **10**

### Agreement classes

- `MODERATE_GLYPH_AGREEMENT`: 25
- `LOW_GLYPH_AGREEMENT`: 17
- `ALL_THREE_EXACT`: 7
- `BOUNDARY_ONLY_DISAGREEMENT`: 5
- `HIGH_BUT_NONEXACT_GLYPH_AGREEMENT`: 2
- `COVERAGE_DIFFERENCE_SPECIAL_MARGINALIA`: 1

## Recommended Manual Checks

For these cases, you are **not** being asked to re-transcribe the whole line. Locate the correct manuscript region and focus only on the disagreement described in `review_reason`. If the glyph distinction is not obvious, record it as unresolved rather than guessing.

### 01. `f72r3.34,@Cc` — MANDATORY_DISAGREEMENT

- Folio: `f72r3`
- Class: `BOUNDARY_ONLY_DISAGREEMENT`
- Reason: All three agree on glyphs but disagree on token boundaries; this is directly relevant to segmentation-sensitive models.
- Primary support: `NO_EXACT_TOKEN_SUPPORT`
- Minimum pairwise glyph similarity: `1.0`

**ZL3b**

```text
<!09:00>okeos.aiin,olaiin.oraiin.octheolarl.okeeody.oteos.aiin,koly
```

**GC2a**

```text
ohcos.am.oeam.oyam.oKcoe.aye.ohcco,89.okcos.am.hoe9
```

**IT2a**

```text
okeos.aiin.olaiin.oraiin.octheol.arl.okeeody.oteos.aiin.koly
```

### 02. `f114r.23,+P0` — MANDATORY_DISAGREEMENT

- Folio: `f114r`
- Class: `BOUNDARY_ONLY_DISAGREEMENT`
- Reason: All three agree on glyphs but disagree on token boundaries; this is directly relevant to segmentation-sensitive models.
- Primary support: `NO_EXACT_TOKEN_SUPPORT`
- Minimum pairwise glyph similarity: `1.0`

**ZL3b**

```text
or,cheo.al.taiin.qokedaiin.oeain.al.s.ain,ches<$>
```

**GC2a**

```text
oy.1co.ae.kam.4ohc8,am.ocan.ae.s.an.1cs<$>
```

**IT2a**

```text
orcheo.al.taiin.qokedaiin.oeain.al.s.ain.ches<$>
```

### 03. `f101v.1,@Lf` — MANDATORY_DISAGREEMENT

- Folio: `f101v`
- Class: `BOUNDARY_ONLY_DISAGREEMENT`
- Reason: All three agree on glyphs but disagree on token boundaries; this is directly relevant to segmentation-sensitive models.
- Primary support: `EXACT_IT2a`
- Minimum pairwise glyph similarity: `1.0`

**ZL3b**

```text
[s:r]airaly
```

**GC2a**

```text
saz,ae9
```

**IT2a**

```text
sairaly
```

### 04. `f111v.51,+P0` — MANDATORY_DISAGREEMENT

- Folio: `f111v`
- Class: `BOUNDARY_ONLY_DISAGREEMENT`
- Reason: All three agree on glyphs but disagree on token boundaries; this is directly relevant to segmentation-sensitive models.
- Primary support: `EXACT_GC2a`
- Minimum pairwise glyph similarity: `1.0`

**ZL3b**

```text
sain.cheal.chckhy.okain.okal,aiin.choty.chckhy<$>
```

**GC2a**

```text
san.1cae.1H9.ohan.ohae,am.1ok9.1H9<$>
```

**IT2a**

```text
sain.cheal.chckhy.okain.okalaiin.choty.chckhy<$>
```

### 05. `f114r.45,+P0` — MANDATORY_DISAGREEMENT

- Folio: `f114r`
- Class: `BOUNDARY_ONLY_DISAGREEMENT`
- Reason: All three agree on glyphs but disagree on token boundaries; this is directly relevant to segmentation-sensitive models.
- Primary support: `EXACT_IT2a`
- Minimum pairwise glyph similarity: `1.0`

**ZL3b**

```text
dain.ched.chodaiin.otain.chdar.chedy.chocthy<$>
```

**GC2a**

```text
8an.1c8.1o,8am.okan.18ay.1c89.1oK9<$>
```

**IT2a**

```text
dain.ched.chodaiin.otain.chdar.chedy.chocthy<$>
```

### 06. `f112r.45,+P0` — TARGETED_DISAGREEMENT

- Folio: `f112r`
- Class: `LOW_GLYPH_AGREEMENT`
- Reason: At least one pair has glyph similarity below 0.85.
- Primary support: `NO_EXACT_TOKEN_SUPPORT`
- Minimum pairwise glyph similarity: `0.738095`

**ZL3b**

```text
yor.aiin.o,keeey,teey.shkar.oteeedyqokeey.okeey.okary.oin.yky<$>
```

**GC2a**

```text
9ay.am.o,hd9.kC(.2hay.okd89,4ohC9.ohC9.ohas9.am.9h9<$>
```

**IT2a**

```text
yar.aiin.okeeey.teey.shkar.oteeedy.qokeey.okeey.okary.oi?n.yky<$>
```

### 07. `f41r.1,@P0` — TARGETED_DISAGREEMENT

- Folio: `f41r`
- Class: `LOW_GLYPH_AGREEMENT`
- Reason: At least one pair has glyph similarity below 0.85.
- Primary support: `NO_EXACT_TOKEN_SUPPORT`
- Minimum pairwise glyph similarity: `0.75`

**ZL3b**

```text
<%>pshey,kedal[ee:ch]y.oked.seekeeey<->opshes.[o:y]p[ch:ee]d.qotchedy.shkakeedy
```

**GC2a**

```text
<%>j2c9,hc8aecc9.ohc8.2chc19.oj2cs.9g18.4ok1c79.2WAh189
```

**IT2a**

```text
<%>pchey.kedaleey.oked.shekchey<->opshes.ypchd.qotcheedy.shkakeedy
```

### 08. `f68r2.31,@Cc` — TARGETED_DISAGREEMENT

- Folio: `f68r2`
- Class: `LOW_GLYPH_AGREEMENT`
- Reason: At least one pair has glyph similarity below 0.85.
- Primary support: `NO_EXACT_TOKEN_SUPPORT`
- Minimum pairwise glyph similarity: `0.783784`

**ZL3b**

```text
<!09:30>okey.okoaiin.okol.o,ky.oeeeo.r.okey.okeeol.cheo.o.@231;@232;@233;@234;
```

**GC2a**

```text
ohc9.ohoam.ohoe.oh9.oCco.y.chc9.ohCoe.1co.A.?oI?
```

**IT2a**

```text
okeo.okoaiin.okol.oky.oeeeo.r.ekey.okchol.cheo.o.koiin
```

### 09. `f69v.16,@Ri` — CONSENSUS_CONTROL

- Folio: `f69v`
- Class: `ALL_THREE_EXACT`
- Reason: All three canonical STA1 token streams agree exactly.
- Primary support: `BOTH_GC_AND_IT_EXACT`
- Minimum pairwise glyph similarity: `1.0`

**ZL3b**

```text
<!02:45>oteol
```

**GC2a**

```text
okcoe
```

**IT2a**

```text
oteol
```

### 10. `f78r.1,@Lt` — CONSENSUS_CONTROL

- Folio: `f78r`
- Class: `ALL_THREE_EXACT`
- Reason: All three canonical STA1 token streams agree exactly.
- Primary support: `BOTH_GC_AND_IT_EXACT`
- Minimum pairwise glyph similarity: `1.0`

**ZL3b**

```text
<!16>okchdldlo
```

**GC2a**

```text
oh18e8eo
```

**IT2a**

```text
okchdldlo
```

