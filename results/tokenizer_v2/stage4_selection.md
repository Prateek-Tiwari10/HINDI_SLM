# Stage 4 selection (v2)

Sample: `{"train_bytes": 17458269539, "bands": [{"fraction": 0.05, "docs": 156297, "bytes": 870605684, "chars": 340273226, "lines": 1207783}, {"fraction": 0.15, "docs": 313402, "bytes": 1735465080, "chars": 678353242, "lines": 2415194}, {"fraction": 0.3, "docs": 472146, "bytes": 2622794295, "chars": 1025124768, "lines": 3646421}, {"fraction": 0.5, "docs": 627319, "bytes": 3485021460, "chars": 1362108946, "lines": 4850935}], "val_docs_available": 7873, "val_docs_used": 7873}`
Eval: `{"val_docs": 7873, "val_words": 3272241, "morph_words": 154}`

## Ladder: tokenizer sample size

- machine: 64 threads, 399 GB RAM available

- the last two rungs are within the tolerances -> the 50% extension is not needed

### Rule

- reference = largest trained rung: 29.9% of train (fertility 1.4329, heavy splits 4.21%)
- 5.0%: fertility 1.4335 (+0.04% vs reference, tolerance 0.5%), heavy splits 4.22% (+0.01 pp, tolerance 0.10 pp) -> within tolerance
  -> choose the 5.0% sample: the smallest within tolerance

### Rungs

| sample_pct | sample_lines | fertility | chars_per_token | byte_fallback_pct | heavy_split_pct | morphology_pct | train_seconds | peak_ram_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.99 | 1207783 | 1.4335 | 3.5591 | 1.4247 | 4.2192 | 46.1039 | 377.9 | 9.6 |
| 14.93 | 3622977 | 1.4335 | 3.559 | 1.4247 | 4.2175 | 46.7532 | 1221.9 | 28.26 |
| 29.95 | 7269398 | 1.4329 | 3.5605 | 1.419 | 4.2095 | 46.7532 | 2813.4 | 88.63 |

## Grid: trained on the 5.0% sample
## Step 1 — vocab size per algorithm

- bpe 8192 -> 10240: fertility 1.4585 -> 1.4147, gain 3.00% (threshold 3%), embedding 7.0%
- bpe 10240 -> 12288: fertility 1.4147 -> 1.3839, gain 2.18% (threshold 3%), embedding 8.7%
  -> bpe: choose 10240
- unigram 8192 -> 10240: fertility 1.4335 -> 1.3934, gain 2.80% (threshold 3%), embedding 7.0%
  -> unigram: choose 8192

## Step 2 — common size

- BPE chose 10240, Unigram chose 8192 -> compare at 10240

## Step 3 — algorithm

- fertility: BPE 1.4147 vs Unigram 1.3934 (diff 1.53%, tie margin 2%)
- heavy splits: BPE 3.62% vs Unigram 3.66%
- morphology: BPE 39.6% vs Unigram 44.2%
  -> tie: goes to unigram (decision log: Unigram preferred on ties)

## Step 4 — optional tie-break

- BPE and Unigram are within 2% on fertility: the optional tie-break (one ~20-minute proxy run each) is available. Not run; Prateek decides.

**Recommended tokenizer: `hi_unigram_10240`**