# Stage 5 — Final tokenized dataset

**Status:** done; **train tokens = 2.824B ≥ 1.8B**.
- **Kaggle run:** `prateek99491/hindi-lm-runner` version 9 (`python -m tokenizer.encode`, final step of the chained run).
- **Kaggle username verified:** `prateek99491`.

## What was run

1. **Encode:** each `final/{train,val,test}/*.parquet` shard encoded with `hi_unigram_8192` as `bos + tokens + eos` per doc.
   - Stored as `uint16`; the code asserts `vocab_size (8192) < 65536`.
   - Per-shard tokens and offsets are uploaded to `work/encoded/hi_unigram_8192/` with done-markers.
2. **Upsampling:** in train, every Wikipedia doc is included twice (D2).
3. **Shuffle:** document order shuffled with seed 1337 (+ split index), then the docs are concatenated.
4. **Shards:** `train_000.bin … train_005.bin` (500M tokens each, the last one shorter), `val.bin`, `test.bin`, and `meta.json`.
   - The tokenizer name, vocab size, bos/eos ids, seed, both config hashes and per-source counts are in `meta.json` (`results/stage5_meta.json`).
5. **Upload:** to `Prateek-Tiwari10/hindi-lm-data` under `tokenized/hi_unigram_8192/`.

## Shards on HF (`tokenized/hi_unigram_8192/`)

| file | tokens | size |
| --- | --- | --- |
| train_000.bin | 500,000,000 | 1000.0 MB |
| train_001.bin | 500,000,000 | 1000.0 MB |
| train_002.bin | 500,000,000 | 1000.0 MB |
| train_003.bin | 500,000,000 | 1000.0 MB |
| train_004.bin | 500,000,000 | 1000.0 MB |
| train_005.bin | 332,742,820 | 665.5 MB |
| val.bin | 6,860,319 | 13.7 MB |
| test.bin | 7,042,140 | 14.1 MB |

Train shards are cut every 500M tokens regardless of doc boundaries. The training loader should treat `train_*.bin` as one continuous stream, using `eos` (id 3) as the doc separator.

## Token counts

| split | docs | tokens incl. bos/eos | content tokens |
| --- | --- | --- | --- |
| train (Wikipedia ×2) | 4,517,509 | 2,832,742,820 | **2,823,707,802** |
| val | 11,157 | 6,860,319 | 6,838,005 |
| test | 11,280 | 7,042,140 | 7,019,580 |

**Train content tokens per source** (with upsampling):

| source | tokens | share |
| --- | --- | --- |
| sangraha_verified | 2,499,389,767 | 88.5% |
| manual | 187,838,169 | 6.7% |
| wikipedia (2 × 68.2M) | 136,479,866 | 4.8% |

**Check against D1:** 2.824B ≥ 1.8B. The D1 run needs about 1.8B tokens, so this is about 1.57 epochs of headroom, or a single pass over 64% of the data.

## Calibration: real bytes per token

| | value |
| --- | --- |
| estimate used in Stages 3–4 | 9.5 bytes/token |
| **measured** (train, each doc once, content tokens) | **8.989 bytes/token** |

The estimate overstated bytes per token by 5.7% (9.5 / 8.989). Real token counts are therefore about 5.9% above the Stage 3 estimates: 2.824B measured vs 2.666B estimated.

## README diff summary

- **Data section:** sources, the funnel table, final token counts, and the HF location of the shards.
- **Tokenizer section:** the choice (`hi_unigram_8192`), the key sweep numbers, the selection rule, and HF links.
- **Links:** now also point to `tokenized/hi_unigram_8192/` and the tokenizer sweep.

## Runtime

About 38 min:
- encoding 114 parquet shards (38 per split);
- assembling the shuffled `.bin` files;
- uploading about 6.7 GB.

## Notes

- **Kernel output size:** the tokenizer sample (946 MB) was written under `/kaggle/working`, so it was saved as kernel output and downloaded with the logs. It's harmless, since `results/kaggle_logs/` is git-ignored. The scratch paths now point at `/tmp` for any rerun.
