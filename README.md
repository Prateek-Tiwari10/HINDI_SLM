# HINDI_SLM — a Hindi language model from scratch

## What it is

A ~60M-parameter decoder-only language model for Hindi. Everything is trained from scratch: the data pipeline, the tokenizer, the model and the training code.
- No pretrained models or tokenizers are used anywhere.
- This repo currently covers decisions D1–D4: data collection, cleaning, dedup, splits, tokenizer selection and the final tokenized dataset.

## Results

_To be filled after pretraining._

## Data

**Sources** (details and licences in [data/SOURCES.md](data/SOURCES.md)):
- **Sangraha Verified, Hindi** (AI4Bharat, CC-BY-4.0): the bulk; 25 of its 100 files were enough.
- **Hindi Wikipedia** `20231101.hi` (CC-BY-SA): all of it, upsampled 2× in train.
- **Our own crawl of Hindi news sites:** private research use only.
- **FineWeb-2 Hindi:** not needed.
- **Sangraha Synthetic** (machine-translated): excluded by design.

**Pipeline** ([data/](data/); runs as Kaggle CPU kernels, every step resumable through done-markers on the Hub):
1. `normalize()`: NFC (not NFKC); ZWJ/ZWNJ kept only between Devanagari letters; `|` after Devanagari becomes `।`; control characters removed; whitespace collapsed. Idempotent.
2. Six heuristic filters, cheapest first:
   - at least 50 words;
   - Devanagari ≥ 80% of letters;
   - unique lines ≥ 70%;
   - symbols/words ≤ 0.30;
   - mean word length 2–15;
   - boilerplate phrases.
   Emails and Indian phone numbers are masked.
3. Exact dedup (xxh64), then MinHash near-dedup:
   - word 5-grams, 128 permutations;
   - LSH with 16 bands × 8 rows;
   - verified at Jaccard ≥ 0.8;
   - union-find; the kept copy is the one from the preferred source.
4. Document-level split by `xxh64(id) mod 10,000`: 99.5 / 0.25 / 0.25. Near-dedup runs before the split, so val/test hold no near-duplicates of train.

**Funnel** (docs / GB of text; full report in [results/stage3_RESULTS.md](results/stage3_RESULTS.md)):

| source | raw | after filters | after exact dedup | after near-dedup | final |
| --- | --- | --- | --- | --- | --- |
| Wikipedia | 163,093 / 0.63 | 93,146 / 0.57 | 93,054 / 0.57 | 86,499 / 0.56 | 86,499 / 0.56 |
| Manual news crawl | 377,324 / 2.59 | 286,741 / 1.68 | 285,503 / 1.68 | 284,508 / 1.67 | 284,508 / 1.67 |
| Sangraha Verified | 4,355,234 / 24.22 | 4,182,186 / 23.24 | 4,097,801 / 22.77 | 4,084,777 / 22.67 | 4,082,876 / 22.66 |

**Final tokenized dataset** (`hi_unigram_8192`, `uint16`, `bos + doc + eos`, shuffled with seed 1337):

| split | docs | tokens |
| --- | --- | --- |
| train (Wikipedia ×2) | 4,517,509 | **2.824B** |
| val | 11,157 | 6.84M |
| test | 11,280 | 7.02M |

- Measured compression: **8.99 UTF-8 bytes per token**.
- Train is 88.5% Sangraha, 6.7% manual news and 4.8% Wikipedia.
- Shards: `tokenized/hi_unigram_8192/train_000.bin … train_005.bin`, `val.bin`, `test.bin`, `meta.json` in the HF data repo.

## Tokenizer

**`hi_unigram_8192`**: SentencePiece Unigram with an 8,192-token vocab.
- **Settings:** byte fallback, split digits, identity normalization (our `normalize()` already ran).
- **Special tokens:** pad/unk/bos/eos = 0/1/2/3, plus chat tokens `<|system|>`, `<|user|>`, `<|assistant|>` and 16 reserved tokens.
- **Selection:** chosen from 8 candidates ({BPE, Unigram} × {8K, 10K, 12K, 16K}, trained on a 946 MB train sample) by an intrinsic-only rule:
  - **Vocab size:** the smallest size after which fertility improves by less than 3%. For both algorithms that is 8,192, and the embedding share there is 7.0%.
  - **Algorithm:** at that size BPE and Unigram are within 2% on fertility (1.447 vs 1.474 tokens/word), so the tie goes to Unigram. Unigram also scores better on the morphology check, landing on the stem/suffix boundary for 47.4% of test words vs BPE's 42.9%.
- **Example:** Unigram splits `जाएँगे` as `जा|एँ|गे`; BPE splits it as `जाए|ँ|गे`.

Full sweep: [results/stage4_RESULTS.md](results/stage4_RESULTS.md), [results/tokenizer_sweep.csv](results/tokenizer_sweep.csv), [results/plots/](results/plots/).

## Reproduce

Heavy work runs on Kaggle CPU kernels (account `prateek99491`), never on the laptop.

```bash
# laptop: thin CLI env for pushing kernels and reading the Hub
python -m venv .venv && .venv/bin/pip install kaggle huggingface_hub

# push one stage (uploads code to the HF repo's code/ folder, then pushes the kernel);
# then press "Save Version" in the Kaggle editor so the run can read the HF_TOKEN secret
kaggle/push.sh "python -m data.pipeline stage1"
kaggle/push.sh "python -m data.pipeline stage2"
kaggle/push.sh "python -m data.pipeline stage3a && python -m data.pipeline stage3b && python -m tokenizer.sweep && python -m tokenizer.encode"
```

All thresholds and seeds are in [configs/data.yaml](configs/data.yaml) and [configs/tokenizer.yaml](configs/tokenizer.yaml). Tests (`pytest`) run inside every kernel before the stage starts.

## Links

- Data (private): https://huggingface.co/datasets/Prateek-Tiwari10/hindi-lm-data
  - final text splits: `final/{train,val,test}/`
  - tokenized: `tokenized/hi_unigram_8192/`
- Tokenizers (private): https://huggingface.co/Prateek-Tiwari10/hindi-lm-tokenizer
  - all 8 sweep models: `sweep/`
  - the chosen one: `sweep/hi_unigram_8192.model`, also committed as `tokenizer/hi_unigram_8192.model`
- Stage reports: [results/](results/)
