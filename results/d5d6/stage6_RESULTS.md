# Stage 6 — Evaluation (D6), Kaggle T4

**Status:** done 2026-10-27. **Gate 6 checks: all pass** (table below).
- **Kernel:** `prateek99491/hindi-lm-d6-eval`, run with `python -m eval.stage6` ([eval/stage6.py](../../eval/stage6.py)). It was started with Save Version in the Kaggle UI after the `HF_TOKEN` secret was attached; API-pushed version 1 had failed without it. Runtime about 6 min on one T4 (`elapsed_s` 346), torch 2.11.0+cu128.
- **Checkpoints scored:** `best/` (step 7,250, chosen during training on the 512-window val subset) and the final `step_007405`. Both are scored on the same full sets because `best/` was chosen on a different set of windows than the final full-set number.
- **Outputs:** [stage6.json](stage6.json), [samples.md](samples.md), [samples.jsonl](samples.jsonl), plots in [plots/](plots/) (`stage6_loss_vs_tokens.png`, `stage6_lr.png`, `stage6_grad_norm.png`). Copies are on HF `hindi-lm-checkpoints/d6_results/`.

## Loss, perplexity, bits per byte (all windows)

Non-overlapping 1024-token windows. BPB is measured from the exact UTF-8 byte count of the decoded windows, excluding bos/eos.

| checkpoint | split | windows | target tokens | loss (nats) | PPL | BPB |
| --- | --- | --- | --- | --- | --- | --- |
| `best/` (7,250) | val | 4,468 | 4,570,764 | 2.8071 | 16.56 | 0.4356 |
| `best/` (7,250) | test | 4,559 | 4,663,857 | 2.8173 | 16.73 | 0.4384 |
| **final (7,405)** | val | 4,468 | 4,570,764 | **2.8050** | **16.53** | **0.4353** |
| **final (7,405)** | test | 4,559 | 4,663,857 | **2.8155** | **16.70** | **0.4381** |

- **The final checkpoint is slightly better than `best/`** on both full sets: −0.0021 nats on val and −0.0018 on test. `best/` won on the 512-window subset (2.7825), but the subset was too small to separate two checkpoints that close; the LR was still decaying over the last 155 steps. **Use `step_007405` as the pretrained model.**
- **Test is 0.010 nats above val** (PPL 16.70 vs 16.53). That gap is small and in the expected direction, so val did not overfit through checkpoint selection.
- The loss curve ([plots/stage6_loss_vs_tokens.png](plots/stage6_loss_vs_tokens.png)) is still falling slowly at the end of the run. On the val subset it went from 2.803 at 1.64B tokens to 2.783 at 1.90B, a drop of 0.02 nats over the last 0.26B tokens. More data would still help; the model is not saturated.

## Generation (`best/`, 10 prompts × 3 settings × 150 new tokens)

Prompts are in [eval/prompts.txt](../../eval/prompts.txt). Sampling seed = 1337 + 10·prompt + setting. The metrics use whitespace-split words of the continuation only.

| setting | 4-gram repetition (mean) | Distinct-1 | Distinct-2 | mean words |
| --- | --- | --- | --- | --- |
| greedy | 0.781 | 0.111 | 0.183 | 114.6 |
| T=0.7, top-k 50 | 0.116 | 0.355 | 0.703 | 108.9 |
| T=1.0 | 0.002 | 0.562 | 0.943 | 106.6 |

What the samples show ([samples.md](samples.md)):
- **Grammar and register are good.** Sentences are well-formed Hindi with correct postpositions and agreement, and the model switches register with the prompt: news style for "प्रधानमंत्री ने आज संसद में कहा कि", folk-tale openings for "एक समय की बात है", encyclopedic style for "गंगा नदी".
- **Greedy decoding loops** on all 10 prompts, usually within 1–2 sentences ("2014 में 100 साल पूरे होने पर …" repeated; "हिन्दी भाषा का इतिहास" repeated as a heading). That is the usual behaviour of a small base model under greedy decoding, and it is what the 0.78 4-gram repetition measures.
- **T=0.7, top-k 50 is the best setting.** It stays on topic with little looping (repetition 0.12), but phrase-level echoes still appear (for example the dialogue in the village story).
- **T=1.0 is fluent but drifts** off topic within 2–3 sentences, with occasional made-up words.
- **Facts are not reliable**, as expected at 61M parameters with no instruction tuning. Examples: Gandhi born "16 अप्रैल 1860 … हिसार" (T=1.0) and "१९ जनवरी १९१७ … तिरुनेलवेली" (T=0.7); the true answer is 2 October 1869, Porbandar. Dates and places come out confidently but are wrong.

## Gate 6 checks

| check | result |
| --- | --- |
| val and test loss / PPL / BPB over **all** windows | ✅ 4,468 val and 4,559 test windows, both checkpoints |
| plots: loss vs tokens, LR vs steps, grad norm vs steps | ✅ `plots/stage6_*.png` (title, axis labels, legend where there are two lines) |
| samples: 10 prompts × {greedy, T=0.7 top-k 50, T=1.0} × 150 tokens → `samples.md` | ✅ 30 samples |
| 4-gram repetition and Distinct-1/2 | ✅ table above |
| `stage6_RESULTS.md`, `PROGRESS.md`, D5 and D6 entries in `DECISIONS.md` | ✅ this file; the other two updated on 2026-10-27 |

## Surprises and notes

- **Missing space between prompt and continuation** in `samples.md` ("दिल्ली**में"). The cause: decoding only the new tokens drops SentencePiece's leading `▁`. It affects display only, not the tokens or the metrics. Fixed in `eval/stage6.py` (decode prompt + continuation, then strip the prompt). This run's files were produced before the fix.
- Samples were generated from `best/` as the spec says. Since the final checkpoint is 0.002 nats better, its samples would be practically the same.
- **Possible later changes:** document-level attention masking (windows currently cross document boundaries, Stage 5 note); a repetition penalty or top-p for demos; more tokens, since the loss is still falling.
