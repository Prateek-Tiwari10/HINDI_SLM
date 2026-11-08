# Stage 7b — Model surgery (D9), Kaggle T4

**Status:** done 2026-11-09. **Gate 7b: PASS.** Every check passed on the real 61M checkpoint.
- **Kaggle run:** `prateek99491/hindi-lm-d7-surgery-check`, version 3 (GPU T4, torch 2.11.0+cu128), 118 s.
  - Started with **Save Version** in the Kaggle UI with `HF_TOKEN` attached.
  - Versions 1 and 2 were API pushes. They cannot read secrets, so downloading the private checkpoint failed with HTTP 401.
  - The code is embedded via `EMBED_CODE=1 kaggle/push.sh "python -m eval.stage7b" d7_surgery_check`. It includes the frozen base tokenizer `tokenizer/hi_unigram_10240.model`.
- **Base checkpoint:** `hindi-lm-checkpoints/pretrain_61m/step_007405` (the final D6 model). Tokenizer sha256 `70e469cd…eea27`, the same as Stage 7a.
- **Outputs:** [stage7b_integration.json](stage7b_integration.json), [stage7b_kernel.log](stage7b_kernel.log), [stage7b_pytest.txt](stage7b_pytest.txt). Nothing was uploaded to HF.

## Tests

- **Full suite before the check:** 104 passed, 1 skipped. This includes all 12 `test_surgery.py` tests.
- **Tokenizer test 1:** `encode_dev` == SentencePiece on 1,000 real v2 sentences. **1 passed.** This is the test Stage 7a had to skip.

## Results

| check | `mean` init | `random` init |
| --- | --- | --- |
| params before → after | 61,128,320 → **61,169,280** (expected 61,169,280) ✅ | same ✅ |
| old embedding rows bit-identical | ✅ | ✅ |
| head tied to embedding (also after save → load) | ✅ | ✅ |
| new rows finite | ✅ | ✅ |
| new / old row norm (mean) | 0.435 / 1.285 = **0.34** | 0.507 / 1.285 = 0.39 |
| base-column logits on base-only input (CPU fp32) | **exact**, max diff 0.0 | **exact**, max diff 0.0 |
| GPU base-logit max abs diff | 0.0 | 0.0 |
| KL(base ‖ resized), mean over 102,300 positions | **2.7e-5** | 6.8e-4 |
| KL p99 / max | 2.3e-4 / 1.4e-3 | 5.0e-3 / 1.7e-2 |
| probability mass on new IDs, mean / max | 2.7e-5 / 1.4e-3 | 6.8e-4 / 1.7e-2 |

**What the KL shows:** base logits are unchanged exactly, so the KL comes only from the probability mass the new rows take. KL ≈ new-ID mass, which matches. With `mean` init the new tokens take about 0.003% of the mass on average. That is 25× less than with `random` init, and the model's Hindi behaviour is effectively untouched. **`mean` stays the default** (`load_base_for_finetune(init="mean")`).

## KV cache

Greedy decoding, 128 tokens from each of 5 Devanagari prompts:

| device | tokens identical with / without cache | tok/s cache | tok/s no cache | speedup |
| --- | --- | --- | --- | --- |
| CUDA (T4) | ✅ | 92.4 | 90.1 | 1.03× |
| CPU | ✅ | 47.3 | 11.6 | 4.08× |

On the GPU the speedup is small. At batch size 1 and these lengths, decoding is limited by kernel-launch overhead, not compute. The result that matters is that the tokens are identical.

Greedy sample: "भारत की राजधानी नई दिल्ली" → "में स्थित है। यह भारत का सबसे बड़ा और विश्व का 10वां सबसे बड़ा राष्ट्रीय उद्यान है। …" It is fluent Hindi but factually wrong, as seen in Stage 6.

## Smoke probe (before any fine-tuning)

Input `<rom> kal meeting cancel ho gayi kya <dev>` (32 IDs), generated with the Devanagari-only output mask (9,703 allowed IDs):
- **Greedy:** "की वजह से 2010 में 100000000000000000000000000000". All 40 IDs are allowed, then it falls into a digit loop.
- **T=1.0:** "।\nबिहार(संस्कृति संरक्षण विधेयक 22020' का प्रयोग …". All 40 IDs are allowed.

The mask works: no Latin or new-token IDs were generated. The outputs are not transliterations, as expected before fine-tuning. The base model has never seen `<rom>`/`<dev>` or character-level Latin.

## Next

Gate 7b passed, so D7 can move on to fine-tuning in two phases (D9):
1. **Phase A:** train only the new rows.
2. **Phase B:** full fine-tune.

Both phases start from `step_007405` with `mean` init and a fresh optimizer.
