# Stage 7d — SFT harness (Kaggle T4)

**Status:** done 2026-11-17. **Gate 7d: PASS. All six gates pass.**
- **G1–G3:** pytest on a tiny CPU model.
- **G4–G6:** the real 61M model on one T4.
- **Kaggle run:** `prateek99491/hindi-lm-d7d-gates`, version 2.
  - It was started with Save Version, with the HF_TOKEN secret ticked.
  - It ran Stage 7c (`all`) first, then `python -m finetune.stage7d`.
  - Environment: torch 2.11.0+cu130, Python 3.13.16, Tesla T4.
  - The runner runs the full test suite first: **149 passed, 1 skipped**.
- **Gate data:** pairs revision `cee750e` (`romanize-v1`).
  - The data was later rebuilt as `romanize-v2`; see [stage7c_RESULTS.md](stage7c_RESULTS.md).
  - The gates check the harness (masking, padding, resume, freezing, decoding), not the spelling content, so they stand for v2 without a rerun.
- **Outputs:** [stage7d_gates.json](stage7d_gates.json).

## Code

| file | what it holds |
| --- | --- |
| [finetune/data.py](../../finetune/data.py) | `PairArrays` (memory-mapped `uint16` arrays from 7c) |
| | `BatchPlan`: a seeded permutation per epoch; sorted by length within chunks of 100 global batches; batch order shuffled. Global step → (epoch, batch), so resume needs only the step. |
| | right padding to buckets {64, 128, 192, 256} |
| | `masked_loss_sum`: fp32 cross-entropy summed over target positions |
| | `decode_budget` |
| [finetune/sft.py](../../finetune/sft.py) | the loop |
| | loss: summed cross-entropy / global target count × world size, because DDP averages gradients |
| | optimizer: AdamW, gradient clip 1.0, fp16 autocast with GradScaler, warmup then cosine decay to 10% |
| | Phase A: `freeze_for_phase_a` and `phase_a_param_groups`, wd 0 |
| | Phase B: the pretraining `param_groups` |
| | DDP through `torchrun` |
| | resume: `CheckpointManager` (latest step with optimizer state) |
| | weights-only exports: `best/` (lowest selection dev CER), `epoch<k>/`, `final/`, then `done.json` |
| [finetune/stage7d.py](../../finetune/stage7d.py) | the real-model gates below |
| [tests/test_sft.py](../../tests/test_sft.py) | G1–G3 plus batching, phase-A, eval-plan and decode-budget checks |

## Gates

| gate | check | result |
| --- | --- | --- |
| G1 mask count | non-ignored labels = Σ (len(dev_ids) + 1) | ✅ `test_g1_mask_count`, `test_shift_convention` |
| G2 padding invariance | a pair's loss alone = inside a padded batch (fp32, atol 1e-6) | ✅ `test_g2_padding_invariance` |
| G3 resume | interrupted + resumed run = uninterrupted run, 10 steps, exact | ✅ `test_g3_resume_matches_uninterrupted` |
| G4 overfit | 32 train pairs, full fine-tune, LR 3e-4: loss < 0.05 **and** 32/32 exact match within 300 steps | ✅ **step 35**: loss 0.0070, 32/32 EM (59 s) |
| G5 phase-A freeze | 50 phase-A steps: base rows and all non-embedding tensors bit-identical; new rows moved | ✅ 0 frozen tensors changed; 64/64 new rows changed (mean ‖Δ‖ 0.156 over the 50 the data uses) |
| G6 eval sanity | step-0 `val_syn` loss and `dev_real` CER/WER/EM, 500 fixed pairs, masked greedy decoding + KV cache | ✅ val_syn loss 4.922; dev CER 9.21, WER 14.13, EM 0.000 (n = 500) |

**G4 loss** (every 10 steps): 4.863 → 1.278 → 0.197 → 0.040.

| step | exact match |
| --- | --- |
| 25 | 14/32 |
| 31 | 29/32 |
| 32 | 31/32 |
| 33 | 30/32 |
| 34 | 31/32 |
| 35 | 32/32 |

**What G6 shows:**
- **CER is above 1 at step 0, as expected.** The base model has never seen `<rom>…<dev>` and never emits `</s>` after the Devanagari part. Every hypothesis runs to the 256-token cap, for example "के लिए एक बार फिर से एक बार फिर से …".
- **Decoding cost 1,762 s for 500 pairs**, because of that same 256-token run-on.
- **Fix:** a per-prompt decode budget from the data (below).

## Decode budget (added after G6)

`max_new = ceil(ratio × 1.5 × rom_chars) + 8`, capped at 256. The terms:
- `rom_chars = prompt_len − 2`. The Latin side is one ID per character.
- `ratio` is the 99.9th percentile, over `train`, of Devanagari tokens (including eos) per romanized character.
- Stage 7c measures `ratio` and writes it to `meta.json` → `decode_cap`. `sft.py` reads it from there.

The ratio itself, and how many real references the cap would cut off, are in [stage7c_RESULTS.md](stage7c_RESULTS.md).

## Dev evaluation and model selection (added after the 7c contamination finding)

- **Contamination tags:** Stage 7c tags every `dev_real` / `test_real` record `exact` / `near` / `clean`, meaning seen in v2 pretraining or not.
- **If the contaminated share exceeds `eval.clean_selection_above` (0.25):**
  - the eval set is 500 clean + 500 contaminated pairs (fixed seed);
  - **model selection uses the clean CER**.
- **Otherwise:** one uniform 500-pair subset, and selection uses all of it.
- **Logging:** every eval logs CER / WER / EM for `all`, `clean` and `contaminated`.

## torch.compile: off

The probe ran 40 phase-B steps, global batch 64, on 1 T4:

| | s/step (steps 21–40) | first step | graphs |
| --- | --- | --- | --- |
| eager | 0.378 | 0.40 s | — |
| compiled (`dynamic=False`) | 1.485 | 30.5 s | 3 (one per bucket shape seen) |

- **Compiled was about 4× slower** over steps 21–40.
- **The probe does not show compile's steady-state speed:**
  - each new bucket shape costs another ~30 s compile;
  - a third shape (192) first appeared inside the timing window;
  - 40 steps is too short to amortize compiling.
- **Decision:** with only a few minutes per thousand eager steps at stake, `compile: false` in `configs/sft.yaml`. It does not affect any result.
