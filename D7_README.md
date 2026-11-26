# D7 — Status and next steps

_As of 2026-11-26_

**In short:** the 61M Hindi model has been fine-tuned to turn Hinglish (Hindi in Latin letters) into Devanagari.
- **Result:** the best checkpoint gets a **character error rate of 0.084** on human-written Dakshina dev sentences (Stage 7e, 22 Nov).
- **Done:** Steps 3–5 (Stages 7c–7e) and Step 6, run once on the held-out test sets. Normalized CER is **0.058** on Dakshina test and **0.040** on chat, vs 0.213 / 0.126 for a word-lookup baseline ([step6](results/d7_hinglish/step6_RESULTS.md)).
- **Released:** public weights at [Prateek-Tiwari10/hinglish-to-devanagari-61m](https://huggingface.co/Prateek-Tiwari10/hinglish-to-devanagari-61m), and a FastAPI demo for Azure Container Apps ([DEPLOY.md](DEPLOY.md)).

## Where we are

| part | status |
| --- | --- |
| Pretraining (D5/D6) | ✅ final checkpoint `step_007405`: test loss 2.816, perplexity 16.7 |
| Stage 7a: tokenizer extension | ✅ done 2026-10-27 |
| Stage 7b: model surgery | ✅ passed 2026-11-09 |
| Stage 7c: pair dataset | ✅ 2026-11-17, rebuilt as `romanize-v2.1` 2026-11-22 |
| Stage 7d: SFT harness + gates | ✅ all six gates pass, 2026-11-17 |
| Stage 7e: training | ✅ done 2026-11-22: dev CER 0.084 |
| Step 6: full evaluation | ✅ 2026-11-26: CER 0.058 / 0.040; ambiguity 0.843 vs baseline 0.785; no retrain |
| Release + demo | ✅ public weights; ✅ live at https://hinglish-translit.gentleground-e8984580.uaenorth.azurecontainerapps.io |

## What D7 is

The goal is to turn Hindi written in Latin letters (Hinglish) into Devanagari:

```
<rom> kal meeting cancel ho gayi kya <dev>  →  कल मीटिंग कैंसिल हो गई क्या </s>
```

Before training could start, two problems had to be solved:

1. **The tokenizer:** it barely knows Latin. Its 463 English fragments split Hinglish words inconsistently.
2. **The model:** it has no embeddings for the new tokens.

## Stage 7a — tokenizer extension (done)

The base `hi_unigram_10240` is **unchanged**, and new IDs are added after it. Code: [tokenizer/hinglish.py](tokenizer/hinglish.py). Config: [tokenizer/tokenizer_ext.json](tokenizer/tokenizer_ext.json).

| added                                              | count | IDs          |
| -------------------------------------------------- | ----- | ------------ |
| `<rom>`, `<dev>`                               | 2     | 10240–10241 |
| letters a–z                                       | 26    | 10242–10267 |
| digits 0–9                                        | 10    | 10268–10277 |
| punctuation `. , ? ! ' " - : ; ( ) /`            | 12    | 10278–10289 |
| `<lsp>` (Latin space), `<lunk>` (unknown char) | 2     | 10290–10291 |
| reserved                                           | 12    | 10292–10303 |

**New vocab: 10,304.**

- **Latin input:** split into single characters, after lowercasing and stripping accents. Spelling variants like "hai/hain/he" now share letters predictably.
- **Devanagari output:** uses the same `normalize()` + SentencePiece as pretraining.
- **Output mask:** only **9,703** IDs can be generated. Latin pieces, new tokens, pad/unk/bos and chat pieces are banned, so the model can only write Devanagari.
- **Training format:** `[<rom>] + latin + [<dev>] + devanagari + [</s>]`. Loss is computed only on the Devanagari part (labels are −100 before it).
- **Tests:** 7/7 passed on Kaggle. Test 1, on 1,000 real sentences, passed later in Stage 7b.
- **Report:** [results/d7_hinglish/stage7a_RESULTS.md](results/d7_hinglish/stage7a_RESULTS.md)

## Stage 7b — model surgery (passed)

The embedding table goes from 10,240 to 10,304 rows. Old rows are copied exactly, and the output head stays tied to the embedding. Each new row starts as the **mean of the old rows** plus small noise (`mean` init). Code: [model/surgery.py](model/surgery.py) (`load_base_for_finetune`). The check script is [eval/stage7b.py](eval/stage7b.py), run on Kaggle T4 (version 3).

| check                                          | result                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| params                                         | 61,128,320 → **61,169,280** (as expected)                    |
| old rows / tied head (also after save → load) | identical ✅                                                        |
| Hindi logits on normal text                    | **exactly the same** as the base model (max diff 0.0)         |
| KL(base ‖ resized), 102,300 positions         | `mean` init **2.7e-5**, `random` init 6.8e-4 (25× worse) |
| KV cache vs no cache                           | identical tokens on GPU and CPU (CPU speedup 4.1×)                 |
| output mask                                    | all generated IDs are Devanagari ✅                                 |
| unit tests                                     | 104 passed, 1 skipped                                               |

- **Init choice:** we keep `mean` init, because it barely affects the model's Hindi.
- **Smoke probe:** before training, the model writes Hindi text but not a transliteration (for example "की वजह से 2010 में…"). This is expected: it has never seen `<rom>`/`<dev>` yet.
- **Report:** [results/d7_hinglish/stage7b_RESULTS.md](results/d7_hinglish/stage7b_RESULTS.md)

## Stage 7c — pair dataset (done)

- **Training pairs are synthetic.** 600k of our own v2 Hindi sentences (70% conversational) were romanized by a generator:
  - chat shortcuts for common words;
  - human-attested spellings from the Dakshina and Aksharantar lexicons;
  - rules only as a fallback;
  - three noise levels.
- **Evaluation data is human:** `dev_real` / `test_real` are Dakshina's human-romanized sentences, and `test_chat` is Prateek's chat set.
- **`romanize-v2.1`** fixes seven mis-weighted frequent words (करने → `karane`, वे → `way`, …).
  - Coverage 0.863; word agreement with humans 0.643 (rule-only 0.620); 0 leakage.
- **About 75% of Dakshina dev/test** also appears in the pretraining corpus, so every result is split into clean and contaminated parts, and model selection uses the clean part.
- **Report:** [results/d7_hinglish/stage7c_RESULTS.md](results/d7_hinglish/stage7c_RESULTS.md)

## Stage 7d — SFT harness (done)

- **Code:** [finetune/sft.py](finetune/sft.py) handles length-bucketed batches, loss on the Devanagari part only, AdamW + fp16, DDP on 2×T4, resume, and greedy decoding with the output mask and a decode budget.
- **Gates:**
  - G1 mask count, G2 padding invariance, G3 exact resume;
  - G4 overfit 32/32 by step 35, G5 phase-A freeze, G6 eval sanity.
- **Report:** [results/d7_hinglish/stage7d_RESULTS.md](results/d7_hinglish/stage7d_RESULTS.md)

## Stage 7e — training (done)

Run as one chained, resumable Kaggle kernel, `hindi-lm-d7e-train` ([finetune/stage7e.py](finetune/stage7e.py)). Total: 5.9 h wall, about 11 GPU-h.

| model | dev CER (clean) | WER | exact match |
| --- | --- | --- | --- |
| before fine-tuning | 2.771 | 4.755 | 0.000 |
| after Phase A (new rows only) | 1.105 | 1.533 | 0.000 |
| main, epoch 1 | 0.110 | 0.251 | 0.114 |
| **main, best (step 17,000)** | **0.084** | **0.212** | **0.148** |
| A2: no Phase A, epoch 1 | 0.108 | 0.247 | 0.100 |
| A1: random init (no pretraining), epoch 1 | 0.126 | 0.269 | 0.100 |

- **LR bracket:** picked **6e-4** (1e-4 0.235, 3e-4 0.171, 6e-4 0.147, 1e-3 0.162 at 1,500 steps).
- **Phase A did not help.** Skipping it (A2) gives the same result.
- **Pretraining helps,** most of all early on. The gap to random init is 8 CER points at ~3k steps and 1.6 at epoch 1.
- **Forgetting:** the model lost its plain-Hindi LM ability (loss 2.79 → 8.75). This does not matter for transliteration, but it is no longer a general Hindi LM.
- **The errors that remain** are mostly rare names, singular vs plural (है/हैं), and badly misspelled input.
  - About a third of strict mismatches are spelling variants that are both correct: nukta, ZWJ, परंपरा/परम्परा, a final `।`.
- **Weights:** `hindi-lm-checkpoints/d7/main/best/`
- **Report:** [results/d7_hinglish/stage7e_RESULTS.md](results/d7_hinglish/stage7e_RESULTS.md)

## Lessons from this round

- **HF storage counts git history.** Pruned checkpoints keep using quota until the repo history is squashed **and** the orphaned LFS objects are deleted. The usedStorage counter updates 15–20 min later. Stage 7e's first attempt stopped on the cap at the last upload; the pipeline now does both after every run.

- **Kaggle secrets:** an API-pushed notebook can't read `HF_TOKEN`, so it fails with a 401. To run one:
  1. Push with `EMBED_CODE=1 kaggle/push.sh ...`.
  2. In the Kaggle UI, tick **HF_TOKEN** under Add-ons → Secrets. It is set separately for each notebook.
  3. Click **Save Version**.
- **"Page not found" on Kaggle:** the notebooks are private, so this means the browser isn't signed in as `prateek99491`.

## What to do next

1. **GitHub repo variables** for automatic deploys ([DEPLOY.md](DEPLOY.md) step 5).
2. **Future work:** plain-Hindi replay during fine-tuning, to keep LM ability alongside transliteration (the forgetting: 2.79 → 8.75).
