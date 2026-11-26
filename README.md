# HINDI_SLM: Hinglish → Devanagari with a 61M model trained from scratch

### ▶ Try it live: **https://hinglish-translit.gentleground-e8984580.uaenorth.azurecontainerapps.io**

API docs (Swagger): [/docs](https://hinglish-translit.gentleground-e8984580.uaenorth.azurecontainerapps.io/docs) · model weights: [huggingface.co/Prateek-Tiwari10/hinglish-to-devanagari-61m](https://huggingface.co/Prateek-Tiwari10/hinglish-to-devanagari-61m)

_The demo scales to zero when idle, so the first request after a quiet period can take about 10 s._

Type Hindi the way people type it in chats, in Latin letters, and get it back in Devanagari:

```
kya haal hai                                    →  क्या हाल है
Ve Shivaji ke ashtapradhanon men se ek the.     →  वे शिवाजी के अष्टप्रधानों में से एक थे।
```

Everything is built from scratch: the data pipeline (10B raw tokens cleaned down to 1.94B), the tokenizer, a 61M-parameter LLaMA-style decoder pretrained on Hindi, and the fine-tune that turns it into a transliterator. No pretrained models or tokenizers are used anywhere. The model runs on half a CPU core behind a FastAPI service on Azure Container Apps.

**Results on held-out human data** (run once, [details](results/d7_hinglish/step6_RESULTS.md)):
- **Dakshina test** (5,000 human-romanized sentences): character error rate **5.8%** vs 21.3% for a word-lookup baseline.
- **Real chat messages** (180): **4.0%** vs 12.6%.
- **Ambiguous words** such as *ki* (कि/की) and *main* (मैं/में): the right word **84%** of the time vs 78%.
- **Speed:** about 0.3 s per chat-length sentence on the live URL, on half a CPU core.

## Try it

```bash
curl -s -X POST https://hinglish-translit.gentleground-e8984580.uaenorth.azurecontainerapps.io/api/transliterate \
  -H 'Content-Type: application/json' \
  -d '{"text": "kya haal hai"}'
# {"output":"क्या हाल है","latency_ms":123.3}
```

`GET /health` reports the model revision and the deployed git commit. Inputs are capped at 500 characters. Longer texts are split at sentence ends and then at word boundaries, and each piece is transliterated separately. The public URL is rate-limited to 30 requests per minute per IP. The app scales to zero when idle, so the first request after a quiet period waits for a cold start. Measured from zero replicas on Azure, the first `/health` answered in **10.1 s**. After that, requests take the times below.

## Architecture

```
 browser / curl
      │  HTTPS
      ▼
 Azure Container Apps ingress (Consumption plan, 0–2 replicas, HTTP scale rule)
      │
      ▼
 container: uvicorn (1 worker) ──▶ FastAPI  (serve/app.py)
      │                              ├─ GET  /                    one static page (serve/static/index.html)
      │                              ├─ POST /api/transliterate   {text} → {output, latency_ms}
      │                              ├─ GET  /health, /docs
      │                              └─ per-IP rate limit, JSON logs with latency_ms
      ▼
 model: 61M decoder on CPU (0.5 vCPU, 1 thread), greedy + KV cache, Devanagari-only output mask
        weights baked into the image at build time from a pinned HF revision (serve/MODEL_ID)

 git push (serve/, model/, finetune/translit.py) ──▶ GitHub Actions:
        pytest ▶ docker build ▶ ghcr.io/…/hinglish-translit:<sha> ▶ az containerapp update (OIDC) ▶ live smoke test
```

The server and the evaluation share one inference module, [finetune/translit.py](finetune/translit.py), so the demo behaves exactly as evaluated. Deployment steps are in [DEPLOY.md](DEPLOY.md).

## Results (D7 Step 6, run once on held-out test sets)

Normalized scores (NFC, nukta ignored, one final `।` ignored; exact rules in the report), 95% bootstrap CIs:

| test set | n | model CER | lookup-baseline CER | model WER | baseline WER | model exact match | baseline exact match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dakshina test, all | 5,000 | **0.058** [0.056, 0.060] | 0.213 | **0.164** | 0.226 | **0.170** | 0.083 |
| Dakshina test, clean (not in pretraining) | 1,247 | **0.062** [0.058, 0.066] | 0.220 | **0.182** | 0.245 | **0.157** | 0.074 |
| chat messages | 180 | **0.040** [0.029, 0.055] | 0.126 | **0.109** | 0.174 | **0.589** | 0.356 |

- **Lookup baseline:** for each Roman word, the most frequent Devanagari spelling in the 600k training pairs. Unknown words stay in Latin.
- **Ambiguous words** (649 Roman words with two common spellings, found automatically): 10,436 occurrences in test. Accuracy is **0.843** for the model vs 0.785 for the baseline:
  - *ki*: 0.968 vs 0.818;
  - *main*: 0.923 vs 0.423;
  - *or* (ओर in Dakshina, not और): 0.828 vs 0.000.
- **Contamination:** 75% of Dakshina test also appears in the pretraining web data. The clean part scores almost the same, so memorization does not explain the result.

Full report: [results/d7_hinglish/step6_RESULTS.md](results/d7_hinglish/step6_RESULTS.md) · metrics: [step6.json](results/d7_hinglish/step6.json).

### Speed

**Local gate before deploying:** the same image, run with Azure's limits (`docker run --cpus 0.5 --memory 1g`, 1 torch thread, greedy + KV cache):

| | result |
| --- | --- |
| image | 744 MB compressed (2.33 GB unpacked); CPU-only torch, weights baked in |
| cold start: container start → first `/health` 200 (model load + warm-up) | **7.1 s** |
| memory | ~550 MiB of 1 GiB |
| chat-length sentence (3–8 words) | p50 **302 ms**, p95 508 ms |
| Dakshina-length sentence (4–17 words) | p50 810 ms, p95 1,418 ms |

Throughput is about 25 generated tokens/s on one core. Latency grows with input length, at roughly 1.3 s per 100 characters.

**Load test of the live URL** (2026-11-26, [`hey`](https://github.com/rakyll/hey) from a laptop in India to Azure UAE North):
- **Setup:** one replica at 0.5 vCPU / 1 GiB (max replicas pinned to 1 for the test, rate limit raised for it).
- **Latency:** times include the network round trip.
- **Result:** all 450 requests returned 200.

| input | concurrency | requests | p50 | p95 | requests/s |
| --- | --- | --- | --- | --- | --- |
| chat sentence, 30 chars (`main kal office nahi aa paunga`) | 1 | 100 | **301 ms** | 658 ms | **2.3** |
| chat sentence, 30 chars | 4 | 200 | 1.88 s | 2.22 s | 2.2 |
| Dakshina sentence, 95 chars | 1 | 50 | 1.11 s | 1.91 s | 0.8 |
| Dakshina sentence, 95 chars | 4 | 100 | 4.94 s | 5.24 s | 0.8 |

- **One replica runs one inference at a time** (1 thread, 0.5 vCPU). So throughput is fixed at about 2.2 chat sentences per second, and extra concurrency only adds queueing.
- **Scaling:** the HTTP scale rule adds a second replica above 4 requests in flight, which doubles capacity.
- **Inference time on the server** for a short sentence is about 120 ms, as reported in `latency_ms`.
- **Cold start on Azure:** from zero replicas (confirmed with `az containerapp replica list`), the first `/health` answered in **10.1 s**. That includes scheduling, pulling the cached image and loading the model. Use min replicas 1 to avoid it (DEPLOY.md).

### Examples

From Dakshina test (clean part), chosen by a fixed rule before looking at the outputs. All 15 are in the [report](results/d7_hinglish/step6_RESULTS.md#examples).

| kind | Hinglish | model | lookup baseline |
| --- | --- | --- | --- |
| best | Jayshakti ke baad Vijayshakti gaddi par baitha. | जयशक्ति के बाद विजयशक्ति गद्दी पर बैठा। | jayshakti के बाद vijayshakti गद्दी पर बैठा। |
| best | vidyalay ke baare mein samanya jaankari is prakar hai:- | विद्यालय के बारे में सामान्य जानकारी इस प्रकार है:- | विद्यालय के baare में सामान्य jaankari इस प्रकार है:- |
| ambiguous | logon ke ye daastaa ka samay rahaa ho lekin daliton ke liye kai maayanon mein svarnakaal tha. | लोगों के ये दासता का समय रहा हो लेकिन दलितों के लिए कई मायनों में स्वर्णकाल था। | लोगों के यह daastaa का समय rahaa हो … |
| ambiguous | Kannojiya shabd men kisi jati vishesh ka aagrah nahin … | कन्नोजिया शब्द में किसी जाति विशेष का आग्रह नहीं … | kannojiya शब्द में किसी जाती विशेष … |
| worst | eka dala ke ka'i phula - vyathita hrdaya | एका दला के का'ई फुला - व्यथिता हरदया (ref: एक डाल के कई फूल - व्यथित हृदय) | एका डाला के कई फुला - vyathita hrdaya |
| worst | mem campa para akramana kara vijaya para adhikara kara liya. | मेम कैंपा पारा आक्रमणा करा विजया पारा अधिकारा करा लिया। (ref: में चंपा पर आक्रमण कर …) | मेम campa पारा akramana … |

The worst cases are scholarly romanizations that write the inherent *a* (`kara`, `para`), a style the chat-trained model never saw.

### Known limitations

- **It is a transliterator now, not a Hindi LM.** Fine-tuning erased the plain-Hindi language-modelling ability: loss on Hindi text went from 2.79 to 8.75. This was accepted for this release. The fix is to mix plain-Hindi batches back in during fine-tuning (replay), and it is future work.
- **Chat-style input only.** It handles chat-style Hinglish (`kya`, `hai`, `nahi`). Scholarly romanization with a written inherent *a* or diacritics comes out poorly.
- **Long input:** inputs over 160 characters are transliterated in independent chunks, and the API caps input at 500 characters.
- **Spelling variants:** where two spellings are both correct (हिंदी/हिन्दी, किए/किये), the model writes the modern form. Strict scoring counts the other as an error.

## How it was built

| step | what | result |
| --- | --- | --- |
| Data v2 (D1–D3) | all 100 Sangraha Verified Hindi files, Hindi Wikipedia and our own news crawl (~10B raw tokens). Cleaning: NFC normalization, line-level boilerplate removal, heuristic + Gopher/FineWeb-style doc filters, exact + MinHash near-dedup (Jaccard ≥ 0.8), document-level split. A fastText quality ranker was tried and rejected (it learned source style, not quality). | **1.937B** train tokens (Sangraha 86%, news 7%, Wikipedia 6% ×2) |
| Tokenizer (D4) | SentencePiece Unigram picked from an 8-candidate grid by an intrinsic rule (fertility knee + morphology) | `hi_unigram_10240`, 9.31 bytes/token |
| Model (D5) | LLaMA-style decoder: d_model 640, 12 layers, GQA 10/5 heads, SwiGLU, RoPE, RMSNorm, tied embeddings, context 1024 | 61,128,320 params |
| Pretraining (D6) | one epoch on 2×T4 (DDP, fp16, `torch.compile`), peak LR 1e-3 from a sweep, bit-exact resume tested | test loss 2.816, perplexity 16.7 |
| Tokenizer extension (D8) | char-level Latin tokens + `<rom>`/`<dev>` added after the base vocab; output mask allows Devanagari only | vocab 10,304 |
| Pair data (D10) | 600k synthetic Hinglish pairs: our v2 sentences romanized with chat shortcuts and human-attested spellings (Dakshina, Aksharantar); evaluation on human data only | word agreement with humans 0.643 |
| Fine-tune (D11) | full fine-tune, LR 6e-4 from a bracket, 2 epochs; selection by clean dev CER, not loss | dev CER **0.084** |
| Evaluation (D12) | Step 6, run once: `test_real` (Dakshina) + `test_chat` (180 real chat sentences), word-lookup baseline, ambiguous-word slices | this page |
| Serving (D13) | FastAPI + Docker → GHCR → Azure Container Apps (UAE North, 0.5 vCPU, scale to zero) via GitHub Actions | live URL above; p50 301 ms, 2.3 requests/s per replica |

Stage reports are in [results/](results/): data and tokenizer v1 (`stage*_RESULTS.md`), tokenizer v2 ([results/tokenizer_v2/](results/tokenizer_v2/)), pretraining ([results/d5d6/](results/d5d6/)) and fine-tuning ([results/d7_hinglish/](results/d7_hinglish/)). Data sources and licences: [data/SOURCES.md](data/SOURCES.md).

## Repo layout

| path | contents |
| --- | --- |
| [data/](data/) | normalization, filters, dedup, splits, tokenization, Hinglish pair generation |
| [tokenizer/](tokenizer/) | tokenizer sweep and the Hinglish extension |
| [model/](model/) | the decoder, KV cache, vocab surgery |
| [train/](train/) | pretraining loop (DDP, resume) |
| [finetune/](finetune/) | SFT harness, Stage 7e pipeline, shared inference (`translit.py`), Step 6 (`step6.py`) |
| [eval/](eval/) | evaluation scripts; Step 6 scoring (`translit_eval.py`) |
| [serve/](serve/) | FastAPI app, static page, Dockerfile, tests |
| [kaggle/](kaggle/) | kernel metadata and the push script |
| [.github/workflows/](.github/workflows/) | deploy and load-test workflows |

## Reproduce

All training and evaluation ran as Kaggle kernels (CPU, 2×T4, TPU VM), never on a laptop. Every stage is resumable through markers on the HF Hub.

```bash
# laptop: thin CLI env for pushing kernels and reading the Hub
python -m venv .venv && .venv/bin/pip install kaggle huggingface_hub

# push a stage: the code goes into the kernel, then press "Save Version" in the Kaggle editor
# with the HF_TOKEN secret ticked (API-pushed versions can't read secrets)
EMBED_CODE=1 kaggle/push.sh "python -m finetune.step6" kaggle/d7_step6
```

Run the demo locally:

```bash
docker build -f serve/Dockerfile -t hinglish-translit .
docker run --rm -p 8000:8000 --cpus 0.5 --memory 1g hinglish-translit   # then open http://localhost:8000
```

Thresholds and seeds are in [configs/](configs/). The test suite (`pytest`) runs inside every kernel before the stage starts, and `serve/tests` runs in CI against the real released model.
