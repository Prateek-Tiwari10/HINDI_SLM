# D7 Step 6 — final evaluation of the Hinglish → Devanagari model

Run once, 2026-11-26, Kaggle kernel `hindi-lm-d7-step6` (2× T4, 21 min wall). The spec (`instructions/D7_step6_SPEC.md`, local) was fixed before
any test number existed, and nothing was changed after the numbers came in. Metrics: [step6.json](step6.json); examples: [step6_examples.json](step6_examples.json).

## Verdict

- **The model beats the word-lookup baseline by a wide margin.** On `test_real` (Dakshina, 5,000 human-romanized sentences) normalized
  CER is **0.058** vs 0.213, and on `test_chat` (180 real chat sentences) **0.040** vs 0.126. Every paired 95% CI on CER excludes zero.
- **Ambiguous words:** pooled accuracy **0.843** vs the baseline's 0.785 over 10,436 occurrences of 649 ambiguous Roman words. *ki* → कि/की
  0.968 vs 0.818; *main* → मैं/में 0.923 vs 0.423.
- **Retrain rule (D11):** retrain only if the model's pooled ambiguity accuracy is below the baseline's. It is above (0.843 > 0.785), so
  **no retrain**. `d7/main/best` is the final model.
- **Contamination does not drive the result.** Clean `test_real` CER is 0.062 vs 0.057 on the 75% that also appears in pretraining.
- **Released:** [`Prateek-Tiwari10/hinglish-to-devanagari-61m@7cf31e65`](https://huggingface.co/Prateek-Tiwari10/hinglish-to-devanagari-61m/tree/7cf31e65e7b9373b60cf84e64edbfd10e610ed9f), public.
  The released safetensors give the same logits as the training checkpoint (max diff 0.0).

## Protocol

- **Model:** `hindi-lm-checkpoints/d7/main/best/model.pt` (sha256 `4e5943fb…74d41f`), selected in Stage 7e on clean dev CER.
- **Decoding:** prompt `<rom> + Latin chars + <dev>`, greedy, KV cache, Devanagari-only output mask, fp16 autocast on GPU, seed 1337.
  - Budget per chunk: `ceil(0.56 × 1.5 × Latin chars) + 8`, capped at 256 new tokens.
  - Inputs longer than 160 characters are split at sentence ends, then at word boundaries; the outputs are joined with a space.
  - All of this lives in [finetune/translit.py](../../finetune/translit.py), the same code the server runs.
- **Prompt-format gate:** for the 4,313 inputs that fit one chunk, the prompts are byte-identical to the Stage 7c training arrays (0 mismatches).
  867 `test_real` inputs were split.
- **Data:** `hindi-lm-hinglish-pairs@74e39da` `romanize-v2.1/`. `test_real` = Dakshina hi test, 5,000 sentences (1,247 clean / 3,753 also in pretraining);
  `test_chat` = 180 chat sentences written by Prateek (private; aggregates only).

### Metric rules (exact)

- **Raw:** NFC + whitespace collapsed, on both sides.
- **Normalized:** applied to both sides, in this order:
  - **N1** NFC.
  - **N2** delete every nukta U+093C, after NFD, so precomposed nukta letters (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़, ऩ ऱ ऴ) lose it too; then NFC again.
  - **N3** collapse whitespace, strip.
  - **N4** delete one final `।` if the text ends with it, then strip again.
  - Nothing else is ignored: ZWJ/ZWNJ, anusvara vs chandrabindu, `॥` and ASCII `.` still count.
- **CER** = character edits / reference characters (spaces included). **WER** = word edits / reference words. **EM** = exact-match share.
  All corpus-level.
- **95% CIs:** sentence bootstrap, 2,000 resamples, seed 1337, percentile; the model − baseline difference is paired.
- **Word-lookup baseline:** for each Roman word, the most frequent Devanagari spelling in the 600k training pairs.
  - Built from 599,910 pairs whose word counts align: 282,170 Roman words.
  - Ties go to the spelling seen first. Unknown words pass through unchanged.
  - Edge punctuation is kept; a final `.` becomes `।`.

## Results

**Normalized** (the primary metric), with 95% CI:

| set | part | n | model CER | baseline CER | model WER | baseline WER | model EM | baseline EM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_real | all | 5,000 | **0.058** [0.056, 0.060] | 0.213 [0.208, 0.217] | **0.164** [0.159, 0.168] | 0.226 [0.222, 0.230] | **0.170** | 0.083 |
| test_real | clean | 1,247 | **0.062** [0.058, 0.066] | 0.220 [0.212, 0.229] | **0.182** [0.173, 0.192] | 0.245 [0.237, 0.253] | **0.157** | 0.074 |
| test_real | contaminated | 3,753 | 0.057 [0.055, 0.059] | 0.210 [0.205, 0.216] | 0.158 [0.154, 0.162] | 0.220 [0.216, 0.224] | 0.174 | 0.086 |
| test_real | fits one chunk | 4,133 | 0.058 [0.056, 0.060] | 0.212 [0.207, 0.217] | 0.163 [0.158, 0.168] | 0.226 [0.222, 0.231] | 0.203 | 0.100 |
| test_real | long, split | 867 | 0.059 [0.055, 0.062] | 0.214 [0.206, 0.223] | 0.165 [0.156, 0.173] | 0.225 [0.217, 0.232] | 0.015 | 0.000 |
| test_chat | all | 180 | **0.040** [0.029, 0.055] | 0.126 [0.103, 0.150] | **0.109** [0.083, 0.141] | 0.174 [0.152, 0.198] | **0.589** | 0.356 |

**Raw:**

| set | part | n | model CER | baseline CER | model WER | baseline WER | model EM | baseline EM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_real | all | 5,000 | **0.060** | 0.212 | **0.171** | 0.232 | **0.152** | 0.077 |
| test_real | clean | 1,247 | **0.064** | 0.220 | **0.191** | 0.252 | **0.140** | 0.064 |
| test_real | contaminated | 3,753 | 0.058 | 0.210 | 0.165 | 0.226 | 0.156 | 0.081 |
| test_chat | all | 180 | **0.076** | 0.129 | 0.232 | **0.187** | 0.183 | **0.317** |

**Model − baseline** (paired bootstrap 95% CI; negative means the model is better):

| set | part | form | ΔCER | ΔWER |
| --- | --- | --- | --- | --- |
| test_real | all | normalized | [−0.158, −0.151] | [−0.065, −0.059] |
| test_real | all | raw | [−0.156, −0.149] | [−0.064, −0.058] |
| test_real | clean | normalized | [−0.166, −0.151] | [−0.070, −0.054] |
| test_real | clean | raw | [−0.164, −0.149] | [−0.068, −0.053] |
| test_chat | all | normalized | [−0.109, −0.062] | [−0.091, −0.032] |
| test_chat | all | raw | [−0.078, −0.029] | **[+0.017, +0.077]** |

**Reading the numbers**

- **Test CER is lower than Stage 7e's dev CER (0.084).** Both use the same raw rules: clean test gives 0.064 under them. Dev and test are
  different Dakshina sentences, so the two numbers differ only because the sentences differ. No test number was used for any choice.
- **Long inputs split into chunks score the same as short ones** (CER 0.059 vs 0.058). Exact match is near zero for them only because
  a long sentence rarely matches in every word.
- **`test_chat`: raw vs normalized.** Raw EM is 0.183 and normalized EM is 0.589, so 73 of the 180 chat outputs differ from the
  reference only in nukta or a final `।`, which are exactly the differences the normalized rules ignore. On raw WER, that one-character difference
  costs a whole word. That is why the baseline wins raw WER on chat (CI [+0.017, +0.077]) while losing on raw CER and on every normalized metric.
- **The baseline's main weakness is coverage, not spelling.** Unknown words stay in Latin (`jaankari`, `ashtapradhanon`); the model
  spells names and rare words it has never seen.

## Ambiguous words

**Rule:** `ki`, `main`, plus every Roman word with ≥ 100 aligned training occurrences whose second most common spelling (after N1–N4) is ≥ 10% of its uses. That gives 649 words.
- **When an occurrence counts:** only when the test sentence's Latin and reference word counts align.
- **When it is correct:** when a word-level Levenshtein alignment of reference vs output pairs it with an identical word.

| set | model | baseline | occurrences | aligned sentences |
| --- | --- | --- | --- | --- |
| test_real | 0.843 | 0.786 | 10,261 | 4,956 / 5,000 |
| test_real clean | 0.829 | 0.766 | 2,333 | 1,233 / 1,247 |
| test_chat | 0.846 | 0.737 | 175 | 180 / 180 |
| **pooled (retrain rule)** | **0.843** | **0.785** | 10,436 | |

`ki`, `main` and the 24 most frequent ambiguous words in test:

| word | n test | model | baseline | baseline pick | top spellings in train (share) |
| --- | --- | --- | --- | --- | --- |
| ki | 2,315 | **0.968** | 0.818 | की | की 0.67, कि 0.33 |
| main | 26 | **0.923** | 0.423 | मैं | मैं 0.52, में 0.48 |
| hai | 3,160 | **0.929** | 0.905 | है | है 0.81, हैं 0.19 |
| me | 908 | 0.856 | **0.956** | में | में 0.80, मैं 0.12, मे 0.08 |
| gaye | 118 | 0.712 | 0.712 | गए | गए 0.75, गये 0.25 |
| ye | 103 | **0.563** | 0.000 | यह | यह 0.67, ये 0.33 |
| hindi | 100 | 0.550 | **0.630** | हिन्दी | हिन्दी 0.55, हिंदी 0.45 |
| jati | 95 | **0.968** | 0.895 | जाती | जाती 0.74, जाति 0.26 |
| huye | 85 | 0.929 | 0.929 | हुए | हुए 0.80, हुये 0.20 |
| kam | 80 | 0.875 | **0.925** | कम | कम 0.56, काम 0.44 |
| yeh | 66 | 0.985 | 0.985 | यह | यह 0.83, ये 0.17 |
| kiye | 60 | 0.467 | **0.517** | किए | किए 0.66, किये 0.34 |
| na | 59 | 0.712 | **0.848** | न | न 0.49, ना 0.49 |
| or | 58 | **0.828** | 0.000 | और | और 0.89, ओर 0.11 |
| va | 56 | 0.821 | **0.946** | व | व 0.89, वा 0.10 |
| yahan | 53 | 0.377 | 0.358 | यहां | यहां 0.72, यहाँ 0.28 |
| san | 48 | 0.729 | **0.792** | सन् | सन् 0.73, सं 0.11, सन 0.08 |
| mana | 45 | 0.933 | 0.956 | माना | माना 0.80, मना 0.20 |
| of | 41 | 0.951 | 0.951 | ऑफ़ | ऑफ 0.70, आफ 0.30 |
| banane | 41 | 0.951 | 0.927 | बनाने | बनाने 0.89, बनने 0.11 |
| mandir | 41 | 0.732 | **0.805** | मंदिर | मंदिर 0.87, मन्दिर 0.13 |
| band | 40 | **0.450** | 0.325 | बंद | बंद 0.83, बन्द 0.13, बैंड 0.03 |
| chahiye | 39 | 0.744 | **0.846** | चाहिए | चाहिए 0.74, चाहिये 0.26 |
| jahan | 38 | 0.579 | 0.553 | जहां | जहां 0.70, जहाँ 0.29 |
| sambandh | 38 | 0.579 | **0.658** | संबंध | संबंध 0.61, सम्बन्ध 0.37 |
| 1 | 36 | 0.722 | 0.722 | 1 | 1 0.65, १ 0.35 |

- **The model wins where context decides the word.**
  - *ki* (की vs कि) and *main* (मैं vs में).
  - *or*: in Dakshina, "or" is ओर, as in *ki or* ("towards"), and और is written "aur". The baseline learned और from chat-style training
    pairs and scores 0.
  - *ye* (ये vs यह).
- **The baseline wins on two kinds of word:**
  - Very frequent short words where one spelling dominates (*me* → में, *va* → व).
  - Spelling-variant pairs that are both correct, such as हिन्दी/हिंदी, किए/किये, मन्दिर/मंदिर, चाहिए/चाहिये and सम्बन्ध/संबंध. Dakshina references
    often use the older forms. Normalization does not merge anusvara with a half nasal consonant, so these count as errors for both systems.
- ***yahan* and *jahan* (≈ 0.4–0.6 for both systems):** these differ only in anusvara vs chandrabindu (यहां/यहाँ), which the rules
  deliberately do not ignore.

## Examples

These 15 come from `test_real` clean, with inputs of at most 120 characters, chosen by the fixed rule in the spec (seed 1337). The output shown is the CPU output of the released files; the server reproduces all 15 exactly.

| # | kind | Hinglish input | model output | reference | CER | lookup baseline |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | best | vidyalay ke baare mein samanya jaankari is prakar hai:- | विद्यालय के बारे में सामान्य जानकारी इस प्रकार है:- | विद्यालय के बारे में सामान्य जानकारी इस प्रकार है:- | 0.000 | विद्यालय के baare में सामान्य jaankari इस प्रकार है:- |
| 2 | best | Ve Shivaji ke ashtapradhanon men se ek the. | वे शिवाजी के अष्टप्रधानों में से एक थे। | वे शिवाजी के अष्टप्रधानों में से एक थे। | 0.000 | वे शिवाजी के ashtapradhanon में से एक थे। |
| 3 | best | aisa hai hamara sharir - Harish yadav | ऐसा है हमारा शरीर - हरीश यादव | ऐसा है हमारा शरीर - हरीश यादव | 0.000 | ऐसा है हमारा शरीर - हरीश यादव |
| 4 | best | antriksh men ghoomte shikshak - Durga Prasad Shukl | अंतरिक्ष में घूमते शिक्षक - दुर्गा प्रसाद शुक्ल | अंतरिक्ष में घूमते शिक्षक - दुर्गा प्रसाद शुक्ल | 0.000 | अंतरिक्ष में घूमते शिक्षक - दुर्गा प्रसाद शुक्ल |
| 5 | best | Jayshakti ke baad Vijayshakti gaddi par baitha. | जयशक्ति के बाद विजयशक्ति गद्दी पर बैठा। | जयशक्ति के बाद विजयशक्ति गद्दी पर बैठा। | 0.000 | jayshakti के बाद vijayshakti गद्दी पर बैठा। |
| 6 | ambiguous | S.S. alluvaliya bharat ki solahavi loksabha me sansad hai. | स.एस. अल्लूवालिया भारत की सोलहवी लोकसभा में सांसद हैं। | एस॰एस॰ अहलूवालिया भारत की सोलहवीं लोकसभा में सांसद हैं। | 0.111 | एसएस. alluvaliya भारत की solahavi लोकसभा में संसद है। |
| 7 | ambiguous | ye pahari loga kauna the? | ये पहाड़ी लोगा कौना थे? | ये पहाड़ी लोग कौन थे? | 0.100 | यह पहाड़ी लोगा कौना थे? |
| 8 | ambiguous | Kannojiya shabd men kisi jati vishesh ka aagrah nahin balki sthan vishesh ka aagrah antnirhit hai. | कन्नोजिया शब्द में किसी जाति विशेष का आग्रह नहीं बल्कि स्थान विशेष का आग्रह अंतर्निहित है। | कन्नौजिया शब्द में किसी जाति विशेष का आग्रह नहीं बल्कि स्थान विशेष का आग्रह अंतर्निहित है। | 0.011 | kannojiya शब्द में किसी जाती विशेष का आग्रह नहीं बल्कि स्थान विशेष का आग्रह अंतर्निहित है। |
| 9 | ambiguous | Ve Sonam Kapoor ke liye bhi ek poshakh design kar chuke hai. | वे सोनम कपूर के लिए भी एक पोषक डिजाइन कर चुके हैं। | वे सोनम कपूर के लिये भी एक पोशाक डिज़ाइन कर चुके हैं। | 0.078 | वे सोनम कपूर के लिए भी एक poshakh डिजाइन कर चुके है। |
| 10 | ambiguous | logon ke ye daastaa ka samay rahaa ho lekin daliton ke liye kai maayanon mein svarnakaal tha. | लोगों के ये दासता का समय रहा हो लेकिन दलितों के लिए कई मायनों में स्वर्णकाल था। | लोगों के ये दासता का समय रहा हो लेकिन दलितों के लिए कई मायनों में स्वर्णकाल था। | 0.000 | लोगों के यह daastaa का समय rahaa हो लेकिन दलितों के लिए कई maayanon में स्वर्णकाल था। |
| 11 | worst | Kitivarma ke pautra Madanvarma ne bhi dekhi ja sakti hai. | किटीवर्मा के पौत्र मदनवर्मा ने भी देखी जा सकती है। | कीतिवर्मा के पौत्र मदनवर्मा ने भी अच्छी ख्याति प्राप्त की। | 0.368 | kitivarma के पौत्र madanvarma ने भी देखी जा सकती है। |
| 12 | worst | eka dala ke ka'i phula - vyathita hrdaya | एका दला के का'ई फुला - व्यथिता हरदया | एक डाल के कई फूल - व्यथित हृदय | 0.367 | एका डाला के कई फुला - vyathita hrdaya |
| 13 | worst | eka avasara para, ra'ita, bainda chora dene ki dhamaki dete hu'e inglainda lauta a'e. | एका अवसरा पारा, रा'इता, बैंदा चोरी देने की धमकी देते हूं'ए इंग्लैंडा लौटा आ'ए। | एक अवसर पर, राइट, बैंड छोड़ देने की धमकी देते हुए इंग्लैंड लौट आए। | 0.281 | एका avasara पारा, ra'ita, बइिंडा चोरा देने की dhamaki देते हुए inglainda लौटा आए। |
| 14 | worst | mem campa para akramana kara vijaya para adhikara kara liya. | मेम कैंपा पारा आक्रमणा करा विजया पारा अधिकारा करा लिया। | में चंपा पर आक्रमण कर विजय पर अधिकार कर लिया। | 0.273 | मेम campa पारा akramana करा विजया पारा adhikara करा लिया। |
| 15 | worst | ismein chaar ikaiyan hain. Chauthi ikai 20 November 2010 ko kraantik (kritical) huyi thi. | इसमें चार इकाइयां हैं. चौथी इकाई 20 नोवेंबर 2010 को क्रान्तिक (कृतिकाल) हुयी थी। | इसमें चार इकाइयाँ हैं। चौथी ईकाई २० नवम्बर २०१० को क्रांतिक (क्रिटिकल) हुई थी। | 0.273 | इसमें चार इकाइयां हैं. चौथी इकाई 20 november 2010 को kraantik (kritical) हुयी थी। |

**What the worst cases have in common:**
- **#11:** the reference does not match the input; it is a Dakshina annotation error.
- **#12–14:** scholarly romanization with a written inherent *a* (`eka`, `para`, `kara`), which the chat-style training pairs never use. The model
  keeps the final *a* as ा.
- **#15:** the reference uses Devanagari digits and different loanword spellings.

## Speed (CPU, KV cache, fp32, the released files)

**Kaggle CPU (Step 6 §7):** 4 cores; 5 warm-up sentences, then 100 sentences each, one at a time.

| inputs | threads | p50 | p95 | mean new tokens | tokens/s |
| --- | --- | --- | --- | --- | --- |
| test_chat (short chat lines) | 1 | 286 ms | 407 ms | 6.8 | 23.5 |
| test_real clean (≤ 160 chars) | 1 | 597 ms | 1,391 ms | 17.2 | 25.3 |
| test_chat | 4 | 221 ms | 300 ms | 6.8 | 30.7 |
| test_real clean | 4 | 437 ms | 961 ms | 17.2 | 35.1 |

- Model load takes 0.96 s.
- CPU and GPU agree on 99% of `test_chat` outputs and 100% of `test_real` outputs (normalized). The one difference is likely precision: the GPU run used fp16 autocast and the CPU run fp32.

**Local Docker gate** (the image as deployed, `docker run --cpus 0.5 --memory 1g`, 1 torch thread, 2026-11-26):

| check | result |
| --- | --- |
| image size | 744 MB compressed (what GHCR stores and Azure pulls), 2.33 GB unpacked |
| cold start (`docker run` → first `/health` 200, incl. model load + warm-up) | **7.1 s** |
| memory after load / after 30 requests | 528 MiB / 553 MiB of 1 GiB |
| short chat lines (10 sentences, 3–8 words) | p50 **302 ms**, p95 508 ms |
| the 15 examples (4–17 words) | p50 **810 ms**, p95 1,418 ms |
| 485-character input (4 chunks) | 6.5 s |
| `/health`, `/`, `/docs`, POST `/api/transliterate` | 200; outputs match Step 6 for 15/15 examples |
| oversize (501 chars), empty, Devanagari-only, missing field, wrong type | 422 for all five |
| Docker HEALTHCHECK | healthy |

**Live URL load test** (Azure Container Apps, 0.5 vCPU): _pending the first deploy_.

## Known limitation: forgetting

Fine-tuning erased the base model's plain-Hindi language-modelling ability. Loss on 200 v2 val windows was 2.79 for the base model and
**8.75** for `d7/main/best` (Stage 7e). The released model is therefore a Hinglish → Devanagari transliterator, not a general Hindi LM.
Prateek accepted this on 2026-11-26.

**Fix (future work):** mix plain-Hindi LM batches back into fine-tuning (replay), possibly with a lower LR. The trigger for doing it now did
not fire, because the ambiguity slice beats the lookup baseline.

## Other limitations

- **Input style:** the model is trained on synthetic chat-style romanization. Scholarly romanization with an explicit inherent *a* (`kara`, `para`)
  or diacritics transliterates poorly (worst examples #12–14).
- **Long inputs:** inputs are split into chunks of ≤ 160 characters, so context does not cross chunk boundaries, and latency grows with length
  (≈ 1.3 s per 100 characters on 0.5 vCPU).
- **Spelling conventions:** for words with two correct spellings the model writes modern forms (हिंदी, किए). Normalized scores still count the other form as an error.
- **`%` is dropped** from the Latin side (D10).
