# Stage 4 — Tokenizer sweep (CPU only, no proxy models)

**Status:** done.
- **Recommended tokenizer:** **`hi_unigram_8192`**.
- **Kaggle run:** `prateek99491/hindi-lm-runner` version 9, the same chained run as Stage 3.
- **Kaggle username verified:** `prateek99491`.

## 4a — Sample

| | value |
| --- | --- |
| train split size | 24.77 GB |
| sampling rate | 3.84% of train docs; every source at the same rate, so train's proportions are kept, with no upsampling |
| training sample | 169,607 docs, 946 MB UTF-8 (target ≈ 950 MB, ≈ 100M tokens) |
| line length cap | 2,000 chars, split at a space; `max_sentence_length=16384` |
| val docs for evaluation | **11,157** (all of val, 4,724,839 words) |
| morphology words | 154 real words (≥ 20 occurrences) from 288 stem×suffix candidates |

**Shortfalls against the instructions:**
- **Val docs:** 11,157, not 20,000. The instructions ask for 20,000 val docs, but the 0.25% val split of 4.45M docs holds only 11,157. All of them are used.
- **Morphology words:** 154, not about 200. Even with 24 stems × 12 suffixes, only 154 constructed words occur ≥ 20 times; the list is `tokenizer/morph_words.tsv`.

The sample (`work/tokenizer/tok_sample.txt.gz`) and the val docs are stored on HF, not in git.

## 4b — Training

- **Settings:** all 8 tokenizers use identical settings: `character_coverage=0.9995`, `byte_fallback`, `split_digits`, `normalization_rule_name="identity"`, pad/unk/bos/eos = 0/1/2/3, and 19 reserved special tokens (3 chat tokens + 16 reserved).
- **Isolation:** each tokenizer trained in its own process.
- **Storage:** all 8 `.model`/`.vocab` files are uploaded to `Prateek-Tiwari10/hindi-lm-tokenizer` under `sweep/`.
- **Peak RAM:** about 4 GB for BPE and about 10.3 GB for Unigram, the same across vocab sizes. Memory is dominated by the loaded sample, not the vocab.
- **Training time:** Unigram takes about 2–3× as long as BPE.
- **RAM limit:** never hit; the machine has 31 GB.

## 4c — Sweep table (`results/tokenizer_sweep.csv`)

| tokenizer | fertility | chars/token | byte-fallback % | heavy splits % | embedding % | morphology % | train time (s) | peak RAM (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hi_bpe_8192 | 1.4738 | 3.476 | 1.55 | 4.31 | 7.0 | 42.9 | 239.4 | 4.02 |
| hi_bpe_10240 | 1.4296 | 3.583 | 1.60 | 3.77 | 8.7 | 39.6 | 267.9 | 4.02 |
| hi_bpe_12288 | 1.3978 | 3.665 | 1.63 | 3.39 | 10.5 | 35.7 | 309.1 | 4.02 |
| hi_bpe_16384 | 1.3550 | 3.780 | 1.69 | 2.89 | 14.0 | 24.7 | 396.1 | 4.02 |
| hi_unigram_8192 | 1.4472 | 3.540 | 1.58 | 4.37 | 7.0 | 47.4 | 678.0 | 10.27 |
| hi_unigram_10240 | 1.4068 | 3.641 | 1.62 | 3.81 | 8.7 | 44.2 | 662.7 | 10.27 |
| hi_unigram_12288 | 1.3783 | 3.717 | 1.66 | 3.40 | 10.5 | 38.3 | 636.9 | 10.27 |
| hi_unigram_16384 | 1.3393 | 3.825 | 1.71 | 2.86 | 14.0 | 30.5 | 628.0 | 10.27 |

**Plots** in `results/plots/`:
- `fertility_vs_vocab.png`
- `heavy_split_vs_vocab.png`
- `morphology_vs_vocab.png`

**Reading the table:**
- **Unigram vs BPE:** Unigram has 1.2–1.8% lower fertility at every size and a 3–6 point higher morphology score. Heavy splits are within 0.1 points.
- **Vocab size:** the morphology score *falls* as vocab grows. Bigger vocabularies store inflected forms (लड़कों, किताबें) as single tokens, so there is no stem/suffix boundary left to hit.
- **Byte fallback:** 1.5–1.7% of tokens (rare characters outside 99.95% coverage).

## 4e — Selection, step by step (rule applied mechanically by `tokenizer/select.py`)
 — vocab size per algorithm

- bpe 8192 -> 10240: fertility 1.4738 -> 1.4296, gain 3.00% (threshold 3%), embedding 7.0%
  -> bpe: choose 8192
- unigram 8192 -> 10240: fertility 1.4472 -> 1.4068, gain 2.79% (threshold 3%), embedding 7.0%
  -> unigram: choose 8192

## Step 2 — common size

- BPE chose 8192, Unigram chose 8192 -> compare at 8192

## Step 3 — algorithm

- fertility: BPE 1.4738 vs Unigram 1.4472 (diff 1.84%, tie margin 2%)
- heavy splits: BPE 4.31% vs Unigram 4.37%
- morphology: BPE 42.9% vs Unigram 47.4%
  -> tie: goes to unigram (decision log: Unigram preferred on ties)

## Step 4 — optional tie-break

- BPE and Unigram are within 2% on fertility: the optional tie-break (one ~20-minute proxy run each) is available. Not run; Prateek decides.

**Recommended tokenizer: `hi_unigram_8192`**

**Borderline note:**
- **BPE gain:** BPE's 8192 → 10240 fertility gain is 2.999%, just under the 3% threshold, so the rule picks 8192.
- **Unigram gain:** Unigram's gain is 2.79%, clearly under 3%, so its choice of 8192 does not depend on this.
- **Algorithm choice:** the two are compared at 8192 either way.

**Tie-break flag:** BPE and Unigram are within 2% on fertility (1.84%). The optional tie-break (one ~20-minute proxy run each) is available, **not run**; Prateek decides. Unigram also has the better morphology score (47.4% vs 42.9%), so the tie rule and the secondary metric point the same way.

## Example tokenizations (the same 10 sentences for every tokenizer)

Unigram keeps suffixes as separate pieces: जा|एँ|गे, निकल|कर, गिर|ती, विविधता|ओं. BPE 8192 splits the same verb as जाए|ँ|गे, across the vowel sign.


### hi_bpe_8192

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविध | ताओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (11) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताब | ें | ▁पढ़ | ी | ं | ।
- (15) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जाए | ँ | गे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आए | ँ | गे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (10) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वा | दिष्ट | ▁खाना | ▁बनाया | ।
- (11) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़ि | ंदगी | ▁बदल | ▁दी | ▁है | ।
- (16) ▁गंगा | ▁नदी | ▁हिमाल | य | ▁से | ▁निकल | कर | ▁बंगाल | ▁की | ▁खा | ड़ी | ▁में | ▁गिर | ती | ▁है | ।

### hi_bpe_10240

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविध | ताओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (11) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताब | ें | ▁पढ़ | ी | ं | ।
- (14) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जाएँ | गे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आए | ँ | गे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (9) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वादिष्ट | ▁खाना | ▁बनाया | ।
- (11) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़ि | ंदगी | ▁बदल | ▁दी | ▁है | ।
- (14) ▁गंगा | ▁नदी | ▁हिमालय | ▁से | ▁निकल | कर | ▁बंगाल | ▁की | ▁खाड़ी | ▁में | ▁गिर | ती | ▁है | ।

### hi_bpe_12288

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविध | ताओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (10) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताबें | ▁पढ़ | ी | ं | ।
- (14) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जाएँ | गे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आए | ँ | गे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (9) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वादिष्ट | ▁खाना | ▁बनाया | ।
- (10) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़िंदगी | ▁बदल | ▁दी | ▁है | ।
- (13) ▁गंगा | ▁नदी | ▁हिमालय | ▁से | ▁निकलकर | ▁बंगाल | ▁की | ▁खाड़ी | ▁में | ▁गिर | ती | ▁है | ।

### hi_bpe_16384

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविध | ताओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (9) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताबें | ▁पढ़ी | ं | ।
- (14) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जाएँ | गे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आए | ँ | गे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (9) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वादिष्ट | ▁खाना | ▁बनाया | ।
- (10) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़िंदगी | ▁बदल | ▁दी | ▁है | ।
- (13) ▁गंगा | ▁नदी | ▁हिमालय | ▁से | ▁निकलकर | ▁बंगाल | ▁की | ▁खाड़ी | ▁में | ▁गिर | ती | ▁है | ।

### hi_unigram_8192

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविधता | ओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (11) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताब | ें | ▁पढ़ | ी | ं | ।
- (15) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जा | एँ | गे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आ | एँ | गे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (9) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वादिष्ट | ▁खाना | ▁बनाया | ।
- (10) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़िंदगी | ▁बदल | ▁दी | ▁है | ।
- (14) ▁गंगा | ▁नदी | ▁हिमालय | ▁से | ▁निकल | कर | ▁बंगाल | ▁की | ▁खाड़ी | ▁में | ▁गिर | ती | ▁है | ।

### hi_unigram_10240

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविधता | ओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (10) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताबें | ▁पढ़ | ी | ं | ।
- (15) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जा | एँ | गे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आ | एँ | गे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (9) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वादिष्ट | ▁खाना | ▁बनाया | ।
- (10) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़िंदगी | ▁बदल | ▁दी | ▁है | ।
- (14) ▁गंगा | ▁नदी | ▁हिमालय | ▁से | ▁निकल | कर | ▁बंगाल | ▁की | ▁खाड़ी | ▁में | ▁गिर | ती | ▁है | ।

### hi_unigram_12288

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविधता | ओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (10) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताबें | ▁पढ़ | ी | ं | ।
- (15) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जा | एँ | गे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आ | एँ | गे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (9) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वादिष्ट | ▁खाना | ▁बनाया | ।
- (10) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़िंदगी | ▁बदल | ▁दी | ▁है | ।
- (13) ▁गंगा | ▁नदी | ▁हिमालय | ▁से | ▁निकलकर | ▁बंगाल | ▁की | ▁खाड़ी | ▁में | ▁गिर | ती | ▁है | ।

### hi_unigram_16384

- (11) ▁भारत | ▁एक | ▁विशाल | ▁और | ▁विविधता | ओं | ▁से | ▁भरा | ▁देश | ▁है | ।
- (9) ▁लड़कियों | ▁ने | ▁स्कूल | ▁में | ▁नई | ▁किताबें | ▁पढ़ी | ं | ।
- (12) ▁हम | ▁कल | ▁सुबह | ▁दिल्ली | ▁जाएँगे | ▁और | ▁शाम | ▁तक | ▁लौट | ▁आ | एँगे | ।
- (17) ▁प्रधानमंत्री | ▁ने | ▁ | 2 | 0 | 2 | 4 | ▁में | ▁ | 1 | 5 | ▁नई | ▁योजनाओं | ▁की | ▁घोषणा | ▁की | ।
- (11) ▁वैज्ञानिकों | ▁ने | ▁जलवायु | ▁परिवर्तन | ▁पर | ▁एक | ▁नया | ▁अध्ययन | ▁प्रकाशित | ▁किया | ।
- (11) ▁क्या | ▁आप | ▁मुझे | ▁रेलवे | ▁स्टेशन | ▁का | ▁रास्ता | ▁बता | ▁सकते | ▁हैं | ?
- (8) ▁बच्चे | ▁मैदान | ▁में | ▁क्रिकेट | ▁खेल | ▁रहे | ▁थे | ।
- (9) ▁उसने | ▁अपनी | ▁माँ | ▁के | ▁लिए | ▁स्वादिष्ट | ▁खाना | ▁बनाया | ।
- (10) ▁कंप्यूटर | ▁और | ▁इंटरनेट | ▁ने | ▁हमारी | ▁ज़िंदगी | ▁बदल | ▁दी | ▁है | ।
- (13) ▁गंगा | ▁नदी | ▁हिमालय | ▁से | ▁निकलकर | ▁बंगाल | ▁की | ▁खाड़ी | ▁में | ▁गिर | ती | ▁है | ।


## Decision at the STOP gate

**Proceeded with `hi_unigram_8192`.**
- Prateek asked for the pipeline to run end to end; this is the rule's mechanical output.
- Stage 5 used it.
- Overriding later means rerunning only Stage 5 with `python -m tokenizer.encode --tokenizer <name>`, since all 8 models are on HF.

## Committed artifacts

- `tokenizer/hi_unigram_8192.model`, `tokenizer/hi_unigram_8192.vocab`: the chosen tokenizer only.
- `tokenizer/morph_words.tsv`
- `results/tokenizer_sweep.csv`, `results/plots/*.png`, `results/tokenizer_examples.md`

## Runtime

About 72 min in total:
- sample: about 10 min;
- 8 trainings: about 57 min;
- morph words + evaluation: about 2 min.
