# Stage 1 — Source inventory

**Status:** done. Kaggle kernel `prateek99491/hindi-lm-runner` (version 3).
- **Kaggle username verified:** `prateek99491` (checked by `kaggle/push.sh` and `k.sh config view`).
- **Runtime:** about 6 min.
- **Tests:** 29/29 passed on Kaggle before the stage ran.

## What was run

`python -m data.pipeline stage1`, using `configs/data.yaml` (hash `d043f3f20afc`).
- **Sampling:** for each source, 10,000 docs were reservoir-sampled from the first 100k docs of up to 3 seeded-random files.
- **Storage:** the samples were uploaded to `Prateek-Tiwari10/hindi-lm-data` under `raw/samples/<source>.jsonl.gz`. They are not in git.
- **Full-size estimates:** doc counts come from parquet footers (for the manual collection, a full line count). Bytes are doc count × mean sample bytes. Tokens are bytes / 9.5.

## Numbers (`results/stage1_source_stats.csv`)

| source | docs sampled | mean chars | median chars | mean bytes | % Devanagari letters | est. total docs | est. total GB | est. tokens (B) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikipedia | 10,000 | 1,488 | 485 | 3,740 | 96.9 | 163,093 | 0.61 | 0.06 |
| manual | 10,000 | 3,347 | 2,431 | 7,062 | 95.2 | 377,324 | 2.66 | 0.28 |
| sangraha_verified | 10,000 | 2,230 | 1,538 | 5,649 | 97.8 | 17,420,932 | 98.42 | 10.36 |
| fineweb2 | 10,000 | 2,496 | 1,699 | 6,226 | 96.4 | 22,095,985 | 137.56 | 14.48 |

**Reading the numbers:**
- The Stage 3 target is 2.1B train tokens, or about 2.6B with the 1.25× overshoot. That is about 25 GB of surviving text.
- Wikipedia (counted 2×) plus the manual collection gives only about 0.4B tokens. Sangraha Verified has about 10.4B before filtering, about 4× what we need even if half is filtered away.
- So FineWeb-2 is not needed.
- Wikipedia's median doc is short (485 chars, many stubs). The `min_words: 50` filter will remove a share of it; Stage 2 measures how much.

## Decisions at the STOP gate

Prateek asked to run all stages end to end. These defaults are applied and can be overridden:

| Question | Decision | Why |
| --- | --- | --- |
| Sources | Wikipedia `20231101.hi` (all), manual collection (all), Sangraha Verified `hin` (streamed until target) | As in D2; all verified to exist. |
| Licences | CC-BY-SA-3.0/GFDL (Wikipedia), CC-BY-4.0 (Sangraha), ODC-BY (FineWeb-2) are all fine for training. Manual: news copyright with the publishers, private research use only, never redistributed (private HF repo). | See `data/SOURCES.md`. |
| FineWeb-2 | **Excluded** (`enabled: false`) | Sangraha alone is about 4× the need. FineWeb-2 is only for the case where Sangraha runs out. |
| Sangraha Synthetic | Excluded; `sources.py` refuses any path containing `synthetic` | D2. |

## What failed / notes

- **Versions 1–2:** failed at `get_secret` because the Kaggle secret was labelled `Hf_token`, not `HF_TOKEN`. Prateek fixed the label.
- **`huggingface_hub` upgrade:** the runner upgraded it to 2.0, which conflicts with Kaggle's preinstalled `datasets` and `transformers`. Our code doesn't import those, so nothing broke. Fixed for later runs: the upgrade is gone and `pyproject.toml` pins `huggingface_hub<2.0`.
- **Manual collection upload:** it hit a network error on the first try, and the retry succeeded (11/11 shards in `raw/manual/`).

## Previews (3 random docs per source, first 300 chars)


### wikipedia — wikipedia:584518

```text
श्री वंशीधर शर्मा अगस्त, 2014 से भारत के प्रमुख अर्धसैनिक बल सशस्त्र सीमा बल के महानिदेशक हैं।

सन्दर्भ 

सशस्त्र सीमा बल के महानिदेशक
```

### wikipedia — wikipedia:764494

```text
दक्पा भाषा भारत की संकटग्रस्त भाषाओं में से एक है। यह निश्चित रूप से खतरे में है। यह भूटान में भी बोली जाती है।
आईएसओ कोड: dka

भौगोलिक वर्गीकरण

नामकरण

क्षेत्रविस्तार

बोलियाँ

भाषा साहित्य

इन्हें भी देखें
भारत में संकटग्रस्त भाषाओं की सूची
भारत की भाषाएँ
भारत में स्थानीय वक्ताओं की संख्यानुसार भ
```

### wikipedia — wikipedia:12801

```text
मिखाइल गोर्बाचेव (2 मार्च 1931 - 30 अगस्त 2022) एक रूसी और सोवियत राजनेता थे जिन्होंने सोवियत संघ के अंतिम नेता के रूप में कार्य किया। वैचारिक रूप से, गोर्बाचेव ने शुरू में मार्क्सवाद-लेनिनवाद का पालन किया, लेकिन 1990 के दशक की शुरुआत में सामाजिक लोकतंत्र की ओर बढ़ गए।

गोर्बाचेव का जन्म प्रिवोलनोय,
```

### manual — manual:https://www.livehindustan.com/jharkhand/ghumla/story-meeting-on-voter-verification-and-training-held-in-block-office-201785863588338.html

```text
Ghumla News: एसआईआर को लेकर बीएलओ को दिया गया प्रशिक्षण
Ghumla News: फोटो नं. 7 बीएलओ के संग बैठक करते बीडीओ अरूण सिंह। फोटो नं. 7 बीएलओ के संग बैठक करते बीडीओ अरूण सिंह।फोटो नं. 7 बीएलओ के संग बैठक करते बीडीओ अरूण सिंह
Ghumla News: भरनो। प्रखंड कार्यालय सभागार में मंगलवार को बीडीओ अरुण कुमार सिंह क
```

### manual — manual:https://www.bhaskar.com/local/rajasthan/barmer/gudamalani/news/bishnoi-talent-award-ceremony-in-barasan-today-dbp-138701168.html

```text
बारासण में विश्नोई प्रतिभा सम्मान समारोह आज
- कॉपी लिंक
गुड़ामालानी | बारासण गांव में बुधवार को हिरणों की रक्षा के लिए प्राण न्योछावर करने वाले शहीद चिमनाराम व प्रतापाराम मांजू की स्मृति में जिला स्तरीय विश्नोई प्रतिभा सम्मान समारोह आयोजित होगा। समारोह में बाड़मेर-बालोतरा जिले की विभिन्न क्षेत्रों म
```

### manual — manual:https://www.bbc.com/hindi/india-53735581

```text
मोदी: इन 10 राज्यों ने कोरोना को हराया, तो भारत जीत जाएगा
प्रधानमंत्री नरेंद्र मोदी ने कहा है कि भारत के 10 राज्यों में ही कोरोना के कुल 80 प्रतिशत एक्टिव मामले हैं. उन्होंने कहा कि कोरोना के ख़िलाफ़ लड़ाई में इन दस राज्यों की अहम भूमिका है.पीएम मोदी ने दस राज्यों के मुख्यमंत्रियों से वीडियो कॉन्फ़्
```

### sangraha_verified — sangraha_verified:9cefc179a93bd96d0bd26131e5198b55f3550061

```text
कई बार ऐसा होता है कि बच्चे जाने-अनजाने ही ऐसी कई गलत बातें सीख जाते हैं, जो उनके व्यक्तित्व पर असर डालती है। कई बार तो बच्चों को अपने द्वारा बोले गए शब्दों का मतलब भी पता नहीं होता है।
 कहते हैं कि बच्चे अपने बड़ों की नकल करते हैं। कई बार ऐसा होता है कि बच्चे जाने-अनजाने ही ऐसी कई गलत बातें सीख जात
```

### sangraha_verified — sangraha_verified:7744387575d3a1b8cc66bc48816d29d4c0f3a0e8

```text
संदीप शर्मा, विदिशा। विदिशा जिले के लटेरी (leteri) में आदिवासियों पर गोली चलाने के मामले में विदिशा DFO का घटना के 48 घंटे के अंदर तबादला कर दिया गया है। विदिशा डीएफओ राजवीर सिंह का ट्रांसफर (Transfer of Vidisha DFO Rajveer Singh) कर दिया गया है। वहीं वन विभाग के 8 कर्मचारियों को सस्पेंड कर दिया गया
```

### sangraha_verified — sangraha_verified:db4d9202980b943f4b78dd9baf79b3aedc7ecc85ba86488b059a36ade2fcf099

```text
गृह मंत्रालय ने राज्य सरकारों केन्द्रशासित प्रदेशों की सरकारों से कहा है कि वे 15 मई, 2010 को जारी राजपत्रित अधिसूचना के प्रावधानों के अनुसार पास पोर्टों की वैधता पर जोर दिए बिना कुछ खास श्रेणियों के वैसे पाकिस्तानी नागरिकों के लिए दीर्घकालिक वीसा (एलटीवी) के विस्तार के मामले पर विचार करें, जो पाकिस
```

### fineweb2 — fineweb2:<urn:uuid:592129ab-b888-4f67-bae7-c4435e2d8259>

```text
अरूण साथी)
जंतर मंतर पे आप पार्टी की रैली में किसान गजेन्द्र जी ने पेड़ से लटक कर आत्महत्या कर ली और आत्महत्या के बाद भी आप नेता केजारीबाल सहित अन्य रैली को संबोधित करते रह गए, यह संवेदनहीनता की पराकाष्ठा है और इसकी जितनी निंदा की जाय कम होगी। इसके पीछे राजनीतिक चमकाने को लेकर भी सवाल उठ रहे है तो इ
```

### fineweb2 — fineweb2:<urn:uuid:698c040a-ec2b-4e26-855e-0694505887ed>

```text
यह तो हम सब जानते हैं कि CFL बल्ब कितनी बिजली बचाते हैं लेकिन ज़्यादातर लोग यह नहीं जानते कि इन बल्बों में पारा पाया जाता है जो कि शरीर में चले जाने पर बहुत ही घातक साबित होता है। स्मिथ ने ऐसे ही एक बल्ब के ठन्डे होने का इंतज़ार नहीं किया और उसे होल्डर से निकालकर बदलने की कोशिश करते हुए उसे ज़मीन पर
```

### fineweb2 — fineweb2:<urn:uuid:3cf6555a-1a17-429a-a8ed-009546b5cdd7>

```text
लोगों ने तुरंत 108 नंबर डायल करके एंबुलेंस को बुलाया और सामुदायिक केंद्र में भर्ती कराया गया था।
मध्य प्रदेश के छतरपुर में मैगी खाने से 9 बच्चों की तबियत बिगड़ गई है। इन सभी बच्चों ने शनिवार रात को मैगी खाई थी, जिसके बाद ये बीमार पड़ गए। सभी बच्चों की बिगड़ती तबियत के बाद उन्हें ग्वालियर के मेडकल कॉ
```
