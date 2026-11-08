# Stage 7a — Tokenizer extension (D8), Kaggle CPU

**Status:** done 2026-10-27. **Gate 7a: PASS.**
- **Tests:** 7 of 7 runnable `HinglishTokenizer` tests pass. Test 1 needs real v2 text and is skipped here; it runs in the Stage 7b kernel. All 84 pre-existing tests still pass, for 91 passed and 1 skipped.
- **Pretraining artifacts:** untouched. The base model's sha256 is pinned, and `tokenizer/hi_unigram_10240.model` is a byte-identical copy of HF `hindi-lm-tokenizer/sweep_v2/hi_unigram_10240.model`.
- **Kaggle run:** `prateek99491/hindi-lm-d7-tests`, version 2 (CPU, 4 cores, 31 GB RAM, Python 3.13.15).
  - It was started by API push with the code embedded (`EMBED_CODE=1 kaggle/push.sh "python -m tokenizer.stage7a" d7_tests`), since no HF access is needed.
  - The runner runs `pytest -v tests/` first, then the inspection.
  - Version 1 passed the same tests, but the inspection crashed because this sentencepiece build has no `is_user_defined`. It was fixed to read piece types from the model proto.
- **Why Kaggle instead of the laptop:** the laptop venv has neither sentencepiece nor torch, and installing packages there is off-limits (low RAM).
- **Outputs:** [stage7a_inspect.json](stage7a_inspect.json) and [stage7a_pytest.txt](stage7a_pytest.txt).

## Base tokenizer (inspection only, nothing changed)

| item | value |
| --- | --- |
| path | `tokenizer/hi_unigram_10240.model` (copy of HF `sweep_v2/hi_unigram_10240.model`) |
| sha256 | `70e469cd87019e5daef3d3f00c7f66a5fc2a3f011af0f76b9ec228bc934eea27` |
| model type / vocab | UNIGRAM / **10,240** ✅ |
| byte_fallback | true |
| character_coverage | 0.9995 |
| split_digits / normalizer | true / `identity` (our `normalize()` runs before SentencePiece) |
| pad / unk / bos / eos | 0 `<pad>` / 1 `<unk>` / 2 `<s>` / 3 `</s>` |
| piece types | 3 control, 1 unknown, 19 user-defined (`<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|reserved_0..15\|>`), 256 byte, 9,961 normal |

**Document-end marker:** `</s>` (id 3). [tokenizer/encode.py](../../tokenizer/encode.py) `encode_docs()` writes every pretraining document as `bos + tokens + eos`, and the tokenized `meta.json` records bos=2, eos=3. The fine-tuning pairs reuse id 3 as end of output.

**Base pieces with ASCII letters:**
- **Normal pieces:** 463. The first 30: `s e i y ▁S o d ▁P ▁B P ▁A er a A h l S al r t n C R an u p M k or ▁T`.
- **Byte pieces that stand for an ASCII letter:** 52 (`<0x41>`–`<0x5A>` and `<0x61>`–`<0x7A>`).
- **A naive regex** over piece *names* matches 742. That count includes all 256 `<0xNN>` byte pieces (the "x" in their names) and the 23 special pieces.

## Probes through the base tokenizer

| probe | chars | base pieces | unk | byte | char-level IDs | pieces |
| --- | --- | --- | --- | --- | --- | --- |
| kal meeting cancel ho gayi kya | 30 | 18 | 0 | 0 | 30 | `▁ k al ▁ me et ing ▁c ance l ▁ ho ▁g ay i ▁ k ya` |
| main ghar ja raha hu | 20 | 12 | 0 | 0 | 20 | `▁ ma in ▁g har ▁ ja ▁ ra ha ▁ hu` |
| mai ghr jaa rha hun | 19 | 14 | 0 | 0 | 19 | `▁m ai ▁g h r ▁ ja a ▁ r ha ▁ h un` |
| tum kaha ho? | 12 | 9 | 0 | 0 | 12 | `▁ t um ▁ ka ha ▁ ho ?` |
| wo aayi ki nahi | 15 | 11 | 0 | 0 | 15 | `▁w o ▁a ay i ▁ k i ▁ na hi` |
| rahul ki kitab | 14 | 11 | 0 | 0 | 14 | `▁ ra h ul ▁ k i ▁ k it ab` |
| mujhe 7 baje uthna h | 20 | 15 | 0 | 0 | 20 | `▁m u j he ▁ 7 ▁b aj e ▁ u th na ▁ h` |
| kya baat hai yaar!! | 19 | 14 | 0 | 0 | 19 | `▁ k ya ▁b a at ▁ ha i ▁ ya ar ! !` |
| Office mein Wi-Fi nahi chal raha | 32 | 20 | 0 | 0 | 32 | `▁O ff ice ▁ me in ▁W i - F i ▁ na hi ▁ ch al ▁ ra ha` |
| aaj bahut garmi hai | 19 | 12 | 0 | 0 | 19 | `▁a aj ▁b a h ut ▁g ar mi ▁ ha i` |
| आज बहुत गर्मी है | 16 | 4 | 0 | 0 | 15 (all `<lunk>`) | `▁आज ▁बहुत ▁गर्मी ▁है` |

**What this shows:**
- The base model *can* encode Latin: there are no unk or byte pieces on any probe. It does so with 463 leftover English fragments from a corpus that is more than 99% Devanagari, and the result has three problems:
  1. **Fragmented and inconsistent.** The same Hindi word splits differently depending on spelling and context. "ki" is `▁ k i`, "kya" is `▁ k ya`, "hai" is `▁ ha i`, "hu" is `▁ hu` but "hun" is `▁ h un`. Spelling variants share no pieces in a predictable way. This is the problem D8 avoids by going to characters.
  2. **Case-sensitive.** "Office" becomes `▁O ff ice`, so "office" and "Office" are different sequences. The Latin side lowercases instead.
  3. **The same IDs would be inputs and banned outputs.** These pieces must be masked from generation. With them also used as inputs, the model would see them as tokens it should copy.
- **Length cost of characters:** over the 10 Latin probes, character-level input is **200 IDs against 136 base pieces (1.47×)**. Against the Devanagari output it is about 4–5× (19 IDs for "aaj bahut garmi hai" against 4 pieces for "आज बहुत गर्मी है"). Sentence-length inputs stay well under 1,024 tokens either way.

## Extension (implemented as specified)

[tokenizer/hinglish.py](../../tokenizer/hinglish.py) `HinglishTokenizer`, saved as [tokenizer/tokenizer_ext.json](../../tokenizer/tokenizer_ext.json) (`"version": "hinglish-v1"`, with the base path and sha256).

| group | tokens | count | IDs |
| --- | --- | --- | --- |
| specials | `<rom>`, `<dev>` | 2 | 10240–10241 |
| letters | a–z | 26 | 10242–10267 |
| digits | 0–9 | 10 | 10268–10277 |
| punctuation | `. , ? ! ' " - : ; ( ) /` | 12 | 10278–10289 |
| Latin space | `<lsp>` | 1 | 10290 |
| unknown Latin-side char | `<lunk>` | 1 | 10291 |
| reserved (never used) | `<reserved_0>` … `<reserved_11>` | 12 | 10292–10303 |

**Final vocab: 10,304** (64 × 161). Base IDs 0–10239 are unchanged.

- **Latin normalization:** NFKC → lowercase → NFD and drop combining marks → collapse whitespace → strip.
- **Devanagari side:** `data.normalize.normalize`, the pretraining function, followed by the base SentencePiece encode.
- **Pair format:** `input_ids = [<rom>] + rom_ids + [<dev>] + dev_ids + [</s>]`, with `labels` = −100 up to and including `<dev>`, then `dev_ids + [</s>]`.
- **Shift convention:** `labels` is aligned with `input_ids` (not pre-shifted). `HindiLlama.forward` does no internal shift. Pretraining passes `x = w[:, :-1]`, `y = w[:, 1:]` ([train/data.py](../../train/data.py)), and fine-tuning does the same with `x = input_ids[:-1]`, `y = labels[1:]`. The logits at `<dev>` then predict the first Devanagari token (checked in test 5). −100 is the default `ignore_index` of `F.cross_entropy`, which `forward` already uses.
- **No BOS:** the pair starts with `<rom>`, not `<s>`, as the spec format says. Pretraining documents started with `<s>` (2), so `<rom>` takes over the "sequence start" role. Phase A trains `<rom>`'s row. *This is a spec choice, noted here, not a conflict.*
- **Allowed outputs:** **9,703** IDs are allowed and 537 banned. The banned set is pad, unk and bos, the 463 Latin normal pieces, the 52 Latin-letter byte pieces, and the 19 chat/reserved pieces (all of which contain letters).
  - **Interpretation, flagged:** the spec says to ban "any base piece containing ASCII letters". Byte pieces are judged by the byte they stand for, not by their `<0xNN>` name. Read by name, all 256 byte pieces would be banned, and the model could never emit the UTF-8 bytes of rare Devanagari characters that fall outside the 0.9995 character coverage.

## Tests ([tests/test_hinglish_tokenizer.py](../../tests/test_hinglish_tokenizer.py))

| # | check | result |
| --- | --- | --- |
| 1 | `encode_dev` == raw SentencePiece on 1,000 v2 sentences | ⏭ skipped (no local v2 text). Runs in the Stage 7b kernel with `HINGLISH_DEV_SAMPLE` |
| 2 | `decode_rom(encode_rom(x)) == normalize_rom(x)` for the probes and 200 random ASCII strings | ✅ |
| 3 | new IDs are exactly 10240–10303, no collision; vocab 10,304 = 64·161 | ✅ |
| 4 | uppercase → lowercase, accents stripped; emoji, `@`, `#`, `₹`, Devanagari → `<lunk>` | ✅ |
| 5 | `build_pair`: non-masked labels = `len(dev_ids) + 1`; first target = first Devanagari token; logits at `<dev>` predict it | ✅ |
| 6 | mismatched sha256 raises (edited JSON, and a changed model file) | ✅ |
| 7 | `output_allowed_ids()` contains eos and no ID ≥ 10240; bans unk/bos/pad, chat pieces and byte 'a'; keeps `<0xE0>` and `▁के` | ✅ |
| — | all 84 pre-existing tests | ✅ |

The "random ASCII strings" in test 2 are drawn from the supported alphabet (letters in both cases, digits, the 12 punctuation marks, space, tab and newline). Characters outside it become `<lunk>` by design, which test 4 covers, so they cannot round-trip.

## Other changes
- [kaggle/push.sh](../../kaggle/push.sh): with `EMBED_CODE=1`, the push now includes `tokenizer/hi_unigram_10240.model` (255 KB). Before, every `.model` file was skipped.
- New kernel folder [kaggle/d7_tests/](../../kaggle/d7_tests/) (CPU, API push, no secrets).
