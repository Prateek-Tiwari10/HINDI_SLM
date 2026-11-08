"""Stage 7a (D8) checks for HinglishTokenizer. Needs sentencepiece (skipped without it).

Test 1 needs real v2 Devanagari text: set HINGLISH_DEV_SAMPLE to a file with one sentence
per line (the Stage 7b Kaggle kernel writes one from final_v2 val); skipped otherwise.
"""

import json
import os
import random
import shutil
import string

import pytest

spm = pytest.importorskip("sentencepiece")

from tokenizer.hinglish import (BASE_VOCAB, DEFAULT_EXT, IGNORE, LUNK_TEXT, HinglishTokenizer,  # noqa: E402
                                normalize_rom)
from tokenizer.stage7a import PROBES  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return HinglishTokenizer.load(DEFAULT_EXT)


# 1. encode_dev == raw SentencePiece on real v2 text
def test_encode_dev_matches_base_on_v2_text(tok):
    path = os.environ.get("HINGLISH_DEV_SAMPLE")
    if not path or not os.path.exists(path):
        pytest.skip("HINGLISH_DEV_SAMPLE not set; run in the Stage 7b kernel")
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln.strip()][:1000]
    assert len(lines) == 1000
    for line in lines:
        assert tok.encode_dev(line) == tok.sp.encode(line), line


# 2. decode_rom(encode_rom(x)) == normalize_rom(x)
def test_rom_roundtrip_probes(tok):
    for text in PROBES[:-1]:                                  # the last probe is Devanagari
        assert tok.decode_rom(tok.encode_rom(text)) == normalize_rom(text)


def test_rom_roundtrip_random_ascii(tok):
    rng = random.Random(0)
    alphabet = string.ascii_letters + string.digits + ".,?!'\"-:;()/" + " \t\n"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        ids = tok.encode_rom(text)
        assert tok.token_to_id["<lunk>"] not in ids
        assert tok.decode_rom(ids) == normalize_rom(text)


# 3. new IDs live in [10240, 10304) and never collide with base IDs
def test_new_ids_range(tok):
    ids = sorted(tok.token_to_id.values())
    assert ids == list(range(BASE_VOCAB, 10304))
    assert tok.vocab_size == 10304 and tok.vocab_size % 64 == 0
    assert tok.sp.get_piece_size() == BASE_VOCAB
    assert (tok.rom_id, tok.dev_id, tok.eos_id) == (10240, 10241, 3)


# 4. lowercase, accents stripped, other characters -> <lunk>
def test_rom_normalization(tok):
    assert tok.encode_rom("KAL Meeting") == tok.encode_rom("kal meeting")
    assert tok.encode_rom("café naïve") == tok.encode_rom("cafe naive")
    assert normalize_rom("  a \t b\n") == "a b"
    lunk = tok.token_to_id["<lunk>"]
    for weird in ("😀", "@", "#", "क", "₹"):
        assert tok.encode_rom(weird) == [lunk], weird
    assert tok.decode_rom(tok.encode_rom("ok 😀")) == f"ok {LUNK_TEXT}"


# 5. build_pair: masking and first target
def test_build_pair(tok):
    rom, dev = "kal meeting cancel ho gayi kya", "कल मीटिंग कैंसल हो गई क्या"
    input_ids, labels = tok.build_pair(rom, dev)
    rom_ids, dev_ids = tok.encode_rom(rom), tok.encode_dev(dev)
    assert len(input_ids) == len(labels) == len(rom_ids) + len(dev_ids) + 3
    assert input_ids[0] == tok.rom_id and input_ids[len(rom_ids) + 1] == tok.dev_id
    assert input_ids[-1] == labels[-1] == tok.eos_id
    targets = [x for x in labels if x != IGNORE]
    assert len(targets) == len(dev_ids) + 1
    first = next(i for i, x in enumerate(labels) if x != IGNORE)
    assert labels[first] == dev_ids[0] and input_ids[first - 1] == tok.dev_id
    # shifted the way training uses it: the logits at <dev> predict the first Devanagari token
    x, y = input_ids[:-1], labels[1:]
    assert y[x.index(tok.dev_id)] == dev_ids[0]
    assert tok.decode_dev(input_ids) == dev


# 6. mismatched base sha256 raises
def test_sha_mismatch_raises(tok, tmp_path):
    meta = json.loads(DEFAULT_EXT.read_text())
    meta["base_sha256"] = "0" * 64
    bad = tmp_path / "tokenizer_ext.json"
    bad.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sha256"):
        HinglishTokenizer.load(bad)
    other = tmp_path / "other.model"
    shutil.copy(tok.base_model, other)
    other.write_bytes(other.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        HinglishTokenizer.load(DEFAULT_EXT, base_model=other)


# 7. allowed output IDs
def test_output_allowed_ids(tok):
    allowed = tok.output_allowed_ids()
    assert tok.eos_id in allowed
    assert max(allowed) < BASE_VOCAB
    sp = tok.sp
    for banned in (sp.unk_id(), sp.bos_id(), sp.pad_id(), sp.piece_to_id("<|user|>")):
        assert banned not in allowed
    assert sp.piece_to_id("<0x61>") not in allowed                    # byte 'a'
    assert sp.piece_to_id("<0xE0>") in allowed                        # UTF-8 lead byte of Devanagari
    for i in allowed:
        if not sp.is_byte(i) and i != tok.eos_id:
            assert not any(c in string.ascii_letters for c in sp.id_to_piece(i))
    assert sp.piece_to_id("▁के") in allowed
