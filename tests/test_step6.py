"""D7 Step 6: scoring rules (eval/translit_eval.py) and shared inference (finetune/translit.py)."""

from collections import Counter

import numpy as np
import pytest
import torch

from eval import translit_eval as te
from finetune.translit import MAX_CHUNK_CHARS, Transliterator, export_release, load_release, prompt_ids, split_input
from model.config import ModelConfig
from model.model import HindiLlama
from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer

CAP = {"ratio": 0.56, "factor": 1.5, "extra": 8}


@pytest.fixture(scope="module")
def tok():
    return HinglishTokenizer.load(DEFAULT_EXT)


def tiny_model(tok) -> HindiLlama:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=128,
                      max_seq_len=512, expected_params=None)
    return HindiLlama(cfg).eval()


# ---------------------------------------------------------------- normalization
def test_norm_dev_ignores_nukta_precomposed_and_combining():
    pre, comb, plain = "क़ाम", "क़ाम", "काम"
    assert te.norm_dev(pre) == te.norm_dev(comb) == te.norm_dev(plain) == "काम"
    assert te.raw_dev(pre) != te.raw_dev(plain)


def test_norm_dev_drops_only_one_final_danda():
    assert te.norm_dev("राम घर गया ।") == "राम घर गया"
    assert te.norm_dev("राम।  घर गया") == "राम। घर गया"
    assert te.norm_dev("राम गया।।") == "राम गया।"
    assert te.norm_dev("ठीक है.") == "ठीक है."               # ASCII dot still counts (spec §3)


def test_raw_and_normalized_scores():
    s = te.score(["राम घर गया"], ["राम घर गया।"], te.raw_dev)
    n = te.score(["राम घर गया"], ["राम घर गया।"], te.norm_dev)
    assert s["cer"] > 0 and s["exact_match"] == 0
    assert n["cer"] == 0 and n["exact_match"] == 1


def test_bootstrap_ci_brackets_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    stats = np.stack([rng.integers(0, 5, 200), rng.integers(20, 40, 200), rng.integers(0, 2, 200),
                      rng.integers(5, 9, 200), rng.integers(0, 2, 200)], axis=1)
    point = te.corpus(stats)
    a, b = te.bootstrap(stats, b=500), te.bootstrap(stats, b=500)
    assert a == b
    assert a["cer_ci95"][0] <= point["cer"] <= a["cer_ci95"][1]
    assert te.bootstrap(stats, stats, b=200)["cer_ci95"] == [0.0, 0.0]


# ---------------------------------------------------------------- input splitting / prompt format
def test_split_input_short_is_one_normalized_chunk():
    assert split_input("  Kal   MEETING hai ") == ["kal meeting hai"]
    assert split_input("   ") == []


def test_split_input_long_respects_limit_and_keeps_words():
    words = ("yeh ek bahut lamba vaakya hai jo kai baar dohraya gaya hai " * 12).split()
    text = " ".join(words)
    chunks = split_input(text)
    assert len(chunks) > 1 and all(0 < len(c) <= MAX_CHUNK_CHARS for c in chunks)
    assert " ".join(chunks).split() == words
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) < 40                         # near-equal pieces


def test_split_input_prefers_sentence_ends_and_cuts_huge_words():
    a, b = "pehla vaakya " * 8 + "khatam.", "doosra vaakya " * 8 + "bhi?"
    assert split_input(f"{a} {b}") == [a.strip(), b.strip()]
    chunks = split_input("x" * 400)
    assert [len(c) for c in chunks] == [160, 160, 80]


def test_prompt_ids_is_the_training_prompt(tok):
    ids, labels = tok.build_pair("Kya haal hai?", "क्या हाल है?")
    p = labels.index(next(x for x in labels if x != -100))
    assert prompt_ids(tok, "kya haal hai?") == ids[:p]


# ---------------------------------------------------------------- baseline + ambiguity
def test_lexicon_alignment_and_lookup():
    pairs = [("main ghar ja raha hoon.", "मैं घर जा रहा हूँ।"), ("ghar main hai", "घर में है"),
             ("main bhi", "मैं भी"), ("ek do teen", "एक दो")]            # last one does not align
    lex, stats = te.build_lexicon(pairs)
    assert stats["aligned_pairs"] == 3
    table = te.lookup_table(lex)
    assert table["main"] == "मैं" and table["hoon"] == "हूँ"
    assert te.baseline("Main ghar xyz, ja raha.", table) == "मैं घर xyz, जा रहा।"


def test_lookup_tie_goes_to_first_seen():
    lex, _ = te.build_lexicon([("ki", "की"), ("ki", "कि")])
    assert te.lookup_table(lex)["ki"] == "की"


def test_ambiguous_words_rule():
    lex = {"me": {"में": 80, "मैं": 30}, "hai": {"है": 95, "हैं": 5}, "kam": {"काम": 50, "कम": 40},
           "zara": {"ज़रा": 60, "जरा": 60}}
    amb = te.ambiguous_words({k: Counter(v) for k, v in lex.items()})
    assert "me" in amb and "hai" not in amb                     # 30/110 >= 10%; 5/100 < 10%
    assert "kam" not in amb                                     # only 90 uses (< 100)
    assert "zara" not in amb                                    # nukta variants merge under normalization
    assert {"ki", "main"} <= set(amb)                           # always included


def test_alignment_hits():
    assert te.matched_ref_positions(list("abc"), list("axc")) == {0, 2}
    assert te.matched_ref_positions(list("abc"), list("abxc")) == {0, 1, 2}
    hits = te.ambiguity_hits("wo ghar main hai", "वो घर में है।", "वो घर मैं है", {"main"})
    assert hits == [("main", False)]
    assert te.ambiguity_hits("wo ghar main hai", "वो घर में है।", "वो घर में है", {"main"}) == [("main", True)]
    assert te.ambiguity_hits("wo main", "वो घर में", "x", {"main"}) is None
    rep = te.slice_report([[("main", True), ("ki", False)], None, [("main", True)]])
    assert rep["occurrences"] == 3 and rep["accuracy"] == pytest.approx(2 / 3, abs=1e-4) and rep["aligned_sentences"] == 2


# ---------------------------------------------------------------- inference + release round trip
def test_transliterator_runs_and_masks(tok):
    t = Transliterator(tiny_model(tok), tok, CAP, "cpu")
    r = t("kal meeting cancel ho gayi kya")
    assert isinstance(r.output, str) and r.chunks == 1 and r.new_tokens <= t.budget(len(t.prompt("kal meeting cancel ho gayi kya")))
    assert not any("a" <= ch <= "z" for ch in r.output)
    long = t("bahut lamba " * 30)
    assert long.chunks > 1


def test_release_round_trip_identical_logits(tok, tmp_path):
    model = tiny_model(tok)
    export_release(model, tok, CAP, tmp_path, {"checkpoint": "test"})
    rel = load_release(tmp_path, "cpu")
    assert rel.model.lm_head.weight is rel.model.tok_emb.weight
    x = torch.tensor([prompt_ids(tok, "kya haal hai")])
    with torch.inference_mode():
        assert torch.equal(model(x)[0], rel.model(x)[0])
    assert rel.cap == CAP and rel("kya haal hai").output == Transliterator(model, tok, CAP)("kya haal hai").output
