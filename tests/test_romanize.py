"""Stage 7c (D10) romanizer checks. Pure Python: no sentencepiece, torch or network."""

import re
from collections import Counter

import pytest

from data.config import load_config
from data.hinglish.romanize import (NOISE_LEVELS, SHORTCUTS, Romanizer, apply_noise, build_sampler,
                                    rule_romanize, seeded_rng)
from data.normalize import normalize
from tokenizer.hinglish import PUNCT

CFG = load_config("configs/hinglish_data.yaml")["generator"]
VALID = re.compile("^[a-z0-9 " + re.escape("".join(PUNCT)) + "]*$")
SENTENCES = [
    "मैं कल तुम्हारे घर आऊँगा, ठीक है?",
    "क्या आप मेरी बात समझ रहे हैं।",
    "उसने 2019 में ३ किताबें लिखीं!",
    "धीरे-धीरे सब ठीक हो गया (शायद)।",
    "ज़िंदगी में कुछ भी आसान नहीं होता॥",
    "यह “ख़ास” दिन है – सच में…",
]
LEXICON = build_sampler({normalize("कल"): {"kal": 5, "kl": 1}, normalize("बात"): {"baat": 3, "bat": 2}})


@pytest.fixture
def romanizer():
    return Romanizer(LEXICON, CFG)


@pytest.mark.parametrize("word, expected", [
    ("कमल", {"kamal"}), ("घर", {"ghar"}), ("बहन", {"bahan", "behan"}), ("करना", {"karna"}),
    ("अपना", {"apna"}), ("लड़की", {"ladki"}), ("समझना", {"samajhna"}), ("आदमी", {"aadmi"}),
    ("नमस्ते", {"namaste"}), ("मिलकर", {"milkar"}), ("समय", {"samay"}), ("जंगल", {"jangal"}),
])
def test_schwa_deletion(word, expected):
    assert rule_romanize(normalize(word)) in expected


@pytest.mark.parametrize("word, expected", [
    ("संबंध", "sambandh"),     # anusvara -> m before a labial, n elsewhere
    ("हिंदी", "hindi"),
    ("माँ", "maan"),           # chandrabindu -> n, no final shortening before it
    ("अतः", "atah"),           # visarga -> h
    ("ज़रूर", "zaroor"),        # nukta letters
    ("फ़िल्म", "film"),
    ("क़लम", "qalam"),
    ("दोस्त", "dost"),          # cluster-final schwa dropped ...
    ("मित्र", "mitra"),         # ... unless the cluster ends in r/y/v/n
    ("कृष्ण", "krishna"),
    ("ज्ञान", "gyaan"),
    ("लिए", "liye"),
    ("जगत्", "jagat"),          # final halant: no vowel
    ("दिल्ली", "dilli"),
])
def test_rule_signs(word, expected):
    assert rule_romanize(normalize(word)) == expected


def test_precomposed_and_decomposed_nukta_agree():
    assert rule_romanize("ज़") == rule_romanize("ज़") == "za"


def test_deterministic_with_seed(romanizer):
    for i, s in enumerate(SENTENCES):
        a = romanizer.sentence(normalize(s), seeded_rng(1337, i))
        b = romanizer.sentence(normalize(s), seeded_rng(1337, i))
        assert (a.rom, a.noise_level, a.tokens) == (b.rom, b.noise_level, b.tokens)


@pytest.mark.parametrize("level", NOISE_LEVELS)
def test_output_charset_every_level(romanizer, level):
    for seed in range(50):
        for s in SENTENCES:
            r = romanizer.sentence(normalize(s), seeded_rng(seed, s), level=level)
            assert r.noise_level == level
            assert r.rom and VALID.match(r.rom), r.rom
            assert "  " not in r.rom and r.rom == r.rom.strip()


def test_digits_and_punctuation(romanizer):
    r = romanizer.sentence(normalize("उसने 2019 में ३ किताबें लिखीं!"), seeded_rng(0), level="clean")
    assert "2019" in r.rom and " 3 " in r.rom and r.rom.endswith("!")
    r = romanizer.sentence(normalize("यह “ख़ास” दिन है – सच में…"), seeded_rng(0), level="clean")
    assert not set("“”–…") & set(r.rom)


def test_danda_becomes_dot_sometimes(romanizer):
    ends = Counter(romanizer.sentence(normalize("वह घर गया।"), seeded_rng(s), level="clean").rom[-1]
                   for s in range(400))
    assert 0.8 < ends["."] / 400 < 0.97 and set(ends) <= {".", "a"}


def test_shortcuts_used_and_sampled(romanizer):
    seen = Counter()
    for seed in range(300):
        r = romanizer.sentence(normalize("क्या है"), seeded_rng(seed), level="clean")
        assert [src for _, srcs in r.tokens for src in srcs] == ["shortcut", "shortcut"]
        seen[r.rom.split()[1]] += 1
    assert set(seen) == set(SHORTCUTS["है"])
    assert seen["hai"] > seen["h"]                              # weighted, not uniform


def test_lexicon_then_rules(romanizer):
    r = romanizer.sentence(normalize("कल बात घर"), seeded_rng(3), level="clean")
    assert [srcs for _, srcs in r.tokens] == [["lexicon"], ["lexicon"], ["rules"]]
    assert r.rom.split()[0] in {"kal", "kl"} and r.rom.split()[2] == "ghar"
    assert r.source_mix() == {"shortcut": 0.0, "lexicon": 0.6667, "rules": 0.3333}


def test_rule_only_baseline_ignores_tables():
    rule_only = Romanizer(LEXICON, CFG, use_shortcuts=False, use_lexicon=False)
    r = rule_only.sentence(normalize("क्या कल"), seeded_rng(0), level="clean")
    assert r.source_counts() == {"shortcut": 0, "lexicon": 0, "rules": 2}
    assert r.rom == "kya kal"


def test_noise_ops():
    rng = seeded_rng(0)
    assert apply_noise("raha", "clean", rng, CFG) == "raha"
    heavy = {**CFG, "heavy_p": 0.0, "heavy_verb_drop_p": 1.0}
    assert apply_noise("raha", "heavy", rng, heavy) == "rha"
    assert apply_noise("karna", "heavy", rng, heavy) == "krna"
    assert apply_noise("ghar", "heavy", rng, heavy) == "ghar"   # not a verb form
    mild = {**CFG, "mild_p": 1.0}
    assert apply_noise("shaadi", "mild", rng, mild) in {"shadi", "saadi"}
    assert apply_noise("123", "mild", rng, mild) == "123"


def test_noise_level_mix(romanizer):
    levels = Counter(romanizer.sentence("घर", seeded_rng(s)).noise_level for s in range(4000))
    for level, share in CFG["noise_levels"].items():
        assert abs(levels[level] / 4000 - share) < 0.03
