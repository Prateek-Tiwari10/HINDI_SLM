"""Stage 7c romanize-v2 lexicon (D10): source precedence, Aksharantar filters, inflection guard, weights."""

import pytest

pytest.importorskip("torch")          # data.hinglish.stage7c imports finetune.data

from data.hinglish.stage7c import build_lexicon, final_vowel, first_sound, lexicon_weights  # noqa: E402
from data.normalize import normalize  # noqa: E402

CFG = {"seed": 1, "lexicon": {"min_len_ratio": 0.5, "max_len_ratio": 3.0, "max_rule_distance": 0.34,
                              "weight_power": 2, "drop_singletons_if_top_ge": 5}}


@pytest.mark.parametrize("lat,cls", [("koi", "k"), ("khoyi", "kh"), ("arrange", "V"), ("karenge", "k"),
                                     ("qalam", "k"), ("chaar", "C"), ("chhota", "Ch"), ("shaam", "s"),
                                     ("sam", "s"), ("phir", "f"), ("fir", "f"), ("waqt", "v"), ("zaroor", "j"),
                                     ("thoda", "th"), ("toda", "t")])
def test_first_sound(lat, cls):
    assert first_sound(lat) == cls


@pytest.mark.parametrize("lat,nasal,cls", [("sakte", False, "E"), ("sakta", False, "A"), ("donon", True, "O"),
                                           ("dono", True, "O"), ("dodoan", True, "A"), ("hain", True, "E"),
                                           ("hai", False, "E"), ("he", False, "E"), ("ki", False, "I"),
                                           ("kee", False, "I"), ("key", False, "E"), ("kamal", False, "C"),
                                           ("yeh", False, "E"), ("ye", False, "E"), ("maan", True, "A"),
                                           ("hoon", True, "U"), ("bhi", False, "I")])
def test_final_vowel(lat, nasal, cls):
    assert final_vowel(lat, nasal) == cls


@pytest.fixture
def lexicon(tmp_path):
    tsv = tmp_path / "train.tsv"
    tsv.write_text("\n".join(["की\tki\t6", "की\tkee\t1", "की\tkey\t1", "कि\tki\t3",
                              "कमल\tkamal\t2", "कमल\tkamla\t1"]) + "\n", encoding="utf-8")
    rows = [("कमल", "kml"), ("सकते", "sakte"), ("सकते", "sakta"), ("सकता", "sakta"), ("दस्त", "dast"),
            ("दोस्त", "dost"), ("दोस्त", "dast"), ("करेंगे", "arrange"), ("कोई", "khoyi")]
    report = {}
    table = build_lexicon(tsv, [{"native word": n, "english word": e} for n, e in rows], CFG, report)
    return {normalize(k): v for k, v in table.items()}, report["lexicon"]


def test_dakshina_precedence_and_singleton_drop(lexicon):
    table, rep = lexicon
    assert dict(table[normalize("की")]) == {"ki": 6}            # top >= 5: count-1 variants dropped
    assert dict(table[normalize("कि")]) == {"ki": 3}            # shared spelling kept for both words
    assert dict(table[normalize("कमल")]) == {"kamal": 2, "kamla": 1}   # top < 5: kept; Aksharantar ignored
    assert rep["rejected_rows"]["aksharantar:word_in_dakshina"] == 1
    assert rep["singleton_spellings_dropped"] == 2


def test_aksharantar_filters(lexicon):
    table, rep = lexicon
    assert "sakta" not in table.get(normalize("सकते"), {})       # final vowel a vs e (inflection)
    assert "sakta" in table[normalize("सकता")]
    assert dict(table[normalize("दोस्त")]) == {"dost": 1}        # dast is दस्त's spelling
    assert rep["rejected_rows"]["aksharantar:inflection_guard"] == 1
    assert "arrange" not in table.get(normalize("करेंगे"), {})
    assert "khoyi" not in table.get(normalize("कोई"), {})


def test_weights_are_count_squared(lexicon):
    table, _ = lexicon
    assert lexicon_weights(table, CFG)[normalize("कमल")] == {"kamal": 4, "kamla": 1}
