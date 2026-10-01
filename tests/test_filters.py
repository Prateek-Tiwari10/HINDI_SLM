from data.config import load_config
from data.filters import apply_filters, devanagari_ratio, pii_mask
from data.normalize import normalize

CFG = load_config("configs/data.yaml")["filters"]

GOOD_SENTENCE = "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं और लोग मिलजुल कर रहते हैं।"


def good_doc(lines=8):
    return "\n".join(f"{GOOD_SENTENCE} यह पंक्ति संख्या {i} है।" for i in range(lines))


def test_good_hindi_doc_kept():
    assert apply_filters(good_doc(), CFG) == (True, "")


def test_short_doc_removed():
    assert apply_filters("बहुत छोटा लेख है।", CFG) == (False, "min_words")


def test_english_doc_removed():
    text = " ".join(["this is plain english text"] * 20)
    assert apply_filters(text, CFG) == (False, "devanagari_ratio")


def test_matras_count_as_devanagari_letters():
    assert devanagari_ratio("किताबें") == 1.0


def test_repeated_lines_removed():
    text = "\n".join([GOOD_SENTENCE] * 10)
    assert apply_filters(text, CFG) == (False, "unique_lines")


def test_boilerplate_repeats_removed():
    text = normalize(good_doc() + "\nयह भी पढ़ें\nयह भी पढ़ें\nयह भी पढ़ें")
    assert apply_filters(text, CFG) == (False, "boilerplate")


def test_symbol_heavy_removed():
    text = good_doc() + "\n" + "@#$%^&*" * 200
    assert apply_filters(text, CFG)[1] in {"symbol_ratio", "mean_word_len"}


def test_pii_mask():
    text = "संपर्क करें test.user@example.com या +91 98765 43210 पर"
    masked = pii_mask(text, CFG)
    assert "<EMAIL>" in masked and "<PHONE>" in masked
    assert "example.com" not in masked and "98765" not in masked
