from data.config import load_config
from data.filters import (apply_filters, devanagari_ratio, dup_ngram_char_frac, pii_mask, stopword_ratio,
                          top_ngram_char_frac)
from data.lines import LineCleaner
from data.normalize import normalize

FULL = load_config("configs/data.yaml")
CFG = FULL["filters"]

GOOD_SENTENCE = "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं और लोग मिलजुल कर रहते हैं।"

# Varied prose: no sentence repeats, so the Gopher repetition rules must keep it.
GOOD_LINES = [
    "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं और लोग मिलजुल कर रहते हैं।",
    "गंगा नदी हिमालय से निकलकर बंगाल की खाड़ी में गिरती है और करोड़ों लोगों को पानी देती है।",
    "पिछले साल गाँव के स्कूल में नई किताबें आईं, जिससे बच्चों की पढ़ाई में काफी सुधार हुआ।",
    "किसानों ने बताया कि इस बार मानसून समय पर आया, इसलिए धान की फसल अच्छी रही।",
    "शहर के पुस्तकालय में अब शाम आठ बजे तक पढ़ने की सुविधा मिलती है।",
    "वैज्ञानिकों ने जलवायु परिवर्तन पर एक नया अध्ययन प्रकाशित किया है, जिसमें कई चेतावनियाँ दी गई हैं।",
    "रविवार को मैदान में क्रिकेट का मैच हुआ और दर्शकों ने खिलाड़ियों का उत्साह बढ़ाया।",
    "उसने अपनी माँ के लिए स्वादिष्ट खाना बनाया, फिर सब ने साथ बैठकर भोजन किया।",
]


def good_doc():
    return "\n".join(GOOD_LINES)


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
    text = normalize(good_doc() + "\nयह भी पढ़ें काम\nयह भी पढ़ें काम\nयह भी पढ़ें काम")
    assert apply_filters(text, CFG) == (False, "boilerplate")


def test_symbol_heavy_removed():
    text = good_doc() + "\n" + "@#$%^&*" * 200
    assert apply_filters(text, CFG)[1] in {"symbol_ratio", "mean_word_len"}


def test_pii_mask():
    text = "संपर्क करें test.user@example.com या +91 98765 43210 पर"
    masked = pii_mask(text, CFG)
    assert "<EMAIL>" in masked and "<PHONE>" in masked
    assert "example.com" not in masked and "98765" not in masked


def test_stopword_ratio_strips_danda():
    assert stopword_ratio(["यह", "हैं।", "किताब"], CFG["stopwords"]) == 2 / 3


def test_keyword_list_fails_stopwords():
    text = "\n".join(" ".join(f"मोबाइल{i} कीमत ऑफर सस्ता" for i in range(j, j + 5)) + "।" for j in range(0, 60, 5))
    assert apply_filters(text, CFG) == (False, "stopwords")


def test_lines_without_sentence_ends_removed():
    text = "\n".join(line.rstrip("।") + " और भी बातें हैं" for line in GOOD_LINES)
    assert apply_filters(text, CFG) == (False, "line_punct")


def test_repeated_phrase_doc_removed():
    spam = " ".join(["सबसे सस्ता मोबाइल यहाँ है और यह भी"] * 30) + "।"
    assert apply_filters(good_doc() + "\n" + spam, CFG)[1] in {"top_ngram", "dup_ngram"}


def test_ngram_fractions():
    words = "क ख ग घ क ख ग घ".split()
    assert top_ngram_char_frac(words, 2) == 3 * 2 / 8   # "क ख" is 3 chars (with its space), seen twice
    assert dup_ngram_char_frac(words, 4) == 4 / 8
    assert dup_ngram_char_frac(GOOD_SENTENCE.split(), 5) == 0.0


def test_line_cleaner_patterns_and_boilerplate():
    cleaner = LineCleaner(FULL["lines"], boilerplate_hashes=[])
    text = normalize(good_doc() + "\nयह भी पढ़ें: बड़ी खबर\nफोटो: सोशल मीडिया\nऐप डाउनलोड करें")
    cleaned, removed = cleaner.clean(text)
    assert cleaned == normalize(good_doc())
    assert [why for why, _ in removed] == ["pattern"] * 3


def test_line_cleaner_drops_known_boilerplate():
    from data.lines import line_hash
    nav = "होम | देश | विदेश | खेल"
    cleaner = LineCleaner(FULL["lines"], boilerplate_hashes=[line_hash(nav), line_hash(GOOD_LINES[0])])
    cleaned, removed = cleaner.clean(nav + "\n" + good_doc())
    assert removed == [("boilerplate", nav), ("boilerplate", GOOD_LINES[0])]
    assert cleaned == "\n".join(GOOD_LINES[1:])


def test_short_lines_weighed_by_chars():
    links = "\n".join(f"अग्नि-{i}" for i in range(8))
    assert apply_filters(good_doc() + "\n" + links, CFG) == (True, "")
    menu = "\n".join(f"मेनू {i} है।" for i in range(200))
    assert apply_filters(menu, CFG) == (False, "short_lines")
