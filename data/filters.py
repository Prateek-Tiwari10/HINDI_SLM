"""Document quality filters (D2), cheapest first. Each returns (keep, reason).

Thresholds come from configs/data.yaml `filters:`. Texts are expected to be normalized already.
pii_mask() is not a filter: it rewrites emails and Indian phone numbers in survivors.

Counting note: Devanagari vowel signs (matras) are Unicode *marks* (category M),
not letters, so str.isalnum() is False for them. The regexes below treat the
whole Devanagari letter/mark range as word characters. Otherwise every matra
would count as a "symbol".
"""

import re
import unicodedata

# Devanagari letters and marks; excludes danda (U+0964/5), digits (U+0966-F) and abbreviation sign (U+0970).
_DEVANAGARI_LETTER = re.compile(r"[ऀ-ॣॱ-ॿ]")
# Letters from any other script: word chars minus digits, underscore and the Devanagari block.
_OTHER_LETTER = re.compile(r"[^\W\d_ऀ-ॿ]")
# Symbols: not a word char, not whitespace, and not a Devanagari letter/mark/digit.
_SYMBOL = re.compile(r"[^\w\sऀ-ॣ०-९ॱ-ॿ]")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Indian mobile: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9.
_PHONE = re.compile(r"(?<!\d)(?:(?:\+|00)?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")


def devanagari_ratio(text):
    """Devanagari letters / all letters. 0.0 for text with no letters."""
    devanagari = len(_DEVANAGARI_LETTER.findall(text))
    other = len(_OTHER_LETTER.findall(text))
    total = devanagari + other
    return devanagari / total if total else 0.0


def check_min_words(text, words, cfg):
    keep = len(words) >= cfg["min_words"]
    return keep, "" if keep else "min_words"


def check_devanagari_ratio(text, words, cfg):
    keep = devanagari_ratio(text) >= cfg["devanagari_ratio_min"]
    return keep, "" if keep else "devanagari_ratio"


def check_unique_lines(text, words, cfg):
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return False, "unique_lines"
    keep = len(set(lines)) / len(lines) >= cfg["unique_lines_ratio_min"]
    return keep, "" if keep else "unique_lines"


def check_symbol_ratio(text, words, cfg):
    symbols = len(_SYMBOL.findall(text))
    keep = symbols / max(len(words), 1) <= cfg["symbol_ratio_max"]
    return keep, "" if keep else "symbol_ratio"


def check_mean_word_len(text, words, cfg):
    mean_len = sum(len(w) for w in words) / max(len(words), 1)
    keep = cfg["mean_word_len_min"] <= mean_len <= cfg["mean_word_len_max"]
    return keep, "" if keep else "mean_word_len"


def check_boilerplate(text, words, cfg):
    """Drop if any listed phrase appears >= min_repeats times, or phrases cover most of the doc."""
    bp = cfg["boilerplate"]
    lowered = text.lower()
    covered_chars = 0
    for phrase in bp["phrases"]:
        # Docs are NFC-normalized, so the phrase must be too (nukta letters differ otherwise).
        phrase = unicodedata.normalize("NFC", phrase)
        count = lowered.count(phrase.lower())
        if count >= bp["min_repeats"]:
            return False, "boilerplate"
        covered_chars += count * len(phrase)
    keep = covered_chars / max(len(text), 1) <= bp["max_fraction"]
    return keep, "" if keep else "boilerplate"


# Cheapest first: word count needs only split(); boilerplate scans for several phrases.
FILTERS = [
    ("min_words", check_min_words),
    ("devanagari_ratio", check_devanagari_ratio),
    ("unique_lines", check_unique_lines),
    ("symbol_ratio", check_symbol_ratio),
    ("mean_word_len", check_mean_word_len),
    ("boilerplate", check_boilerplate),
]
FILTER_NAMES = [name for name, _ in FILTERS]


def apply_filters(text, cfg):
    """Run filters in order. Returns (keep, reason); reason names the first failing filter."""
    words = text.split()
    for _, check in FILTERS:
        keep, reason = check(text, words, cfg)
        if not keep:
            return False, reason
    return True, ""


def pii_mask(text, cfg):
    """Replace emails and 10-digit Indian phone numbers with placeholder tokens."""
    text = _EMAIL.sub(cfg["pii"]["email_token"], text)
    return _PHONE.sub(cfg["pii"]["phone_token"], text)
