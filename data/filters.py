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
from collections import Counter
from functools import lru_cache

# Devanagari letters and marks; excludes danda (U+0964/5), digits (U+0966-F) and abbreviation sign (U+0970).
_DEVANAGARI_LETTER = re.compile(r"[ऀ-ॣॱ-ॿ]")
# Letters from any other script: word chars minus digits, underscore and the Devanagari block.
_OTHER_LETTER = re.compile(r"[^\W\d_ऀ-ॿ]")
# Symbols: not a word char, not whitespace, and not a Devanagari letter/mark/digit.
_SYMBOL = re.compile(r"[^\w\sऀ-ॣ०-९ॱ-ॿ]")

# Punctuation stripped from a word before the stopword lookup ("हैं।" -> "हैं").
_WORD_EDGE_PUNCT = "।॥.,!?;:\"'()[]{}-–—“”‘’"
_BULLET = re.compile(r"^\s*[•●▪■◦\-*–—►✓✔]")
_ELLIPSIS = re.compile(r"(\.\.\.|…)\s*$")

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


def check_max_words(text, words, cfg):
    keep = len(words) <= cfg["max_words"]
    return keep, "" if keep else "max_words"


@lru_cache(maxsize=4)
def _stopword_set(stopwords):
    return frozenset(unicodedata.normalize("NFC", w) for w in stopwords)


def stopword_ratio(words, stopwords):
    """Share of words that are Hindi function words. Prose is ~30-40%; lists and keyword spam are far lower."""
    stop = _stopword_set(tuple(stopwords))
    hits = sum(1 for w in words if w.strip(_WORD_EDGE_PUNCT) in stop)
    return hits / max(len(words), 1)


def check_stopwords(text, words, cfg):
    keep = stopword_ratio(words, cfg["stopwords"]) >= cfg["stopword_ratio_min"]
    return keep, "" if keep else "stopwords"


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


def _nonempty_lines(text):
    return [line.strip() for line in text.split("\n") if line.strip()]


def dup_line_char_frac(text):
    """Chars in lines that already appeared earlier in the doc / chars in all lines (Gopher)."""
    seen, dup_chars, total = set(), 0, 0
    for line in _nonempty_lines(text):
        total += len(line)
        if line in seen:
            dup_chars += len(line)
        seen.add(line)
    return dup_chars / max(total, 1)


def check_dup_line_chars(text, words, cfg):
    keep = dup_line_char_frac(text) <= cfg["dup_line_char_frac_max"]
    return keep, "" if keep else "dup_line_chars"


def check_line_structure(text, words, cfg):
    """FineWeb/Gopher line rules: enough lines end like sentences; not mostly short, bulleted or truncated lines."""
    lines = _nonempty_lines(text)
    if not lines:
        return False, "line_punct"
    n = len(lines)
    if sum(1 for line in lines if line[-1] in cfg["terminal_punct"]) / n < cfg["line_punct_ratio_min"]:
        return False, "line_punct"
    # Short lines are weighed by characters, not counted: a Wikipedia stub of one paragraph
    # plus a list of short "see also" links is fine; a page of nothing but menu items is not.
    short_chars = sum(len(line) for line in lines if len(line) < cfg["short_line_chars"])
    if short_chars / sum(len(line) for line in lines) > cfg["short_line_char_frac_max"]:
        return False, "short_lines"
    if sum(1 for line in lines if _BULLET.match(line)) / n > cfg["bullet_line_ratio_max"]:
        return False, "bullet_lines"
    if sum(1 for line in lines if _ELLIPSIS.search(line)) / n > cfg["ellipsis_line_ratio_max"]:
        return False, "ellipsis_lines"
    return True, ""


def top_ngram_char_frac(words, n):
    """Chars covered by the most frequent word n-gram / chars in all words (Gopher)."""
    if len(words) < n:
        return 0.0
    grams = Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    return len(gram) * count / max(sum(len(w) for w in words), 1)


def dup_ngram_char_frac(words, n):
    """Chars in word n-grams that repeat an earlier n-gram, counted without overlap / all word chars (Gopher)."""
    seen, dup_chars, i = set(), 0, 0
    while i <= len(words) - n:
        gram = " ".join(words[i : i + n])
        if gram in seen:
            dup_chars += sum(len(w) for w in words[i : i + n])
            i += n
        else:
            seen.add(gram)
            i += 1
    return dup_chars / max(sum(len(w) for w in words), 1)


def check_repetition(text, words, cfg):
    for n, limit in cfg["top_ngram_char_frac_max"].items():
        if top_ngram_char_frac(words, int(n)) > limit:
            return False, "top_ngram"
    for n, limit in cfg["dup_ngram_char_frac_max"].items():
        if dup_ngram_char_frac(words, int(n)) > limit:
            return False, "dup_ngram"
    return True, ""


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


# Cheapest first: word count needs only split(); the n-gram repetition checks are the most expensive.
FILTERS = [
    ("min_words", check_min_words),
    ("max_words", check_max_words),
    ("devanagari_ratio", check_devanagari_ratio),
    ("stopwords", check_stopwords),
    ("unique_lines", check_unique_lines),
    ("dup_line_chars", check_dup_line_chars),
    ("line_structure", check_line_structure),
    ("symbol_ratio", check_symbol_ratio),
    ("mean_word_len", check_mean_word_len),
    ("boilerplate", check_boilerplate),
    ("repetition", check_repetition),
]
# Reasons a doc can be removed for: one per filter, except that line_structure and
# repetition report which of their rules failed.
FILTER_NAMES = ["min_words", "max_words", "devanagari_ratio", "stopwords", "unique_lines", "dup_line_chars",
                "line_punct", "short_lines", "bullet_lines", "ellipsis_lines", "symbol_ratio",
                "mean_word_len", "boilerplate", "top_ngram", "dup_ngram"]


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
