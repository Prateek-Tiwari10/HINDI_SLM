"""Stage 7c: sentence units from v2 documents, the candidate filter and the match key."""

import re
import unicodedata

_SENTENCE = re.compile(r"[^।॥?!\n]+[।॥?!]*")
_LATIN = re.compile(r"[A-Za-z]")
_KEY_DROP = re.compile(r"[^ऀ-ॣ०-ॿ0-9A-Za-z]+")
_EDGE_PUNCT = "।॥?!.,;:'\"()[]{}-–—“”‘’…"


def split_sentences(text):
    """Sentences on danda, double danda, ?, ! and newlines; the terminator stays with its sentence."""
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


def sentence_key(text):
    """Match key for exact dedup and leakage: letters and digits only, single spaces."""
    return _KEY_DROP.sub(" ", text).strip()


def _is_devanagari_letter(ch):
    cp = ord(ch)
    return 0x0900 <= cp <= 0x097F and not 0x0964 <= cp <= 0x096F


def devanagari_ratio(text):
    """Devanagari letters / all letters, ignoring spaces, digits, punctuation, symbols and joiners."""
    dev = other = 0
    for ch in text:
        if _is_devanagari_letter(ch):
            dev += 1
        elif ch.isspace() or ch.isdigit() or ch in "‌‍" or unicodedata.category(ch)[0] in "PSZ":
            continue
        else:
            other += 1
    return dev / (dev + other) if dev + other else 0.0


def is_conversational(text, words):
    """True if any whitespace token, stripped of edge punctuation, is a conversational marker word."""
    return any(tok.strip(_EDGE_PUNCT) in words for tok in text.split())


def classify(text, cfg, conv_words):
    """None if the sentence is not a candidate; else True (conversational) / False (general).

    Candidate: min_words..max_words words, no Latin letter, Devanagari ratio >= min_devanagari_ratio.
    """
    n = len(text.split())
    if n < cfg["min_words"] or n > cfg["max_words"] or _LATIN.search(text):
        return None
    if devanagari_ratio(text) < cfg["min_devanagari_ratio"]:
        return None
    return is_conversational(text, conv_words)
