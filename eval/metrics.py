"""Diversity and repetition metrics for generated text, computed on whitespace words.

- repetition rate (per sample): share of 4-gram occurrences that repeat an earlier
  4-gram of the same sample, i.e. 1 - unique/total. 0 = no loops; near 1 = a loop.
- Distinct-n (pooled over a set of samples, Li et al. 2016): unique n-grams / all n-grams.
"""


def ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """All n-grams of a word list, in order."""
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def repetition_rate(words: list[str], n: int = 4) -> float:
    """1 - unique/total n-grams of one sample; 0.0 if it has no n-gram."""
    grams = ngrams(words, n)
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def distinct_n(samples: list[list[str]], n: int) -> float:
    """Unique n-grams / all n-grams over all samples together; 0.0 if there are none."""
    grams = [g for words in samples for g in ngrams(words, n)]
    return len(set(grams)) / len(grams) if grams else 0.0


# ---------- transliteration metrics (D7 Stage 7d) ----------
def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or word lists); every edit costs 1."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def canonical_dev(text: str) -> str:
    """NFC, single spaces, stripped: both sides of CER/WER/EM go through this."""
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", text).split())


def transliteration_scores(hyps: list[str], refs: list[str]) -> dict:
    """Corpus CER (char edits / ref chars, spaces included), WER (word edits / ref words), exact match."""
    char_e = char_n = word_e = word_n = exact = 0
    for h, r in zip(hyps, refs, strict=True):
        h, r = canonical_dev(h), canonical_dev(r)
        char_e += edit_distance(h, r)
        char_n += len(r)
        word_e += edit_distance(h.split(), r.split())
        word_n += len(r.split())
        exact += h == r
    n = len(refs)
    return {"cer": char_e / max(char_n, 1), "wer": word_e / max(word_n, 1), "exact_match": exact / max(n, 1), "n": n}
