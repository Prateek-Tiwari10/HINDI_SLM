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
