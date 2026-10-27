"""Stage 6 text metrics on hand-checked examples."""

import pytest

from eval.metrics import distinct_n, ngrams, repetition_rate


def test_ngrams():
    assert ngrams(list("abcd"), 2) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert ngrams(list("ab"), 4) == []


def test_repetition_rate():
    assert repetition_rate("क ख ग घ ङ च".split()) == 0.0
    # "a b c d" repeated 3x: 9 four-grams, 4 unique
    assert repetition_rate(("a b c d " * 3).split()) == pytest.approx(1 - 4 / 9)
    assert repetition_rate(["a", "b"]) == 0.0


def test_distinct_n_pools_samples():
    s = [["a", "b", "a"], ["a", "c"]]
    assert distinct_n(s, 1) == pytest.approx(3 / 5)          # a b c out of 5 words
    assert distinct_n(s, 2) == pytest.approx(3 / 3)          # ab, ba, ac
    assert distinct_n([[]], 2) == 0.0
