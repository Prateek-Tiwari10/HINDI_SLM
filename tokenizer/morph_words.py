"""Stage 4d: morphology spot-check words, built by construction so the boundary is known.

word = stem + suffix, e.g. लड़क + ों = लड़कों, boundary = len(stem) (in code points, after NFC).
We keep only constructed words seen >= min_count times in the train sample, so they are real words.
Score = % of words where some token boundary falls exactly on the stem/suffix boundary.
"""

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

from data.config import REPO_ROOT, load_config

# Runs of Devanagari letters/marks: a word with surrounding punctuation stripped.
_DEVANAGARI_WORD = re.compile(r"[ऀ-ॣॱ-ॿ]+")


def candidate_words(stems, suffixes):
    """{word: (stem, suffix, boundary)} for every stem x suffix, NFC-normalized like the corpus."""
    candidates = {}
    for stem in stems:
        stem = unicodedata.normalize("NFC", stem)
        for suffix in suffixes:
            word = unicodedata.normalize("NFC", stem + suffix)
            if word.startswith(stem):  # NFC could merge across the boundary; skip if so
                candidates[word] = (stem, unicodedata.normalize("NFC", suffix), len(stem))
    return candidates


def count_in_sample(sample_path, words):
    """Occurrences of each candidate word in the sample, streaming line by line."""
    counts = Counter()
    wanted = set(words)
    with open(sample_path, encoding="utf-8") as f:
        for line in f:
            for token in _DEVANAGARI_WORD.findall(line):
                if token in wanted:
                    counts[token] += 1
    return counts


def build(cfg, sample_path):
    """Write tokenizer/morph_words.tsv and return rows of (word, stem, suffix, boundary, count)."""
    m = cfg["morphology"]
    candidates = candidate_words(m["stems"], m["suffixes"])
    counts = count_in_sample(sample_path, candidates)
    rows = [
        (word, stem, suffix, boundary, counts[word])
        for word, (stem, suffix, boundary) in candidates.items()
        if counts[word] >= m["min_count_in_sample"]
    ]
    rows.sort(key=lambda r: -r[4])
    rows = rows[: m["target_words"]]
    out_path = REPO_ROOT / m["out_path"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["word", "stem", "suffix", "boundary", "count"])
        writer.writerows(rows)
    print(f"[morph] {len(candidates)} candidates -> {len(rows)} real words", flush=True)
    return rows


def load(cfg):
    with open(REPO_ROOT / cfg["morphology"]["out_path"], encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [(r["word"], int(r["boundary"])) for r in reader]


def token_boundaries(sp, word):
    """Character offsets where token pieces end, found by decoding growing prefixes of the ids.

    Decoding (rather than summing piece lengths) handles the '▁' prefix and byte-fallback pieces.
    """
    ids = sp.encode(word)
    return {len(sp.decode(ids[:k])) for k in range(1, len(ids))}


def morphology_score(sp, words):
    """% of (word, boundary) pairs where a token boundary lands exactly on the boundary."""
    if not words:
        return 0.0
    hits = sum(1 for word, boundary in words if boundary in token_boundaries(sp, word))
    return 100.0 * hits / len(words)


if __name__ == "__main__":
    config = load_config("configs/tokenizer.yaml")
    build(config, Path(config["sample"]["band_dir"]) / "band_0.txt")  # the smallest sample
