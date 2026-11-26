"""D7 Step 6 scoring: normalized CER/WER, bootstrap CIs, word-lookup baseline, ambiguity slices.

All rules are fixed in instructions/D7_step6_SPEC.md (and repeated in step6_RESULTS.md) before the run.
Pure Python + numpy: no model, no HF, so tests/test_step6.py covers every rule.
"""

import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from eval.metrics import canonical_dev, edit_distance
from tokenizer.hinglish import normalize_rom

NUKTA = "़"
DANDA = "।"
BOOT_B = 2000
SEED = 1337
AMBIG_ALWAYS = ("ki", "main")
AMBIG_MIN_COUNT = 100
AMBIG_SECOND_SHARE = 0.10


# ---------------------------------------------------------------- normalization (spec §3)
def raw_dev(text: str) -> str:
    """Raw scoring form: NFC + whitespace collapsed (Stage 7e canonical_dev)."""
    return canonical_dev(text)


def _drop_nukta(text: str) -> str:
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).replace(NUKTA, ""))


def norm_dev(text: str) -> str:
    """Normalized scoring form: N1 NFC, N2 no nukta (also inside precomposed letters), N3 whitespace, N4 no final danda."""
    t = " ".join(_drop_nukta(unicodedata.normalize("NFC", text)).split())
    if t.endswith(DANDA):
        t = t[:-1].rstrip()
    return t


# ---------------------------------------------------------------- corpus scores + bootstrap
def sentence_stats(hyps: list[str], refs: list[str], form=raw_dev) -> np.ndarray:
    """(n, 5) int array per sentence: char edits, ref chars, word edits, ref words, exact."""
    rows = []
    for h, r in zip(hyps, refs, strict=True):
        h, r = form(h), form(r)
        rows.append((edit_distance(h, r), len(r), edit_distance(h.split(), r.split()), len(r.split()), int(h == r)))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 5)


def corpus(stats: np.ndarray) -> dict:
    s = stats.sum(axis=0)
    n = len(stats)
    return {"n": int(n), "cer": float(s[0] / max(s[1], 1)), "wer": float(s[2] / max(s[3], 1)),
            "exact_match": float(s[4] / max(n, 1))}


def _rates(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return sums[:, 0] / np.maximum(sums[:, 1], 1), sums[:, 2] / np.maximum(sums[:, 3], 1)


def bootstrap(stats: np.ndarray, other: np.ndarray | None = None, b: int = BOOT_B, seed: int = SEED) -> dict:
    """Percentile 95% CIs of corpus CER / WER over sentence resamples; with `other`, of (stats - other), paired."""
    n = len(stats)
    rng = np.random.default_rng(seed)
    cer, wer = [], []
    for start in range(0, b, 250):
        idx = rng.integers(0, n, size=(min(250, b - start), n))
        c, w = _rates(stats[idx].sum(axis=1).astype(np.float64))
        if other is not None:
            c2, w2 = _rates(other[idx].sum(axis=1).astype(np.float64))
            c, w = c - c2, w - w2
        cer.append(c)
        wer.append(w)
    cer, wer = np.concatenate(cer), np.concatenate(wer)
    q = lambda x: [round(float(np.percentile(x, 2.5)), 5), round(float(np.percentile(x, 97.5)), 5)]  # noqa: E731
    return {"cer_ci95": q(cer), "wer_ci95": q(wer), "resamples": b, "seed": seed}


def summarize(stats: np.ndarray, ci: bool = True) -> dict:
    """Corpus CER / WER / EM (+ bootstrap CIs) of precomputed sentence_stats rows."""
    out = corpus(stats)
    if ci and len(stats):
        out.update(bootstrap(stats))
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in out.items()}


def score(hyps: list[str], refs: list[str], form=raw_dev, ci: bool = False) -> dict:
    return summarize(sentence_stats(hyps, refs, form), ci)


# ---------------------------------------------------------------- words (spec §4, §5)
def rom_words(text: str) -> list[str]:
    """Latin words for alignment: normalize_rom, split, [^a-z0-9] removed, empties dropped."""
    return [w for w in (re.sub("[^a-z0-9]", "", t) for t in normalize_rom(text).split()) if w]


def _is_edge(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PSZC"


def edge_strip(word: str) -> str:
    """Devanagari word with leading/trailing punctuation, symbols, separators and format chars removed."""
    i, j = 0, len(word)
    while i < j and _is_edge(word[i]):
        i += 1
    while j > i and _is_edge(word[j - 1]):
        j -= 1
    return word[i:j]


def word_key(word: str) -> str:
    """Comparison key of a Devanagari word: letters / marks / numbers only, no nukta, NFC."""
    t = unicodedata.normalize("NFD", word).replace(NUKTA, "")
    return unicodedata.normalize("NFC", "".join(c for c in t if unicodedata.category(c)[0] in "LMN"))


def dev_words(text: str) -> list[str]:
    """Edge-stripped Devanagari words with a non-empty key (the units aligned 1:1 with rom_words)."""
    return [w for w in (edge_strip(t) for t in unicodedata.normalize("NFC", text).split()) if word_key(w)]


# ---------------------------------------------------------------- word-lookup baseline (spec §4)
def build_lexicon(pairs) -> tuple[dict[str, Counter], dict]:
    """{roman word: Counter(devanagari spelling)} from (rom, dev) pairs whose word counts align."""
    lex: dict[str, Counter] = defaultdict(Counter)
    seen = aligned = 0
    for rom, dev in pairs:
        seen += 1
        rw, dw = rom_words(rom), dev_words(dev)
        if len(rw) != len(dw):
            continue
        aligned += 1
        for a, b in zip(rw, dw):
            lex[a][b] += 1
    return dict(lex), {"pairs": seen, "aligned_pairs": aligned, "roman_words": len(lex),
                       "aligned_word_occurrences": int(sum(sum(c.values()) for c in lex.values()))}


def lookup_table(lex: dict[str, Counter]) -> dict[str, str]:
    """Most frequent spelling per roman word; ties go to the spelling seen first (Counter keeps insertion order)."""
    return {w: c.most_common(1)[0][0] for w, c in lex.items()}


_TOKEN = re.compile(r"^([^a-z0-9]*)(.*?)([^a-z0-9]*)$")


def baseline(text: str, table: dict[str, str]) -> str:
    """Word-by-word lookup; unknown words pass through (Latin); edge punctuation kept; a final '.' becomes '।'."""
    out = []
    for tok in normalize_rom(text).split():
        lead, core, trail = _TOKEN.match(tok).groups()
        key = re.sub("[^a-z0-9]", "", core)
        out.append(lead + (table.get(key, core) if key else core) + trail)
    joined = " ".join(out)
    return joined[:-1] + DANDA if joined.endswith(".") else joined


# ---------------------------------------------------------------- ambiguity slices (spec §5)
def ambiguous_words(lex: dict[str, Counter], min_count: int = AMBIG_MIN_COUNT, share: float = AMBIG_SECOND_SHARE,
                    always=AMBIG_ALWAYS) -> dict[str, dict]:
    """Roman words whose 2nd most common (normalized) spelling has >= share of >= min_count uses, plus `always`."""
    out = {}
    for w, c in lex.items():
        k = Counter()
        for dev, n in c.items():
            k[word_key(dev)] += n
        tot = sum(k.values())
        top = k.most_common(3)
        second = top[1][1] / tot if len(top) > 1 else 0.0
        if w in always or (tot >= min_count and second >= share):
            out[w] = {"n_train": int(tot), "second_share": round(second, 4),
                      "top": [[d, round(n / tot, 4)] for d, n in top]}
    for w in always:
        out.setdefault(w, {"n_train": 0, "second_share": 0.0, "top": []})
    return out


def matched_ref_positions(ref: list[str], hyp: list[str]) -> set[int]:
    """Indices of ref words paired with an identical hyp word by a word-level Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, hits = n, m, set()
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            hits.add(i - 1)
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return hits


def ambiguity_hits(rom: str, ref: str, hyp: str, ambiguous) -> list[tuple[str, bool]] | None:
    """[(roman word, correct)] for each ambiguous word in an aligned sentence; None if rom/ref do not align."""
    rw = rom_words(rom)
    refk = [word_key(w) for w in dev_words(ref)]
    if len(rw) != len(refk):
        return None
    hypk = [word_key(w) for w in dev_words(hyp)]
    hits = matched_ref_positions(refk, hypk)
    return [(w, i in hits) for i, w in enumerate(rw) if w in ambiguous]


def slice_report(hits_by_sentence: list[list[tuple[str, bool]] | None]) -> dict:
    """Pooled accuracy + per-word counts from ambiguity_hits outputs (None = unaligned sentence)."""
    per = defaultdict(lambda: [0, 0])
    aligned = 0
    for hits in hits_by_sentence:
        if hits is None:
            continue
        aligned += 1
        for w, ok in hits:
            per[w][0] += 1
            per[w][1] += ok
    n = sum(v[0] for v in per.values())
    return {"aligned_sentences": aligned, "sentences": len(hits_by_sentence), "occurrences": n,
            "accuracy": round(sum(v[1] for v in per.values()) / n, 5) if n else None,
            "per_word": {w: {"n": v[0], "correct": v[1]} for w, v in sorted(per.items(), key=lambda kv: -kv[1][0])}}
