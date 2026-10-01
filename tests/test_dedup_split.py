import numpy as np

from data import dedup_minhash as mh
from data.config import load_config
from data.dedup_exact import ExactDedup, text_hash
from data.split import assign_split, split_bounds

CFG = load_config("configs/data.yaml")
PERMS = mh.make_permutations(128, 42)
WORDS = [f"शब्द{i}" for i in range(300)]


def test_exact_dedup_first_seen_wins():
    exact = ExactDedup()
    h = text_hash("एक ही पाठ")
    assert exact.is_new(h)
    assert not exact.is_new(h)


def test_signature_is_deterministic():
    text = " ".join(WORDS)
    assert np.array_equal(mh.signature(text, 5, PERMS), mh.signature(text, 5, PERMS))


def test_near_duplicates_agree_and_different_docs_do_not():
    a = " ".join(WORDS)
    b = " ".join(WORDS[:-3] + ["नया", "अलग", "अंत"])       # ~98% shingle overlap
    c = " ".join(f"अन्य{i}" for i in range(300))
    sigs = np.stack([mh.signature(t, 5, PERMS) for t in (a, b, c)])
    agree = mh.signature_agreement(sigs, np.array([0, 0]), np.array([1, 2]))
    assert agree[0] > 0.9
    assert agree[1] < 0.1


def test_lsh_finds_pair_and_union_find_keeps_preferred_source():
    a = " ".join(WORDS)
    b = " ".join(WORDS[:-2] + ["नया", "अंत"])
    c = " ".join(f"अन्य{i}" for i in range(300))
    sigs = np.stack([mh.signature(t, 5, PERMS) for t in (a, b, c)])
    keys = mh.band_keys(sigs, 16, 8, 42)
    i, j = mh.candidate_pairs(keys)
    assert (0, 1) in set(zip(i.tolist(), j.tolist()))
    # doc 1 is from a preferred source (rank 0), so doc 0 is dropped
    drops, _, _ = mh.choose_drops(3, np.array([0]), np.array([1]),
                                  source_rank=np.array([2, 0, 2]), n_bytes=np.array([100, 50, 10]))
    assert drops.tolist() == [0]


def test_split_bounds_and_determinism():
    assert split_bounds(CFG["split"]) == (9950, 9975)
    ids = [f"wikipedia:{i}" for i in range(20000)]
    parts = [assign_split(x, CFG["split"]) for x in ids]
    assert parts == [assign_split(x, CFG["split"]) for x in ids]
    assert 0.99 < parts.count("train") / len(parts) < 0.999
