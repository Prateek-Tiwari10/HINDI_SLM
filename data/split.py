"""Document-level train/val/test split (D3), made before tokenization.

bucket = xxh64(doc id) mod 10,000. With the 99.5 / 0.25 / 0.25 split:
buckets [0, 9950) are train, [9950, 9975) are val, [9975, 10000) are test.
The split depends only on the id, so it is the same on every run, machine and
shard order. Near-dedup ran globally before the split, so val/test cannot hold a
near-duplicate of a train doc.
"""

import xxhash


def bucket(doc_id, modulus):
    return xxhash.xxh64_intdigest(doc_id.encode("utf-8")) % modulus


def split_bounds(cfg):
    """(train_end, val_end) bucket boundaries from the split config."""
    modulus = cfg["modulus"]
    train_end = round(cfg["train"] * modulus)
    val_end = train_end + round(cfg["val"] * modulus)
    assert val_end + round(cfg["test"] * modulus) == modulus, "split fractions must sum to 1"
    return train_end, val_end


def assign_split(doc_id, cfg):
    """'train', 'val' or 'test' for one document id."""
    train_end, val_end = split_bounds(cfg)
    b = bucket(doc_id, cfg["modulus"])
    if b < train_end:
        return "train"
    if b < val_end:
        return "val"
    return "test"
