"""Near-dedup (D3): MinHash signatures, banding LSH, verification, union-find.

Signature: word 5-gram shingles, each hashed to 32 bits; 128 hash functions
h_k(x) = high 32 bits of (a_k * x + b_k) mod 2^64 (multiply-shift hashing, a_k odd).
Signature[k] = min over shingles of h_k. Two docs agree at position k with
probability = Jaccard similarity of their shingle sets.

LSH: 16 bands x 8 rows. Two docs become candidates if all 8 values of some band match.
Candidates are then verified: signature agreement >= 0.80 counts as a duplicate.
"""

import numpy as np
import xxhash

SHINGLE_CHUNK = 8_192  # shingles processed per numpy step, bounds memory on very long docs


def make_permutations(num_perm, seed):
    """Seeded (a, b) parameters for num_perm hash functions. a is odd, as multiply-shift needs."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
    return a, b


def shingles(text, n):
    """Set of word n-grams. A doc shorter than n words is a single shingle."""
    words = text.split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def signature(text, n, perms):
    """uint32 MinHash signature of shape [num_perm]."""
    a, b = perms
    hashes = np.fromiter(
        (xxhash.xxh32_intdigest(s.encode("utf-8")) for s in shingles(text, n)),
        dtype=np.uint64,
    )
    result = np.full(len(a), np.iinfo(np.uint32).max, dtype=np.uint64)
    for start in range(0, len(hashes), SHINGLE_CHUNK):
        chunk = hashes[start : start + SHINGLE_CHUNK]
        # uint64 arithmetic wraps mod 2^64; the top 32 bits are the hash value.
        permuted = (a[:, None] * chunk[None, :] + b[:, None]) >> np.uint64(32)
        result = np.minimum(result, permuted.min(axis=1))
    return result.astype(np.uint32)


def band_keys(signatures, bands, rows, seed):
    """Collapse each band's `rows` uint32 values into one uint64 key -> shape [N, bands].

    Key = sum of value * random odd multiplier (wrapping mod 2^64). Different bands
    can occasionally share a key by chance. That only adds a candidate, which the
    verify step rejects.
    """
    assert signatures.shape[1] == bands * rows
    rng = np.random.default_rng(seed)
    mult = rng.integers(1, 2**63, size=rows, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    keys = np.empty((signatures.shape[0], bands), dtype=np.uint64)
    for band in range(bands):
        block = signatures[:, band * rows : (band + 1) * rows].astype(np.uint64)
        keys[:, band] = (block * mult[None, :]).sum(axis=1, dtype=np.uint64)
    return keys


def candidate_pairs(keys):
    """(i, j) index pairs sharing a band key.

    Within a group of equal keys, every member is paired with the group's first
    member (a star), not with every other member. After union-find this gives the
    same clusters with O(group size) pairs instead of O(size^2).
    """
    all_i, all_j = [], []
    for band in range(keys.shape[1]):
        order = np.argsort(keys[:, band], kind="stable")
        sorted_keys = keys[order, band]
        same_as_prev = np.empty(len(order), dtype=bool)
        same_as_prev[0] = False
        same_as_prev[1:] = sorted_keys[1:] == sorted_keys[:-1]
        group_start = np.maximum.accumulate(np.where(~same_as_prev, np.arange(len(order)), 0))
        members = np.nonzero(same_as_prev)[0]
        all_i.append(order[group_start[members]])
        all_j.append(order[members])
    i = np.concatenate(all_i)
    j = np.concatenate(all_j)
    pairs = np.unique(np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def signature_agreement(signatures, i, j, chunk=1_000_000):
    """Fraction of equal positions for each pair (estimated Jaccard)."""
    out = np.empty(len(i), dtype=np.float32)
    for start in range(0, len(i), chunk):
        a = signatures[i[start : start + chunk]]
        b = signatures[j[start : start + chunk]]
        out[start : start + chunk] = (a == b).mean(axis=1)
    return out


class UnionFind:
    """Disjoint sets with path halving and union by size.

    Plain lists, not numpy: this runs element by element in Python, where list
    indexing is several times faster than numpy scalar indexing.
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]


def choose_drops(n_docs, dup_i, dup_j, source_rank, n_bytes):
    """Indices to drop: keep one doc per cluster.

    The kept doc has the best source_rank (lower = preferred). Ties go to the
    longest doc (most bytes), then the lowest index.
    """
    uf = UnionFind(n_docs)
    for x, y in zip(dup_i.tolist(), dup_j.tolist()):
        uf.union(x, y)
    members = np.unique(np.concatenate([dup_i, dup_j]))
    roots = np.array([uf.find(int(m)) for m in members], dtype=np.int64)
    # Sort members by (root, rank, -bytes, index); the first of each root is the keeper.
    order = np.lexsort((members, -n_bytes[members].astype(np.int64), source_rank[members], roots))
    sorted_roots = roots[order]
    is_keeper = np.empty(len(order), dtype=bool)
    is_keeper[0] = True
    is_keeper[1:] = sorted_roots[1:] != sorted_roots[:-1]
    return np.sort(members[order][~is_keeper]), roots, members
