"""Exact dedup (D3): a 64-bit xxhash of the normalized text, checked against a global seen-set.

Sources are processed in priority order (Wikipedia, manual, Sangraha, FineWeb-2). With
first-seen-wins, an exact duplicate therefore keeps its highest-priority copy.
"""

import numpy as np
import xxhash


def text_hash(text):
    """64-bit xxhash of the UTF-8 text, as a Python int."""
    return xxhash.xxh64_intdigest(text.encode("utf-8"))


class ExactDedup:
    """Global set of hashes already kept. Rebuilt from finished shards on a rerun."""

    def __init__(self):
        self.seen = set()

    def load(self, hashes):
        """Add hashes from shards that finished in an earlier session."""
        self.seen.update(int(h) for h in np.asarray(hashes, dtype=np.uint64))

    def is_new(self, h):
        """True the first time a hash is seen; records it."""
        if h in self.seen:
            return False
        self.seen.add(h)
        return True

    def __len__(self):
        return len(self.seen)
