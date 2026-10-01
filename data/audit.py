"""Filter audit: keep a uniform random sample of removed documents per (stage, source).

Reservoir sampling keeps exactly k items from a stream of unknown length.
Every removed doc has the same chance to be kept, and memory stays at k.
"""

import random
from collections import defaultdict
from pathlib import Path


class Reservoir:
    """Uniform sample of k items from a stream (Algorithm R)."""

    def __init__(self, k, rng):
        self.k = k
        self.rng = rng
        self.seen = 0
        self.items = []

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
            return
        slot = self.rng.randrange(self.seen)
        if slot < self.k:
            self.items[slot] = item


class FilterAudit:
    """One reservoir per (stage, source) of removed docs."""

    def __init__(self, k, chars_shown, seed):
        self.k = k
        self.chars_shown = chars_shown
        self.rng = random.Random(seed)
        self.reservoirs = defaultdict(lambda: Reservoir(self.k, self.rng))

    def record(self, stage, source, doc_id, reason, text):
        entry = {"id": doc_id, "reason": reason, "text": text[: self.chars_shown]}
        self.reservoirs[(stage, source)].add(entry)

    def write(self, out_dir):
        """Write results/filter_audit/<stage>/<source>.md files."""
        for (stage, source), reservoir in sorted(self.reservoirs.items()):
            path = Path(out_dir) / stage / f"{source}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"# Removed by `{stage}` — {source}",
                "",
                f"{len(reservoir.items)} random docs of {reservoir.seen} removed.",
                "",
            ]
            for entry in reservoir.items:
                lines += [
                    f"## {entry['id']}",
                    f"reason: `{entry['reason']}`",
                    "",
                    "```text",
                    entry["text"],
                    "```",
                    "",
                ]
            path.write_text("\n".join(lines), encoding="utf-8")
