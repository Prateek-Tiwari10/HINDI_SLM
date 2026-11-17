"""D7 Stage 7d: pre-tokenized pair arrays, length-bucketed batches, the masked SFT loss.

Arrays (from Stage 7c, data/hinglish/stage7c.py): <split>.ids.npy (uint16, every pair's
input_ids concatenated), <split>.offsets.npy (int64, n+1), <split>.prompt_len.npy (uint16,
len(rom_ids) + 2 = the position of the first Devanagari token).

Shift convention as in pretraining (train/data.py): x = input_ids[:-1], y = labels[1:], where
labels are -100 before prompt_len. So y[k] = -100 for k < prompt_len - 1 and the logits at
<dev> predict the first Devanagari token; each pair has len(dev_ids) + 1 targets (incl. eos).

Batches: an epoch is a seeded permutation, cut into chunks of chunk_batches global batches;
each chunk is sorted by length and sliced into global batches, and the batch order is
shuffled. Global step s -> epoch s // steps_per_epoch, batch s % steps_per_epoch: a resumed
run needs only s. Right padding to the smallest bucket that fits; with a causal model real
tokens never attend to padding after them, so no attention mask is needed.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IGNORE = -100
PAD_ID = 0                      # base <pad>; never a target (its labels are IGNORE)


def decode_budget(prompt_len: int, cap: dict) -> int:
    """Greedy-decoding budget for one prompt (Stage 7c meta.json "decode_cap", D10/7d):
    ceil(ratio * factor * rom_chars) + extra, rom_chars = prompt_len - 2 (one id per romanized char),
    ratio = a high quantile of Devanagari tokens (incl. eos) per romanized char over train."""
    return math.ceil(cap["ratio"] * cap["factor"] * max(1, prompt_len - 2)) + cap["extra"]


class PairArrays:
    """Memory-mapped pre-tokenized pairs of one split."""

    def __init__(self, directory: str | Path, split: str) -> None:
        d = Path(directory)
        self.ids = np.load(d / f"{split}.ids.npy", mmap_mode="r")
        self.offsets = np.load(d / f"{split}.offsets.npy")
        self.prompt_len = np.load(d / f"{split}.prompt_len.npy").astype(np.int64)
        assert len(self.offsets) == len(self.prompt_len) + 1

    def __len__(self) -> int:
        return len(self.prompt_len)

    @property
    def lengths(self) -> np.ndarray:
        """Built pair length (input_ids) per pair."""
        return np.diff(self.offsets)

    @property
    def n_targets(self) -> np.ndarray:
        """Supervised positions per pair: len(dev_ids) + 1."""
        return self.lengths - self.prompt_len

    def example(self, i: int) -> tuple[np.ndarray, int]:
        """(input_ids int64, prompt_len) of pair i."""
        return np.asarray(self.ids[self.offsets[i]:self.offsets[i + 1]], dtype=np.int64), int(self.prompt_len[i])

    def prompt(self, i: int) -> list[int]:
        """[<rom>] + rom_ids + [<dev>]: the generation prompt of pair i."""
        ids, p = self.example(i)
        return ids[:p].tolist()


def make_xy(input_ids: np.ndarray, prompt_len: int) -> tuple[np.ndarray, np.ndarray]:
    """x = input_ids[:-1], y = labels[1:] with -100 before the Devanagari part."""
    x = input_ids[:-1]
    y = input_ids[1:].copy()
    y[: prompt_len - 1] = IGNORE
    return x, y


def bucket_length(n: int, buckets: list[int]) -> int:
    """Smallest bucket >= n; error if none (Stage 7c keeps train pairs <= 256 tokens)."""
    for b in buckets:
        if n <= b:
            return b
    raise ValueError(f"sequence of {n} tokens is longer than the largest bucket {buckets[-1]}")


def collate(arrays: PairArrays, indices, buckets: list[int], device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded (x, y) int64 tensors for the given pairs, padded to one bucket length."""
    pairs = [make_xy(*arrays.example(int(i))) for i in indices]
    T = bucket_length(max(len(x) for x, _ in pairs), buckets)
    X = np.full((len(pairs), T), PAD_ID, dtype=np.int64)
    Y = np.full((len(pairs), T), IGNORE, dtype=np.int64)
    for r, (x, y) in enumerate(pairs):
        X[r, : len(x)] = x
        Y[r, : len(y)] = y
    X, Y = torch.from_numpy(X), torch.from_numpy(Y)
    if device is not None:
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
    return X, Y


def masked_loss_sum(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of fp32 cross-entropy over target positions (y != -100)."""
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE,
                           reduction="sum")


class BatchPlan:
    """Deterministic, resumable global batches with length bucketing (see module docstring)."""

    def __init__(self, lengths: np.ndarray, global_batch: int, chunk_batches: int, seed: int) -> None:
        self.lengths = np.asarray(lengths)
        self.global_batch, self.chunk_batches, self.seed = global_batch, chunk_batches, seed
        self.steps_per_epoch = len(self.lengths) // global_batch
        if self.steps_per_epoch == 0:
            raise ValueError(f"{len(self.lengths)} pairs < one global batch of {global_batch}")
        self._epoch, self._batches = None, None

    def epoch_batches(self, epoch: int) -> np.ndarray:
        """(steps_per_epoch, global_batch) pair indices of one epoch."""
        if epoch != self._epoch:
            rng = np.random.default_rng([self.seed, epoch])
            perm = rng.permutation(len(self.lengths))[: self.steps_per_epoch * self.global_batch]
            chunk = self.chunk_batches * self.global_batch
            parts = []
            for start in range(0, len(perm), chunk):
                c = perm[start:start + chunk]
                parts.append(c[np.argsort(self.lengths[c], kind="stable")])
            batches = np.concatenate(parts).reshape(self.steps_per_epoch, self.global_batch)
            self._batches = batches[rng.permutation(self.steps_per_epoch)]
            self._epoch = epoch
        return self._batches

    def global_indices(self, step: int) -> np.ndarray:
        """Pair indices of the global batch at 0-based optimizer step `step`."""
        epoch, k = divmod(step, self.steps_per_epoch)
        return self.epoch_batches(epoch)[k]

    def micro_indices(self, step: int, micro: int, rank: int, world: int, micro_batch: int) -> np.ndarray:
        """This rank's slice for micro-step `micro`: global[(micro*world + rank)*mb : +mb]."""
        start = (micro * world + rank) * micro_batch
        return self.global_indices(step)[start:start + micro_batch]
