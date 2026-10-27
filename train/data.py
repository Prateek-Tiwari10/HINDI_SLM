"""Memmap loader over the tokenized shards with a deterministic, resumable batch order.

The 4 train shards are one token stream, cut into non-overlapping windows of
seq_len + 1 = 1025 tokens (input 1024 + the same window shifted by one as target).
Windows may cross document boundaries; bos/eos mark them. Document-level attention
masking is out of scope.

Order: perm = default_rng(1337).permutation(n_windows). At global step s, micro-step m,
rank r, the micro-batch is perm[((s*accum + m)*world + r)*B : +B]. A resumed run needs
only s to get exactly the same batches, and every window is used at most once.
"""

import json
from pathlib import Path

import numpy as np
import torch

from data import hub


def download_split(cfg: dict) -> tuple[dict, dict[str, list[Path]]]:
    """Download meta.json and every .bin shard to cfg['data']['local_dir'].

    Returns (meta, {"train": [paths...], "val": [path], "test": [path]}).
    Files already present with the right size are not downloaded again.
    """
    d = cfg["data"]
    local = Path(d["local_dir"])
    local.mkdir(parents=True, exist_ok=True)
    meta_path = hub.download(d["hf_repo"], f"{d['prefix']}meta.json", local)
    meta = json.loads(Path(meta_path).read_text())
    paths = {}
    for split, info in meta["splits"].items():
        paths[split] = []
        for name, n_tokens in info["files"]:
            target = local / d["prefix"] / name
            if not (target.exists() and target.stat().st_size == 2 * n_tokens):
                hub.download(d["hf_repo"], f"{d['prefix']}{name}", local)
            assert target.stat().st_size == 2 * n_tokens, f"{name}: size mismatch"
            paths[split].append(target)
    return meta, paths


class TokenStream:
    """Several uint16 memmaps read as one contiguous array."""

    def __init__(self, paths: list[Path]) -> None:
        self.maps = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.starts = np.cumsum([0] + [len(m) for m in self.maps])
        self.n_tokens = int(self.starts[-1])

    def read(self, start: int, length: int) -> np.ndarray:
        """Tokens [start, start+length), possibly spanning two shards."""
        out = np.empty(length, dtype=np.uint16)
        pos, filled = start, 0
        while filled < length:
            i = int(np.searchsorted(self.starts, pos, side="right")) - 1
            off = pos - self.starts[i]
            take = min(length - filled, len(self.maps[i]) - off)
            out[filled:filled + take] = self.maps[i][off:off + take]
            filled += take
            pos += take
        return out


class TrainBatches:
    """Deterministic micro-batches of (input, target) windows for DDP training."""

    def __init__(self, paths: list[Path], seq_len: int, micro_batch: int, grad_accum: int,
                 world: int, rank: int, seed: int) -> None:
        self.stream = TokenStream(paths)
        self.window = seq_len + 1
        self.n_windows = self.stream.n_tokens // self.window
        self.B, self.accum, self.world, self.rank = micro_batch, grad_accum, world, rank
        self.perm = np.random.default_rng(seed).permutation(self.n_windows)

    @property
    def global_batch_windows(self) -> int:
        """Windows per optimizer step across all ranks and micro-steps."""
        return self.world * self.B * self.accum

    @property
    def steps_per_epoch(self) -> int:
        """floor(n_windows / global_batch_windows): one pass over the data."""
        return self.n_windows // self.global_batch_windows

    def window_ids(self, step: int, micro: int) -> np.ndarray:
        """Window indices of this rank's micro-batch at (step, micro)."""
        start = ((step * self.accum + micro) * self.world + self.rank) * self.B
        assert start + self.B <= self.n_windows, "step beyond one epoch"
        return self.perm[start:start + self.B]

    def get(self, step: int, micro: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """(x, y) int64 tensors (B, seq_len) on device; y is x shifted by one."""
        ids = self.window_ids(step, micro)
        arr = np.stack([self.stream.read(int(i) * self.window, self.window) for i in ids])
        t = torch.from_numpy(arr.astype(np.int32))
        if device.type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        t = t.long()
        return t[:, :-1], t[:, 1:]


class ValWindows:
    """All non-overlapping seq_len-token windows of val.bin (or test.bin).

    Each window predicts its own tokens 1..seq_len-1 from 0..seq_len-2, i.e.
    seq_len - 1 scored targets per window.
    """

    def __init__(self, path: Path, seq_len: int) -> None:
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n_windows = len(self.tokens) // seq_len

    def window(self, i: int) -> np.ndarray:
        """Tokens of window i."""
        return np.asarray(self.tokens[i * self.seq_len:(i + 1) * self.seq_len])

    def batch(self, ids: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """(x, y) for the given window ids."""
        t = torch.from_numpy(np.stack([self.window(i) for i in ids]).astype(np.int64)).to(device)
        return t[:, :-1], t[:, 1:]


def token_byte_lengths(sp) -> np.ndarray:
    """UTF-8 bytes each token id contributes to decoded text.

    Normal pieces: their text with '▁' as a space. Byte-fallback pieces (<0xNN>): 1 byte.
    Control and unknown tokens (pad/unk/bos/eos, chat and reserved tokens): 0 bytes.
    Summing this table over tokens is exact and additive, unlike decoding window
    fragments (which drops a leading space and can split multi-byte characters).
    """
    out = np.zeros(sp.get_piece_size(), dtype=np.int64)
    for i in range(sp.get_piece_size()):
        if sp.is_byte(i):
            out[i] = 1
        elif not (sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i)):
            out[i] = len(sp.id_to_piece(i).replace("▁", " ").encode("utf-8"))
    return out


def target_bytes(windows: ValWindows, ids: range, byte_len: np.ndarray) -> tuple[int, int]:
    """(n_target_tokens, n_bytes) of the scored targets of these windows, bos/eos at 0 bytes."""
    n_tok = n_bytes = 0
    for i in ids:
        targets = windows.window(i)[1:]
        n_tok += len(targets)
        n_bytes += int(byte_len[targets].sum())
    return n_tok, n_bytes
