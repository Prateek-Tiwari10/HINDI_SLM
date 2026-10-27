"""D6 training-logic checks on CPU: data order, cross-shard reads, schedule, param groups."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from model.config import ModelConfig
from model.model import HindiLlama
from train.data import TokenStream, TrainBatches, ValWindows, target_bytes
from train.optim import lr_multiplier, param_groups


def _shards(tmp_path, sizes):
    """uint16 shards holding the token stream 0, 1, 2, ... (mod 65536)."""
    paths, start = [], 0
    for i, n in enumerate(sizes):
        p = tmp_path / f"train_{i:03d}.bin"
        (np.arange(start, start + n) % 65536).astype(np.uint16).tofile(p)
        paths.append(p)
        start += n
    return paths


def test_stream_reads_across_shards(tmp_path):
    s = TokenStream(_shards(tmp_path, [100, 37, 250]))
    assert s.n_tokens == 387
    assert s.read(90, 60).tolist() == list(range(90, 150))
    assert s.read(0, 387).tolist() == list(range(387))


def test_batches_are_deterministic_disjoint_and_shifted(tmp_path):
    paths = _shards(tmp_path, [5000, 3000, 4000])
    seq, B, accum, world = 15, 2, 3, 2
    ranks = [TrainBatches(paths, seq, B, accum, world, r, 1337) for r in range(world)]
    assert ranks[0].n_windows == 12000 // 16
    seen = []
    for step in range(ranks[0].steps_per_epoch):
        for m in range(accum):
            for r in range(world):
                ids = ranks[r].window_ids(step, m)
                seen.extend(ids.tolist())
                x, y = ranks[r].get(step, m, torch.device("cpu"))
                assert x.shape == (B, seq) and x.dtype == torch.int64
                assert torch.equal(x[:, 1:], y[:, :-1])
                assert torch.equal(x[:, 0], torch.as_tensor(ids * 16, dtype=torch.int64))
    assert len(seen) == len(set(seen))                        # every window at most once
    again = TrainBatches(paths, seq, B, accum, world, 1, 1337)
    assert torch.equal(again.get(7, 2, torch.device("cpu"))[0], ranks[1].get(7, 2, torch.device("cpu"))[0])


def test_window_beyond_epoch_is_rejected(tmp_path):
    b = TrainBatches(_shards(tmp_path, [1600]), 15, 2, 1, 1, 0, 0)
    with pytest.raises(AssertionError):
        b.window_ids(b.steps_per_epoch, 0)


def test_val_windows_and_target_bytes(tmp_path):
    p = tmp_path / "val.bin"
    np.array([2, 5, 6, 3, 2, 7, 8, 9, 3, 2], dtype=np.uint16).tofile(p)
    v = ValWindows(p, 4)
    assert v.n_windows == 2
    byte_len = np.zeros(16, dtype=np.int64)
    byte_len[5:10] = [1, 2, 3, 4, 5]                          # bos=2, eos=3 stay at 0 bytes
    # window 0 = [2,5,6,3] -> targets [5,6,3]; window 1 = [2,7,8,9] -> targets [7,8,9]
    assert target_bytes(v, range(2), byte_len) == (6, (1 + 2) + (3 + 4 + 5))


def test_lr_schedule():
    peak_mult = [lr_multiplier(s, 200, 7405, 0.1) for s in range(7405)]
    assert peak_mult[0] == pytest.approx(1 / 200)
    assert peak_mult[199] == pytest.approx(1.0)
    assert peak_mult[200] == pytest.approx(1.0)
    assert peak_mult[-1] == pytest.approx(0.1)
    assert all(a >= b for a, b in zip(peak_mult[200:], peak_mult[201:]))


def test_param_groups_decay_only_matrices():
    m = HindiLlama(replace(ModelConfig(), vocab_size=512, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                           ffn_hidden=320, max_seq_len=64, expected_params=None))
    decay, no_decay = param_groups(m, 0.1)
    assert all(p.dim() == 2 for p in decay["params"]) and all(p.dim() == 1 for p in no_decay["params"])
    assert no_decay["weight_decay"] == 0.0 and decay["weight_decay"] == 0.1
    n = sum(p.numel() for g in (decay, no_decay) for p in g["params"])
    assert n == m.num_params()                                 # tied embedding counted once
    assert len(no_decay["params"]) == 2 * 2 + 1                # 2 norms per layer + final norm
