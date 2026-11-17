"""Stage 7d gates G1-G3 (+ batching and phase-A checks) on a tiny CPU model. No HF, no GPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from finetune.data import IGNORE, BatchPlan, PairArrays, collate, make_xy, masked_loss_sum  # noqa: E402
from finetune.sft import Hparams, build_optimizer, make_state, restore_state, train_step  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from model.model import HindiLlama  # noqa: E402
from model.surgery import freeze_for_phase_a  # noqa: E402
from train.checkpoint import rng_state  # noqa: E402
from train.optim import build_scheduler  # noqa: E402

VOCAB, BASE = 96, 80                     # tiny "extended" vocab: ids >= 80 play the new rows
BUCKETS = [16, 32, 48, 64]
OCFG = {"betas": [0.9, 0.95], "eps": 1e-8}


def tiny_cfg():
    return ModelConfig(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
                       max_seq_len=64, expected_params=None)


@pytest.fixture
def arrays(tmp_path):
    """60 random pairs: [rom=80] rom ids (81..95) [dev=81]... as prompt, then dev ids (4..79) + eos 3."""
    rng = np.random.default_rng(0)
    ids, offsets, prompt = [], [0], []
    for _ in range(60):
        rom = rng.integers(82, VOCAB, rng.integers(3, 20)).tolist()
        dev = rng.integers(4, BASE, rng.integers(2, 25)).tolist()
        seq = [80] + rom + [81] + dev + [3]
        ids += seq
        offsets.append(offsets[-1] + len(seq))
        prompt.append(len(rom) + 2)
    np.save(tmp_path / "train.ids.npy", np.array(ids, dtype=np.uint16))
    np.save(tmp_path / "train.offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(tmp_path / "train.prompt_len.npy", np.array(prompt, dtype=np.uint16))
    return PairArrays(tmp_path, "train")


def hp(phase="B", accum=2, micro=4, total=10):
    return Hparams(run="t", phase=phase, init="random", peak_lr=3e-3, warmup_steps=2, total_steps=total,
                   global_pairs=accum * micro, micro_batch=micro, accum=accum, world=1,
                   weight_decay=0.0 if phase == "A" else 0.1, seed=0, eval_every=100, save_every=100,
                   max_train_pairs=None, compile=False)


def fresh(phase="B", h=None):
    torch.manual_seed(0)
    model = HindiLlama(tiny_cfg())
    if phase == "A":
        freeze_for_phase_a(model, BASE)
    h = h or hp(phase)
    opt = build_optimizer(model, phase, h, OCFG)
    sched = build_scheduler(opt, h.warmup_steps, h.total_steps, 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, opt, sched, scaler


def run_steps(state, arrays, plan, h, start, end):
    model, opt, sched, scaler = state
    for step in range(start, end):
        train_step(model, model, opt, sched, scaler, arrays, plan, step, h, BUCKETS, 0, torch.device("cpu"), None, 1.0)


# ---------- G1: mask count ----------
def test_g1_mask_count(arrays):
    idx = list(range(12))
    _, y = collate(arrays, idx, BUCKETS)
    expected = sum(len(arrays.example(i)[0]) - arrays.example(i)[1] for i in idx)   # len(dev_ids) + 1 each
    assert int((y != IGNORE).sum()) == expected == int(arrays.n_targets[idx].sum())


def test_shift_convention(arrays):
    ids, p = arrays.example(0)
    x, y = make_xy(ids, p)
    assert x[p - 1] == 81 and y[p - 1] == ids[p]          # logits at <dev> predict the first dev token
    assert (y[: p - 1] == IGNORE).all() and y[-1] == 3     # eos is a target


# ---------- G2: padding invariance ----------
def test_g2_padding_invariance(arrays):
    torch.manual_seed(0)
    model = HindiLlama(tiny_cfg()).eval()
    idx = list(range(8))
    with torch.no_grad():
        X, Y = collate(arrays, idx, BUCKETS)
        logits, _ = model(X)
        for r, i in enumerate(idx):
            x1, y1 = collate(arrays, [i], BUCKETS)
            alone = masked_loss_sum(model(x1)[0], y1)
            in_batch = masked_loss_sum(logits[r:r + 1], Y[r:r + 1])
            assert torch.allclose(alone, in_batch, atol=1e-6, rtol=0), (i, alone.item(), in_batch.item())


# ---------- G3: resume ----------
def test_g3_resume_matches_uninterrupted(arrays):
    h = hp("B")
    plan = BatchPlan(arrays.lengths, h.global_pairs, chunk_batches=2, seed=1337)
    a = fresh("B", h)
    run_steps(a, arrays, plan, h, 0, 10)

    b = fresh("B", h)
    run_steps(b, arrays, plan, h, 0, 5)
    state = make_state(b[0], b[1], b[2], b[3], 5, [rng_state()], {"dev_cer": None}, {}, h)
    c = fresh("B", h)                                     # new objects, as in a new Kaggle session
    step, _ = restore_state(state, *c, rank=0)
    assert step == 5
    run_steps(c, arrays, plan, h, 5, 10)
    for (k, va), (_, vc) in zip(a[0].state_dict().items(), c[0].state_dict().items()):
        assert torch.equal(va, vc), k
    assert a[2].get_last_lr() == c[2].get_last_lr()


# ---------- batching ----------
def test_batch_plan_deterministic_and_complete(arrays):
    plan = BatchPlan(arrays.lengths, 8, chunk_batches=2, seed=7)
    again = BatchPlan(arrays.lengths, 8, chunk_batches=2, seed=7)
    assert plan.steps_per_epoch == 7
    e0 = plan.epoch_batches(0).copy()
    assert np.array_equal(e0, again.epoch_batches(0))
    assert len(set(e0.ravel().tolist())) == 56             # each pair at most once per epoch
    assert not np.array_equal(e0, plan.epoch_batches(1))   # new order next epoch
    assert np.array_equal(plan.global_indices(7), plan.epoch_batches(1)[0])
    micro = [plan.micro_indices(3, m, r, 2, 2) for m in range(2) for r in range(2)]
    assert np.array_equal(np.concatenate(micro), plan.global_indices(3))


def test_bucketing_pads_to_bucket(arrays):
    X, Y = collate(arrays, [0, 1, 2], BUCKETS)
    assert X.shape[1] in BUCKETS and X.shape[1] >= max(len(arrays.example(i)[0]) - 1 for i in range(3))


# ---------- phase A at tiny scale (G5 runs on the real model in the gates kernel) ----------
def test_phase_a_only_new_rows_move(arrays):
    h = hp("A", total=6)
    plan = BatchPlan(arrays.lengths, h.global_pairs, chunk_batches=2, seed=1)
    st = fresh("A", h)
    before = {k: v.clone() for k, v in st[0].state_dict().items()}
    run_steps(st, arrays, plan, h, 0, 6)
    after = st[0].state_dict()
    for k, v in after.items():
        if k in ("tok_emb.weight", "lm_head.weight"):
            assert torch.equal(v[:BASE], before[k][:BASE]), k
            assert not torch.equal(v[BASE:], before[k][BASE:]), k
        else:
            assert torch.equal(v, before[k]), k


def test_decode_budget():
    from finetune.data import decode_budget

    cap = {"ratio": 1.0, "factor": 1.5, "extra": 8}
    assert decode_budget(12, cap) == 23                 # 10 rom chars -> ceil(15) + 8
    assert decode_budget(2, cap) == 10                  # never below one char


def test_dev_eval_plan_clean_selection(arrays):
    from finetune.sft import dev_eval_plan

    ecfg = {"dev_subset": 10, "subset_seed": 0, "clean_selection_above": 0.25}
    records = [{"contamination": "exact" if i % 3 else "clean"} for i in range(len(arrays))]
    plan = dev_eval_plan(arrays, records, ecfg, max_tokens=64)
    assert plan["select"] == "clean"
    clean, cont = plan["parts"]["clean"], plan["parts"]["contaminated"]
    assert len(clean) == 10 and len(cont) == 10
    assert all(records[i]["contamination"] == "clean" for i in clean)
    assert all(records[i]["contamination"] != "clean" for i in cont)
    assert plan["indices"] == sorted(clean + cont)
    assert plan == dev_eval_plan(arrays, records, ecfg, max_tokens=64)      # fixed across runs

    few = [{"contamination": "near" if i < 5 else "clean"} for i in range(len(arrays))]
    plan = dev_eval_plan(arrays, few, ecfg, max_tokens=64)
    assert plan["select"] == "all" and len(plan["indices"]) == 10
    assert sorted(plan["parts"]["clean"] + plan["parts"]["contaminated"]) == plan["indices"]
