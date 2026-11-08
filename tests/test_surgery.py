"""Stage 7b (D9) checks for model surgery on a tiny config. CPU, fp32.

Tiny model: 2 layers, d_model 64, 4 query / 2 K/V heads, base vocab 256 extended to 320.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from model.attention import KVCache
from model.config import ModelConfig
from model.model import HindiLlama
from model.surgery import (finetune_meta, freeze_for_phase_a, load_finetuned, mask_from_ids,
                           phase_a_param_groups, resize_embeddings, save_finetuned)

BASE_V, NEW_V = 256, 320
TINY = replace(ModelConfig(), vocab_size=BASE_V, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
               ffn_hidden=160, max_seq_len=128, expected_params=None)
TOK = SimpleNamespace(vocab_size=NEW_V, base_vocab_size=BASE_V, base_sha256="a" * 64, version="hinglish-v1")


def tiny(seed: int = 0) -> HindiLlama:
    torch.manual_seed(seed)
    return HindiLlama(TINY).eval()


def resized(init: str = "mean", seed: int = 0) -> tuple[HindiLlama, HindiLlama]:
    """(base model, a resized deep copy of it)."""
    base = tiny(seed)
    big = tiny(seed)
    big.load_state_dict(base.state_dict())
    return base, resize_embeddings(big, NEW_V, init=init, seed=seed)


def other_params(model: HindiLlama) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if n != "tok_emb.weight"}


def batch(step: int, B: int = 4, T: int = 32) -> torch.Tensor:
    """Deterministic batch for one step; tokens from the whole extended vocab."""
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randint(0, NEW_V, (B, T + 1), generator=g)


def phase_a_step(model: HindiLlama, opt: torch.optim.Optimizer, step: int) -> float:
    w = batch(step)
    _, loss = model(w[:, :-1], w[:, 1:])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


# 1. old rows copied bit for bit; 2. head is the same Parameter as the embedding
@pytest.mark.parametrize("init", ["mean", "random"])
def test_resize_copies_old_rows_and_reties(init):
    base, big = resized(init)
    w = big.tok_emb.weight
    assert w.shape == (NEW_V, TINY.d_model)
    assert torch.equal(w[:BASE_V], base.tok_emb.weight)
    assert big.lm_head.weight is w
    assert big.cfg.vocab_size == NEW_V
    assert big.num_params() == base.num_params() + (NEW_V - BASE_V) * TINY.d_model


def test_tie_survives_save_load(tmp_path):
    _, big = resized()
    path = tmp_path / "ft.pt"
    save_finetuned(path, big, finetune_meta(TOK, "tiny", "A"))
    loaded, ckpt = load_finetuned(path, TOK)
    assert loaded.lm_head.weight is loaded.tok_emb.weight
    assert torch.equal(loaded.tok_emb.weight, big.tok_emb.weight)
    assert ckpt["finetune"]["phase"] == "A" and ckpt["finetune"]["vocab_size"] == NEW_V


# 3. new rows finite, with the expected size
def test_new_rows_finite_and_sized():
    base, big = resized("mean")
    old, new = base.tok_emb.weight, big.tok_emb.weight[BASE_V:]
    assert torch.isfinite(new).all()
    # mean init: row ≈ μ + N(0, (0.1σ)²), so E‖row‖² = ‖μ‖² + 0.01·Σσ². For this untrained
    # tiny model μ ≈ 0, so the rows are much shorter than old rows (expected, see RESULTS).
    expected = (old.mean(0).pow(2).sum() + 0.01 * old.std(0).pow(2).sum()).sqrt()
    ratio = new.norm(dim=1).mean() / expected
    assert 0.5 < ratio < 2.0
    # random init: N(0, 0.02²) rows are the same size as the (untrained) old rows
    _, rnd = resized("random")
    ratio_rnd = rnd.tok_emb.weight[BASE_V:].norm(dim=1).mean() / old.norm(dim=1).mean()
    assert 0.5 < ratio_rnd < 2.0


# 4. base-only inputs: the network up to lm_head is unchanged, so base-column logits are too
def final_hidden(model: HindiLlama, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(hidden state after the final norm, logits) for one forward pass."""
    seen = {}
    handle = model.norm.register_forward_hook(lambda _m, _i, out: seen.setdefault("h", out))
    with torch.no_grad():
        logits, _ = model(idx)
    handle.remove()
    return seen["h"], logits


@pytest.mark.parametrize("init", ["mean", "random"])
def test_base_logits_unchanged(init):
    base, big = resized(init)
    idx = torch.randint(0, BASE_V, (2, 40), generator=torch.Generator().manual_seed(3))
    h_a, a = final_hidden(base, idx)
    h_b, b = final_hidden(big, idx)
    assert b.shape[-1] == NEW_V
    assert torch.equal(h_a, h_b)                                   # bit-identical up to lm_head
    # same matmul shape -> bit-identical base logits
    assert torch.equal(a, torch.nn.functional.linear(h_b, big.lm_head.weight[:BASE_V]))
    # the wider (320-column) matmul may pick a different BLAS kernel: last-bit differences only
    assert torch.allclose(a, b[..., :BASE_V], atol=1e-6, rtol=0), (a - b[..., :BASE_V]).abs().max()


# 5. phase A: only new rows move
def test_phase_a_only_new_rows_train():
    _, model = resized()
    model.train()
    before_emb = model.tok_emb.weight.detach().clone()
    before_other = other_params(model)
    freeze_for_phase_a(model, BASE_V)
    assert [n for n, p in model.named_parameters() if p.requires_grad] == ["tok_emb.weight"]
    opt = torch.optim.AdamW(phase_a_param_groups(model), lr=1e-2)
    for step in range(20):
        phase_a_step(model, opt, step)
    after = model.tok_emb.weight.detach()
    assert torch.equal(after[:BASE_V], before_emb[:BASE_V])
    assert not torch.equal(after[BASE_V:], before_emb[BASE_V:])
    for name, p in model.named_parameters():
        if name != "tok_emb.weight":
            assert torch.equal(p, before_other[name]), name


# 6. the weight-decay trap (expected behaviour; documents why phase A uses wd = 0)
def test_weight_decay_trap_moves_frozen_rows():
    _, model = resized()
    model.train()
    before = model.tok_emb.weight.detach().clone()
    freeze_for_phase_a(model, BASE_V)
    groups = phase_a_param_groups(model)
    groups[0]["weight_decay"] = 0.1
    opt = torch.optim.AdamW(groups, lr=1e-2)
    for step in range(20):
        phase_a_step(model, opt, step)
    assert not torch.equal(model.tok_emb.weight.detach()[:BASE_V], before[:BASE_V])


# 7. output mask
def test_output_mask_blocks_disallowed_ids():
    _, model = resized()
    g = torch.Generator().manual_seed(7)
    allowed = sorted(torch.randperm(BASE_V, generator=g)[:120].tolist())
    mask = mask_from_ids(allowed, NEW_V)
    assert mask.shape == (NEW_V,) and int(mask.sum()) == len(allowed) and not mask[BASE_V:].any()
    prompt = torch.randint(0, NEW_V, (200, 5), generator=g)          # prompts may contain any ID
    out = model.generate(prompt, 24, temperature=1.5, generator=g, logits_mask=mask)
    new = out[:, 5:]
    assert mask[new].all()
    # unmasked sampling at the same temperature does produce disallowed IDs
    free = model.generate(prompt, 24, temperature=1.5, generator=torch.Generator().manual_seed(7))[:, 5:]
    assert not mask[free].all()


def test_generate_rejects_overlong():
    model = tiny()
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(torch.zeros(1, 100, dtype=torch.long), 29)


# 8. KV cache on the resized model
def test_kv_cache_matches_no_cache():
    _, model = resized()
    prompt = torch.randint(0, NEW_V, (2, 9), generator=torch.Generator().manual_seed(11))
    with_cache = model.generate(prompt, 64, temperature=0, use_cache=True)
    without = model.generate(prompt, 64, temperature=0, use_cache=False)
    assert torch.equal(with_cache, without)
    # logits: incremental (cached) vs one full forward over the same sequence
    seq = with_cache
    cache = KVCache(TINY.n_layers)
    with torch.no_grad():
        logits, _ = model(seq[:, :9], kv_cache=cache)
        steps = [logits]
        for t in range(9, seq.shape[1]):
            step_logits, _ = model(seq[:, t:t + 1], kv_cache=cache)
            steps.append(step_logits)
        inc = torch.cat(steps, dim=1)
        full, _ = model(seq)
    assert torch.allclose(inc, full, atol=1e-5, rtol=0)
    assert cache.k[0].shape == (2, TINY.n_kv_heads, seq.shape[1], TINY.head_dim)
    assert cache.k[0].shape[1] == 2


# 9. resume of phase A is exact; mismatched checkpoints are rejected
def test_phase_a_resume_is_exact(tmp_path):
    def fresh():
        _, m = resized(seed=5)
        m.train()
        freeze_for_phase_a(m, BASE_V)
        return m, torch.optim.AdamW(phase_a_param_groups(m), lr=1e-2)

    model, opt = fresh()
    straight = [phase_a_step(model, opt, s) for s in range(20)]

    model, opt = fresh()
    for s in range(10):
        phase_a_step(model, opt, s)
    path = tmp_path / "step10.pt"
    save_finetuned(path, model, finetune_meta(TOK, "tiny", "A"), optimizer=opt.state_dict(), step=10)
    del model, opt

    model, ckpt = load_finetuned(path, TOK)          # a "new process": everything rebuilt from the file
    model.train()
    freeze_for_phase_a(model, BASE_V)
    opt = torch.optim.AdamW(phase_a_param_groups(model), lr=1e-2)
    opt.load_state_dict(ckpt["optimizer"])
    resumed = [phase_a_step(model, opt, s) for s in range(ckpt["step"], 20)]
    assert resumed == straight[10:]

    bad_vocab = SimpleNamespace(**{**vars(TOK), "vocab_size": 384})
    with pytest.raises(ValueError, match="vocab_size"):
        load_finetuned(path, bad_vocab)
    bad_sha = SimpleNamespace(**{**vars(TOK), "base_sha256": "b" * 64})
    with pytest.raises(ValueError, match="sha256"):
        load_finetuned(path, bad_sha)


def test_resize_rejects_untied_and_shrink():
    model = HindiLlama(replace(TINY, tie_embeddings=False))
    with pytest.raises(ValueError, match="tied"):
        resize_embeddings(model, NEW_V)
    with pytest.raises(ValueError):
        resize_embeddings(tiny(), BASE_V)
