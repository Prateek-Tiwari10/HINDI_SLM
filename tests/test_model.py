"""Stage 1 (D5) correctness checks for HindiLlama. CPU, fp32."""

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from model.attention import GQAttention, KVCache
from model.config import ModelConfig
from model.layers import RotaryEmbedding, apply_rope
from model.model import HindiLlama

CFG = ModelConfig.from_yaml("configs/model_61m.yaml")
SMALL = replace(CFG, vocab_size=512, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                ffn_hidden=320, max_seq_len=256, expected_params=None)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return HindiLlama(CFG).eval()


def _attn_pair(cfg, seed=0):
    """Two attention modules with identical weights: manual and sdpa."""
    torch.manual_seed(seed)
    manual = GQAttention(replace(cfg, attn_impl="manual"), 0).eval()
    sdpa = GQAttention(replace(cfg, attn_impl="sdpa"), 0).eval()
    sdpa.load_state_dict(manual.state_dict())
    return manual, sdpa


# 1. parameter count and tying
def test_param_count_and_tied_embedding(model):
    assert model.num_params() == 61_128_320
    assert model.num_params(non_embedding=True) == 54_574_080
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    report = model.param_report()
    assert report["per_layer"] == 4_547_840


def test_wrong_expected_params_is_rejected():
    with pytest.raises(AssertionError):
        HindiLlama(replace(SMALL, expected_params=123))


def test_no_biases(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, name


# 2. output shapes
@pytest.mark.parametrize("T", [1, 17, 1024])
def test_output_shapes(model, T):
    idx = torch.randint(0, CFG.vocab_size, (2, T))
    with torch.no_grad():
        logits, loss = model(idx, idx)
    assert logits.shape == (2, T, CFG.vocab_size)
    assert loss.shape == () and torch.isfinite(loss)


# 3. causality
def test_causality(model):
    torch.manual_seed(1)
    T = 64
    idx = torch.randint(0, CFG.vocab_size, (2, T))
    with torch.no_grad():
        base, _ = model(idx)
        for t in [0, 7, 31, 62]:
            changed = idx.clone()
            changed[:, t + 1] = (changed[:, t + 1] + 1) % CFG.vocab_size
            out, _ = model(changed)
            assert (out[:, : t + 1] - base[:, : t + 1]).abs().max().item() < 1e-6, t
            assert (out[:, t + 1] - base[:, t + 1]).abs().max().item() > 1e-4, t   # the change is visible


# 4. RoPE relative-position property
def test_rope_depends_only_on_offset():
    torch.manual_seed(2)
    rope = RotaryEmbedding(64, 1024, 10000.0)
    q, k = torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64)

    def score(m, n):
        cm, sm = rope(m, 1)
        cn, sn = rope(n, 1)
        return (apply_rope(q, cm, sm) * apply_rope(k, cn, sn)).sum().item()

    for m, n in [(3, 0), (10, 4), (50, 50), (7, 30)]:
        ref = score(m, n)
        for shift in [0, 5, 100]:
            assert score(m + shift, n + shift) == pytest.approx(ref, abs=1e-4), (m, n, shift)
    assert abs(score(3, 0) - score(4, 0)) > 1e-3     # and it does depend on the offset


def test_rope_preserves_norm():
    rope = RotaryEmbedding(64, 1024, 10000.0)
    x = torch.randn(2, 3, 17, 64)
    cos, sin = rope(100, 17)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-4)


# 5. manual vs sdpa
@pytest.mark.parametrize("T", [1, 17, 256])
def test_manual_matches_sdpa(T):
    manual, sdpa = _attn_pair(CFG)
    x = torch.randn(2, T, CFG.d_model)
    cos, sin = RotaryEmbedding(CFG.head_dim, CFG.max_seq_len)(0, T)
    with torch.no_grad():
        a, w = manual(x, cos, sin, return_attn=True)
        b, _ = sdpa(x, cos, sin)
    assert (a - b).abs().max().item() < 1e-5
    assert w.shape == (2, CFG.n_heads, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(()), atol=1e-5)
    assert (w.triu(diagonal=1) == 0).all()


def test_full_model_manual_matches_sdpa():
    torch.manual_seed(3)
    m1 = HindiLlama(replace(CFG, attn_impl="manual")).eval()
    m2 = HindiLlama(replace(CFG, attn_impl="sdpa")).eval()
    m2.load_state_dict(m1.state_dict())
    idx = torch.randint(0, CFG.vocab_size, (2, 128))
    with torch.no_grad():
        assert (m1(idx)[0] - m2(idx)[0]).abs().max().item() < 1e-4


# 6. GQA sanity
def _reference_mha(x, attn, cos, sin):
    """Plain multi-head attention written independently (einsum, explicit mask), no K/V sharing."""
    B, T, _ = x.shape
    H, hd = attn.n_heads, attn.head_dim
    q = (x @ attn.q_proj.weight.T).view(B, T, H, hd).transpose(1, 2)
    k = (x @ attn.k_proj.weight.T).view(B, T, H, hd).transpose(1, 2)
    v = (x @ attn.v_proj.weight.T).view(B, T, H, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(hd)
    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    y = torch.einsum("bhts,bhsd->bhtd", s.softmax(-1), v)
    return y.transpose(1, 2).reshape(B, T, H * hd) @ attn.o_proj.weight.T


def test_gqa_with_all_kv_heads_equals_mha():
    cfg = replace(CFG, n_kv_heads=CFG.n_heads, expected_params=None)
    for impl in ("manual", "sdpa"):
        torch.manual_seed(4)
        attn = GQAttention(replace(cfg, attn_impl=impl), 0).eval()
        x = torch.randn(2, 33, cfg.d_model)
        cos, sin = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len)(0, 33)
        with torch.no_grad():
            assert (attn(x, cos, sin)[0] - _reference_mha(x, attn, cos, sin)).abs().max().item() < 1e-5


def test_gqa_equals_mha_with_duplicated_kv_heads():
    """GQA(5 K/V heads) == MHA(10) whose K/V head 2j and 2j+1 both copy GQA head j."""
    torch.manual_seed(5)
    gqa = GQAttention(CFG, 0).eval()
    mha = GQAttention(replace(CFG, n_kv_heads=CFG.n_heads, expected_params=None), 0).eval()
    hd = CFG.head_dim
    with torch.no_grad():
        mha.q_proj.weight.copy_(gqa.q_proj.weight)
        mha.o_proj.weight.copy_(gqa.o_proj.weight)
        for proj in ("k_proj", "v_proj"):
            w = getattr(gqa, proj).weight.view(CFG.n_kv_heads, hd, CFG.d_model)
            getattr(mha, proj).weight.copy_(w.repeat_interleave(CFG.n_rep, dim=0).reshape(-1, CFG.d_model))
        x = torch.randn(2, 40, CFG.d_model)
        cos, sin = RotaryEmbedding(hd, CFG.max_seq_len)(0, 40)
        assert (gqa(x, cos, sin)[0] - mha(x, cos, sin)[0]).abs().max().item() < 1e-5


# 7. KV-cache generation
def test_kv_cache_greedy_matches_no_cache(model):
    torch.manual_seed(6)
    prompt = torch.randint(0, CFG.vocab_size, (2, 12))
    with_cache = model.generate(prompt, 50, temperature=0, use_cache=True)
    without = model.generate(prompt, 50, temperature=0, use_cache=False)
    assert with_cache.shape == (2, 62)
    assert torch.equal(with_cache, without)


def test_kv_cache_stores_kv_heads_only(model):
    cache = KVCache(CFG.n_layers)
    with torch.no_grad():
        model(torch.randint(0, CFG.vocab_size, (2, 9)), kv_cache=cache)
        model(torch.randint(0, CFG.vocab_size, (2, 1)), kv_cache=cache)
    assert cache.length == 10
    assert cache.k[0].shape == (2, CFG.n_kv_heads, 10, CFG.head_dim)


def test_cached_chunk_matches_full_forward():
    """Prefill 20 tokens, then feed 7 at once with the cache: matches one 27-token forward."""
    torch.manual_seed(7)
    for impl in ("manual", "sdpa"):
        m = HindiLlama(replace(SMALL, attn_impl=impl)).eval()
        idx = torch.randint(0, SMALL.vocab_size, (2, 27))
        cache = KVCache(SMALL.n_layers)
        with torch.no_grad():
            m(idx[:, :20], kv_cache=cache)
            part, _ = m(idx[:, 20:], kv_cache=cache)
            full, _ = m(idx)
        assert (part - full[:, 20:]).abs().max().item() < 1e-5, impl


def test_sampling_respects_top_k():
    torch.manual_seed(8)
    m = HindiLlama(SMALL).eval()
    out = m.generate(torch.zeros(1, 1, dtype=torch.long), 20, temperature=1.0, top_k=1)
    greedy = m.generate(torch.zeros(1, 1, dtype=torch.long), 20, temperature=0)
    assert torch.equal(out, greedy)


# 8. init std
def test_init_std(model):
    target = 0.02 / math.sqrt(2 * CFG.n_layers)
    for block in model.blocks:
        for w in (block.attn.o_proj.weight, block.ffn.w2.weight):
            assert w.std().item() == pytest.approx(target, rel=0.03)
        assert block.attn.q_proj.weight.std().item() == pytest.approx(0.02, rel=0.03)
        assert torch.equal(block.norm1.weight, torch.ones(CFG.d_model))
    assert model.tok_emb.weight.std().item() == pytest.approx(0.02, rel=0.03)


# 9. loss at init
def test_init_loss_near_uniform(model):
    torch.manual_seed(9)
    idx = torch.randint(0, CFG.vocab_size, (4, 256))
    tgt = torch.randint(0, CFG.vocab_size, (4, 256))
    with torch.no_grad():
        _, loss = model(idx, tgt)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3, loss.item()


def test_config_yaml_round_trip(tmp_path):
    path = tmp_path / "m.yaml"
    CFG.to_yaml(path)
    assert ModelConfig.from_yaml(path) == CFG


def test_loss_accepts_shifted_slices():
    """Training passes x[:, :-1], x[:, 1:] of one window: non-contiguous targets must work."""
    m = HindiLlama(SMALL).eval()
    w = torch.randint(0, SMALL.vocab_size, (3, 17))
    with torch.no_grad():
        _, loss = m(w[:, :-1], w[:, 1:])
        _, ref = m(w[:, :-1].contiguous(), w[:, 1:].contiguous())
    assert loss.item() == ref.item()


def test_loss_matches_manual_cross_entropy():
    torch.manual_seed(10)
    m = HindiLlama(SMALL).eval()
    idx = torch.randint(0, SMALL.vocab_size, (2, 16))
    with torch.no_grad():
        logits, loss = m(idx, idx)
    ref = -F.log_softmax(logits.float(), -1).gather(-1, idx[..., None]).mean()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-6)
