"""D9: model surgery for Hinglish fine-tuning (Stage 7b).

- resize_embeddings: grow the tied embedding / lm_head from 10,240 to 10,304 rows. Old
  rows are copied bit for bit; new rows are mean-initialized (or random, for comparison).
- Phase A: only the new rows train (freeze_for_phase_a + phase_a_param_groups).
- build_output_mask: the bool mask passed to HindiLlama.generate(logits_mask=...).
- Checkpoints: load_base_for_finetune (pretrained 10,240 weights -> resized model, no
  optimizer state), save_finetuned / load_finetuned (built directly at the saved vocab,
  checked against the tokenizer extension).

`tok` arguments need .vocab_size, .base_vocab_size, .base_sha256 and .version; a
tokenizer.hinglish.HinglishTokenizer has them.
"""

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from model.config import ModelConfig
from model.model import HindiLlama


def resize_embeddings(model: HindiLlama, new_vocab_size: int, init: str = "mean", seed: int = 0) -> HindiLlama:
    """Resize the tied embedding in place to new_vocab_size rows and re-tie lm_head to it.

    init "mean": new row = μ + ε, μ the mean of the old rows, ε ~ N(0, (0.1·σ)²) per
    dimension with σ the per-dimension std of the old rows. Because the rows are tied, they
    are also output logits; near-average rows give near-average logits, so the pretrained
    next-token distribution barely moves. init "random": N(0, 0.02²), the pretraining init.
    """
    cfg = model.cfg
    if not cfg.tie_embeddings or model.lm_head.weight is not model.tok_emb.weight:
        raise ValueError("resize_embeddings expects tied embeddings (D5)")
    old = model.tok_emb.weight.data
    old_v, d = old.shape
    if new_vocab_size <= old_v:
        raise ValueError(f"new vocab {new_vocab_size} must be larger than {old_v}")
    # the only vocab-sized tensor must be the tied embedding (RoPE buffers are per position)
    sized = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if old_v in t.shape]
    assert sized == ["tok_emb.weight"], f"unexpected vocab-sized tensors: {sized}"

    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_new = new_vocab_size - old_v
    if init == "mean":
        mu, sigma = old.float().mean(0).cpu(), old.float().std(0).cpu()
        rows = mu + torch.randn(n_new, d, generator=gen) * (0.1 * sigma)
    elif init == "random":
        rows = torch.randn(n_new, d, generator=gen) * 0.02
    else:
        raise ValueError(f"unknown init {init!r}")

    emb = nn.Embedding(new_vocab_size, d, device=old.device, dtype=old.dtype)
    with torch.no_grad():
        emb.weight[:old_v].copy_(old)
        emb.weight[old_v:].copy_(rows.to(old.device, old.dtype))
    model.tok_emb = emb
    model.lm_head = nn.Linear(d, new_vocab_size, bias=False, device=old.device, dtype=old.dtype)
    model.lm_head.weight = model.tok_emb.weight                    # re-tie: the same Parameter object
    expected = None if cfg.expected_params is None else cfg.expected_params + n_new * d
    model.cfg = replace(cfg, vocab_size=new_vocab_size, expected_params=expected)
    return model


# ---------- phase A / phase B ----------
def freeze_for_phase_a(model: HindiLlama, base_vocab_size: int) -> None:
    """Train only embedding rows >= base_vocab_size: freeze everything else, zero base-row grads.

    The hook zeroes rows < base_vocab_size of every gradient reaching the tied weight (from
    both the input embedding and the lm_head). Pair with phase_a_param_groups (wd = 0).
    """
    unfreeze_all(model)
    for p in model.parameters():
        p.requires_grad_(False)
    w = model.tok_emb.weight
    w.requires_grad_(True)

    def zero_base_rows(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[:base_vocab_size] = 0
        return grad

    model._phase_a_hook = w.register_hook(zero_base_rows)


def phase_a_param_groups(model: HindiLlama) -> list[dict]:
    """One group, the embedding, with weight_decay 0.

    AdamW's decoupled decay (w -= lr·wd·w) runs even when the gradient is zero, so any
    wd > 0 would shrink the frozen base rows. With zero gradients from step 0 Adam's
    moments stay zero and the base-row update is exactly 0.
    """
    return [{"params": [model.tok_emb.weight], "weight_decay": 0.0}]


def unfreeze_all(model: HindiLlama) -> None:
    """Phase B: every parameter trainable, the phase-A hook removed."""
    hook = getattr(model, "_phase_a_hook", None)
    if hook is not None:
        hook.remove()
        model._phase_a_hook = None
    for p in model.parameters():
        p.requires_grad_(True)


# ---------- output mask ----------
def mask_from_ids(allowed_ids: list[int], vocab_size: int) -> torch.Tensor:
    """Bool (vocab_size,) tensor, True at the allowed IDs."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[torch.tensor(allowed_ids, dtype=torch.long)] = True
    return mask


def build_output_mask(tok) -> torch.Tensor:
    """Mask over the extended vocab from tok.output_allowed_ids() (Devanagari pieces + eos)."""
    return mask_from_ids(tok.output_allowed_ids(), tok.vocab_size)


# ---------- checkpoints ----------
def finetune_meta(tok, base_checkpoint: str, phase: str) -> dict:
    """The fine-tune block stored with every fine-tune checkpoint."""
    if phase not in ("A", "B"):
        raise ValueError(f"phase must be 'A' or 'B', not {phase!r}")
    return {"vocab_size": tok.vocab_size, "tokenizer_version": tok.version, "tokenizer_sha256": tok.base_sha256,
            "base_checkpoint": base_checkpoint, "phase": phase}


def load_base_for_finetune(base_ckpt: str | Path, tok, base_cfg: ModelConfig | None = None,
                           init: str = "mean", seed: int = 0) -> HindiLlama:
    """Pretrained model resized for fine-tuning.

    Builds at the base vocab (10,240), loads the weights with strict=True from either a
    best/model.pt (a plain state dict) or a step ckpt.pt (its "model" entry), then resizes
    to tok.vocab_size. The pretraining optimizer / scheduler / scaler states are ignored:
    shapes and objective differ, so fine-tuning starts with a fresh optimizer.
    """
    cfg = base_cfg or ModelConfig.from_yaml("configs/model_61m.yaml")
    if cfg.vocab_size != tok.base_vocab_size:
        raise ValueError(f"base model vocab {cfg.vocab_size} != tokenizer base vocab {tok.base_vocab_size}")
    state = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model = HindiLlama(cfg)
    model.load_state_dict(state, strict=True)
    return resize_embeddings(model, tok.vocab_size, init=init, seed=seed)


def save_finetuned(path: str | Path, model: HindiLlama, meta: dict, **extra) -> None:
    """torch.save {model, model_config, finetune: meta, **extra (optimizer, step, rng, ...)}."""
    if meta["vocab_size"] != model.cfg.vocab_size:
        raise ValueError(f"meta vocab {meta['vocab_size']} != model vocab {model.cfg.vocab_size}")
    torch.save({"model": model.state_dict(), "model_config": model.cfg.to_dict(), "finetune": meta, **extra}, path)


def load_finetuned(path: str | Path, tok=None) -> tuple[HindiLlama, dict]:
    """(model built at the saved vocab size, the whole checkpoint dict).

    With tok given, the saved vocab size and tokenizer sha256 must match it.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("finetune")
    if meta is None:
        raise ValueError(f"{path} is not a fine-tune checkpoint (no 'finetune' block); use load_base_for_finetune")
    cfg = ModelConfig(**ckpt["model_config"])
    if cfg.vocab_size != meta["vocab_size"]:
        raise ValueError(f"checkpoint config vocab {cfg.vocab_size} != finetune vocab {meta['vocab_size']}")
    if tok is not None:
        if meta["vocab_size"] != tok.vocab_size:
            raise ValueError(f"checkpoint vocab_size {meta['vocab_size']} != tokenizer vocab_size {tok.vocab_size}")
        if meta["tokenizer_sha256"] != tok.base_sha256:
            raise ValueError(f"checkpoint tokenizer sha256 {meta['tokenizer_sha256']} != tokenizer {tok.base_sha256}")
    model = HindiLlama(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    assert model.lm_head.weight is model.tok_emb.weight
    return model, ckpt
