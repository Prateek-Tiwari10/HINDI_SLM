"""AdamW with decay / no-decay parameter groups and the warmup + cosine LR schedule."""

import math

import torch
from torch import nn


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on 2-D weights (Linear, tied embedding); none on 1-D RMSNorm gains.

    named_parameters() yields the tied embedding/lm_head tensor once.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, optim_cfg: dict, peak_lr: float) -> torch.optim.AdamW:
    """AdamW(β=(0.9, 0.95), eps 1e-8), fused on CUDA when available."""
    fused = torch.cuda.is_available() and next(model.parameters()).is_cuda
    return torch.optim.AdamW(
        param_groups(model, optim_cfg["weight_decay"]),
        lr=peak_lr,
        betas=tuple(optim_cfg["betas"]),
        eps=optim_cfg["eps"],
        fused=fused,
    )


def lr_multiplier(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """LR / peak at a 0-based optimizer step.

    Linear warmup from peak/warmup to peak over `warmup` steps, then cosine decay that
    reaches min_ratio · peak exactly at the last step (total - 1).
    """
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - 1 - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(opt: torch.optim.Optimizer, warmup: int, total: int,
                    min_ratio: float) -> torch.optim.lr_scheduler.LambdaLR:
    """LambdaLR wrapping lr_multiplier; its state_dict (last_epoch) goes into checkpoints."""
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, warmup, total, min_ratio))
