"""HindiLlama: the D5 LLaMA-style decoder-only Transformer.

Parameter count of the fixed config (d=640, 12 layers, 10 query / 5 K/V heads of 64,
SwiGLU hidden 1728, vocab 10240, tied embeddings, no biases):

- attention per layer = q 640·640 + k 640·320 + v 640·320 + o 640·640 = 1,228,800
- SwiGLU per layer    = 3 · 640 · 1728                                = 3,317,760
- 2 RMSNorm gains per layer                                           =     1,280
- per layer = 4,547,840; × 12 layers                                  = 54,574,080 (non-embedding)
- tied embedding 10240 · 640 = 6,553,600; final norm 640
- total                                                               = 61,128,320

Run `python -m model.model` to print the per-module table.
"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch import nn

from model.attention import GQAttention, KVCache
from model.config import ModelConfig
from model.layers import RMSNorm, RotaryEmbedding, SwiGLU


class Block(nn.Module):
    """Pre-norm Transformer block: h = x + attn(norm1(x)); out = h + ffn(norm2(h))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = GQAttention(cfg, layer_idx)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x (B, T, d) -> (B, T, d), plus attention weights if requested."""
        a, attn = self.attn(self.norm1(x), cos, sin, kv_cache, return_attn)
        h = x + a
        return h + self.ffn(self.norm2(h)), attn


class HindiLlama(nn.Module):
    """Decoder-only LM: token embedding -> 12 blocks -> RMSNorm -> tied lm_head.

    No position embedding table: positions enter only through RoPE inside attention.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self._init_weights()
        if cfg.expected_params is not None:
            n = self.num_params()
            assert n == cfg.expected_params, f"parameter count {n:,} != expected {cfg.expected_params:,}"

    def _init_weights(self) -> None:
        """N(0, 0.02) for Linear/Embedding; N(0, 0.02/√(2·n_layers)) for o_proj and w2; gains 1.

        The smaller std on the two projections that write into the residual stream keeps
        its variance from growing with depth (GPT-2 style scaling).
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        resid_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=resid_std)

    def num_params(self, non_embedding: bool = False) -> int:
        """Unique parameters (the tied embedding is counted once)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.norm.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        return_attn: bool = False,
    ):
        """idx (B, T) int64 -> (logits (B, T, V), loss or None).

        Loss is mean cross-entropy over all positions, on logits upcast to fp32.
        With a kv_cache, idx holds only the new tokens and positions continue after the
        cached ones. With return_attn=True a third element is returned: the list of
        per-layer attention weights (B, H, T, S), computed on the manual path.
        """
        B, T = idx.shape
        start = kv_cache.length if kv_cache is not None else 0
        cos, sin = self.rope(start, T)
        x = self.tok_emb(idx)
        attns = []
        for block in self.blocks:
            x, attn = block(x, cos, sin, kv_cache, return_attn)
            attns.append(attn)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            # reshape, not view: targets are often a shifted slice (x[:, 1:]) and not contiguous
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        if return_attn:
            return logits, loss, attns
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Extend idx (B, T) by up to max_new_tokens tokens.

        temperature 0 means greedy (argmax); otherwise logits are divided by temperature
        and, if top_k is set, cut to the k largest before sampling. use_cache=False
        recomputes the whole sequence each step (the reference for testing the cache).
        Generation stops early if every row has produced eos_id. The total length must
        fit in max_seq_len (positions are absolute RoPE angles; there is no sliding window).
        """
        assert idx.shape[1] + max_new_tokens <= self.cfg.max_seq_len, "prompt + new tokens > max_seq_len"
        was_training = self.training
        self.eval()
        cache = KVCache(self.cfg.n_layers) if use_cache else None
        out = idx
        step_input = idx
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits, _ = self(step_input if use_cache else out, kv_cache=cache)
            logits = logits[:, -1, :].float()
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
            out = torch.cat((out, nxt), dim=1)
            step_input = nxt
            if eos_id is not None:
                done |= nxt.squeeze(1) == eos_id
                if bool(done.all()):
                    break
        self.train(was_training)
        return out

    def param_report(self) -> dict[str, int]:
        """Print a per-module parameter table and the total; assert the expected count."""
        cfg = self.cfg
        b = self.blocks[0]
        per_layer = {
            "attn.q_proj": b.attn.q_proj.weight.numel(),
            "attn.k_proj": b.attn.k_proj.weight.numel(),
            "attn.v_proj": b.attn.v_proj.weight.numel(),
            "attn.o_proj": b.attn.o_proj.weight.numel(),
            "ffn.w1": b.ffn.w1.weight.numel(),
            "ffn.w3": b.ffn.w3.weight.numel(),
            "ffn.w2": b.ffn.w2.weight.numel(),
            "norm1 + norm2": b.norm1.weight.numel() + b.norm2.weight.numel(),
        }
        layer_total = sum(p.numel() for p in b.parameters())
        assert layer_total == sum(per_layer.values())
        rows = [(f"block.{k}", v, v * cfg.n_layers) for k, v in per_layer.items()]
        rows.append(("block (all)", layer_total, layer_total * cfg.n_layers))
        tied = "tied with lm_head" if cfg.tie_embeddings else "untied"
        rows.append((f"tok_emb ({tied})", self.tok_emb.weight.numel(), self.tok_emb.weight.numel()))
        if not cfg.tie_embeddings:
            rows.append(("lm_head", self.lm_head.weight.numel(), self.lm_head.weight.numel()))
        rows.append(("final norm", self.norm.weight.numel(), self.norm.weight.numel()))
        total = self.num_params()
        print(f"{'module':<28}{'per layer':>14}{'total':>14}")
        for name, one, all_ in rows:
            print(f"{name:<28}{one:>14,}{all_:>14,}")
        print(f"{'TOTAL':<28}{'':>14}{total:>14,}")
        print(f"{'non-embedding':<28}{'':>14}{self.num_params(non_embedding=True):>14,}")
        if cfg.expected_params is not None:
            assert total == cfg.expected_params, f"{total:,} != {cfg.expected_params:,}"
        return {"total": total, "non_embedding": self.num_params(non_embedding=True), "per_layer": layer_total}


def main() -> None:
    """Build the model from a YAML config and print its parameter report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_61m.yaml")
    args = parser.parse_args()
    model = HindiLlama(ModelConfig.from_yaml(args.config))
    model.param_report()
    print("tied:", model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr())


if __name__ == "__main__":
    main()
