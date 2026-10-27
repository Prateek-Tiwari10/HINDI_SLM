"""Model hyperparameters (D5) as a dataclass, loaded from and saved to YAML."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Hyperparameters of HindiLlama. Defaults are the fixed D5 config (configs/model_61m.yaml).

    `expected_params` is asserted when the model is built; set it to None only for the
    small or modified configs used in tests.
    """

    vocab_size: int = 10240
    d_model: int = 640
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: int = 5
    ffn_hidden: int = 1728
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    attn_impl: str = "sdpa"
    expected_params: int | None = 61_128_320

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        assert self.attn_impl in ("manual", "sdpa"), f"unknown attn_impl {self.attn_impl!r}"

    @property
    def head_dim(self) -> int:
        """Size of one attention head (640 / 10 = 64)."""
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """Query heads per K/V head (10 / 5 = 2)."""
        return self.n_heads // self.n_kv_heads

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load a config; unknown keys are an error so typos cannot pass silently."""
        raw = yaml.safe_load(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        assert not unknown, f"unknown model config keys: {sorted(unknown)}"
        return cls(**raw)

    def to_dict(self) -> dict:
        """Plain dict, e.g. for checkpoints."""
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write the config as YAML."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
