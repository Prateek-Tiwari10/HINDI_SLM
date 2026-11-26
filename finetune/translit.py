"""Hinglish -> Devanagari inference, shared by D7 Step 6 (finetune/step6.py) and the demo server (serve/).

One code path for both, so the served model decodes exactly as it was evaluated:

    normalize_rom -> split_input (chunks of <= MAX_CHUNK_CHARS) -> [<rom>] + encode_rom(chunk) + [<dev>]
    -> masked greedy decoding with the KV cache, budget min(max_new, decode_budget(prompt_len, cap),
       max_seq_len - prompt_len) -> decode_dev; chunk outputs joined with one space.

Release format (the public HF model repo, written by export_release, read by load_release):
    model.safetensors     fp32 weights; the tied lm_head / tok_emb is stored once (safetensors save_model)
    config.json           model_config, tokenizer sha256, decode settings, source checkpoint
    tokenizer_ext.json    the D8 Latin table
    hi_unigram_10240.model  the frozen base SentencePiece model
"""

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from finetune.data import decode_budget
from model.config import ModelConfig
from model.model import HindiLlama
from model.surgery import build_output_mask
from tokenizer.hinglish import HinglishTokenizer, normalize_rom

MAX_CHUNK_CHARS = 160            # training pairs are <= 256 tokens; 160 Latin chars + Devanagari fit inside
MAX_NEW_TOKENS = 256             # configs/sft.yaml eval.max_new_tokens
RELEASE_FORMAT = "hinglish-translit-v1"
_SENT_END = re.compile(r"(?<=[.?!])\s+")


def split_input(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Normalized Latin text -> chunks of at most max_chars characters (spec 6 §2).

    Text that fits is one chunk, unchanged. Longer text is cut after sentence-ending . ? ! first;
    a sentence still longer than max_chars is cut at word boundaries into near-equal chunks
    (a single word longer than max_chars is cut hard). Empty input gives [].
    """
    text = normalize_rom(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    for sent in _SENT_END.split(text):
        if len(sent) <= max_chars:
            chunks.append(sent)
            continue
        pieces = []
        for word in sent.split(" "):
            while len(word) > max_chars:
                pieces.append(word[:max_chars])
                word = word[max_chars:]
            if word:
                pieces.append(word)
        target = len(sent) / math.ceil(len(sent) / max_chars)
        cur: list[str] = []
        for word in pieces:
            joined = " ".join(cur)
            if cur and (len(joined) + 1 + len(word) > max_chars or len(joined) >= target):
                chunks.append(joined)
                cur = []
            cur.append(word)
        if cur:
            chunks.append(" ".join(cur))
    return chunks


def prompt_ids(tok: HinglishTokenizer, chunk: str) -> list[int]:
    """[<rom>] + one id per normalized Latin char + [<dev>]: the training prompt (tokenizer/hinglish.build_pair)."""
    return [tok.rom_id] + tok.encode_rom(chunk) + [tok.dev_id]


@dataclass
class Result:
    output: str
    chunks: int
    new_tokens: int
    seconds: float


class Transliterator:
    """A fine-tuned HindiLlama + HinglishTokenizer wrapped as text -> Devanagari text."""

    def __init__(self, model: HindiLlama, tok: HinglishTokenizer, cap: dict, device: str | torch.device = "cpu",
                 max_new: int = MAX_NEW_TOKENS, max_chunk_chars: int = MAX_CHUNK_CHARS) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tok = tok
        self.cap = cap
        self.max_new = max_new
        self.max_chunk_chars = max_chunk_chars
        self.mask = build_output_mask(tok).to(self.device)

    def prompt(self, chunk: str) -> list[int]:
        return prompt_ids(self.tok, chunk)

    def budget(self, prompt_len: int) -> int:
        return min(self.max_new, decode_budget(prompt_len, self.cap), self.model.cfg.max_seq_len - prompt_len)

    @torch.inference_mode()
    def decode_prompt(self, prompt: list[int]) -> tuple[str, int]:
        """(Devanagari text, generated tokens excl. eos) for one prompt; masked greedy with the KV cache."""
        x = torch.tensor([prompt], dtype=torch.long, device=self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            y = self.model.generate(x, self.budget(len(prompt)), temperature=0, use_cache=True,
                                    eos_id=self.tok.eos_id, logits_mask=self.mask)
        gen = y[0, len(prompt):].tolist()
        if self.tok.eos_id in gen:
            gen = gen[: gen.index(self.tok.eos_id)]
        return self.tok.decode_dev(gen), len(gen)

    def __call__(self, text: str) -> Result:
        t0 = time.perf_counter()
        outs, n_new = [], 0
        chunks = split_input(text, self.max_chunk_chars)
        for chunk in chunks:
            out, n = self.decode_prompt(self.prompt(chunk))
            outs.append(out.strip())
            n_new += n
        return Result(" ".join(o for o in outs if o), len(chunks), n_new, time.perf_counter() - t0)


# ---------------------------------------------------------------- release files
def export_release(model: HindiLlama, tok: HinglishTokenizer, cap: dict, out_dir: str | Path, source: dict) -> Path:
    """Write the public release folder (see the module docstring); returns out_dir."""
    from safetensors.torch import save_model

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_model(model.cpu().float(), str(out / "model.safetensors"))
    tok.save(out / "tokenizer_ext.json")
    (out / tok.base_model.name).write_bytes(tok.base_model.read_bytes())
    config = {
        "format": RELEASE_FORMAT,
        "model_config": model.cfg.to_dict(),
        "params": model.num_params(),
        "tokenizer": {"ext": "tokenizer_ext.json", "base_model": tok.base_model.name, "base_sha256": tok.base_sha256,
                      "version": tok.version, "vocab_size": tok.vocab_size},
        "decode": {"strategy": "greedy, Devanagari-only output mask, KV cache", "cap": cap,
                   "max_new_tokens": MAX_NEW_TOKENS, "max_chunk_chars": MAX_CHUNK_CHARS,
                   "prompt": "[<rom>] + one id per normalized Latin char + [<dev>]", "stop": "eos (id 3)"},
        "source": source,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n")
    return out


def load_release(directory: str | Path, device: str | torch.device = "cpu") -> Transliterator:
    """Transliterator from a release folder (export_release / the public HF repo snapshot)."""
    from safetensors.torch import load_model

    d = Path(directory)
    config = json.loads((d / "config.json").read_text())
    if config.get("format") != RELEASE_FORMAT:
        raise ValueError(f"{d}/config.json format {config.get('format')!r}, expected {RELEASE_FORMAT!r}")
    tcfg = config["tokenizer"]
    tok = HinglishTokenizer.load(d / tcfg["ext"], base_model=d / tcfg["base_model"])
    if tok.base_sha256 != tcfg["base_sha256"] or tok.vocab_size != tcfg["vocab_size"]:
        raise ValueError("release tokenizer does not match config.json")
    model = HindiLlama(ModelConfig(**config["model_config"]))
    load_model(model, str(d / "model.safetensors"), strict=True)   # handles the tied lm_head / tok_emb pair
    assert model.lm_head.weight is model.tok_emb.weight
    dec = config["decode"]
    return Transliterator(model, tok, dec["cap"], device, dec["max_new_tokens"], dec["max_chunk_chars"])
