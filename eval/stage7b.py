"""Stage 7b integration check (Kaggle T4, needs HF_TOKEN): surgery on the real 61M checkpoint.

    python -m eval.stage7b

1. Download the final pretrained checkpoint (latest.txt of pretrain_61m/) and run
   load_base_for_finetune with init "mean" and "random".
2. Parameter count before / after (expected 61,128,320 -> 61,169,280).
3. Unit tests 1, 2, 4 at full size: old rows bit-identical, head tied (also after
   save -> load), base-column logits unchanged on base-only input (CPU fp32, exact;
   the GPU max diff is reported too).
4. KL(base || resized) of next-token distributions over 100 val windows of 1024 tokens,
   for both inits; mean and max per position, plus the probability mass on new IDs.
5. KV cache: greedy 128 tokens from 5 Devanagari prompts with and without the cache;
   tokens must match. Tokens/s for both, on GPU and CPU.
6. Smoke probe: <rom> kal meeting cancel ho gayi kya <dev> with the output mask.
7. Tokenizer test 1 (encode_dev == SentencePiece on 1,000 v2 sentences).
Everything goes to /kaggle/working/stage7b_integration.json. Nothing is uploaded.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml

from data import hub
from model.config import ModelConfig
from model.model import HindiLlama
from model.surgery import build_output_mask, finetune_meta, load_base_for_finetune, load_finetuned, save_finetuned
from tokenizer.hinglish import DEFAULT_EXT, HinglishTokenizer
from train.data import ValWindows

OUT = Path("/kaggle/working/stage7b_integration.json")
DL = Path("/kaggle/tmp/stage7b")
N_KL_WINDOWS = 100
GEN_TOKENS = 128
PROBE = "kal meeting cancel ho gayi kya"
VAL_TEXT_FILES = ["final_v2/val/sangraha_verified-0000-00.parquet", "final_v2/val/wikipedia-0000-00.parquet"]


def base_model(path: Path, cfg: ModelConfig) -> HindiLlama:
    """The unmodified pretrained model (step ckpt.pt -> its 'model' entry)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = HindiLlama(cfg)
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


@torch.no_grad()
def final_hidden(model: HindiLlama, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(hidden state after the final norm, logits) for one forward pass."""
    seen = {}
    handle = model.norm.register_forward_hook(lambda _m, _i, out: seen.setdefault("h", out))
    logits, _ = model(idx)
    handle.remove()
    return seen["h"], logits


@torch.no_grad()
def kl_stats(base: HindiLlama, resized: HindiLlama, windows: ValWindows, device: torch.device) -> dict:
    """KL(p_base || p_resized) per position over the first N_KL_WINDOWS val windows.

    p_base has no mass on the new IDs, so KL = Σ_{i<10240} p_i (log p_i - log q_i).
    """
    base_v = base.cfg.vocab_size
    kls, new_mass = [], []
    for start in range(0, N_KL_WINDOWS, 10):
        x, _ = windows.batch(list(range(start, start + 10)), device)
        lp = F.log_softmax(base(x)[0].float(), dim=-1)
        lq = F.log_softmax(resized(x)[0].float(), dim=-1)
        kl = (lp.exp() * (lp - lq[..., :base_v])).sum(-1)
        kls.append(kl.flatten().cpu())
        new_mass.append(lq[..., base_v:].exp().sum(-1).flatten().cpu())
    kl, mass = torch.cat(kls), torch.cat(new_mass)
    return {"positions": kl.numel(), "kl_mean": kl.mean().item(), "kl_max": kl.max().item(),
            "kl_p99": kl.quantile(0.99).item(), "new_id_mass_mean": mass.mean().item(),
            "new_id_mass_max": mass.max().item()}


def time_generate(model: HindiLlama, prompts: list[list[int]], use_cache: bool, device: torch.device) -> dict:
    """Greedy GEN_TOKENS per prompt (batch 1, no eos stop); outputs and tokens/s."""
    outs, secs = [], 0.0
    for ids in prompts:
        x = torch.tensor([ids], device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model.generate(x, GEN_TOKENS, temperature=0, use_cache=use_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        outs.append(y[0, len(ids):].tolist())
    return {"outputs": outs, "seconds": round(secs, 3), "tokens_per_s": round(len(prompts) * GEN_TOKENS / secs, 1)}


def write_dev_sample(path: Path) -> int:
    """1,000 Devanagari sentences from two small final_v2 val files, one per line."""
    sents = []
    for remote in VAL_TEXT_FILES:
        local = hub.download("Prateek-Tiwari10/hindi-lm-data", remote, DL / "text")
        for doc in pq.read_table(local, columns=["text"]).column("text").to_pylist():
            for s in re.split(r"(?<=।)|\n", doc):
                s = s.strip()
                dev = sum("ऀ" <= c <= "ॿ" for c in s)
                if 20 <= len(s) <= 300 and dev >= 0.5 * len(s):
                    sents.append(s)
    path.write_text("\n".join(sents[:1000]) + "\n", encoding="utf-8")
    return min(len(sents), 1000)


def main() -> None:
    """Run every check and write stage7b_integration.json."""
    t_start = time.time()
    DL.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path("configs/pretrain_61m.yaml").read_text())
    mcfg = ModelConfig.from_yaml(cfg["model_config"])
    repo, prefix = cfg["checkpoint"]["hf_repo"], cfg["checkpoint"]["hf_prefix"]
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    tok = HinglishTokenizer.load(DEFAULT_EXT)
    res: dict = {"device": str(gpu), "torch": torch.__version__, "tokenizer_sha256": tok.base_sha256}

    # 1. base checkpoint -> resized models
    latest = Path(hub.download(repo, f"{prefix}latest.txt", DL, repo_type="model")).read_text().strip()
    ckpt = Path(hub.download(repo, f"{prefix}{latest}/ckpt.pt", DL, repo_type="model"))
    base_id = f"{repo}/{prefix}{latest}"
    res["base_checkpoint"] = base_id
    base = base_model(ckpt, mcfg)
    models = {init: load_base_for_finetune(ckpt, tok, mcfg, init=init, seed=0).eval() for init in ("mean", "random")}

    # 2. parameter counts
    res["params"] = {"before": base.num_params(), "after": models["mean"].num_params(),
                     "expected_after": 61_128_320 + 64 * 640}
    assert res["params"]["after"] == res["params"]["expected_after"]

    # 3. unit tests 1, 2, 4 at full size
    checks = {}
    old = base.tok_emb.weight
    for init, m in models.items():
        w = m.tok_emb.weight
        new = w[tok.base_vocab_size:]
        checks[init] = {
            "old_rows_identical": bool(torch.equal(w[:tok.base_vocab_size], old)),
            "head_is_embedding": m.lm_head.weight is w,
            "new_rows_finite": bool(torch.isfinite(new).all()),
            "new_row_norm_mean": new.norm(dim=1).mean().item(),
            "old_row_norm_mean": old.norm(dim=1).mean().item(),
            "mu_norm": old.mean(0).norm().item(),
        }
        checks[init]["new_over_old_norm"] = checks[init]["new_row_norm_mean"] / checks[init]["old_row_norm_mean"]
    path = DL / "resized_mean.pt"
    save_finetuned(path, models["mean"], finetune_meta(tok, base_id, "A"))
    loaded, _ = load_finetuned(path, tok)
    checks["save_load"] = {"head_is_embedding": loaded.lm_head.weight is loaded.tok_emb.weight,
                           "weights_equal": bool(torch.equal(loaded.tok_emb.weight, models["mean"].tok_emb.weight))}
    del loaded

    d = cfg["data"]
    meta = json.loads(Path(hub.download(d["hf_repo"], f"{d['prefix']}meta.json", DL)).read_text())
    (val_name, _), = meta["splits"]["val"]["files"]
    windows = ValWindows(Path(hub.download(d["hf_repo"], f"{d['prefix']}{val_name}", DL)), d["seq_len"])
    x, _ = windows.batch([0, 1], cpu)
    h_a, a = final_hidden(base, x[:, :256])
    logit_checks = {}
    for init, m in models.items():
        h_b, full = final_hidden(m, x[:, :256])
        b = full[..., :tok.base_vocab_size]
        same_shape = F.linear(h_b, m.lm_head.weight[:tok.base_vocab_size])
        logit_checks[init] = {"hidden_exact": bool(torch.equal(h_a, h_b)),
                              "logits_same_matmul_shape_exact": bool(torch.equal(a, same_shape)),
                              "logits_exact": bool(torch.equal(a, b)), "logits_max_abs_diff": (a - b).abs().max().item()}
    checks["base_logits"] = logit_checks
    res["unit_checks_full_size"] = checks
    print(f"[stage7b] checks {json.dumps(checks)}", flush=True)

    # 4. KL on GPU (fp32)
    base.to(gpu)
    res["kl"] = {}
    for init, m in models.items():
        m.to(gpu)
        res["kl"][init] = kl_stats(base, m, windows, gpu)
        with torch.no_grad():
            xg, _ = windows.batch([0, 1], gpu)
            diff = (base(xg)[0] - m(xg)[0][..., :tok.base_vocab_size]).abs().max().item()
        res["kl"][init]["gpu_base_logit_max_abs_diff"] = diff
        print(f"[stage7b] kl {init}: {json.dumps(res['kl'][init])}", flush=True)
        if init == "random":
            m.to(cpu)
    del models["random"]
    model = models["mean"]

    # 5. KV cache generation, GPU then CPU
    prompts_txt = [ln.strip() for ln in Path("eval/prompts.txt").read_text().splitlines() if ln.strip()][:5]
    prompts = [[tok.sp.bos_id()] + tok.encode_dev(p) for p in prompts_txt]
    res["kv_cache"] = {}
    for dev in (gpu, cpu) if gpu.type == "cuda" else (cpu,):
        model.to(dev)
        cached = time_generate(model, prompts, True, dev)
        plain = time_generate(model, prompts, False, dev)
        res["kv_cache"][dev.type] = {
            "identical_tokens": cached["outputs"] == plain["outputs"],
            "tokens_per_s_cache": cached["tokens_per_s"], "tokens_per_s_no_cache": plain["tokens_per_s"],
            "speedup": round(cached["tokens_per_s"] / plain["tokens_per_s"], 2)}
        if dev.type == "cuda":
            res["kv_cache"]["sample_greedy"] = {"prompt": prompts_txt[0],
                                                "continuation": tok.decode_dev(cached["outputs"][0][:60])}
        print(f"[stage7b] kv {dev.type}: {json.dumps(res['kv_cache'][dev.type])}", flush=True)

    # 6. smoke probe with the output mask (untrained: content is meaningless)
    model.to(gpu)
    mask = build_output_mask(tok)
    ids = [tok.rom_id] + tok.encode_rom(PROBE) + [tok.dev_id]
    x = torch.tensor([ids], device=gpu)
    probe = {}
    for name, temp in (("greedy", 0.0), ("T=1.0", 1.0)):
        y = model.generate(x, 40, temperature=temp, eos_id=tok.eos_id, logits_mask=mask,
                           generator=torch.Generator(device=gpu).manual_seed(0))
        new = y[0, len(ids):].tolist()
        probe[name] = {"n_new": len(new), "all_allowed": bool(mask[torch.tensor(new, dtype=torch.long)].all()),
                       "max_id": max(new, default=None), "ids": new, "text": tok.decode_dev(new)}
    res["smoke_probe"] = {"input": PROBE, "input_ids_len": len(ids), "n_allowed": int(mask.sum()), **probe}
    print(f"[stage7b] probe {json.dumps(res['smoke_probe'], ensure_ascii=False)}", flush=True)

    # 7. tokenizer test 1 on real v2 text
    sample = DL / "dev_sample.txt"
    n = write_dev_sample(sample)
    run = subprocess.run([sys.executable, "-m", "pytest", "-v",
                          "tests/test_hinglish_tokenizer.py::test_encode_dev_matches_base_on_v2_text"],
                         env={**os.environ, "HINGLISH_DEV_SAMPLE": str(sample)}, capture_output=True, text=True)
    res["tokenizer_test_1"] = {"sentences": n, "returncode": run.returncode,
                               "summary": run.stdout.strip().splitlines()[-1] if run.stdout else run.stderr[-500:]}
    print(run.stdout[-2000:], flush=True)

    res["elapsed_s"] = round(time.time() - t_start, 1)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"[stage7b] {json.dumps(res, indent=1, ensure_ascii=False)}", flush=True)
    ok = (all(checks[i]["old_rows_identical"] and checks[i]["head_is_embedding"]
              and checks["base_logits"][i]["hidden_exact"] for i in ("mean", "random"))
          and checks["save_load"]["head_is_embedding"] and all(v["identical_tokens"] for k, v in res["kv_cache"].items()
                                                                 if k in ("cuda", "cpu"))
          and all(p["all_allowed"] for p in probe.values()) and run.returncode == 0)
    if not ok:
        sys.exit("[stage7b] a required check failed; see stage7b_integration.json")


if __name__ == "__main__":
    main()
