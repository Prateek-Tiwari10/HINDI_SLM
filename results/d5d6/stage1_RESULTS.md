# Stage 1 — Model correctness (D5), Kaggle CPU

**Status:** done. **Gate 1: PASS**, all 9 required checks pass.
- **Kaggle run:** `prateek99491/hindi-lm-d6-tests`, version 1 (CPU, 4 cores, 31 GB RAM, Python 3.13.15).
- **Started by API push**, with no Save Version: the code is embedded in `run.py` (`EMBED_CODE=1 kaggle/push.sh ... d6_tests`), and this stage needs no HF access.
- **Commands:** `pytest -v tests/` (74 tests, 26.4 s), then `python -m model.model --config configs/model_61m.yaml`.
- **Full pytest output:** [stage1_pytest.txt](stage1_pytest.txt).

## What was built

| file | content |
| --- | --- |
| `model/config.py` | `ModelConfig` dataclass, YAML load (unknown keys rejected) and save |
| `model/layers.py` | `RMSNorm` (fp32 statistic), `RotaryEmbedding` (fp32 cos/sin buffers, non-persistent), `rotate_half`, `apply_rope` (rotate-half / GPT-NeoX convention), `SwiGLU` |
| `model/attention.py` | `GQAttention` (manual + SDPA paths), `KVCache` (stores the 5 K/V heads), `causal_mask` |
| `model/model.py` | `Block` (pre-norm), `HindiLlama` (forward + fp32 loss, `generate` with and without cache, `param_report`) |
| `configs/model_61m.yaml` | the fixed D5 table |
| `configs/pretrain_61m.yaml` | the D6 table; `peak_lr: null` until Stage 4 |
| `tests/test_model.py` | 23 tests: the 9 required checks plus extra ones |

## Parameter report (from the Kaggle log)

```
module                           per layer         total
block.attn.q_proj                  409,600     4,915,200
block.attn.k_proj                  204,800     2,457,600
block.attn.v_proj                  204,800     2,457,600
block.attn.o_proj                  409,600     4,915,200
block.ffn.w1                     1,105,920    13,271,040
block.ffn.w3                     1,105,920    13,271,040
block.ffn.w2                     1,105,920    13,271,040
block.norm1 + norm2                  1,280        15,360
block (all)                      4,547,840    54,574,080
tok_emb (tied with lm_head)      6,553,600     6,553,600
final norm                             640           640
TOTAL                                         61,128,320
non-embedding                                 54,574,080
tied: True
```

## Gate checks

| # | check | test(s) | threshold | result |
| --- | --- | --- | --- | --- |
| 1 | params = 61,128,320; embedding and lm_head share a tensor | `test_param_count_and_tied_embedding`, `test_wrong_expected_params_is_rejected`, `test_no_biases` | exact; `data_ptr` equal | **PASS** |
| 2 | output shapes, B=2, T ∈ {1, 17, 1024} | `test_output_shapes[1/17/1024]` | (2, T, 10240), finite loss | **PASS** |
| 3 | causality: changing token t+1 leaves logits ≤ t unchanged, t ∈ {0, 7, 31, 62} | `test_causality` | max abs diff < 1e-6 (fp32); position t+1 does change (> 1e-4) | **PASS** |
| 4 | RoPE: ⟨rope(q,m), rope(k,n)⟩ depends only on m−n, shifts +0/+5/+100 | `test_rope_depends_only_on_offset`, `test_rope_preserves_norm` | abs diff < 1e-4 | **PASS** |
| 5 | manual vs SDPA attention | `test_manual_matches_sdpa[1/17/256]`, `test_full_model_manual_matches_sdpa` | module < 1e-5; full 12-layer model < 1e-4 | **PASS** |
| 6 | GQA with n_kv_heads = n_heads equals standard MHA | `test_gqa_with_all_kv_heads_equals_mha` (independent einsum MHA, both impls), `test_gqa_equals_mha_with_duplicated_kv_heads` | < 1e-5 | **PASS** |
| 7 | greedy 50 tokens identical with and without the KV cache | `test_kv_cache_greedy_matches_no_cache`, `test_kv_cache_stores_kv_heads_only`, `test_cached_chunk_matches_full_forward` | exact token equality | **PASS** |
| 8 | o_proj / w2 std ≈ 0.02/√24 = 0.00408 | `test_init_std` (also q_proj and the embedding ≈ 0.02, norm gains = 1) | within 3% | **PASS** |
| 9 | loss at init ≈ ln(10240) = 9.23 | `test_init_loss_near_uniform` | ± 0.3 | **PASS** |

Extra tests: YAML round trip, loss equal to a hand-written cross-entropy, top-k = 1 sampling equal to greedy, and a cached 7-token chunk after a 20-token prefill matching a full forward. The last one covers the explicit-mask SDPA branch for T > 1 with a cache.

The 51 earlier data and tokenizer tests also still pass.

## Notes and surprises

- **Kaggle's CPU image is now Python 3.13**, not 3.11/3.12 as in D1–D4; all tests still pass.
- **`python -m model.model` warned with a `RuntimeWarning`**, because `model/__init__.py` imported `model.model`. It was harmless. `__init__.py` is now docstring-only, and all imports name the submodule.
- **`return_attn=True` always uses the manual path**, even when `attn_impl` is `sdpa`, since SDPA cannot return weights. Training never sets it.
- **SDPA with a cache:** with a cached prefix and T > 1, the SDPA path passes an explicit bottom-right-aligned mask, because `is_causal=True` aligns the mask top-left. For T = 1 no mask is needed.
- **`generate()` limit:** prompt plus new tokens must fit in 1,024 tokens (absolute RoPE positions, no sliding window).
- **Output download:** the Kaggle output download hit a DNS error on the extracted code folder. Fetching only the log (`--file-pattern '.*\.log$'`) worked.
