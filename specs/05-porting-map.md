# ChromoFold — Porting Map (Warp prototype → native engine)

> The measured Python/Warp prototype (`warp-solar-system-shaders/warp_compress`, 341 tests) is the **correctness
> oracle and performance floor** for the native port. Each native component must reproduce its prototype's
> bit-exact output and match-or-beat its measured numbers before the step is considered done.

## Prototype → native component

| Prototype module (Warp) | What it proves | Native home | Milestone |
|---|---|---|---|
| `gpu_wavelet.py` `GPUWavelet` | packed wavelet `access`/`rank` on GPU | `src/cuda/access.cu`, `rank.cu` | M1, M3 |
| `gpu_rrr.py` `GPURRR` | RRR bitvector, in-register combinatorial decode, two-level samples | `src/cuda/rrr.cu`, `include/.../wavelet.hpp` | M4 |
| `gpu_rrr_wavelet.py` `RRRWaveletGPU` | RRR-backed wavelet (5.80 b/tok on BWT, below H₀) | `src/cuda/rrr.cu` | M4 |
| `gpu_rrr_huffman.py` | Huffman-coded class stream, in-kernel decode | `rrr.cu` (class-stream variant) | M4 |
| `gpu_block_huffman.py` `BlockHuffmanArray` | LUT block decode, fixed-count blocks, random access | `src/cuda` block decoder; feeds fused ops | M4, M6 |
| `gpu_rans.py` `BlockRANSArray` | rANS block coder (skewed/low-entropy regime) | block coder variant | M4 |
| `gpu_fm_index.py` / `GPURRRFMIndex` | FM `count`/`locate`/`predict_next` over compressed BWT | `src/cuda/fm_search.cu` | M7 |
| `gpu_suffix.py` `gpu_suffix_array` | GPU suffix-array build (prefix-doubling, 17–32× CPU) | `src/builder.cpp` + CUB-based GPU build | M5, M7 |
| `gpu_delta.py` `GPUDeltaCluster` | reference-delta decode, O(depth) fetch | `src/cuda/delta_apply.cu` | M8 |
| `weight_store.py` | quantized value stream + outliers + `channel_scale` | value-stream + side-channel format | M4, (M6 fused) |
| `kv_store.py` | KIVI per-axis KV + entropy + windowed fetch | KV path; fused decode+dequant | M6, M9 |
| `gpu_fused_matmul.py` | fused decode-in-matmul (10.6× less VRAM in compute) | `src/cuda/fused_embedding.cu` (thesis) | M6 |
| `format.py` | self-describing versioned container | `src/format.cpp`, `include/.../format.hpp` | M0, M5 |
| `hf_cache.py` `ChromoFoldCache` | drop-in `transformers` KV cache, real generation | `bindings/python` + M9 integration | M9 |
| `spec_decode.py` | FM-index draft (2.18× fewer forwards) | consumer of `fm_search.cu` | post-M7 |
| `awq.py`, `autotune.py`, `dedup.py`, `multi_seed.py`, `lora_library.py`, `moe_store.py`, `model_store.py` | host-side calibration / config / orchestration | Python + C++ build side | as needed |

## Frozen reference numbers (the floor to match or beat)

Carry the (2×RTX 2080 Ti sm_75, Warp 1.15, torch 2.13-cpu, transformers 5.14) envelope.

- **access/rank:** ~1.1 B/tok index (V=256, 4M tok); kernel ~1.24 B/s, user-facing ~460 M/s (transfer-bound — the
  device-native API must remove this gap).
- **random access vs decompress-all:** 68–111× (sparse, 4M stream).
- **RRR wavelet on BWT:** 5.80 b/tok (< H₀ 5.96); block-Huffman bulk decode 12–21× faster than the wavelet.
- **KV:** KIVI int4 attn MSE 1.5e-4; **8.1–8.9× VRAM**; window latency flat 1K→64K; **264× vs decode-all at 64K
  (Qwen2.5-1.5B)**; drop-in `transformers` generation coherent.
- **FM-index build:** 17–32× vs CPU, bit-identical.
- **fused decode-in-matmul:** 10.6× less VRAM held during compute (PoC, ~4 GFLOP/s — correctness/memory proof;
  the native version must add throughput).
- **weights:** int4+class-Huffman ~1.33 b/w; SpQR outliers make int4 beat int8 accuracy at int8 size; whole gpt2
  int8 near-lossless, int4 needs group-128 **and** int8 embedding (mixed precision).

## Known honest boundaries to carry forward (P7, brief §9)

- **Dense quantized weights barely compress losslessly** — compose with quantization; target structure (sparse
  masks, outliers, cold experts, adapter libraries, repeated blocks, metadata, token streams, KV history).
- **Index overhead is real** — for small alphabets a packed wavelet can exceed optimally packed raw tokens; the
  win must come from search, entropy coding, cross-sequence dedup, or avoided decompression.
- **Construction is expensive** — build offline/CPU, deploy a compact immutable GPU representation (P9).
- **Token index ≠ KV cache** — exact token memory, compressed KV values, and full contextual attention state are
  distinct objects; claims must say which.

## Verification discipline

For every ported operation: (1) bit-identical to the Warp/CPU reference on the fuzz corpus, (2) benchmarked with
the full envelope against a real baseline, (3) archived so a second run / second GPU generation reproduces it.
The native engine is not "faster because CUDA" until the harness says so.
