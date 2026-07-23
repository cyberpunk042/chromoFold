# KV backend — the honest baseline finding (read before building)

**Stop-and-check result (P7, and "honesty is the brand"): llama.cpp already ships most of what a ChromoFold KV
backend would provide.** Discovered while scoping the end-to-end backend ([KV_BACKEND_DESIGN.md](KV_BACKEND_DESIGN.md)).
Recorded here so we don't build redundant work.

## What llama.cpp already does (pinned tree 76f46ad2)
1. **Quantized KV storage** — `-ctk/-ctv {q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl, …}` (`common/arg.cpp:301`). So the
   **storage-VRAM win is already available in stock llama**: q4_0 KV is ~4× smaller than f16, the same ballpark as
   ChromoFold int4. No storage advantage.
2. **Fused decode-in-attention** — ggml-cuda flash attention is templated by K/V type and accepts quantized types
   (`fattn.cu:245` `FATTN_VEC_CASE(D, type_K, type_V)`). llama reads q4_0 K/V **directly in the FA kernel**, never
   materializing dense KV — which is exactly ChromoFold's P3 "decode only inside the consumer" thesis, shipped and
   **tensor-core-optimized** (WMMA). Our warp-cooperative `cf_kv_paged_attention` essentially reimplements this.

So a ChromoFold KV backend built to "win VRAM/latency" would reproduce a mature llama feature. **The capacity number
(2.67×) is real vs f16 — but the honest comparison is vs llama's q4_0, where the gap ≈ 1×.**

## Where ChromoFold *is* measurably different (head-to-head, `gpu-quantizer-frontier`)
- **Quantizer accuracy at equal bits.** ChromoFold's KIVI grouping (per-channel K over tokens / per-token V over
  dims) vs q4_0's uniform per-32-block: KIVI attention error **0.039 vs 0.047 → 1.19× more accurate** at the same
  4 bits. Real, but modest — not, by itself, worth a whole custom backend competing with ggml's fattn.
- **Searchability — the actual ChromoFold thesis, which KV-decode does not exercise.** The FM-index (count/locate),
  wavelet rank/select/access — llama's q4_0 KV has **none** of this. A *searchable* compressed KV (e.g. FM-search
  over attended history, rank/select on the token stream, cross-sequence prefix dedup via the FM-index) is
  something llama fundamentally cannot do. This is where ChromoFold is unique, not in raw KV compression.

## Recommendation
1. **Do not build the redundant VRAM/latency KV backend.** llama's `-ctk q4_0` + quantized fattn already deliver
   it, better-optimized. Reproducing it would violate P7 (rebuilding a known baseline) and P10 (the proof must be a
   workload llama can't already serve).
2. **If the KV path continues, justify it on the two real differentiators, measured:** (a) the ~1.19× KIVI accuracy
   edge at equal bits — quantify end-to-end (perplexity/quality at q4_0-equal VRAM); and/or (b) a **searchable-KV
   capability** llama can't offer — the ChromoFold-native direction.
3. **The engine's real home is the searchable succinct-index thesis** (M1–M7: access/rank/RRR-wavelet/FM-search,
   all bit-exact and measured) applied to a workload that *needs* search while compressed — not KV-cache
   compression, which the ecosystem has already commoditized.

## What this session's KV work still bought (not wasted)
- A **verified compressed-attention engine** (warp-cooperative, bit-exact head_dim 32–128 × int4/int8) and the
  **honest measurements** that located this finding: the capacity number, the latency crossover, and now the
  q4_0 baseline. The value was **learning where the real frontier is** — and it isn't KV compression.
- The `cb_eval` shadow integration + the mapped llama seam remain a correct, reusable harness if (2a/2b) is pursued.
