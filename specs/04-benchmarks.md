# ChromoFold Native Engine — Measurement Protocol

> **How we prove it** (P7, P10). No headline ships without its reproducibility envelope and an honest baseline.
> This document is the contract for every benchmark the engine reports.

## 1. Reproducibility envelope (required on every record)

- **Machine:** GPU model + arch (sm_XX), CPU, driver, CUDA, compiler + flags, Warp/Triton versions, power mode,
  PCIe link, **git commit SHA**.
- **Workload:** corpus, vocabulary size V, token count N, query distribution, batch size.
- **Statistics:** ≥ **20 warm reps**; report **median, p5, p95, stddev, CI** — never a single wall-clock.
- **Memory:** resident bytes, transient/peak allocation bytes.
- **Rates:** queries/s, ns/access, effective bandwidth (GB/s).

## 2. The four timing layers (report separately)

| Layer | Includes | Answers |
|---|---|---|
| Kernel-only | preallocated device in/out | the actual algorithm cost |
| GPU-API | kernel launch + sync | launch overhead |
| Upload + kernel | host→device positions copied | input-transfer cost |
| Round trip | upload + kernel + result copied back | today's user-facing path |

The prototype's honest finding: the `access` **kernel** is ~1.24 B/s (~0.8 ns) but the user-facing round trip is
~460 M/s — **transfer-bound** (H2D+D2H ≈ ⅔ of wall time). The device-native API (P5) exists to remove that layer;
the benchmark must show it.

## 3. Baselines that matter (never a scalar Python loop)

- **Raw GPU gather** from uint8 / uint16 / uint32 token arrays — the direct price of addressability.
- **Optimized C++ CPU wavelet** (AVX2/AVX-512, multithreaded) — the real CPU baseline.
- **gzip and zstd** — whole-stream and block access (they win ratio; we win navigation — state both).
- **A GPU compression library** (e.g. nvCOMP ANS/GDeflate) where the workload is compatible.
- **Warp vs CUDA C++ vs Triton** implementations of the same operation — keep the fastest maintainable one.
- **packed wavelet vs RRR wavelet vs full FM-index** — the internal memory–latency frontier.

## 4. Workload sweep (V=256 Zipfian is not enough)

- **Vocabulary:** 4, 16, 256, 32K, 64K, 128K.
- **Streams:** uniform and Zipfian synthetic; real code; real prose; chat histories; repeated system prompts;
  RAG chunks; near-duplicate tokenized datasets.
- **Query pattern:** uniform, sorted, clustered, contiguous, attention-derived sparse.
- **Batch:** 1, 8, 32, 128, 1K, 16K, 1M.

## 5. Named experiments

- **A — Raw gather vs wavelet access.** Price of addressability. N ∈ {1M,4M,16M,64M}, V ∈ {256,32K,64K,128K},
  batch/pattern as above. Out: resident B/token, ns/access, qps, effective bandwidth, p95.
- **B — Shared-prefix prompt cache.** 64–256 requests, one large shared prefix + unique suffixes. Compare
  duplicated vs content-addressed vs full ChromoFold. Out: resident + peak transient memory, TTFT, prefill
  throughput, max batch at fixed memory, cost to resume from arbitrary positions, cost to add one suffix.
- **C — Near-duplicate dataset.** Reference-delta tree where it should genuinely beat per-file compression.
  Base docs + controlled edits + real versioned corpora. Out: ratio, random minibatch access, build time,
  update cost.
- **D — Fused embedding gather.** The architectural thesis (P3). Compare (raw IDs + gather) vs (ChromoFold decode
  + gather) vs (one fused decode-and-gather). Judge by **final embeddings/s and bytes moved**, not decoded
  tokens/s.
- **E — Effective-gain table.** The one-glance summary:

  | Representation | B/token | 1-token access | 1K slice | Search | Host trip |
  |---|---|---|---|---|---|
  | raw uint32 | 4 | ✓ (gather) | ✓ | ✗ | maybe |
  | gzip / zstd | best | ✗ | ✗ | ✗ | usually |
  | GPU block codec | low | ✗ | ✗ | ✗ | no |
  | packed wavelet | ~raw | ✓ | ✓ | rank | no |
  | RRR wavelet | ≤ H₀ | ✓ | ✓ | rank | no |
  | **full ChromoFold** | **≤ H₀** | **✓** | **✓** | **✓** | **no** |

## 6. Frozen reference numbers (Warp prototype — the oracle to match/beat)

These carry their own (2×RTX 2080 Ti sm_75, Warp 1.15) envelope and are the correctness + lower-bound targets:

| Primitive | Result |
|---|---|
| wavelet access/rank | ~1.1 B/tok index (V=256, 4M tok); ~400M access/s user-facing, ~1.24 B/s kernel |
| random access vs decompress-all | 68–111× (sparse reads, 4M stream) |
| RRR wavelet on BWT | 5.80 b/tok (below H₀ 5.96), two-level rank |
| block-Huffman bulk decode | 12–21× faster than the wavelet |
| KV KIVI int4 + entropy | attn MSE 1.5e-4; 8.1–8.9× VRAM |
| KV O(1) window latency | flat 1K→64K; **264× vs decode-all at 64K (Qwen2.5-1.5B)** |
| GPU FM-index build | 17–32× vs CPU, bit-identical |
| fused decode-in-matmul | 10.6× less VRAM held during compute (PoC, ~4 GFLOP/s) |

## 7. The decisive metric (P10)

Every experiment above is supporting evidence. The **ship metric** is Experiment-B/M9-style: a real LLM workload
that **fits a larger batch or longer context in the same GPU memory at equal-or-better latency**, output quality
intact. Ratio is not the proof; **useful work per GB at fixed latency** is.
