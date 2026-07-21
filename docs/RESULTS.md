# Results — what was measured, and what it means

Every result here was **verified bit-for-bit** against an independent reference before any speed was reported
(principle P0). Numbers are on an **NVIDIA RTX 2080 Ti** (a 2018 gaming GPU, `sm_75`), CUDA 12.6. Each row is
reproducible with the `make` target named in its heading.

If a term is unfamiliar, see [`GLOSSARY.md`](GLOSSARY.md); for how the pieces connect, [`CONCEPTS.md`](CONCEPTS.md);
for the raw milestone log, [`../specs/03-roadmap.md`](../specs/03-roadmap.md).

**How to read a result.** Each entry says: *what was built*, *the plain-English meaning*, *the measured number*,
and *the honest caveat*. "Bit-identical" means the GPU output exactly matched the reference — correctness first,
speed second.

---

## M1 — Read a compressed token (`access`) · `make bench`

- **What.** Jump to any position in a compressed token sequence and return its symbol, on the GPU.
- **Result.** Bit-identical to the reference; **1.23 billion reads/second** (0.81 nanoseconds each) in the kernel.
  The API is *device-native* — it takes GPU pointers and never copies to the CPU, which is what makes the memory
  savings real.
- **Caveat.** The *user-facing* number (with a CPU round-trip) was 3–4× slower — proving why the device-native,
  no-copy API matters: copying to the CPU would erase the whole point.

## M2 — The price of addressability · `make experiment-a`

- **What.** An honest head-to-head: reading from the compressed wavelet vs. the cheapest possible thing (a raw
  memory fetch), across vocabulary sizes and access patterns.
- **Result.** Raw fetch is the floor (~0.1 ns, does nothing but load). The wavelet costs more (1.7×–13×), scaling
  with `log₂(vocabulary)` — because it walks that many bit-layers. Sorted/contiguous access roughly halves the gap.
- **Meaning (the honest line).** Raw fetch wins for dense, small-alphabet, plain reads. **The wavelet's extra cost
  buys capabilities a raw fetch simply cannot do**: rank, search, and — for big vocabularies — a smaller footprint.
  You pay for addressability; you get search. That's the trade, stated plainly.

## M3 — Fast `rank` with a two-level directory · `make rank`

- **What.** `rank(c, i)` — count a symbol's occurrences before position `i` — the primitive search is built on. Two
  directory layouts compared.
- **Result.** A **two-level directory** (coarse signposts + fine deltas) is **1.6–1.8× faster** than scanning, and
  bit-identical. It uses a bigger directory but eliminates work per query — the right trade for a query-bound job.

## M4 — The compressed, searchable index and its coders

This is the heart of the project: an index that is **entropy-sized *and* searchable**, plus the entropy coders that
make it so.

### RRR bit-vector rank · `make rrr`
- **What.** Store a bit-vector near its entropy while keeping `rank` fast (the RRR method).
- **Result.** On skewed data, **2.2–2.8× smaller than a packed layout** while `rank` stays ~0.7 ns. Bit-identical.
- **Caveat.** On 50/50 data RRR is actually *bigger* than packed — so those layers stay packed. We only RRR-code
  the skewed ones (which is exactly what the BWT produces).

### RRR-backed wavelet — the searchable entropy-sized index · `make rrr-wavelet`
- **What.** Every layer of the search wavelet stored with RRR → the whole self-index, entropy-sized.
- **Result.** On a BWT, the index is **5.80 bits/token — below `H₀` (5.96)** and 1.25–1.36× smaller than a packed
  wavelet, **while still supporting `access` and `rank`** on the GPU. Bit-identical. `access` costs ~5 ns/read.
- **Meaning.** A single object that is *both* compressed *and* instantly searchable — the thing the FM-index rides
  on. **Optimization (later round):** `access` originally decoded each layer's block twice; a one-decode-per-layer
  trick made it **1.9× faster** (9.7 → 5.0 ns), bit-identical, which also sped up `locate` (1.6×) and sparse gather.

### Two entropy coders compared — rANS vs Huffman · `make rans`
- **What.** A block-rANS decoder (near the entropy floor) vs block-Huffman, across data skew and block size.
- **Result (bit-identical, the honest crossover):**

  | data | entropy `H₀` | block | rANS | Huffman | winner |
  |---|---|---|---|---|---|
  | multi-bit (peaky) | 3.06 | 64 | 4.51 | 3.62 | Huffman |
  | multi-bit (peaky) | 3.06 | 1024 | 3.16 | 3.15 | Huffman |
  | very skewed | 0.45 | 64 | 1.88 | 1.68 | Huffman |
  | **very skewed** | **0.45** | **1024** | **0.55** | 1.21 | **rANS — 2.2× smaller** |

- **Meaning.** rANS wins **only** on very skewed data with large blocks (where Huffman's ~1-bit-per-symbol floor
  bites and rANS reaches near `H₀`). Everywhere else Huffman wins. We show the whole table, not the flattering row.
  The block size is a per-deployment dial (bigger = smaller but less parallel decode).

## M5 — Building the index natively (no Python) and on the GPU

### Native C++ builder + CPU oracle · `make build-index`
- **What.** Build the *entire* searchable index (suffix array → BWT → RRR wavelet → FM-index) in plain C++ — no
  Python, no research prototype.
- **Result.** The C++ builder writes the index; the GPU kernels answer `access`/`rank`/`count`/`locate` on it
  **bit-identically** to a CPU reference. Build, query, and oracle all agree, with the prototype removed from the
  build path. Build cost (offline): ~1.3 s for 2M tokens. CPU baseline `access` ~1.4 µs/read (single-thread) — so
  the GPU is ~130–200× faster; the CPU path is the *correctness oracle*, not the speed path.

### GPU suffix-array build · `make suffix-array`
- **What.** Move the expensive construction step (sorting all suffixes) onto the GPU.
- **Result.** **21–23× faster** than the CPU at 2M tokens (1.0 s → 0.05 s), **bit-identical** (the suffix array is
  unique, so correct = equals the CPU's). Verified across sizes/vocabs/seeds.
- **Caveat.** The win grows with size — only ~2× at 100K tokens, where GPU launch overhead dominates. Small inputs
  don't benefit.

## M6 — Fusion: decode *inside* the computation (the differentiator)

This is where chromoFold stops being "a compressor" and becomes a *runtime*. We proved it on three consumers and,
crucially, learned **when it pays**.

### First attempt — embedding gather — an honest negative · `make fused`
- **What.** Fuse token-decode with an embedding lookup.
- **Result.** The fused version **lost** at every size. Why: the thing it avoided building (a list of token ids) is
  tiny, so there was no memory to save, and the two halves want opposite GPU layouts.
- **Why we keep it.** It taught the rule for the wins below: *fuse only when the avoided intermediate is large, or
  the consumer is sparse.* (Principle P7 — report negatives.)

### Fused decode-in-matmul — the positive case · `make fused-matmul`
- **What.** Multiply by a 4-bit, entropy-coded weight matrix, decoding each weight inside the multiply so the dense
  matrix never exists.
- **Result (bit-identical to decoding-then-multiplying, and exact to the prototype):**

  | weight matrix | compressed (resident) | dense (avoided) | memory saved | fused vs dense speed |
  |---|---|---|---|---|
  | 2048×2048 | 1.58 MB | 16.78 MB | **10.6×** | 1.8× slower |
  | 4096×4096 | 6.30 MB | 67.11 MB | **10.6×** | **1.3× faster** |

- **Meaning.** Fusion is *numerically free* (bit-identical). At large sizes it's also *faster*, because it avoids
  writing and re-reading a 67 MB intermediate. **Caveat:** if you reuse the same weight matrix across many
  multiplies, decoding it once (dense) and reusing wins — fusion re-decodes each time. It's for
  large-intermediate, memory-bound, single-use multiplies.

### Fused KV-cache attention — the long-context brick · `make kv-attention`
- **What.** Attention that decodes each Key/Value from a compressed KV cache inline (never building the dense
  K/V tiles).
- **Result (bit-identical to decode-then-attend):**

  | KV cache | compressed | dense | KV memory saved |
  |---|---|---|---|
  | 4096×64 | 0.27 MB | 2.10 MB | **7.7×** |
  | 8192×128 | 1.02 MB | 8.39 MB | **8.3×** |

- **Meaning.** ~8× less KV memory → **fit a much longer context / more layers in the same GPU.** This is the
  concrete brick for the final workload proof (M9). **Caveat:** the win here is *capacity*, not speed — the fused
  path re-decodes per query (compute-for-memory); it flips to a *speed* win when the dense KV tile is too big to
  cache or the attention is genuinely sparse.

### Fused sparse gather — random access beats decompress-all · `make sparse-gather`
- **What.** When a consumer touches only K of N positions, read *just those K* from the compressed index vs.
  decompressing all N first.
- **Result (bit-identical):**

  | fraction touched | read-only-what's-needed | decompress-all | speedup |
  |---|---|---|---|
  | 0.1% | 0.12 ms | 2.10 ms | **17.8×** |
  | 1% | 0.13 ms | 2.14 ms | 16.7× |
  | 10% | 0.61 ms | 3.68 ms | 6.0× |
  | 100% | 6.42 ms | 23.7 ms | 3.7× |

- **Meaning.** This is principle **P2 in numbers**: because the index stays randomly addressable, a sparse consumer
  never pays to reconstruct the whole sequence — the win scales ~N/K. **Caveat:** the "decompress-all" baseline
  here uses the *same* per-element wavelet walk; a genuinely dense job should use the fast bulk block-coder (M4),
  which beats per-element walks as you approach 100%.

## M7 — Substring search on the compressed index · `make fm-search`

- **What.** FM-index `count` ("how many times does this pattern occur?") and `locate` ("where?"), running entirely
  on the compressed BWT wavelet in VRAM.
- **Result.** Both **bit-identical to ground truth** (a naive brute-force count). `count` batches to **~19 million
  pattern-searches/second**. `locate` walks each occurrence back to a landmark.
- **Meaning.** The *same compact index* is now read, searched, and (via batched count) usable as an n-gram
  predictor — search without leaving the GPU or decompressing.
- **Caveat.** `locate` is decode-heavy (each step is a full walk), and its per-occurrence latency is meaningful
  only at high occurrence counts (at low counts it's dominated by GPU launch overhead, not real work — we say so
  rather than quoting a flattering number).

## M8 — Cross-request dedup (delta clusters) · `make delta`

- **What.** Store many near-identical sequences (shared prompts, conversation turns, adapter libraries) as one base
  + per-sequence differences; reconstruct any token on the GPU by binary-searching its diffs.
- **Result (bit-identical reconstruction of all 2.06M tokens):**

  | | value |
  |---|---|
  | 256 near-identical requests, stored as a cluster | **329 KB** |
  | …stored as 256 full copies | 8.26 MB → **25× less memory** |
  | cost to add one conversation turn | +512 B vs +32 KB copy → **63× cheaper** |

- **Meaning.** The shared prompt is stored *once* and amortized across the batch — the prompt-cache / dedup win.
- **Caveat.** Sparse diffs cost 8 bytes/token, so a long *unique* suffix is better as a plain array; the win is the
  *shared* part, not the unique part.

---

## The scoreboard

| Milestone | State | Headline |
|---|---|---|
| M0 harness + frozen reference | ✅ | golden files + reproducibility harness |
| M1 `access` (device-native) | ✅ | bit-identical, 1.23 B reads/s |
| M2 addressability frontier | ◐ | price-of-addressability measured (Python binding remains) |
| M3 `rank` + two-level directory | ✅ | bit-identical, 1.6–1.8× faster |
| M4 RRR rank + wavelet + rANS | ✅ | 5.80 b/tok (< `H₀`), searchable; rANS/Huffman crossover mapped |
| M5 native builder + GPU suffix array | ◐ | no-Python build; GPU==CPU bit-identical; SA build 21–23× |
| M6 fusion (matmul, KV, sparse) | ✅ | 10.6× / 7.7–8.3× / 17.8× — both thesis branches, + one honest negative |
| M7 FM-index `count`/`locate` | ✅ | bit-identical; ~19 M pattern-searches/s |
| M8 delta clusters (dedup) | ◐ | 25× less memory, 63× cheaper/turn (runtime metrics need M9) |
| M9 real-model proof | ◐ | integration built (llama.cpp compressed-KV backend + RC1 qualification); a validated hardware **PASS** is the remaining gate |
| M10 Rust wrapper · M11 PTX tuning | ☐ | deliberately last |

**The finish line (M9)** is the only result that would prove the thesis on a real workload: a live model fitting a
longer context or bigger batch in the same GPU memory at equal-or-better latency. The integration to do this
exists — a llama.cpp **compressed paged-KV backend**, a `libchromofold.so` package, and an **RC1 release
qualification** workflow — mapped in [`INTEGRATION.md`](INTEGRATION.md). Its CPU-side contracts pass; the remaining
gate is a **validated PASS on real hardware** (a machine that owns the GPU + a weight model + the running serving
runtime). On a box without those, the qualification honestly reports **INCOMPLETE** rather than fabricating a
pass — which is exactly what it did when run here.
