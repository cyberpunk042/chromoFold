# How it works — a guided tour

This walks through chromoFold end to end in plain language: what problem each piece solves, and how they connect.
If a word is unfamiliar, it's defined in [`GLOSSARY.md`](GLOSSARY.md). For the measured numbers, see
[`RESULTS.md`](RESULTS.md).

The whole system is one pipeline:

```
   BUILD  ─────────────▶  STORE  ─────────────▶  USE
   (offline, once)        (compact, in VRAM)      (online, every query)

   raw data               a searchable            read one token,
   → suffix array         compressed index        search for a pattern,
   → BWT                  that stays usable        or feed values straight
   → entropy-coded        while compressed         into a matmul / attention
     wavelet index                                 — decoding only what's touched
```

A guiding rule runs through all of it (the **build ≠ query** split, principle P9): **construction is allowed to be
expensive and offline; queries must be cheap and online.** You build the compact index once, then serve millions
of fast queries from it.

---

## 1. The problem, restated concretely

A GPU has two very different budgets:

- **Compute**: enormous. Trillions of math ops per second, often idle.
- **Fast memory (VRAM)**: small and fixed. This is what runs out.

For an LLM, VRAM fills with the KV cache, the weights, and repetitive token/prompt data. When it's full, you shrink
the model, the context, or the batch. chromoFold's bet: **shrink the *data* instead, and pay for it with the idle
compute.** But naive compression fails, because to *use* compressed data you normally decompress it all first —
recreating the very buffer you were trying to avoid. So we need compression that stays **usable while compressed.**

---

## 2. Making data small *and* readable: the wavelet index

The foundation is a **wavelet tree/matrix** (→ glossary). It stores a sequence of symbols as `log₂(V)` layers of
plain bits, and can answer three questions directly on that form:

- **`access(i)`** — what symbol is at position `i`? (walk down the layers, one bit per layer)
- **`rank(c, i)`** — how many `c`s occur before `i`? (the primitive that powers search)
- and by extension `select`, `count`, `locate`.

That already gives us **navigability while compressed** (principle P2). But a plain wavelet stores each layer as
raw bits — no smaller than a tight naive layout. To actually *shrink*, we compress each layer.

### Squeezing each layer with RRR

Each layer is a bit-vector. We store it with **RRR** (→ glossary): cut it into tiny 15-bit blocks and record each
as `(class, offset)` — "how many 1s, and which arrangement." When the bits are **skewed** (mostly 0s or mostly 1s),
this is far smaller than 15 bits. And we can still `rank` quickly because a small "superblock directory" (→
two-level directory) lets us jump near the answer, then decode just one block.

Why does skew show up? Because we don't index the raw text — we index its **BWT** (→ glossary), a reversible
reshuffle whose bit-layers are skewed *by construction*. The result (milestone **M4**): a searchable index that is
**smaller than the information-theoretic floor of a naive layout** (5.8 bits/token, below `H₀`), yet still supports
instant random access. One object that is *entropy-sized and searchable at the same time* — the core artifact
everything else builds on. Its device code is `cf_rrrw_*` in
[`include/chromofold/detail/rrr_wavelet_device.cuh`](../include/chromofold/detail/rrr_wavelet_device.cuh).

The price, honestly: decoding a compressed block per layer makes `access`/`rank` slower than a plain wavelet
(nanoseconds, not sub-nanosecond). We spend compute to save memory — exactly the trade. (We later halved the
access cost with a one-decode-per-layer trick; see M4 in [`RESULTS.md`](RESULTS.md).)

---

## 3. Two entropy coders, and being honest about which wins

The wavelet is for **search + random access**. But sometimes you just want to **decode a whole array fast** (e.g.
a stream of quantized weight values). For that, chromoFold has two **block coders** — cut the stream into
independent blocks so one GPU thread decodes each:

- **Block-Huffman** (→ glossary): fast, simple, great for small-block random access. But it wastes up to ~1
  bit/symbol.
- **Block-rANS** (→ glossary): reaches much closer to the entropy floor, but carries a fixed per-block cost.

Which is better? *It depends*, and we measured the exact crossover (milestone **M4**): rANS wins decisively on
**low-entropy data with large blocks** (2.2× smaller than Huffman there), and loses elsewhere. We report the whole
table, winners and losers, rather than the one row that flatters rANS. The **block size** is a dial the deployer
sets per data type and access pattern (principle P9).

---

## 4. Searching the compressed index: the FM-index

Once the BWT lives in a `rank`-able wavelet, **full-text search comes almost for free** — that's the whole point of
the FM-index (→ glossary). Two operations (milestone **M7**):

- **`count(pattern)`** — how many times does the pattern occur? Done by **backward search**: start with "anything
  could match," feed the pattern's symbols last-to-first, and each one shrinks a range via `rank`. One GPU thread
  per pattern → search a whole batch at once (~19 million patterns/second).
- **`locate(pattern)`** — *where* are the occurrences? For each one, **LF-walk** (→ glossary) back to a sampled
  landmark to recover the text position.

So the *same compact index* is now **read** (`access`), **searched** (`count`/`locate`), and — via batched count —
used as an **n-gram predictor**, all without ever leaving the GPU or decompressing anything.

---

## 5. The thesis move: fusion (decode *inside* the consumer)

Everything so far keeps data small and navigable. **Fusion** is where it becomes a *runtime*, not just a
compressor (principle P3). The idea: when a GPU computation needs a value, decode it *in registers at that instant*
— so the decompressed data never exists as a memory buffer.

We proved this on three consumers, and — importantly — found the **rule for when it pays**:

- **Fused decode-in-matmul (M6).** Multiply by a compressed weight matrix, decoding each weight inside the multiply.
  The dense weight matrix (megabytes) is never built → **10.6× less VRAM during the compute**, and even *faster* at
  large sizes because it skips writing/reading that huge intermediate.
- **Fused KV-cache attention (M6/M9).** Attention that decodes each Key/Value from the compressed KV cache inline.
  **7.7–8.3× less KV memory** → fit a much longer context. (This is the brick for the final workload proof, M9.)
- **Fused sparse gather (M6/P2).** When you only touch K of N positions, read *just those K* directly from the
  compressed index instead of decompressing all N — up to ~18× faster while K ≪ N.

**The rule (learned from a failure).** Our *first* fusion attempt — on an embedding lookup — **lost**, because the
thing it avoided building (a list of token ids) was tiny. So:

> **Fuse when the avoided intermediate is large (a weight/KV tile) *or* the consumer is sparse (touches a
> fraction). Don't fuse when the intermediate is small.**

Keeping that negative result front-and-center is deliberate (principle P7).

---

## 6. Storing many near-identical things once: delta clusters

A different memory win, for a different shape of data (milestone **M8**). In serving, you often have **many
near-identical sequences**: requests sharing a long system prompt, turns of one conversation, or a library of
fine-tuning adapters. Storing them separately duplicates the shared part N times.

A **reference-delta cluster** (→ glossary) stores **one base + each sequence's small list of differences**.
Reconstructing a token is "use the base, unless this sequence overrides it here" (a quick binary search over its
diffs). For 256 near-identical requests: **25× less memory**, and adding a conversation turn is **63× cheaper**
than duplicating. Honest caveat: sparse diffs cost more per token than a plain array, so the win is the *shared
prefix*, not long unique suffixes.

---

## 7. Building the index — natively, and on the GPU

Queries must be fast; **building** the index can be slow and offline (build ≠ query, P9). Two milestones make the
build self-sufficient:

- **Native C++ builder (M5).** The whole searchable index (suffix array → BWT → RRR wavelet → FM-index) builds in
  plain C++ with **no Python dependency**. A CPU reference implementation doubles as the correctness oracle *and*
  an honest single-thread baseline.
- **GPU suffix-array build (M5).** The expensive step — sorting all suffixes — runs on the GPU (prefix-doubling +
  radix sort), **21–23× faster** than the CPU, and bit-identical (the suffix array is unique, so "correct" means
  "equals the CPU's").

The verification loop is the same throughout: the C++ builder writes the index, the GPU kernels answer queries on
it, and their answers must be **bit-identical** to the CPU oracle's. Build, query, and oracle all agree.

---

## 8. How correctness is guaranteed (why you can trust the numbers)

Every capability follows the same discipline (principle P0, "build ≠ query" P9, "measured not asserted" P7):

1. A **reference generator** (Python prototype, or the native C++/numpy builder) freezes a **golden** file: the
   inputs *and* the exact expected outputs.
2. The **CUDA kernel** runs on those inputs.
3. The benchmark checks the kernel's output equals the golden **bit for bit** — printing `BIT-IDENTICAL ✓` — and
   *only then* reports timing (median over many repetitions, with the GPU/CUDA version noted).

So a speed number in this repo always comes attached to a proof that the fast thing computed the *right* thing.

---

## 9. What's still open

The one milestone that would *prove the thesis on a real workload* is **M9**: wire the fused KV-cache attention
into an actual inference runtime and show a longer context or bigger batch fitting in the same GPU at
equal-or-better latency. Everything above is the machinery; M9 is the payoff, and it needs a GPU-enabled Python/ML
stack to plug into. The current status board is in [`../specs/03-roadmap.md`](../specs/03-roadmap.md).
