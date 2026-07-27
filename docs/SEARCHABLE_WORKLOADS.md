# Where ChromoFold is unique: search-while-compressed (workload fit)

> Companion memo (visual): **[ChromoFold and the O(n) wall](https://claude.ai/code/artifact/9137a101-4076-4afa-a822-ae71ce1c69f2)** —
> the strategic framing of this doc (coefficient vs. exponent; substrate not algorithm). Indexed in [`ARTIFACTS.md`](ARTIFACTS.md).
>
> Downstream, verified: **[SOVEREIGN_SEARCH_INTEGRATION.md](SOVEREIGN_SEARCH_INTEGRATION.md)** — sovereign-os
> committed to bind this search lane *first* (SDD-400 Lane A); the ABI matches and `libchromofold.so` serves
> `cf_fm_count`/`locate` bit-identically through the shared library. The search direction is the real integration.

Follows the KV finding ([`../integrations/llama.cpp/runtime/KV_BACKEND_FINDINGS.md`](../integrations/llama.cpp/runtime/KV_BACKEND_FINDINGS.md)):
llama.cpp already ships quantized + fused-decode KV, so KV *compression* is commoditized. ChromoFold's genuinely
non-redundant capability is its **searchable succinct index** (M1–M7, all bit-exact). This doc maps that capability
to LLM workloads and is honest about where it wins and where existing baselines already suffice.

## The measured, unique capability (RTX 2080 Ti, `make fm-search`)
FM-index count + locate over the entropy-sized RRR-wavelet BWT, verified **BIT-IDENTICAL** to naive ground truth:

| corpus | ChromoFold FM-index (searchable, VRAM) | plain suffix array (same search) | raw tokens | throughput |
|---|---|---|---|---|
| **2,000,000 tokens**, vocab 64 | **1.45 MB** | 2M·4 = **8 MB** | ~1.5 MB (6 b/tok) | count 19 M pat/s, locate 0.6 µs |

**The headline: arbitrary-pattern search (count + locate) over 2M tokens in 1.45 MB — ~5.5× smaller than the
uncompressed suffix array that supports the same search, and GPU-resident.** This is the FM-index's classic win
(succinct self-index < SA), delivered on-GPU, entropy-sized, decoded-in-kernel. llama/vLLM have no compressed
searchable token index — their search structures are uncompressed.

## Candidate LLM workloads, with honest baselines
1. **N-gram / prompt-lookup speculative-decoding drafts** *(shipped in llama: `--spec-type ngram-*`)*. Find the
   longest suffix of the current context that occurred earlier → propose its continuation as the draft.
   - Baseline: an **uncompressed** hash map of fixed-order n-grams → next token. Fast, but fixed order and grows
     with the corpus.
   - ChromoFold: FM backward-search finds the **longest** match (any length) over a **compressed** index; `locate`
     yields the continuation. Win when the draft corpus is large / must fit VRAM / benefits from variable-length
     matches. Honest caveat: for small contexts a hash map is simpler and plenty fast — the win is at scale.
2. **Prefix-cache / RadixAttention dedup across requests** *(vLLM: radix trie on raw token ids)*. Find the longest
   shared prefix of a new prompt against a pool of cached prompts.
   - Baseline: **uncompressed** radix trie / hash of token sequences.
   - ChromoFold: FM-search over a compressed corpus of prior prompts → longest match + location, in a fraction of
     the memory. Win for large shared-prefix pools kept resident.
3. **Long-context retrieval / "needle" search** *(agentic long-context, RAG-in-context)*. Locate occurrences of a
   pattern in a very long context already resident on GPU.
   - Baseline: linear scan, or an uncompressed index rebuilt per context.
   - ChromoFold: count/locate over the compressed context index without leaving VRAM.

## Honest assessment (P7/P10)
- **Where it clearly wins:** any workload that needs **arbitrary-length pattern search over a large token corpus
  that must stay resident in VRAM**, at less memory than an uncompressed index. That's the succinct-index thesis,
  and it's real and measured.
- **Where existing baselines already suffice:** small/fixed-order n-gram lookup with a hash map (fast, simple) —
  ChromoFold's edge there is only memory, and only at scale. Don't oversell it.
- **The proof must be a workload llama/vLLM can't already serve well** (P10). The strongest such case is #1/#2 at
  **scale** (large draft/prefix corpora kept GPU-resident) — where "compressed + searchable + on-GPU" is
  categorically different from an uncompressed hash/trie, not just incrementally cheaper.

## Demonstration — built and measured (`make spec-draft`, `tests/spec_draft_demo.cu`)
The compressed n-gram speculative-draft lookup is now a running demo: build the FM-index over a 200k-token corpus,
and for 4000 query suffixes do FM backward-search → `locate` → argmax next-token draft, vs an **uncompressed hash
n-gram** baseline. Both predict from *prior* occurrences only (the query's own occurrence excluded — the honest
fair comparison). RTX 2080 Ti, vocab 64:

| context L | draft hit-rate (FM == hash) | FM-index (ONE index, ANY L) | hash order-L: min .. real-histogram |
|---|---|---|---|
| 2 | 0.601 | 215 KB | 12 KB .. 1.0 MB |
| 4 | 0.578 | 215 KB | 185 KB .. 9.5 MB |
| 8 | 0.529 | 215 KB | 1.1 MB .. 33 MB |

- **Correctness:** FM occurrence sets == brute-force corpus scan on every gated query; **FM predictions bit-match
  the hash baseline** (identical hit-rates) — the FM-index implements the n-gram lookup exactly.
- **The win:** *one* 215 KB compressed GPU-resident index serves **every** order L (and `locate` + arbitrary-length
  patterns); a hash table is fixed-order and grows with L — at L=8 the FM index is **5× smaller than even a
  charitable argmax-only hash** and **150× smaller than the real histogram map**.
- **Honest caveat (kept):** for a single *small* fixed order with an argmax-only packed encoding, a hash is tinier
  (L=2: 12 KB vs 215 KB). ChromoFold's edge is generality (any L, one index), the richer `locate`/count capability,
  and GPU-residency at scale — not raw size at a small fixed order.

This is the honest shape of the searchable thesis: same accuracy, one compressed index for any pattern length,
GPU-resident — a capability profile llama's quantized KV (or a per-order hash) does not offer.

## Index footprint, measured honestly (`build_index` memory line)

`build_index --fm out.cffm` now prints an authoritative memory breakdown separating the **shippable searchable
index** (RRR-wavelet BWT + C-table + sampled SA) from the **golden test vectors** the `.cffm` also carries. The
distinction matters: the `.cffm` *file* is dominated by those test vectors, so **file size is not index size**
(e.g. n=1M: 850 KB index vs 2.4 MB of golden occurrence positions in the same file). The demo's `fm_bytes` and the
sovereign `HostFmSearch` both load only the index arrays — never the golden vectors.

Index-only footprint vs the raw int32 token stream it replaces (vocab 64, sa=1/16, `make build/build_index`):

| corpus n | searchable index | vs raw int32 | b/token |
|---|---|---|---|
| 2 000 | 2.4 KB | 3.37× | 9.50 |
| 50 000 | 43 KB | 4.64× | 6.89 |
| 1 000 000 | 850 KB | 4.71× | 6.80 |

**The honest read (P7):** "4.7× smaller than raw" uses a *weak* baseline — int32 spends 32 bits on a 6-bit token
(`log₂64`). Against a **bit-packed** raw stream (6.0 b/tok) the self-index is ~6.8 b/tok, i.e. **~0.8 b/tok (~13%)
larger** — and that packed stream is **not searchable** (no count/`locate` without decompressing and scanning). So
the index's real claim is *searchability at roughly packed-raw footprint*, converging to ~6.8 b/tok, **not** a raw
size win. The genuine size win is against an *uncompressed* search structure (a plain suffix array is n·4 = the raw
int32 size again, ~5× the index; a per-order hash n-gram map is far larger — see the table above). The SA sampling
rate is the knob: denser (sa=1/8) costs more bits but locates faster; sparser trades the other way.

## The compute↔memory frontier of the index (`CF_RRR_S` sweep, measured)

The RRR rank sample rate `CF_RRR_S` (blocks per superblock) is the index's core compute↔memory dial: a backward-search
`count` runs ~`len × 2 × levels` two-level RRR rank queries per pattern, and each rank **scans up to `CF_RRR_S`
in-superblock blocks** (then decodes each). Denser samples (small `S`) shorten that scan → faster search, but the rank
directory grows → larger index. Swept end-to-end (n=200k, vocab 64, sa=1/16; each point rebuilds `build_index` + the
`.so`, gated on `parity_smoke` = provenance-A == provenance-B == oracle; device-native search @ batch 4096, RTX 2080 Ti):

| `CF_RRR_S` | index b/tok | GPU search ms | correctness |
|---|---|---|---|
| 16 | 7.56 | 1.118 | A==B==oracle |
| 32 | 7.07 | 1.190 | A==B==oracle |
| **64 (default)** | **6.82** | **1.359** | A==B==oracle |
| 128 | 6.70 | 1.743 | A==B==oracle |

**Two honest findings (P7/P10):**
1. **The frontier is asymmetric.** Over an 8× range of `S` the index moves only **7.56→6.70 b/tok (−11%)** while
   latency moves **1.118→1.743 ms (+56%)**. Memory is *insensitive* to `S` (the RRR class/offset streams dominate the
   index; the rank directory is a thin slice), but latency is *sensitive* (scan length ∝ `S`). So `S` is a latency
   fine-tune, barely a size knob. `S=32` (7.07 b/tok, 1.19 ms) is arguably a better default than 64 if search latency
   matters — 12% faster for 3.7% more memory — but that's the consumer's call.
2. **You cannot tune out of the ~1 ms floor with `S`.** Even the densest point (`S=16`) is 1.12 ms. The floor is the
   entropy decode itself — each backward-search step does `2 × levels` RRR rank scans (two `cf_rrrw_rank_one`
   interval queries, for `lo` and `hi`), and `S` only shortens each *scan*, not the *number* of scans. The lever is
   **fewer scans**, not sampling. This is P1 (compute-for-memory) with its price tag shown: the index stays ~6.8 b/tok
   and searchable, and that costs ~1–1.7 ms of GPU decode per batch, tunable but not eliminable via sampling.

   *Attempted and refuted (honest negative).* The obvious "fewer scans" move — replace `C[c] + rank(c,i)` with a
   single-position wavelet descent of `i` (halving the rank1 scans per step) — assumes the image of position 0 down
   symbol `c`'s path equals `C[c]`. It does **not** in this whole-level-bitvector layout: the descent lands at
   `C[c] + rank(c,i) + δ_c` for a per-symbol offset `δ_c` that cancels in `hi − lo` (so `count` stays correct) but
   corrupts the *absolute* `lo`/`hi` (so `ranges`/`locate` break). The `parity_smoke` gate caught exactly this —
   count PASS, locate FAIL — and it was reverted. The correct single-descent form needs a precomputed
   `D[c] = C[c] − lf(c,0)` table in `cf_fm_view`, i.e. an **ABI/format change** — real, but unjustified while no
   consumer needs sub-ms compressed search. Recorded so it is not re-attempted naively.
