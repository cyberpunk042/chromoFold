# Where ChromoFold is unique: search-while-compressed (workload fit)

> Companion memo (visual): **[ChromoFold and the O(n) wall](https://claude.ai/code/artifact/9137a101-4076-4afa-a822-ae71ce1c69f2)** —
> the strategic framing of this doc (coefficient vs. exponent; substrate not algorithm). Indexed in [`ARTIFACTS.md`](ARTIFACTS.md).

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
