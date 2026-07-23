# Where ChromoFold is unique: search-while-compressed (workload fit)

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

## Bounded next demonstration (proposed)
A **compressed n-gram speculative-draft lookup** micro-benchmark: build the FM-index over a realistic (order-k
Markov / real-text) token corpus; for many query suffixes, FM backward-search → `locate` → next-token draft;
compare **draft hit-rate** (vs ground-truth next token) and **memory** against an uncompressed hash-map n-gram
baseline. Success = same/again-better hit rate at a fraction of the memory, GPU-resident. Reuses the verified
`cf_fm_count/ranges/locate` kernels — the primitive is done; this frames it as the workload.
