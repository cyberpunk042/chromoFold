# M9 — Page sealing and paged-attention reference

## Purpose

This milestone closes the gap between the non-owning runtime descriptors and the future CUDA/llama.cpp path. It provides an executable reference for the two lifecycle operations that must be correct before GPU optimization:

1. seal an appendable dense KV tail into an immutable quantized page;
2. attend across any number of sealed pages plus the still-active tail without changing softmax semantics at page boundaries.

## What this round implements

- An owning `SealedKvPage` C++ reference type.
- Deterministic symmetric signed-int4 packing with two values per byte.
- K scaling per channel and V scaling per token, matching the existing M6/M9 orientation.
- Dequantization for correctness comparisons.
- Paged causal attention over sealed pages and active tails.
- Numerically stable online-softmax accumulation.
- Grouped-query attention mapping from query heads to KV heads.
- Optional causal windows.
- Page-order invariance and overlap rejection.
- Diagnostics separating sealed and active values consumed.
- Sanitizer-backed tests.

## Explicit boundary

The reference encoder intentionally stops at packed int4. It does not pretend to be the final block-Huffman page encoder. Entropy coding is a second transformation over the already frozen quantized-value and scale orientation.

This separation is deliberate:

```text
active fp16/fp32 KV
        |
        v
per-channel K + per-token V quantization
        |
        v
signed-int4 reference page       <- implemented here
        |
        v
block-Huffman device page        <- next GPU/codec round
```

The CUDA implementation must reproduce the output of this reference after accounting for the same quantization values. It must not normalize each page separately.

## GQA contract

For `Hq` query heads and `Hkv` KV heads, `Hq` must be divisible by `Hkv`. Query head `h` maps to:

```text
kv_head = floor(h * Hkv / Hq)
```

This is tested with four query heads and two KV heads.

## Causal and active-tail contract

For query token `q`, visible tokens satisfy:

```text
earliest <= token <= q
```

where `earliest` is zero for full history or `q + 1 - window` for a bounded window. Sealed history and the active tail participate in one shared online-softmax state.

## What remains

- Device allocation and ownership.
- CUDA page sealing and block-Huffman encoding.
- Device-visible page descriptor packing.
- CUDA paged attention that decodes values inside the consumer.
- Multi-layer launch orchestration.
- llama.cpp adapter and cache hooks.
- End-to-end GPU capacity and latency evidence.

## Acceptance criteria for the next CUDA round

The CUDA path must:

- accept the same logical page boundaries;
- support GQA and active-tail merging;
- match the reference output within a declared tolerance;
- remain invariant to page ordering;
- perform no host allocation or synchronization in the query path;
- avoid materializing historical dense KV;
- expose exact resident-byte accounting.
