# M9 — Real inference runtime integration

## Status

Proposed implementation contract for the first end-to-end ChromoFold workload proof.

## Goal

Integrate the native compressed-KV path into one real inference runtime and prove that, at a fixed VRAM budget, it enables a longer context or larger batch while preserving usable generation quality and reporting the latency trade-off honestly.

The first target runtime is **llama.cpp**. The integration is optional, CUDA-only, and disabled by default.

## Ship criterion

M9 is complete only when a reproducible run demonstrates all of the following:

1. The same model and prompt complete with the baseline KV cache and ChromoFold.
2. ChromoFold fits a longer context or larger batch in the same measured VRAM budget.
3. Historical KV remains compressed through the attention consumer; no full dense historical KV buffer is materialized.
4. The query path uses caller-owned CUDA streams without host synchronization or PCIe round trips.
5. Output quality and numerical error are measured, not asserted.
6. Peak VRAM, prefill throughput, decode throughput, TTFT, and per-token latency are emitted as machine-readable JSON.

## Deliberate non-goals

M9 does not include vLLM, TensorRT-LLM, ROCm, multi-GPU, a Rust wrapper, Python packaging, FM-index-assisted generation, a generic compressed-tensor framework, or a new quantizer.

## Runtime architecture

### Two-tier KV lifecycle

Inference KV is append-heavy while ChromoFold query views are immutable. M9 therefore uses two tiers:

- **active page** — appendable quantized or dense K/V for the newest tokens;
- **sealed pages** — immutable entropy-coded K/V resident on the GPU.

When the active page reaches its configured token capacity, the runtime encodes it, publishes an immutable page descriptor, and starts a new active page. Page capacity is benchmark-selected; initial candidates are 128 and 256 tokens.

### Page identity

Every page is identified by sequence, layer, KV head, and logical token range. Page boundaries must not change numerical results beyond the declared tolerance.

### Attention execution

The production path must not use a fixed-size score array. Each page produces online-softmax state:

- local maximum score;
- local exponential sum;
- weighted value accumulator.

Page states are merged with the standard numerically stable online-softmax recurrence. The active page participates through the same merge contract, even if its values are not entropy-coded yet.

### GQA and batching

The first supported model must use grouped-query attention. The runtime descriptor therefore distinguishes query heads from KV heads and maps each query head to a KV head. Batch and sequence identities are explicit; no global mutable singleton is permitted.

## Proposed public runtime surface

The existing low-level `cf_kv_attn_*` functions remain kernel seams. Runtime consumers should use a higher-level page/cache API so they do not directly assemble Huffman words, offsets, lookup tables, and scales.

```c
typedef struct cf_kv_page_view {
    const uint32_t *k_words;
    const int32_t  *k_block_offsets;
    const int32_t  *k_decode_lut;
    const uint32_t *v_words;
    const int32_t  *v_block_offsets;
    const int32_t  *v_decode_lut;
    const float    *k_scales;
    const float    *v_scales;

    uint32_t token_begin;
    uint32_t token_count;
    uint32_t head_dim;
    uint32_t block_size;
    uint32_t k_max_code_len;
    uint32_t v_max_code_len;
    uint32_t flags;
} cf_kv_page_view;

typedef struct cf_kv_attention_desc {
    uint32_t batch_size;
    uint32_t query_count;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t causal_window;
    float    scale;
} cf_kv_attention_desc;
```

The implementation PR may adjust field order for ABI alignment, but must preserve the ownership and execution contract.

## Ownership rules

- All page payload pointers are device pointers.
- The page descriptor array may be host or device resident only if the API names this explicitly; v0 should choose one and not infer.
- ChromoFold does not free caller memory.
- Successful asynchronous calls mean launch acceptance, not completion.
- The caller owns stream ordering and lifetime.
- Cache destruction must be safe after caller synchronization and must release every allocation it owns.

## Error model

Runtime APIs must distinguish:

- invalid arguments;
- unsupported model/layout;
- capacity exhaustion;
- malformed encoded page;
- CUDA launch/runtime failure;
- numerical validation failure in test-only paths.

No unsupported configuration may silently fall back while still reporting ChromoFold as enabled.

## llama.cpp integration boundary

The adapter lives under `integrations/llama.cpp/` and is built only when explicitly enabled. It should expose one runtime option initially:

```text
--chromo-kv
```

Additional options may include page size and causal window, but unsupported combinations must fail clearly.

The adapter owns:

- translating llama.cpp tensor/layout metadata;
- allocating the active page;
- sealing full pages;
- invoking paged attention;
- reporting memory statistics;
- resetting per-sequence state.

The core library owns:

- page format validation;
- encoding/decoding primitives;
- online-softmax page merging;
- compressed attention kernels;
- stable status codes.

## Baseline and experiment matrix

Freeze these fields in every result:

- ChromoFold commit;
- llama.cpp commit;
- model filename and SHA-256;
- GPU name and compute capability;
- CUDA driver/runtime versions;
- context size;
- batch size;
- page size;
- causal window;
- warmup and measured token counts;
- baseline and ChromoFold peak VRAM;
- TTFT;
- prefill tokens/s;
- decode tokens/s;
- p50/p95 decode latency;
- numerical error and generation-quality check.

Minimum context sweep: 4K, 8K, 16K, 32K, 64K where supported.

Minimum capacity experiments:

1. Maximum context at a fixed VRAM budget.
2. Maximum parallel sequence count at a fixed context and VRAM budget.

## Correctness gates

Before end-to-end generation:

1. Encode then reconstruct every page and compare with the quantized source.
2. Compare paged compressed attention with a dense reference over multiple page sizes and boundary positions.
3. Test empty cache, partially filled active page, one sealed page, many sealed pages, reset, and reuse.
4. Test GQA mappings and multiple batches.
5. Run Compute Sanitizer on a small deterministic fixture.
6. Verify that no full historical dense K/V allocation appears in the ChromoFold path.

## PR decomposition inside this large scope

This foundation PR establishes the contract and tooling. Subsequent implementation commits on the M9 line should proceed in this order:

1. page descriptors and validation;
2. cache lifecycle and memory accounting;
3. online-softmax reference implementation;
4. CUDA paged compressed attention;
5. llama.cpp adapter;
6. end-to-end benchmark evidence.

## Exit conditions

M9 is not complete because an adapter compiles or a prompt generates. It is complete only when the committed benchmark artifacts support the workload claim and can be reproduced on a second compatible NVIDIA machine.
