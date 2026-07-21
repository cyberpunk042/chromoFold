# M9 paged CUDA attention

## Purpose

This round introduces the first CUDA consumer for the paged KV runtime. It consumes immutable block-Huffman pages and an optional dense active tail without materializing historical dense K/V buffers.

## Execution model

`cf_kv_paged_attention_async` launches one CUDA thread for each `(query token, query head)` pair.

Each thread:

1. maps its query head to a KV head using `gqa_group_size`;
2. computes the visible causal interval;
3. scans matching immutable pages;
4. decodes K values in-place and computes a score;
5. decodes the matching V row;
6. merges that row into a numerically stable online-softmax state;
7. repeats for the active dense tail;
8. writes one output head.

There is no standalone decode kernel and no dense historical KV allocation.

## Public boundary

`include/chromofold/kv_cuda.h` defines two POD structures:

- `cf_kv_device_page`: one device-resident compressed page for one KV head;
- `cf_kv_paged_attention_desc`: one launch over a layer, query batch, and optional active tail.

The launch descriptor is copied by value into the kernel. Its page array and all tensor/page payload pointers are device pointers.

The API is asynchronous and performs no allocation, synchronization, or transfer.

## Numerical contract

All visible tokens, regardless of page or active-tail membership, contribute to one online-softmax state:

- running maximum;
- running exponential sum;
- running weighted V vector.

Pages are never normalized independently. This is the same invariant frozen by the CPU reference in `kv_reference.cpp`.

## GQA mapping

For query head `h`:

```text
kv_head = h / gqa_group_size
```

The descriptor is valid only when:

```text
query_head_count == kv_head_count * gqa_group_size
```

## Current implementation envelope

- NVIDIA CUDA only;
- fp32 queries, active K/V, accumulation, and output;
- head dimension at most `CF_KV_MAX_HEAD_DIM`;
- at most `CF_KV_CUDA_MAX_PAGES_PER_LAUNCH` page descriptors;
- one logical layer per launch;
- host validation checks descriptor shape but cannot inspect device page contents;
- page ordering does not affect normalization, but callers remain responsible for non-overlap and page metadata integrity;
- kernel is correctness-first and not yet warp-cooperative or tensor-core optimized.

## Validation layers

### No-GPU CI

The CUDA development container:

- compiles the translation unit with warnings as errors;
- links the public ABI;
- executes host-side descriptor validation without launching a kernel.

### Required GPU validation

The next hardware-gated benchmark must compare this kernel to the merged CPU paged-attention reference across:

- head dimensions 32, 64, 128;
- MHA and GQA topologies;
- page sizes 64, 128, 256;
- one to many pages;
- active-tail sizes from zero to a full page minus one;
- causal windows and unbounded history;
- shuffled non-overlapping page descriptor order;
- extreme score ranges;
- sequence lengths through at least 64K.

Report maximum absolute error, MSE, decoded values consumed, launch latency, and resident page bytes.

## Non-claims

This round does not yet provide:

- a production page encoder;
- host/device page ownership helpers;
- a llama.cpp adapter;
- multi-layer scheduling;
- a throughput or capacity claim;
- a tuned serving kernel.

It establishes the device execution contract and compile-validated CUDA implementation that the next GPU correctness harness will exercise.
