# M9 device-native KV data plane

## Purpose

This milestone moves llama.cpp K/V ingestion from host-oriented fixtures to explicit CUDA device tensors. It introduces stream-ordered active-page ownership and conversion kernels while preserving a strict fail-closed boundary around unvalidated GPU sealing and compressed attention publication.

## Implemented

- CUDA tensor descriptors for FP16, BF16, and FP32;
- explicit token, head, and element strides;
- topology and stride validation;
- caller-stream conversion into FP32 device-resident active pages;
- asynchronous allocation through `cudaMallocAsync` when requested;
- contiguous-token enforcement;
- page-boundary enforcement;
- per-layer active K/V ownership;
- device append and conversion counters;
- memory and peak-owned-byte counters;
- llama.cpp device graph bridge;
- hosted CUDA compilation and contract checks.

## Fail-closed boundary

A full active page is not published as compressed history until hardware validation covers all transactional stages:

1. candidate allocation;
2. quantization;
3. symbol packing;
4. offsets and decoder metadata;
5. descriptor validation;
6. stream-ordered publication;
7. replacement of the active page.

Until those stages exist, `cf_device_kv_seal_full_pages_async` returns an explicit error for a full page. It does not drop active K/V, publish a partial descriptor, increment success counters, or allow dense fallback.

Likewise, `cf_device_kv_attention_async` remains unavailable until the device page table and active-tail merge are connected to the validated paged-attention kernel.

## Required invariants

- No full-page host staging.
- No `cudaDeviceSynchronize` in append, seal, or attention APIs.
- Caller stream ordering is preserved.
- Appends are contiguous by token position.
- A single append cannot silently cross a page boundary.
- Failed sealing retains the active page.
- Dense fallback is always zero when ChromoFold is selected.
- Counters originate from actual CUDA paths.
- Evidence cannot claim compressed execution before positive seal and attention counters exist.

## Hardware completion gates

The PR becomes an end-to-end execution claim only after a self-hosted NVIDIA run proves:

```text
device_append_calls > 0
seal_successes > 0
descriptor_publications > 0
compressed_attention_launches > 0
sealed_values_consumed > 0
dense_fallback_launches == 0
cuda_errors == 0
```

Compute Sanitizer must cover append, transactional sealing, descriptor growth, attention, clear, and destruction with pending stream work.
