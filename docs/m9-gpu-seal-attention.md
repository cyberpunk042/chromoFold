# M9 transactional GPU sealing and compressed attention

This round closes the fail-closed boundary introduced by the device KV dataplane. It adds a correctness-first CUDA implementation for page validation, signed-int4 quantization, deterministic four-bit packing, candidate validation, page-table publication, active-page rollover, and compressed attention over sealed pages plus the active tail.

## Support envelope

- CUDA device memory
- FP16, BF16, and FP32 query tensors
- FP32 active-page storage
- causal attention
- MHA and integer-ratio GQA
- page sizes selected by the parent dataplane
- deterministic signed-int4 symbols
- fixed four-bit codebook represented as nibble packing
- one caller stream per operation
- no dense fallback

The fixed four-bit format is not adaptive entropy coding. Adaptive histograms and canonical variable-length codebooks remain later work.

## Transactional seal

A full active page follows this state machine:

```text
ACTIVE_FULL
  -> VALIDATING
  -> QUANTIZING
  -> PACKING
  -> PUBLISHING
  -> ACTIVE_FILLING
```

Publication is allowed only after the candidate owns valid K/V payloads, scales, metadata, and a replacement active page. Any failure returns the layer to `ACTIVE_FULL`; the existing page remains intact and the descriptor-table generation is unchanged.

The API includes deterministic fault injection for allocation, validation, quantization, packing, descriptor validation, page-table growth, publication, and replacement-page allocation.

## Quantization contract

Each block uses a symmetric signed-int4 representation:

```text
scale = max(abs(values)) / 7
symbol = clamp(round(value / scale), -8, 7) + 8
```

K and V have independent scales. NaN and Inf values are rejected before quantization. Saturation is counted and reported.

## Packing contract

Two four-bit symbols are packed into each byte. The resulting storage preserves the fixed-codebook descriptor contract but does not claim adaptive Huffman compression.

## Attention contract

The initial kernel is deliberately correctness-first. It:

- decodes sealed pages directly from device payloads;
- reads the uncompressed active tail directly from device memory;
- applies causal and optional sliding-window filtering;
- maps query heads to KV heads using the configured GQA ratio;
- combines all visible tokens through online softmax;
- writes FP16, BF16, or FP32 strided output;
- never executes a dense fallback after ChromoFold selection.

Performance optimization is not claimed by this round. The reference kernel is a proof path for end-to-end correctness and sanitizer validation.

## Required evidence

A real hardware artifact may claim compressed execution only when it reports:

```text
seal_successes > 0
descriptor_publications == seal_successes
compressed_attention_launches > 0
sealed_pages_consumed > 0
sealed_values_consumed > 0
dense_fallback_launches == 0
cuda_errors == 0
```

It must also report numerical error against a dense reference and actual resident bytes for payloads, scales, descriptors, workspaces, and active pages.

## Hardware validation

Run:

```bash
make -f m9-gpu-seal-attention.mk gpu-seal-contract
make -f m9-gpu-seal-attention.mk gpu-seal-sanitize
```

The self-hosted workflow is responsible for real kernel execution. Hosted CI proves compilation and source-level contract anchors only.

## Remaining work

- real pinned llama.cpp call-site invocation of the seal and attention APIs;
- real GGUF paired evidence;
- page descriptor offsets/LUTs for variable-length coding;
- multi-stream retirement queues;
- optimized warp-cooperative attention;
- adaptive codebooks;
- production sequence and server semantics.
