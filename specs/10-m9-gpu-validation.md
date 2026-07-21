# M9 GPU validation and evidence

## Scope

This round turns the paged CUDA attention kernel into a reproducible hardware experiment. It adds the missing host pipeline, ownership boundary, numerical harness, sanitizer target, and machine-readable benchmark output.

## Deterministic fixture codec

The initial GPU fixture codec uses signed int4 quantization with a canonical fixed-length Huffman table:

- symbols are in `[0, 15]` with zero point 8;
- every symbol has a four-bit canonical code;
- the decoder LUT has 16 entries;
- streams are packed MSB-first;
- block offsets are explicit bit positions;
- K scales are per channel;
- V scales are per token.

This is deliberately deterministic. It proves the page contract and GPU decoder without introducing adaptive-codebook variability. It is not claimed to achieve the entropy ratio of the future adaptive builder.

## Ownership

`DeviceKvPage` owns all CUDA allocations referenced by one `cf_kv_device_page`. `DeviceKvPageArray` owns the device descriptor array. Both are move-only RAII types.

Upload is asynchronous with respect to the caller stream, but destruction currently uses `cudaFree`; callers must preserve stream lifetime and ordering before releasing pages.

## Validation commands

```bash
make -f m9-gpu.mk gpu-fixture-test
make -f m9-gpu.mk gpu-correctness
make -f m9-gpu.mk gpu-sanitize
make -f m9-gpu.mk gpu-benchmark
```

The default hosted workflow runs the CPU fixture suite and compiles both GPU binaries in a CUDA 12.6 development container. Hardware execution is isolated to the manually dispatched self-hosted runner labelled:

```text
self-hosted, linux, x64, cuda, chromofold-gpu
```

## Correctness threshold

The first hardware test compares a 128-token, 64-dimensional compressed page against a CPU attention calculation over the exact dequantized fixture values. It fails when maximum absolute error exceeds `2e-4`.

The executable emits JSON containing maximum absolute error, MSE, token count, and head dimension.

## Benchmark evidence

The benchmark emits JSON with:

- GPU name and compute capability;
- token and head dimensions;
- dense KV bytes;
- compressed page bytes;
- compression ratio;
- mean synchronized kernel latency.

These results are isolated-kernel evidence only. They are not an end-to-end inference, throughput, TTFT, or maximum-context claim.

## Required follow-up before llama.cpp integration

- expand correctness to page sizes 64, 128, and 256;
- cover multiple pages, GQA, active tails, bounded windows, and shuffled descriptors;
- add adaptive length-limited Huffman tables;
- record hardware artifacts on at least one supported NVIDIA GPU;
- validate 64K token indexing and descriptor scaling;
- decide whether page destruction requires stream-ordered `cudaFreeAsync`.
