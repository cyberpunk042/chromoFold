# ChromoFold Native Engine — Architecture

> **How** the engine is built. Governed by [00-constitution.md](00-constitution.md); realizes
> [01-spec.md](01-spec.md); sequenced by [03-roadmap.md](03-roadmap.md).

## 1. The layered stack (P6)

```
                 Python API / experiments / notebooks
                              │  (pybind11)
                              ▼
                 ┌──────────────────────────────┐
                 │        C++20 core library     │   format · builders · views ·
                 │     + stable, versioned C ABI │   dispatch · ownership · serialization
                 └──────────────────────────────┘
                    │            │             │
        ┌───────────┘            │             └───────────────┐
        ▼                        ▼                             ▼
  CUDA C++ kernels        optimized CPU backend        optional Triton kernels
  (production hot path)   (AVX2 / AVX-512:             (experimental fused,
                           baseline · build · verify)   autotuned)
        │
        └── optional Rust wrapper (safe host: parsing, mmap, corpus, CLI, servers) — added only after
            the C ABI + on-device format stabilize.
```

**Language rationale (from the brief, condensed):**
- **C++20 core** — the practical integration center for CUDA, PyTorch extensions, llama.cpp, TensorRT-class
  runtimes, custom allocators, device streams, and zero-copy tensor interfaces. Owns the binary format, memory
  views, builders, CPU implementation, dispatch, and Python bindings.
- **CUDA C++ hot path** — the only model that exposes what packed-bit succinct structures need: warp shuffles,
  `popc`/`brev`, shared memory, cooperative loading, compile-time specialization, cache-control loads,
  occupancy tuning, architecture-aware profiling.
- **Not plain C** — C would still need CUDA extensions plus hand-rolled, inferior replacements for templates,
  RAII, typed views, and compile-time specialization. Use a C *ABI*, implemented in C++.
- **Rust** — excellent for the safe control plane (parsing, integrity, mmap, corpus construction, async I/O,
  CLI, server glue) and can wrap the C ABI via `cudarc`-style bindings; **not** the first GPU-kernel language.
- **Triton** — an experimental backend for tensor-shaped fused ops (access+gather, batched delta); benchmarked
  against CUDA C++, keep the fastest maintainable kernel per op. Less natural for irregular bit-level rank.
- **PTX / SASS** — targeted inline PTX only on profiler evidence (a missed `popc`/funnel-shift/vector load);
  SASS never as foundation (fragile across GPU generations).

## 2. Directory layout

```
chromofold/
├── include/chromofold/         # public C++ headers + the C ABI header
│   ├── format.hpp   index.hpp   wavelet.hpp   delta.hpp   fm_index.hpp   runtime.hpp
│   └── chromofold.h            # the stable C ABI (C-surface)
├── src/
│   ├── format.cpp  builder.cpp # binary format, offline builders
│   ├── cpu/  wavelet_avx2.cpp  wavelet_avx512.cpp
│   └── cuda/ access.cu  rank.cu  rrr.cu  fm_search.cu  delta_apply.cu  fused_embedding.cu
├── bindings/ python/ (pybind11)   c/ (ABI shim)
├── rust/ chromofold-rs/         # later
├── python/ chromofold/          # research surface, mirrors the Warp prototype API
└── benchmarks/ gpu_access.cpp  raw_gather.cpp  fm_search.cpp  end_to_end_llm.cpp
```

## 3. The device-native API (P5)

The query path consumes and returns **device pointers**, runs on the caller's `cudaStream_t`, and does **no
allocation, no synchronization, no host copy**.

```cpp
struct ChromoFoldView {              // immutable POD descriptor over device-resident index memory
    const uint32_t* bitplanes;
    const uint32_t* superblocks;     // rank directory (see §6)
    const uint32_t* zero_counts;     // per-level zero count (wavelet descent)
    uint64_t        token_count;
    uint32_t        levels;          // = ceil(log2(vocab))
};

void chromofold_access_async(
    ChromoFoldView   index,
    const uint32_t*  device_positions,
    uint32_t*        device_output,
    size_t           count,
    cudaStream_t     stream);        // async; caller owns lifetime + ordering
```

The C ABI (P8) wraps this with status codes and opaque handles so Python/Rust/servers bind without touching the
execution model:

```c
cf_status cf_access(const cf_index* index, const uint32_t* d_positions,
                    uint32_t* d_output, size_t count, cf_stream stream);
```

## 4. Build ≠ query split (P9)

```
BUILD  (offline / CPU / non-serving accelerator)      QUERY / SERVE  (GPU)
  suffix array + BWT construction                       compact immutable ChromoFoldView
  RRR encoding, wavelet layout                          async access / rank / search / fused decode
  reference-delta tree, serialization                   no construction, no host round trip
```

Suffix-array/BWT construction is expensive and need not run on the inference GPU. Eventually GPU construction
can use CUB radix sort + scans (the Warp prototype already builds the suffix array on-GPU at 17–32× CPU), but it
must never block runtime work. Append-heavy workloads prefer reference-delta + chunked indexes over full rebuild.

## 5. Compile-time specialization by vocabulary width

`levels` (= bit depth) is small and known per corpus. Compile specialized traversals and dispatch at runtime;
keep a dynamic fallback.

```cpp
template<int LEVELS> __global__ void access_kernel(/* ... */);   // unrolled, constants propagated
switch (index.levels) {
    case 2:  launch_access<2>(...);  break;   // V ≤ 4
    case 8:  launch_access<8>(...);  break;   // V ≤ 256
    case 15: launch_access<15>(...); break;   // ~32K vocab
    case 16: launch_access<16>(...); break;
    case 17: launch_access<17>(...); break;
    default: launch_access_dynamic(...);
}
```

Specialization lets the compiler unroll levels, remove loop control, propagate constants, and schedule loads.

## 6. Rank directory redesign (two-level)

The prototype stores a cumulative rank every eight 32-bit words and may scan several words per superblock. Test
a **two-level directory** so each rank is roughly one superblock load + one block load + one word load + one
popcount, instead of a serial word scan.

```cpp
struct RankDirectory {
    uint32_t* words;        // packed bitplane
    uint32_t* super_rank;   // cumulative rank every ~2048 bits
    uint16_t* block_rank;   // relative rank every ~256 bits
};
```

Exact block sizes are benchmark-decided; extra directory memory may be worth fewer serialized loads. (The Warp
prototype already validated the *encoding-side* two-level idea — int32 anchors + uint16 deltas — dropping the
resident sample table ~1.9×; this is the CUDA-side latency counterpart.)

## 7. Query mapping (thread- vs warp-cooperative)

Benchmark at least two mappings; select by workload via dispatch.

- **Thread-per-query** — one thread walks all levels. Best for uniform-random, independent queries; no sync.
- **Warp-per-query / warp-per-group** — a warp cooperatively loads and shares the rank directory. Wins when
  queries are sorted / clustered / contiguous / attention-derived-sparse (cache reuse, fewer divergent loads).

NVIDIA GPUs execute 32-thread warps; deliberately warp-cooperative layouts change memory behavior materially.
The optimal mapping depends on whether requests are uniform, sorted, clustered, contiguous, or sparse attention
sets — so it is a **benchmark-driven dispatch decision**, not a fixed choice.

## 8. The thesis: never fully decompress (P3)

The largest architectural win is not an exotic language — it is **eliminating the standalone decompression
pass**. Decompression becomes a private step *inside* the consumer kernel.

```
Conventional:  compressed → decode to temp token array → embedding gather → output
Fused:         compressed index + positions + embedding table
                   → decode token ID in-kernel → gather embedding row → write embedding
               (no full decompressed token buffer ever exists)
```

**Candidate fused operations** (roadmap-ordered): wavelet `access` + embedding gather · RRR `rank` + sparse
value gather · FM backward-search + candidate retrieval · reference-delta decode + adapter apply · compressed
KV decode + attention-page selection · quantized-value decode + dequant + matmul staging.

Why this beats micro-optimization: avoiding a global-memory write, a temporary allocation, a second kernel
launch, and a second global-memory read is a **larger end-to-end gain** than shaving cycles off an isolated
decoder. The compressed representation should survive until the exact op that needs the value.

## 9. CPU backend (not a Python fallback)

A serious engine ships a real CPU implementation in C++ with AVX2 / AVX-512, `std::popcount`, prefetching,
huge-page-friendly layouts, and multithreaded batch queries. It serves as: an honest benchmark baseline (P7),
the offline build path (P9), a CPU-only deployment option, and the fuzzing/verification oracle for the CUDA
kernels.

## 10. Interop surface

The stable C ABI (P8) makes the engine consumable from Python (pybind11), Rust (`cudarc`/FFI), Go, C#,
llama.cpp, TensorRT-class runtimes, and custom inference servers — all riding the same versioned C surface with
a C++ implementation underneath.
