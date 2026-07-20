# packaging/ — `libchromofold` for native integration

The outward-facing C-ABI packaging of the ChromoFold native engine, for a native runtime to link — originally
built for **sovereign-os SDD-500** (`../sovereign-os/docs/sdd/500-chromofold-compressed-domain-integration.md`).
Everything here is **additive** — it does not modify the engine's algorithms or the top-level benchmark build.

## What's here

- **`../include/chromofold/chromofold.h`** — the primary stable C ABI:
  - packed-wavelet `access` / `rank`;
  - fused decode + embedding gather;
  - **compressed-KV causal windowed attention** (`cf_kv_attn_fused_async`), which decodes and dequantizes K/V
    inside the consumer so dense KV tiles never exist;
  - the bit-identical dense-reference path (`cf_kv_attn_dense_async`) with caller-provided scratch buffers.
- **`../include/chromofold/chromofold_search.h`** — the compressed-domain search surface: the RRR-backed wavelet
  self-index (`cf_rrrw_access_async` / `cf_rrrw_rank_async`) and the FM-index (`cf_fm_count_async` /
  `cf_fm_ranges_async` / `cf_fm_locate_async`).
- **`chromofold_capability.json`** — the committed capability descriptor: ABI version, exported functions, views,
  fixture formats, and honest-degrade contract. Consumers can inspect this without loading CUDA code.
- **`conformance.c`** — the **FFI-signature seam**: links the `.so` and asserts every public entry point is exported,
  the signatures match, and the NULL-argument error contract holds — **without a GPU or driver**. This includes
  the compressed-KV APIs; their guards return `CF_ERR_INVALID_ARGUMENT` before any CUDA call.
- **`seam_check.c` + `fixtures/tiny.cf{wv,rr,rw}`** — the `.cfold`-header seam: small committed reference fixtures
  and a pure-CPU reader that validates each stable header prefix.
- **`functional.cu` + `fixtures/tiny.cffm`** — the hardware-gated GPU functional run: FM-index count/ranges/locate
  through the shared library, verified against golden results.

Together these provide both pure CI seams and a GPU functional seam. A future M9 runtime integration should link
only these public headers and `libchromofold.so`; it must not redeclare private symbols from CUDA translation units.

## Build

```sh
make lib          # -> build/libchromofold.so   (nvcc; ARCH?=sm_75, override for other GPUs)
make conformance  # link the public ABI seam test + run it (no GPU)
make seams        # conformance + the .cfold-header seam (no GPU)
make functional   # GPU: FM-index count/locate through the .so, verified vs golden
```

## ABI at a glance (v0)

- **Device-native:** all view/tensor pointers are device pointers; queries run on the caller's `cudaStream_t`
  (`void*`, `NULL` = default). No host allocation, implicit synchronization, or PCIe round trip.
- **Views:** `cf_wavelet_view`, `cf_rrrw_view`, and `cf_fm_view` are POD / `#[repr(C)]`-safe and passed by value.
- **KV v0 bounds:** `dim <= 128`, `window <= 512`, `0 <= nq <= seq`, positive block size. The public header is the
  source of truth.
- **Errors:** `CF_OK=0`, `CF_ERR_INVALID_ARGUMENT=1`, `CF_ERR_UNSUPPORTED=2`, `CF_ERR_CUDA=3`.
- **Asynchronous contract:** a successful call means the launch was accepted; the caller owns stream ordering and
  synchronization.

Non-goals here: a safe Rust wrapper, llama.cpp/vLLM adapter, and production scheduler. Those belong after the ABI
surface is stable and the real-runtime M9 experiment is selected.
