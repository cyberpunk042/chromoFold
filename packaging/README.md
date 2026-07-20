# packaging/ — `libchromofold` for native (Rust FFI) integration

The outward-facing C-ABI packaging of the ChromoFold native engine, for a native runtime to link — built for
**sovereign-os SDD-500** (`../sovereign-os/docs/sdd/500-chromofold-compressed-domain-integration.md`), whose
`sovereign-chromofold-sys` FFI crate binds this `.so` behind a safe wrapper. Everything here is **additive** — it
does not modify the engine's kernels or the top-level build.

## What's here
- **`../include/chromofold/chromofold.h`** + **`chromofold_search.h`** — the stable C ABI. `chromofold.h` is the
  base (access / rank / embedding-gather); `chromofold_search.h` is the **compressed-domain search** surface:
  the RRR-backed wavelet self-index (`cf_rrrw_access_async` / `cf_rrrw_rank_async`) and the **FM-index**
  (`cf_fm_count_async` / `cf_fm_ranges_async` / `cf_fm_locate_async`) — the net-new capability with no analogue
  in a plain KV/quant stack (SDD-500 Q-500-B: search first).
- **`chromofold_capability.json`** — the committed **capability descriptor**: ABI version, exported functions +
  views, `.cfwv`/`.cfrr`/`.cfrw` fixture formats, the honest-degrade contract (`CHROMOFOLD_ROOT` defaulting to
  `WARP_SHADERS_ROOT`), and the offline guarantee stated as a *contract to test*. This is the source-of-truth a
  consumer commits so a box with no ChromoFold checkout degrades honestly (exit-3), never fabricates success.
- **`conformance.c`** — a **pure-seam** conformance test: it links the `.so` and asserts every entry point is
  exported, the signatures match, and the NULL-argument error contract holds — **without a GPU or driver** (the
  argument guards return `CF_ERR_INVALID_ARGUMENT` before any CUDA call). This is the CI seam for a box without
  SAIN-01 hardware (SDD-500 Q-500-C: hardware steps gated, pure seams CI-tested).

## Build
```sh
make lib          # -> build/libchromofold.so   (nvcc; ARCH?=sm_75, override for other GPUs)
make conformance  # build the .so + link the C seam test + run it (no GPU needed)
```

## ABI at a glance (v0)
- **Device-native**: all view pointers are device pointers; queries run on the caller's `cudaStream_t`
  (passed as `void*`, `NULL` = default). No host allocation, no implicit sync, no PCIe round trip.
- **Views** (`cf_wavelet_view`, `cf_rrrw_view`, `cf_fm_view`) are POD / `#[repr(C)]`-safe, passed by value.
  Verified sizes on x86-64: 48 / 96 / 144 bytes.
- **Errors**: `CF_OK=0`, `CF_ERR_INVALID_ARGUMENT=1`, `CF_ERR_UNSUPPORTED=2`, `CF_ERR_CUDA=3`.

Non-goals here (per SDD-500): the Rust `-sys`/wrapper crates, the cockpit panel, and the osctl verb live in
sovereign-os; this side only provides the linkable library, the header, the descriptor, and the seam test.
