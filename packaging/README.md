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
- **`conformance.c`** — the **FFI-signature seam**: links the `.so` and asserts every entry point is exported,
  the signatures match, and the NULL-argument error contract holds — **without a GPU or driver** (the argument
  guards return `CF_ERR_INVALID_ARGUMENT` before any CUDA call).
- **`seam_check.c` + `fixtures/tiny.cf{wv,rr,rw}`** — the **`.cfold`-header seam**: small committed reference
  fixtures (1–10 KB, golden-verified) and a pure-CPU reader that validates each one's stable header prefix
  (4-byte magic + `u32` version) with **no CUDA at all**.

Together these are SDD-500 Q-500-C's "pure seams" (descriptor parse via the JSON, FFI-signature, `.cfold`
header) — everything a CI job on a box **without** SAIN-01 hardware can run.

- **`functional.cu` + `fixtures/tiny.cffm`** — the **GPU functional run** (hardware-gated half of Q-500-C):
  loads a committed FM-index fixture, builds a `cf_fm_view`, and calls `cf_fm_count` / `cf_fm_ranges` /
  `cf_fm_locate` **through the shared library**, verifying the results bit-identical to the fixture's golden.
  This proves the packaged `.so` *does compressed-domain search correctly* across the FFI boundary — not just
  that its symbols link. Requires an NVIDIA GPU.

## Build
```sh
make lib          # -> build/libchromofold.so   (nvcc; ARCH?=sm_75, override for other GPUs)
make conformance  # link the FFI-signature seam test + run it (no GPU)
make seams        # conformance + the .cfold-header seam over the committed fixtures (no GPU) — the full seam suite
make functional   # GPU: FM-index count/locate through the .so, verified vs golden (hardware-gated)
```

## ABI at a glance (v0)
- **Device-native**: all view pointers are device pointers; queries run on the caller's `cudaStream_t`
  (passed as `void*`, `NULL` = default). No host allocation, no implicit sync, no PCIe round trip.
- **Views** (`cf_wavelet_view`, `cf_rrrw_view`, `cf_fm_view`) are POD / `#[repr(C)]`-safe, passed by value.
  Verified sizes on x86-64: 48 / 96 / 144 bytes.
- **Errors**: `CF_OK=0`, `CF_ERR_INVALID_ARGUMENT=1`, `CF_ERR_UNSUPPORTED=2`, `CF_ERR_CUDA=3`.

Non-goals here (per SDD-500): the Rust `-sys`/wrapper crates, the cockpit panel, and the osctl verb live in
sovereign-os; this side only provides the linkable library, the header, the descriptor, and the seam test.
