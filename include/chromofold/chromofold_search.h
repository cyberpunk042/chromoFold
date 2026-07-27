/* chromofold_search.h — stable C ABI for the RRR-backed wavelet + FM-index compressed-domain SEARCH.
 *
 * Consumer-facing extension of chromofold.h: the entropy-sized, GPU-searchable self-index (M4 RRR wavelet) and
 * FM-index backward search (M7 count / ranges / locate). This is the surface a Rust `-sys` crate or any C caller
 * binds to for compressed-domain search — the net-new capability with no analogue in a plain KV/quant stack.
 *
 * IMPORTANT (single-definition rule): this header re-declares `cf_rrrw_view` / `cf_fm_view` with layouts
 * identical to the engine's internal `detail/*.cuh`. It is for EXTERNAL CONSUMERS only. Do NOT include it from a
 * translation unit that already includes `detail/rrr_wavelet_device.cuh` or `detail/fm_search_device.cuh`
 * (that TU already has the structs) — you would get a redefinition. Callers include only this header + the .so.
 *
 * All pointers are DEVICE pointers; the query path is device-native (no host alloc/sync/copy); `stream` is a
 * cudaStream_t passed as void* (NULL = default stream). Views are POD / #[repr(C)]-safe, passed by value.
 */
#ifndef CHROMOFOLD_SEARCH_H
#define CHROMOFOLD_SEARCH_H

#include "chromofold/chromofold.h" /* cf_status, fixed-width ints */

#ifdef __cplusplus
extern "C" {
#endif

/* RRR-backed wavelet index: every wavelet level is an RRR bitvector with two-level superblock samples.
 * Constants (fixed): T=15-bit blocks, S=64 blocks/superblock, K=32 superblocks/anchor. Layout matches
 * detail/rrr_wavelet_device.cuh exactly. */
typedef struct cf_rrrw_view {
  const uint32_t *classes; /* [bits * cwords]      4-bit block class stream per level, LSB-first */
  const uint32_t *offsets; /* [owords]             flat enumerative-offset stream; per-level base in offbase[] */
  const int32_t *rank_a;   /* [bits * na]          two-level rank sample: int32 anchor every K superblocks */
  const uint16_t *rank_d;  /* [bits * (nsb + 1)]   two-level rank sample: uint16 delta per superblock */
  const int32_t *off_a;    /* [bits * na]          two-level offset-bit sample: int32 anchor */
  const uint16_t *off_d;   /* [bits * (nsb + 1)]   two-level offset-bit sample: uint16 delta */
  const int32_t *offbase;  /* [bits]               bit offset of each level's slice within offsets[] */
  const int32_t *zeros;    /* [bits]               number of 0-bits per level (1-child descent base) */
  const int *width;        /* [16] constant        offset bit-width per class */
  const int *binom;        /* [16 * 16] constant   Pascal's triangle for the combinatorial decode */
  int bits;                /* levels = ceil(log2(vocab)) */
  int cwords;              /* class words per level */
  int nsb;                 /* superblocks per level (delta/sample rows are nsb + 1) */
  int na;                  /* rank/offset anchors per level */
} cf_rrrw_view;

/* FM-index over the RRR-backed BWT wavelet + a succinct sampled suffix array. Layout matches
 * detail/fm_search_device.cuh exactly. */
typedef struct cf_fm_view {
  cf_rrrw_view w;         /* the RRR-backed wavelet of the BWT */
  const int32_t *C;       /* [sigma]      cumulative symbol counts (FM C-table) */
  const uint32_t *mwords; /* [mwords_len] packed sampled-SA mark plane (bit p set => SA[p] is sampled) */
  const int32_t *msb;     /* [msb_len]    its superblock directory (SB=8 words), ranked by cf_rank1 */
  const int32_t *sval;    /* [nsval]      sampled suffix-array values (text positions), in SA order */
  int sigma;              /* alphabet size (incl. sentinel) */
  int n;                  /* BWT length = |s| */
  int sa_sample;          /* an LF-walk hits a mark within sa_sample steps */
} cf_fm_view;

/* --- RRR-backed wavelet (entropy-sized self-index) --- */
/* Batched access: decode the token id at each of `count` `positions` into `out`. */
cf_status cf_rrrw_access_async(cf_rrrw_view v, const uint32_t *positions, uint32_t *out, size_t count,
                               void *stream);
/* Batched rank: occurrences of `symbols[t]` in the first `positions[t]` tokens, for each t. */
cf_status cf_rrrw_rank_async(cf_rrrw_view v, const uint32_t *symbols, const uint32_t *positions, uint32_t *out,
                             size_t count, void *stream);

/* --- FM-index backward search (compressed-domain search) --- */
/* count: for each of `npat` patterns (flattened in `pat`, per-pattern start/len in `pstart`/`plen`),
 * write the number of occurrences to `out[t]`. */
cf_status cf_fm_count_async(cf_fm_view v, const int32_t *pat, const int32_t *pstart, const int32_t *plen,
                            uint32_t *out, size_t npat, void *stream);
/* ranges: like count, but write the suffix-array [lo, hi) interval per pattern (occurrences = hi - lo). */
cf_status cf_fm_ranges_async(cf_fm_view v, const int32_t *pat, const int32_t *pstart, const int32_t *plen,
                             int32_t *lo_out, int32_t *hi_out, size_t npat, void *stream);
/* locate: for each suffix-array row index `r_in[t]` (from a [lo, hi) range), write its text position to out[t]. */
cf_status cf_fm_locate_async(cf_fm_view v, const int32_t *r_in, int32_t *out, size_t nocc, void *stream);

/* --- FM-index host-pointer convenience layer (device marshalling hidden in the engine) ---
 * For a caller that has a `.cffm` blob + host query arrays and does NOT want to do CUDA device marshalling
 * itself (e.g. a Rust `-sys` FFI that would otherwise need cudart). `cf_fm_host_load` uploads the index to the
 * device ONCE (build != query, P9); each query uploads the small pattern arrays, runs the device-native kernels
 * above, and downloads the results. The device-native hot path is unchanged; this trades a host copy of the
 * (small) query + result for the caller not needing cudart. Every entry point returns CF_ERR_INVALID_ARGUMENT
 * on a NULL required pointer BEFORE any CUDA call (the null_arg_contract). */
typedef struct cf_fm_host_index cf_fm_host_index;

/* Build a device-resident FM-index from a `.cffm` container blob (host bytes). On CF_OK, `*out` owns device
 * memory until `cf_fm_host_free`. */
cf_status cf_fm_host_load(const uint8_t *cffm, size_t nbytes, cf_fm_host_index **out);
/* count: occurrences of each of `npat` patterns (flattened `pat`, per-pattern `pstart`/`plen`), host arrays;
 * `counts_out` is host memory of length `npat`. */
cf_status cf_fm_host_count(const cf_fm_host_index *ix, const int32_t *pat, const int32_t *pstart,
                           const int32_t *plen, uint32_t npat, uint32_t *counts_out);
/* ranges: suffix-array [lo, hi) interval per pattern (host in/out, each of length `npat`). */
cf_status cf_fm_host_ranges(const cf_fm_host_index *ix, const int32_t *pat, const int32_t *pstart,
                            const int32_t *plen, uint32_t npat, int32_t *lo_out, int32_t *hi_out);
/* locate: text position of each of `nocc` suffix-array row indices in `rows` (host in/out). */
cf_status cf_fm_host_locate(const cf_fm_host_index *ix, const int32_t *rows, uint32_t nocc, int32_t *pos_out);
/* release the device-resident index. */
void cf_fm_host_free(cf_fm_host_index *ix);

/* --- FM-index DEVICE-NATIVE path (P5: zero host round-trip in the hot loop) ---
 * The host layer above copies the (small) query + result every call. For a caller that keeps query data resident
 * on the device across calls — or simply wants to amortize allocation — these helpers expose the device-native
 * async API (`cf_fm_*_async` above) without the caller linking cudart. The engine still owns all CUDA; the caller
 * holds opaque device pointers (void*) and drives the async kernels on a stream. Copy in/out only when it chooses.
 * All entry points honor the null_arg_contract (CF_ERR_INVALID_ARGUMENT on a NULL required pointer before any CUDA
 * call). ADDITIVE to abi_version 0 (backward-compatible: no existing symbol changed). */

/* Expose the resident device-side FM view built by `cf_fm_host_load`, so the caller can drive `cf_fm_count_async`
 * / `cf_fm_ranges_async` / `cf_fm_locate_async` directly. The view's pointers are owned by `ix` and valid until
 * `cf_fm_host_free(ix)`; copy it by value, do not free its contents. */
cf_status cf_fm_host_view(const cf_fm_host_index *ix, cf_fm_view *out);

/* Device-memory helpers (thin cudaMalloc/Free/Memcpy wrappers). `*out`/`dptr` are DEVICE pointers. `cf_device_alloc`
 * of 0 bytes still returns a unique freeable pointer. h2d = host→device (upload), d2h = device→host (download). */
cf_status cf_device_alloc(size_t nbytes, void **out);
void cf_device_free(void *dptr);
cf_status cf_device_h2d(void *ddst, const void *hsrc, size_t nbytes);
cf_status cf_device_d2h(void *hdst, const void *dsrc, size_t nbytes);

/* Stream helpers. A NULL stream everywhere means the default stream. `cf_stream_sync(NULL)` syncs the default stream. */
cf_status cf_stream_create(void **out);
void cf_stream_destroy(void *stream);
cf_status cf_stream_sync(void *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHROMOFOLD_SEARCH_H */
