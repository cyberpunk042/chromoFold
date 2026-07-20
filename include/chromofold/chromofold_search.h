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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHROMOFOLD_SEARCH_H */
