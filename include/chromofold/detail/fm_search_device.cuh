// fm_search_device.cuh — device-side FM-index backward search over the RRR-backed BWT wavelet (milestone M7).
// The search half of the thesis: `count` and `locate` run GPU-resident over the SAME compact, entropy-sized,
// GPU-searchable index built in M4. Backward search is serial within a pattern but embarrassingly parallel across
// patterns; locate adds one LF-walk thread per occurrence over a succinct sampled suffix array. Header-only
// __device__ inlines so the search rides directly on cf_rrrw_rank_one / cf_rrrw_access_one (constitution P3).
//
// Faithful port of warp_compress.gpu_fm_index (_bwsearch_k / _ranges_k / _locate_walk_k) with the wavelet's
// rank/access swapped for the RRR-backed versions. Ground-truth verified (the prototype's RRR FM has count only).
#ifndef CHROMOFOLD_DETAIL_FM_SEARCH_DEVICE_CUH
#define CHROMOFOLD_DETAIL_FM_SEARCH_DEVICE_CUH

#include "chromofold/detail/access_device.cuh"        // cf_rank1 over the packed sampled-SA mark plane (SB=8)
#include "chromofold/detail/rrr_wavelet_device.cuh"   // cf_rrrw_view + rank/access over the RRR-backed BWT

#include <cstdint>

// An immutable, device-resident RRR-backed FM-index. All pointers are DEVICE pointers.
struct cf_fm_view {
  cf_rrrw_view w;         // the RRR-backed wavelet of the BWT (M4)
  const int32_t *C;       // [sigma]      cumulative symbol counts (FM C-table)
  const uint32_t *mwords; // [mwords_len] packed sampled-SA mark plane (bit p set => SA[p] is sampled)
  const int32_t *msb;     // [msb_len]    its superblock directory (CF_WAVELET_SB = 8 words), ranked by cf_rank1
  const int32_t *sval;    // [nsval]      sampled suffix-array values (text positions in s), in SA order
  int sigma;              // alphabet size (incl. sentinel)
  int n;                  // BWT length = text length |s|
  int sa_sample;          // an LF-walk hits a mark within sa_sample steps
};

// Backward search: narrow the suffix-array range [lo, hi) for a pattern consumed right-to-left. pat symbols are
// already shifted into the sentinel'd alphabet. Out-of-alphabet symbols empty the range.
template <int LEVELS>
__device__ __forceinline__ void cf_fm_backward(const cf_fm_view &v, const int32_t *pat, int st, int len,
                                               int &lo, int &hi) {
  lo = 0;
  hi = v.n;
  for (int k = 0; k < len; ++k) {
    int c = pat[st + len - 1 - k];
    if (c >= 0 && c < v.sigma) {
      if (lo < hi) {
        lo = v.C[c] + (int)cf_rrrw_rank_one<LEVELS>(v.w, c, lo, v.w.bits);
        hi = v.C[c] + (int)cf_rrrw_rank_one<LEVELS>(v.w, c, hi, v.w.bits);
      }
    } else {
      lo = 0;
      hi = 0;
    }
  }
}

// One occurrence's text position: LF-walk the SA position `p` until a sampled position, then
// text_pos = (SA[sampled] + steps) mod n. Mirrors fm_index.FMIndex.locate, RRR-decoded, in VRAM.
template <int LEVELS>
__device__ __forceinline__ int cf_fm_locate_one(const cf_fm_view &v, int p) {
  int steps = 0;
  while (!((v.mwords[p >> 5] >> (p & 31)) & 1u)) {
    int c = (int)cf_rrrw_access_one<LEVELS>(v.w, p, v.w.bits);          // BWT[p]
    p = v.C[c] + (int)cf_rrrw_rank_one<LEVELS>(v.w, c, p, v.w.bits);    // LF-mapping
    ++steps;
  }
  int idx = cf_rank1(v.mwords, v.msb, p);                              // mark index -> sampled-value slot
  return (int)(((long long)v.sval[idx] + steps) % v.n);
}

#ifdef __cplusplus
extern "C" {
#endif
// experimental FM-search over the RRR-backed index (not in the public header yet — benchmark-driven, like the
// other cf_rrr* entry points). Views passed by value; all pointers are device pointers.
cf_status cf_fm_count_async(cf_fm_view v, const int32_t *pat, const int32_t *pstart, const int32_t *plen,
                            uint32_t *out, size_t npat, void *stream);
cf_status cf_fm_ranges_async(cf_fm_view v, const int32_t *pat, const int32_t *pstart, const int32_t *plen,
                             int32_t *lo_out, int32_t *hi_out, size_t npat, void *stream);
cf_status cf_fm_locate_async(cf_fm_view v, const int32_t *r_in, int32_t *out, size_t nocc, void *stream);
#ifdef __cplusplus
}
#endif

#endif // CHROMOFOLD_DETAIL_FM_SEARCH_DEVICE_CUH
