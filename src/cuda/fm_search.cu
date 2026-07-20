// fm_search.cu — FM-index backward search on the GPU over the RRR-backed BWT wavelet (milestone M7). count and
// locate run GPU-resident over the SAME compact, entropy-sized index from M4: search AND generate without leaving
// VRAM. Backward search is one thread per pattern (parallel across a batch); locate adds one LF-walk thread per
// occurrence over a succinct sampled suffix array. Device logic lives in the shared header (constitution P3).

#include "chromofold/detail/fm_search_device.cuh"

#include <cuda_runtime.h>

template <int LEVELS>
__global__ void cf_fm_count_kernel(cf_fm_view v, const int32_t *pat, const int32_t *pstart, const int32_t *plen,
                                   uint32_t *out, size_t npat) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= npat) return;
  int lo, hi;
  cf_fm_backward<LEVELS>(v, pat, pstart[t], plen[t], lo, hi);
  out[t] = (uint32_t)(hi - lo);
}

template <int LEVELS>
__global__ void cf_fm_ranges_kernel(cf_fm_view v, const int32_t *pat, const int32_t *pstart, const int32_t *plen,
                                    int32_t *lo_out, int32_t *hi_out, size_t npat) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= npat) return;
  int lo, hi;
  cf_fm_backward<LEVELS>(v, pat, pstart[t], plen[t], lo, hi);
  lo_out[t] = lo;
  hi_out[t] = hi;
}

template <int LEVELS>
__global__ void cf_fm_locate_kernel(cf_fm_view v, const int32_t *r_in, int32_t *out, size_t nocc) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= nocc) return;
  out[t] = cf_fm_locate_one<LEVELS>(v, r_in[t]);
}

// dispatch the level count (unrolled walk) like the other RRR entry points; default 0 = dynamic level count.
#define CF_FM_DISPATCH(LAUNCH)                                                                                  \
  switch (v.w.bits) {                                                                                          \
    case 2:  LAUNCH(2);  break;                                                                                \
    case 4:  LAUNCH(4);  break;                                                                                \
    case 6:  LAUNCH(6);  break;                                                                                \
    case 7:  LAUNCH(7);  break;                                                                                \
    case 8:  LAUNCH(8);  break;                                                                                \
    case 16: LAUNCH(16); break;                                                                                \
    case 17: LAUNCH(17); break;                                                                                \
    default: LAUNCH(0);  break;                                                                                \
  }

extern "C" cf_status cf_fm_count_async(cf_fm_view v, const int32_t *pat, const int32_t *pstart,
                                       const int32_t *plen, uint32_t *out, size_t npat, void *stream) {
  if (!v.w.classes || !pat || !pstart || !plen || !out) return CF_ERR_INVALID_ARGUMENT;
  if (npat == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (npat + threads - 1) / threads;
#define CF_LAUNCH(L) cf_fm_count_kernel<L><<<blocks, threads, 0, s>>>(v, pat, pstart, plen, out, npat)
  CF_FM_DISPATCH(CF_LAUNCH)
#undef CF_LAUNCH
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

extern "C" cf_status cf_fm_ranges_async(cf_fm_view v, const int32_t *pat, const int32_t *pstart,
                                        const int32_t *plen, int32_t *lo_out, int32_t *hi_out, size_t npat,
                                        void *stream) {
  if (!v.w.classes || !pat || !pstart || !plen || !lo_out || !hi_out) return CF_ERR_INVALID_ARGUMENT;
  if (npat == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (npat + threads - 1) / threads;
#define CF_LAUNCH(L) cf_fm_ranges_kernel<L><<<blocks, threads, 0, s>>>(v, pat, pstart, plen, lo_out, hi_out, npat)
  CF_FM_DISPATCH(CF_LAUNCH)
#undef CF_LAUNCH
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

extern "C" cf_status cf_fm_locate_async(cf_fm_view v, const int32_t *r_in, int32_t *out, size_t nocc,
                                        void *stream) {
  if (!v.w.classes || !r_in || !out) return CF_ERR_INVALID_ARGUMENT;
  if (nocc == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (nocc + threads - 1) / threads;
#define CF_LAUNCH(L) cf_fm_locate_kernel<L><<<blocks, threads, 0, s>>>(v, r_in, out, nocc)
  CF_FM_DISPATCH(CF_LAUNCH)
#undef CF_LAUNCH
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
