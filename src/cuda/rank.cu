// rank.cu — packed-wavelet `rank(c, i)` = occurrences of symbol c in [0, i), on the GPU (milestone M3).
//
// Two rank-directory designs, benchmarked head-to-head (architecture §6, brief §5.2):
//   linear   : superblock sample every SB=8 words + scan up to 7 words + one partial popcount (the M1 layout).
//   two-level: coarse int32 superblock (every CB words) + per-word uint16 within-superblock prefix -> ZERO
//              word scan, one partial popcount. Bigger directory, fewer serialized loads.

#include "chromofold/chromofold.h"
#include "chromofold/detail/access_device.cuh"

#include <cuda_runtime.h>

// ---- linear directory (reuses cf_rank1 from access_device.cuh) ----
template <int LEVELS>
__global__ void cf_rank_kernel(const uint32_t *bitplanes, const int32_t *superblocks, const int32_t *zeros,
                               int nwords, int nblocks, int levels_dyn, const uint32_t *c_in,
                               const uint32_t *i_in, uint32_t *out, size_t count) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= count) return;
  const int L = (LEVELS > 0) ? LEVELS : levels_dyn;
  int c = (int)c_in[t], p = 0, q = (int)i_in[t];
#pragma unroll
  for (int lvl = 0; lvl < L; ++lvl) {
    const uint32_t *w = bitplanes + (size_t)lvl * nwords;
    const int32_t *s = superblocks + (size_t)lvl * (nblocks + 1);
    int bitc = (c >> (L - 1 - lvl)) & 1;
    int rp = cf_rank1(w, s, p), rq = cf_rank1(w, s, q);
    if (bitc) { p = zeros[lvl] + rp; q = zeros[lvl] + rq; }
    else      { p = p - rp;         q = q - rq; }
  }
  out[t] = (uint32_t)(q - p);
}

extern "C" cf_status cf_rank_async(cf_wavelet_view idx, const uint32_t *sym, const uint32_t *pos, uint32_t *out,
                                   size_t count, void *stream) {
  if (!idx.bitplanes || !sym || !pos || !out) return CF_ERR_INVALID_ARGUMENT;
  if (count == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  const int nw = (int)idx.nwords, nb = (int)idx.nblocks, lv = (int)idx.levels;
#define CF_RANK(L)                                                                                           \
  cf_rank_kernel<L><<<blocks, threads, 0, s>>>(idx.bitplanes, idx.superblocks, idx.zero_counts, nw, nb, lv,   \
                                               sym, pos, out, count)
  switch (lv) {
    case 2:  CF_RANK(2);  break;
    case 4:  CF_RANK(4);  break;
    case 8:  CF_RANK(8);  break;
    case 15: CF_RANK(15); break;
    case 16: CF_RANK(16); break;
    case 17: CF_RANK(17); break;
    default: CF_RANK(0);  break;
  }
#undef CF_RANK
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

// ---- two-level directory: coarse[word/CB] + per-word within-superblock prefix -> no word scan ----
__device__ __forceinline__ int cf_rank1_2level(const uint32_t *w, const int32_t *coarse, const uint16_t *fine,
                                               int CB, int pos) {
  int word = pos >> 5, bit = pos & 31;
  int r = coarse[word / CB] + (int)fine[word];
  if (bit > 0) r += __popc(w[word] & ((1u << bit) - 1u));
  return r;
}

template <int LEVELS>
__global__ void cf_rank2_kernel(const uint32_t *bitplanes, const int32_t *coarse, const uint16_t *fine,
                                int ncoarse, int CB, const int32_t *zeros, int nwords, int levels_dyn,
                                const uint32_t *c_in, const uint32_t *i_in, uint32_t *out, size_t count) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= count) return;
  const int L = (LEVELS > 0) ? LEVELS : levels_dyn;
  int c = (int)c_in[t], p = 0, q = (int)i_in[t];
#pragma unroll
  for (int lvl = 0; lvl < L; ++lvl) {
    const uint32_t *w = bitplanes + (size_t)lvl * nwords;
    const int32_t *co = coarse + (size_t)lvl * ncoarse;
    const uint16_t *fi = fine + (size_t)lvl * (nwords + 1);  // +1 sentinel: a query at pos=n hits word=nwords
    int bitc = (c >> (L - 1 - lvl)) & 1;
    int rp = cf_rank1_2level(w, co, fi, CB, p), rq = cf_rank1_2level(w, co, fi, CB, q);
    if (bitc) { p = zeros[lvl] + rp; q = zeros[lvl] + rq; }
    else      { p = p - rp;         q = q - rq; }
  }
  out[t] = (uint32_t)(q - p);
}

// experimental two-level rank (not in the public header yet — a benchmark-driven design decision)
extern "C" cf_status cf_rank2_async(cf_wavelet_view idx, const int32_t *coarse, const uint16_t *fine,
                                    int ncoarse, int CB, const uint32_t *sym, const uint32_t *pos, uint32_t *out,
                                    size_t count, void *stream) {
  if (count == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  const int nw = (int)idx.nwords, lv = (int)idx.levels;
#define CF_RANK2(L)                                                                                          \
  cf_rank2_kernel<L><<<blocks, threads, 0, s>>>(idx.bitplanes, coarse, fine, ncoarse, CB, idx.zero_counts,    \
                                                nw, lv, sym, pos, out, count)
  switch (lv) {
    case 2:  CF_RANK2(2);  break;
    case 4:  CF_RANK2(4);  break;
    case 8:  CF_RANK2(8);  break;
    case 15: CF_RANK2(15); break;
    case 16: CF_RANK2(16); break;
    case 17: CF_RANK2(17); break;
    default: CF_RANK2(0);  break;
  }
#undef CF_RANK2
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
