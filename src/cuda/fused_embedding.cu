// fused_embedding.cu — the ChromoFold thesis (P3): decode a token INSIDE the embedding-gather kernel, so the
// decompressed token stream never exists in global memory. This is the "never fully decompress" op from the
// research brief — one kernel, no intermediate buffer, no extra launch, no extra global-memory round trip.

#include "chromofold/chromofold.h"
#include "chromofold/detail/access_device.cuh"

#include <cuda_runtime.h>

// One block per query: thread 0 decodes the token id (in-register wavelet walk), the block then copies that
// token's `dim`-wide embedding row to the output. The token id lives in a register/shared — never in DRAM.
template <int LEVELS>
__global__ void cf_fused_embed_kernel(const uint32_t *bitplanes, const int32_t *superblocks, const int32_t *zeros,
                                      int nwords, int nblocks, int levels_dyn, const float *E, uint32_t dim,
                                      const uint32_t *positions, float *out, size_t count) {
  size_t q = blockIdx.x;
  if (q >= count) return;
  __shared__ uint32_t tok;
  if (threadIdx.x == 0)
    tok = cf_access_one<LEVELS>(bitplanes, superblocks, zeros, nwords, nblocks, (int)positions[q], levels_dyn);
  __syncthreads();
  const float *row = E + (size_t)tok * dim;
  float *dst = out + q * dim;
  for (uint32_t e = threadIdx.x; e < dim; e += blockDim.x) dst[e] = row[e]; // coalesced row copy
}

// Alternative mapping: one THREAD per query. Decode is fully parallel (all threads active, unlike the
// block-per-query variant), but the row copy is strided across threads (uncoalesced). Which mapping wins
// depends on the decode/copy balance — measured in the benchmark.
template <int LEVELS>
__global__ void cf_fused_embed_tpq_kernel(const uint32_t *bitplanes, const int32_t *superblocks,
                                          const int32_t *zeros, int nwords, int nblocks, int levels_dyn,
                                          const float *E, uint32_t dim, const uint32_t *positions, float *out,
                                          size_t count) {
  size_t q = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (q >= count) return;
  uint32_t tok =
      cf_access_one<LEVELS>(bitplanes, superblocks, zeros, nwords, nblocks, (int)positions[q], levels_dyn);
  const float *row = E + (size_t)tok * dim;
  float *dst = out + q * dim;
  for (uint32_t e = 0; e < dim; ++e) dst[e] = row[e];
}

extern "C" cf_status cf_embedding_gather_tpq_async(cf_wavelet_view idx, const float *E, uint32_t dim,
                                                   const uint32_t *positions, float *out, size_t count,
                                                   void *stream) {
  if (count == 0 || dim == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  const int nw = (int)idx.nwords, nb = (int)idx.nblocks, lv = (int)idx.levels;
#define CF_TPQ(L)                                                                                            \
  cf_fused_embed_tpq_kernel<L><<<blocks, threads, 0, s>>>(idx.bitplanes, idx.superblocks, idx.zero_counts,   \
                                                          nw, nb, lv, E, dim, positions, out, count)
  switch (lv) {
    case 2:  CF_TPQ(2);  break;
    case 4:  CF_TPQ(4);  break;
    case 8:  CF_TPQ(8);  break;
    case 15: CF_TPQ(15); break;
    case 16: CF_TPQ(16); break;
    case 17: CF_TPQ(17); break;
    default: CF_TPQ(0);  break;
  }
#undef CF_TPQ
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

extern "C" cf_status cf_embedding_gather_async(cf_wavelet_view idx, const float *E, uint32_t dim,
                                               const uint32_t *positions, float *out, size_t count,
                                               void *stream) {
  if (!idx.bitplanes || !E || !positions || !out) return CF_ERR_INVALID_ARGUMENT;
  if (count == 0 || dim == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = (dim < 256u) ? (int)dim : 256;
  const int nw = (int)idx.nwords, nb = (int)idx.nblocks, lv = (int)idx.levels;
#define CF_FUSE(L)                                                                                             \
  cf_fused_embed_kernel<L><<<count, threads, 0, s>>>(idx.bitplanes, idx.superblocks, idx.zero_counts, nw, nb,  \
                                                     lv, E, dim, positions, out, count)
  switch (lv) {
    case 2:  CF_FUSE(2);  break;
    case 4:  CF_FUSE(4);  break;
    case 8:  CF_FUSE(8);  break;
    case 15: CF_FUSE(15); break;
    case 16: CF_FUSE(16); break;
    case 17: CF_FUSE(17); break;
    default: CF_FUSE(0);  break;
  }
#undef CF_FUSE
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
