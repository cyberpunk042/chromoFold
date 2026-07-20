// sparse_gather.cu — the light/sparse-consumer branch of the fusion thesis (P3), a direct demonstration of P2
// (navigable while compressed). When a consumer touches only K of N positions, the FUSED path decodes each
// touched token id from the compressed RRR-wavelet and gathers its embedding row inline (decode-only-what-you-
// touch), so the N-length decoded sequence never exists. The DECOMPRESS-ALL baseline reconstructs every token id
// into a full buffer, then gathers K rows. For K << N the fused/sparse path does K wavelet walks vs the baseline's
// N — random access beats decompress-all; the crossover is the deliverable. Reuses cf_rrrw_access_one (M4).

#include "chromofold/detail/rrr_wavelet_device.cuh"

#include <cuda_runtime.h>

// FUSED sparse: one thread per query — decode the token id at pos[t], gather its embedding row. No id buffer.
template <int LEVELS>
__global__ void cf_sparse_fused_kernel(cf_rrrw_view v, const float *emb, int dim, const uint32_t *pos, float *out,
                                       size_t K) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= K) return;
  uint32_t id = cf_rrrw_access_one<LEVELS>(v, (int)pos[t], v.bits);
  const float *row = emb + (size_t)id * dim;
  float *o = out + t * (size_t)dim;
  for (int c = 0; c < dim; ++c) o[c] = row[c];
}

// DECOMPRESS-ALL stage 1: reconstruct every token id into a full N-length buffer (the intermediate fusion avoids).
template <int LEVELS>
__global__ void cf_decode_all_kernel(cf_rrrw_view v, uint32_t *ids, size_t n) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= n) return;
  ids[t] = cf_rrrw_access_one<LEVELS>(v, (int)t, v.bits);
}

// DECOMPRESS-ALL stage 2: gather K rows from the decoded id buffer.
__global__ void cf_gather_kernel(const uint32_t *ids, const float *emb, int dim, const uint32_t *pos, float *out,
                                 size_t K) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= K) return;
  uint32_t id = ids[pos[t]];
  const float *row = emb + (size_t)id * dim;
  float *o = out + t * (size_t)dim;
  for (int c = 0; c < dim; ++c) o[c] = row[c];
}

#define CF_SG_DISPATCH(LAUNCH)                                                                                  \
  switch (v.bits) {                                                                                            \
    case 2:  LAUNCH(2);  break;                                                                                \
    case 4:  LAUNCH(4);  break;                                                                                \
    case 6:  LAUNCH(6);  break;                                                                                \
    case 7:  LAUNCH(7);  break;                                                                                \
    case 8:  LAUNCH(8);  break;                                                                                \
    case 16: LAUNCH(16); break;                                                                                \
    case 17: LAUNCH(17); break;                                                                                \
    default: LAUNCH(0);  break;                                                                                \
  }

extern "C" cf_status cf_sparse_gather_fused_async(cf_rrrw_view v, const float *emb, int dim, const uint32_t *pos,
                                                  float *out, size_t K, void *stream) {
  if (!v.classes || !emb || !pos || !out) return CF_ERR_INVALID_ARGUMENT;
  if (K == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 128;
  const size_t blocks = (K + threads - 1) / threads;
#define CF_LAUNCH(L) cf_sparse_fused_kernel<L><<<blocks, threads, 0, s>>>(v, emb, dim, pos, out, K)
  CF_SG_DISPATCH(CF_LAUNCH)
#undef CF_LAUNCH
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

// decode every token id into `ids` (the materialised intermediate) — the decompress-all path's stage 1.
extern "C" cf_status cf_sparse_decode_all_async(cf_rrrw_view v, uint32_t *ids, size_t n, void *stream) {
  if (!v.classes || !ids) return CF_ERR_INVALID_ARGUMENT;
  if (n == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 128;
  const size_t blocks = (n + threads - 1) / threads;
#define CF_LAUNCH(L) cf_decode_all_kernel<L><<<blocks, threads, 0, s>>>(v, ids, n)
  CF_SG_DISPATCH(CF_LAUNCH)
#undef CF_LAUNCH
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

extern "C" cf_status cf_sparse_gather_async(const uint32_t *ids, const float *emb, int dim, const uint32_t *pos,
                                            float *out, size_t K, void *stream) {
  if (!ids || !emb || !pos || !out) return CF_ERR_INVALID_ARGUMENT;
  if (K == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 128;
  const size_t blocks = (K + threads - 1) / threads;
  cf_gather_kernel<<<blocks, threads, 0, s>>>(ids, emb, dim, pos, out, K);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
