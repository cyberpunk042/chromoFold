// rrr_wavelet.cu — access/rank over an RRR-backed wavelet matrix on the GPU (milestone M4, wavelet wiring). The
// resident index is entropy-sized (every level an RRR bitvector, two-level superblock samples) yet still fully
// GPU-searchable: one object, entropy-sized AND navigable. Device logic lives in the shared header so consumer
// kernels can decode-in-register (constitution P3). Faithful port of warp_compress.gpu_rrr_wavelet.RRRWaveletGPU.

#include "chromofold/detail/rrr_wavelet_device.cuh"

#include <cuda_runtime.h>

template <int LEVELS>
__global__ void cf_rrrw_access_kernel(cf_rrrw_view v, const uint32_t *pos, uint32_t *out, size_t count) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= count) return;
  out[t] = cf_rrrw_access_one<LEVELS>(v, (int)pos[t], v.bits);
}

template <int LEVELS>
__global__ void cf_rrrw_rank_kernel(cf_rrrw_view v, const uint32_t *sym, const uint32_t *pos, uint32_t *out,
                                    size_t count) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= count) return;
  out[t] = cf_rrrw_rank_one<LEVELS>(v, (int)sym[t], (int)pos[t], v.bits);
}

extern "C" cf_status cf_rrrw_access_async(cf_rrrw_view v, const uint32_t *positions, uint32_t *out, size_t count,
                                          void *stream) {
  if (!v.classes || !positions || !out) return CF_ERR_INVALID_ARGUMENT;
  if (count == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
#define CF_ACC(L) cf_rrrw_access_kernel<L><<<blocks, threads, 0, s>>>(v, positions, out, count)
  switch (v.bits) {
    case 2:  CF_ACC(2);  break;
    case 4:  CF_ACC(4);  break;
    case 6:  CF_ACC(6);  break;
    case 7:  CF_ACC(7);  break;
    case 8:  CF_ACC(8);  break;
    case 16: CF_ACC(16); break;
    case 17: CF_ACC(17); break;
    default: CF_ACC(0);  break;  // dynamic level count
  }
#undef CF_ACC
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

extern "C" cf_status cf_rrrw_rank_async(cf_rrrw_view v, const uint32_t *symbols, const uint32_t *positions,
                                        uint32_t *out, size_t count, void *stream) {
  if (!v.classes || !symbols || !positions || !out) return CF_ERR_INVALID_ARGUMENT;
  if (count == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
#define CF_RNK(L) cf_rrrw_rank_kernel<L><<<blocks, threads, 0, s>>>(v, symbols, positions, out, count)
  switch (v.bits) {
    case 2:  CF_RNK(2);  break;
    case 4:  CF_RNK(4);  break;
    case 6:  CF_RNK(6);  break;
    case 7:  CF_RNK(7);  break;
    case 8:  CF_RNK(8);  break;
    case 16: CF_RNK(16); break;
    case 17: CF_RNK(17); break;
    default: CF_RNK(0);  break;
  }
#undef CF_RNK
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
