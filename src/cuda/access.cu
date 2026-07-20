// access.cu — packed-wavelet `access` as a CUDA C++ kernel (milestone M1).
//
// Reproduces the frozen Warp reference bit-for-bit: each thread walks `levels` wavelet levels, at each level
// computing rank1 (superblock jump + blocked __popc) and descending into the 0- or 1-child. Compile-time
// specialization by level count (architecture §5) unrolls the traversal for common vocabulary widths.

#include "chromofold/chromofold.h"
#include "chromofold/detail/access_device.cuh"

#include <cuda_runtime.h>

template <int LEVELS>
__global__ void cf_access_kernel(const uint32_t *bitplanes, const int32_t *superblocks, const int32_t *zeros,
                                 int nwords, int nblocks, const uint32_t *positions, uint32_t *out, size_t count,
                                 int levels_dyn) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= count) return;
  out[t] = cf_access_one<LEVELS>(bitplanes, superblocks, zeros, nwords, nblocks, (int)positions[t], levels_dyn);
}

extern "C" cf_status cf_access_async(cf_wavelet_view idx, const uint32_t *device_positions,
                                     uint32_t *device_output, size_t count, void *stream) {
  if (!idx.bitplanes || !idx.superblocks || !idx.zero_counts || !device_positions || !device_output)
    return CF_ERR_INVALID_ARGUMENT;
  if (count == 0) return CF_OK;

  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  const int nw = (int)idx.nwords, nb = (int)idx.nblocks, lv = (int)idx.levels;

// Compile-time specialization for common vocabulary widths; dynamic fallback otherwise.
#define CF_LAUNCH(L)                                                                                          \
  cf_access_kernel<L><<<blocks, threads, 0, s>>>(idx.bitplanes, idx.superblocks, idx.zero_counts, nw, nb,     \
                                                 device_positions, device_output, count, lv)
  switch (lv) {
    case 2:  CF_LAUNCH(2);  break; // V <= 4
    case 4:  CF_LAUNCH(4);  break;
    case 8:  CF_LAUNCH(8);  break; // V <= 256
    case 15: CF_LAUNCH(15); break; // ~32K vocab
    case 16: CF_LAUNCH(16); break;
    case 17: CF_LAUNCH(17); break;
    default: CF_LAUNCH(0);  break; // dynamic (LEVELS==0 -> use levels_dyn)
  }
#undef CF_LAUNCH
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
