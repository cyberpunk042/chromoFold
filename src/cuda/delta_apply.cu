// delta_apply.cu — reference/delta cluster decode on the GPU (milestone M8): cross-sequence dedup for the
// shared-prefix / conversation-history / LoRA-library / near-duplicate-context path. A cluster of near-identical
// sequences is stored as ONE base + per-sequence sparse deltas; fetch(seq, pos) = base[pos] overridden by the
// sequence's own delta at pos (a binary search over its sorted delta positions). Partial unfold of a compressed
// cluster, batched, never leaving the GPU. Faithful flat port of warp_compress.gpu_delta._fetch_k.

#include "chromofold/chromofold.h"

#include <cuda_runtime.h>

// one thread per (leaf, pos) query: base default, overridden by the leaf's delta at pos if present.
__global__ void cf_delta_fetch_kernel(const int32_t *base, int nbase, const int32_t *dpos, const int32_t *dval,
                                      const int32_t *dstart, const int32_t *dlen, const int32_t *leaf_in,
                                      const int32_t *pos_in, int32_t *out, size_t count) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= count) return;
  int leaf = leaf_in[t], pos = pos_in[t];
  int res = (pos < nbase) ? base[pos] : -1;
  int ds = dstart[leaf], dl = dlen[leaf];
  int lo = 0, hi = dl;                                     // binary-search this leaf's sorted delta positions
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (dpos[ds + mid] < pos) lo = mid + 1;
    else hi = mid;
  }
  if (lo < dl && dpos[ds + lo] == pos) res = dval[ds + lo];
  out[t] = res;
}

// Batched reference/delta fetch: reconstruct `count` tokens, one per (device_leaf[t], device_pos[t]).
// All arrays are device pointers; runs on `stream`, no host sync/alloc.
extern "C" cf_status cf_delta_fetch_async(const int32_t *base, int nbase, const int32_t *dpos, const int32_t *dval,
                                          const int32_t *dstart, const int32_t *dlen, const int32_t *device_leaf,
                                          const int32_t *device_pos, int32_t *out, size_t count, void *stream) {
  if (!base || !dstart || !dlen || !device_leaf || !device_pos || !out) return CF_ERR_INVALID_ARGUMENT;
  if (count == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  cf_delta_fetch_kernel<<<blocks, threads, 0, s>>>(base, nbase, dpos, dval, dstart, dlen, device_leaf, device_pos,
                                                   out, count);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
