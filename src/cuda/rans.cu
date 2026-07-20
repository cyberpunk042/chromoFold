// rans.cu — block-rANS value decode on the GPU (milestone M4, entropy-coder frontier). rANS approaches H0 with no
// per-symbol overhead (unlike Huffman); the stream is cut into independent fixed-count blocks so whole-tensor
// decode is one thread per block (parallel) and random access still works within a block — same shape as the
// block-Huffman coder, tighter ratio on low-entropy data. Faithful port of warp_compress.gpu_rans._decode_k /
// _fetch_k: decode is a slot lookup (state & (M-1) -> symbol) + a state update + an 8-bit renormalisation.

#include "chromofold/chromofold.h"

#include <cuda_runtime.h>

#define CF_RANS_PROB_BITS 12
#define CF_RANS_M (1u << CF_RANS_PROB_BITS)   // frequencies normalise to M = 4096
#define CF_RANS_L (1u << 23)                  // 32-bit state renorm bound (8-bit renorm, ryg_rans style)

// decode `cnt` values of block `b` starting from its saved state, advancing an 8-bit renorm read pointer.
__device__ __forceinline__ void cf_rans_decode_block(const uint8_t *data, int pos, uint32_t state,
                                                     const int32_t *slot2sym, const int32_t *freq,
                                                     const int32_t *cum, int cnt, int32_t *out) {
  for (int j = 0; j < cnt; ++j) {
    int slot = (int)(state & (CF_RANS_M - 1));
    int s = slot2sym[slot];
    state = (uint32_t)freq[s] * (state >> CF_RANS_PROB_BITS) + (uint32_t)(slot - cum[s]);
    while (state < CF_RANS_L) { state = (state << 8) | (uint32_t)data[pos]; ++pos; }
    out[j] = s;
  }
}

__global__ void cf_rans_decode_kernel(const uint8_t *data, const int32_t *byte_off, const uint32_t *state0,
                                      const int32_t *slot2sym, const int32_t *freq, const int32_t *cum, int block,
                                      int n, int32_t *out) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int base = b * block;
  if (base >= n) return;
  int cnt = min(block, n - base);
  cf_rans_decode_block(data, byte_off[b], state0[b], slot2sym, freq, cum, cnt, out + base);
}

// Whole-tensor block-rANS decode (one thread per block). All pointers are device pointers; runs on `stream`.
extern "C" cf_status cf_rans_decode_async(const uint8_t *data, const int32_t *byte_off, const uint32_t *state0,
                                          const int32_t *slot2sym, const int32_t *freq, const int32_t *cum,
                                          int block, int n, int32_t *out, void *stream) {
  if (!data || !byte_off || !state0 || !slot2sym || !freq || !cum || !out) return CF_ERR_INVALID_ARGUMENT;
  if (n == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int nblocks = (n + block - 1) / block;
  const int threads = 128;
  cf_rans_decode_kernel<<<(nblocks + threads - 1) / threads, threads, 0, s>>>(data, byte_off, state0, slot2sym,
                                                                             freq, cum, block, n, out);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
