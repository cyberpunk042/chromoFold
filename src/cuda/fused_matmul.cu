// fused_matmul.cu — the thesis op (constitution P3, roadmap M6 re-target): decode int4 block-Huffman weights
// INSIDE the GEMM so the dequantized (M×K) matrix never exists in VRAM — only the compressed store is resident
// during compute. This is the LARGE-intermediate fusion the M6 embedding-gather experiment pointed to: there the
// avoided intermediate (token ids, 4 B) was tiny and fusion lost; here it is the whole dequantized weight matrix,
// so fusion wins the memory it was designed to win. Faithful port of warp_compress.gpu_fused_matmul.
//
// Three kernels: the fused decode-in-GEMM (no dense W), and a decode-to-dense + plain GEMM reference that
// materialises W — same float ops in the same k-order, so the two agree BIT-FOR-BIT (fusion is numerically free).

#include "chromofold/chromofold.h"
#include "chromofold/detail/block_huffman_device.cuh"

#include <cuda_runtime.h>

// FUSED: one thread per output column m. Decode W's row m from the compressed stream and multiply-accumulate it
// into y[:, m] for the whole batch. The dequantized row lives only in registers, never in global memory.
__global__ void cf_fused_matmul_kernel(const uint32_t *words, const int32_t *block_off, const int32_t *lut,
                                       int maxlen, int block, const float *x, float *y, int B, int M, int K,
                                       float scale, int zero) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;
  long start = (long)m * K;                         // flat index of W[m, 0] (row-major)
  int bs = (int)(start / block);
  int pos = block_off[bs];
  int skip = (int)(start - (long)bs * block);
  for (int i = 0; i < skip; ++i) {                  // advance to the row start (mid-block)
    int sl = cf_bh_decode_at(words, lut, maxlen, pos);
    pos += sl >> 8;
  }
  for (int k = 0; k < K; ++k) {
    int sl = cf_bh_decode_at(words, lut, maxlen, pos);
    int q = sl & 0xFF;
    pos += sl >> 8;
    float w = (float)(q - zero) * scale;            // dequantized weight, in a register
    for (int b = 0; b < B; ++b) y[(long)b * M + m] += x[(long)b * K + k] * w;
  }
}

// REFERENCE stage 1: decode the whole compressed store to a dense fp32 W (one thread per block). This is the
// materialised (M×K) matrix the fused path avoids — measured to show the memory it costs.
__global__ void cf_qw_decode_dense_kernel(const uint32_t *words, const int32_t *block_off, const int32_t *lut,
                                          int maxlen, int block, int n, float scale, int zero, float *W) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int base = b * block;
  if (base >= n) return;
  int cnt = min(block, n - base);
  int pos = block_off[b];
  for (int j = 0; j < cnt; ++j) {
    int sl = cf_bh_decode_at(words, lut, maxlen, pos);
    int q = sl & 0xFF;
    pos += sl >> 8;
    W[base + j] = (float)(q - zero) * scale;
  }
}

// REFERENCE stage 2: plain GEMM over the dense W, one thread per column m, SAME k-outer/b-inner order as fused ->
// bit-identical accumulation. y must be pre-zeroed by the caller.
__global__ void cf_gemm_dense_kernel(const float *W, const float *x, float *y, int B, int M, int K) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;
  for (int k = 0; k < K; ++k) {
    float w = W[(long)m * K + k];
    for (int b = 0; b < B; ++b) y[(long)b * M + m] += x[(long)b * K + k] * w;
  }
}

extern "C" cf_status cf_fused_matmul_async(const uint32_t *words, const int32_t *block_off, const int32_t *lut,
                                           int maxlen, int block, const float *x, float *y, int B, int M, int K,
                                           float scale, int zero, void *stream) {
  if (!words || !block_off || !lut || !x || !y) return CF_ERR_INVALID_ARGUMENT;
  if (M == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 128;
  const int blocks = (M + threads - 1) / threads;
  cudaMemsetAsync(y, 0, (size_t)B * M * sizeof(float), s);   // accumulator starts at zero
  cf_fused_matmul_kernel<<<blocks, threads, 0, s>>>(words, block_off, lut, maxlen, block, x, y, B, M, K, scale,
                                                    zero);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

// Reference (materialising) path: decode to the dense buffer `W`, then GEMM. Two kernels, one big intermediate.
extern "C" cf_status cf_dense_matmul_async(const uint32_t *words, const int32_t *block_off, const int32_t *lut,
                                           int maxlen, int block, const float *x, float *W, float *y, int B,
                                           int M, int K, float scale, int zero, void *stream) {
  if (!words || !block_off || !lut || !x || !W || !y) return CF_ERR_INVALID_ARGUMENT;
  if (M == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  int n = M * K;
  const int nblocks = (n + block - 1) / block;
  const int td = 128;
  cf_qw_decode_dense_kernel<<<(nblocks + td - 1) / td, td, 0, s>>>(words, block_off, lut, maxlen, block, n, scale,
                                                                  zero, W);
  cudaMemsetAsync(y, 0, (size_t)B * M * sizeof(float), s);
  cf_gemm_dense_kernel<<<(M + td - 1) / td, td, 0, s>>>(W, x, y, B, M, K);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
