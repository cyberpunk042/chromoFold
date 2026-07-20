// rrr.cu — RRR (Raman-Raman-Rao) bitvector rank1 on the GPU (milestone M4). The entropy frontier: skewed
// planes cost far below 1 bit/bit while rank stays O(1) — a superblock sample + a short in-block class scan +
// ONE combinatorial block decode, all in registers. Faithful port of warp_compress.gpu_rrr._rank1_k.

#include "chromofold/chromofold.h"

#include <cuda_runtime.h>

#define CF_RRR_T 15   // block size in bits: C(15,k) <= 6435 => offsets fit in 13 bits
#define CF_RRR_SB 64  // blocks per superblock (sparse rank samples)

// class stream: 4 bits per block, LSB-first, never spans a word (4 | 32).
__device__ __forceinline__ int cf_classat(const uint32_t *classes, int j) {
  int bp = j * 4;
  return (int)((classes[bp >> 5] >> (bp & 31)) & 15u);
}

// read `width` bits at bit position `bitpos`, LSB-first (may span two words).
__device__ __forceinline__ int cf_readbits(const uint32_t *stream, int bitpos, int width) {
  if (width == 0) return 0;
  int wi = bitpos >> 5, b = bitpos & 31;
  uint32_t val = stream[wi] >> b;
  if (b + width > 32) val |= stream[wi + 1] << (32 - b);
  uint32_t mask = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
  return (int)(val & mask);
}

// combinatorial unrank: the T-bit word with `cl` ones whose enumerative rank is `off`. binom is flattened 16x16.
__device__ __forceinline__ uint32_t cf_decode_word(const int *binom, int cl, int off) {
  uint32_t word = 0;
  int r = off, i = cl;
  while (i >= 1) {
    int c = CF_RRR_T - 1;
    while (binom[c * 16 + i] > r) --c;
    word |= (1u << c);
    r -= binom[c * 16 + i];
    --i;
  }
  return word;
}

// rank1 over [0, pos): superblock jump + in-block class scan + one combinatorial block decode + popcount.
__device__ __forceinline__ int cf_rrr_rank1(const uint32_t *classes, const uint32_t *offsets,
                                            const int32_t *sbrank, const int32_t *sboff, const int *width,
                                            const int *binom, int pos) {
  int blk = pos / CF_RRR_T, b = pos % CF_RRR_T, sbi = blk / CF_RRR_SB;
  int r = sbrank[sbi], obit = sboff[sbi];
  int j = sbi * CF_RRR_SB;
  while (j < blk) {  // sum the classes (and advance the offset bit position) of the in-superblock blocks
    int cl = cf_classat(classes, j);
    r += cl;
    obit += width[cl];
    ++j;
  }
  if (b > 0) {  // partial popcount inside the target block
    int cl = cf_classat(classes, blk);
    int off = cf_readbits(offsets, obit, width[cl]);
    uint32_t word = cf_decode_word(binom, cl, off);
    uint32_t mask = (1u << b) - 1u;
    r += __popc(word & mask);
  }
  return r;
}

__global__ void cf_rrr_rank_kernel(const uint32_t *classes, const uint32_t *offsets, const int32_t *sbrank,
                                   const int32_t *sboff, const int *width, const int *binom,
                                   const uint32_t *positions, uint32_t *out, size_t count) {
  size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= count) return;
  out[t] = (uint32_t)cf_rrr_rank1(classes, offsets, sbrank, sboff, width, binom, (int)positions[t]);
}

// experimental RRR rank (not in the public header yet — the entropy-frontier variant of rank)
extern "C" cf_status cf_rrr_rank_async(const uint32_t *classes, const uint32_t *offsets, const int32_t *sbrank,
                                       const int32_t *sboff, const int *width, const int *binom,
                                       const uint32_t *positions, uint32_t *out, size_t count, void *stream) {
  if (!classes || !offsets || !positions || !out) return CF_ERR_INVALID_ARGUMENT;
  if (count == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  cf_rrr_rank_kernel<<<blocks, threads, 0, s>>>(classes, offsets, sbrank, sboff, width, binom, positions, out,
                                                count);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
