// access_device.cuh — device-side packed-wavelet decode, shared by the standalone access kernel and the
// fused decode-and-consume kernels (constitution P3). Header-only __device__ inlines so a token can be decoded
// in-register inside any consumer kernel without a separate decompression pass.
#ifndef CHROMOFOLD_DETAIL_ACCESS_DEVICE_CUH
#define CHROMOFOLD_DETAIL_ACCESS_DEVICE_CUH

#include "chromofold/chromofold.h"

#include <cstdint>

// Set bits of one level's bitplane over [0, pos): superblock sample + word scan + partial-word popcount.
__device__ __forceinline__ int cf_rank1(const uint32_t *w, const int32_t *s, int pos) {
  int word = pos >> 5;
  int bit = pos & 31;
  int blk = word / CF_WAVELET_SB;
  int r = s[blk];
  for (int k = blk * CF_WAVELET_SB; k < word; ++k) r += __popc(w[k]);
  if (bit > 0) {
    uint32_t mask = (1u << bit) - 1u;
    r += __popc(w[word] & mask);
  }
  return r;
}

// Decode the token at position `i` by walking the wavelet levels. LEVELS>0 unrolls; LEVELS==0 uses levels_dyn.
template <int LEVELS>
__device__ __forceinline__ uint32_t cf_access_one(const uint32_t *bitplanes, const int32_t *superblocks,
                                                 const int32_t *zeros, int nwords, int nblocks, int i,
                                                 int levels_dyn) {
  int v = 0;
  const int L = (LEVELS > 0) ? LEVELS : levels_dyn;
#pragma unroll
  for (int lvl = 0; lvl < L; ++lvl) {
    const uint32_t *w = bitplanes + (size_t)lvl * nwords;
    const int32_t *s = superblocks + (size_t)lvl * (nblocks + 1);
    int r0 = cf_rank1(w, s, i);
    int b = (w[i >> 5] >> (i & 31)) & 1;
    if (b) {
      v = (v << 1) | 1;
      i = zeros[lvl] + r0; // descend into the 1-child block
    } else {
      v = v << 1;
      i = i - r0; // rank0(i) = i - rank1(i)
    }
  }
  return (uint32_t)v;
}

#endif // CHROMOFOLD_DETAIL_ACCESS_DEVICE_CUH
