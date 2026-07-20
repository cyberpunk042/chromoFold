// rrr_wavelet_device.cuh — device-side RRR-backed wavelet decode: access/rank over a wavelet matrix whose every
// level is an RRR bitvector with two-level superblock samples (milestone M4, wavelet wiring). This is the whole
// self-index, entropy-sized: each level costs toward its bitplane's entropy (skewed BWT planes -> below H0),
// yet access/rank still run entirely in registers on the GPU. Header-only __device__ inlines so a token can be
// decoded in-register inside any consumer kernel (constitution P3) with no separate decompression pass.
//
// Faithful port of warp_compress.gpu_rrr_wavelet.RRRWaveletGPU (_rank1_lvl / _access_k / _rank_k). Constants must
// match the prototype exactly: T=15-bit blocks, S=64 blocks/superblock, K=32 superblocks/anchor.
#ifndef CHROMOFOLD_DETAIL_RRR_WAVELET_DEVICE_CUH
#define CHROMOFOLD_DETAIL_RRR_WAVELET_DEVICE_CUH

#include "chromofold/chromofold.h"

#include <cstdint>

#define CF_RRR_T 15  // block size in bits: C(15,k) <= 6435 => offsets fit in 13 bits
#define CF_RRR_S 64  // blocks per superblock (sparse rank samples)
#define CF_RRR_K 32  // superblocks per anchor (two-level rank: int32 anchor + uint16 delta)

// An immutable, device-resident RRR-backed wavelet index. All pointers are DEVICE pointers. Shapes are identical
// across levels except the variable-width offset stream, which is concatenated flat with a per-level bit base.
struct cf_rrrw_view {
  const uint32_t *classes; // [bits * cwords]     4-bit block class stream per level, LSB-first
  const uint32_t *offsets; // [owords]            flat enumerative-offset stream; level base (in bits) in offbase[]
  const int32_t *rank_a;   // [bits * na]         two-level rank sample: int32 anchor every K superblocks
  const uint16_t *rank_d;  // [bits * (nsb + 1)]  two-level rank sample: uint16 delta per superblock
  const int32_t *off_a;    // [bits * na]         two-level offset-bit sample: int32 anchor
  const uint16_t *off_d;   // [bits * (nsb + 1)]  two-level offset-bit sample: uint16 delta
  const int32_t *offbase;  // [bits]              bit offset of each level's slice within offsets[]
  const int32_t *zeros;    // [bits]              number of 0-bits per level (the 1-child descent base)
  const int *width;        // [16] constant       offset bit-width per class (rebuilt on host, not stored)
  const int *binom;        // [16 * 16] constant  Pascal's triangle for the combinatorial decode
  int bits;                // levels = ceil(log2(vocab))
  int cwords;              // class words per level
  int nsb;                 // superblocks per level (delta/sample rows are nsb + 1)
  int na;                  // rank/offset anchors per level
};

// class stream: 4 bits per block, LSB-first, never spans a word (4 | 32).
__device__ __forceinline__ int cf_rrr_classat(const uint32_t *classes, int j) {
  int bp = j * 4;
  return (int)((classes[bp >> 5] >> (bp & 31)) & 15u);
}

// read `width` bits at bit position `bitpos`, LSB-first (may span two words).
__device__ __forceinline__ int cf_rrr_readbits(const uint32_t *stream, int bitpos, int width) {
  if (width == 0) return 0;
  int wi = bitpos >> 5, b = bitpos & 31;
  uint32_t val = stream[wi] >> b;
  if (b + width > 32) val |= stream[wi + 1] << (32 - b);
  uint32_t mask = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
  return (int)(val & mask);
}

// combinatorial unrank: the T-bit word with `cl` ones whose enumerative rank is `off`. binom is flattened 16x16.
__device__ __forceinline__ uint32_t cf_rrr_decode_word(const int *binom, int cl, int off) {
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

// RRR rank1 over [0, pos) for ONE wavelet level: superblock jump (two-level sample) + in-block class scan +
// ONE combinatorial block decode + popcount. The heart of the RRR-wavelet — everything else is descent.
__device__ __forceinline__ int cf_rrrw_rank1_lvl(const cf_rrrw_view &v, int lvl, int pos) {
  const uint32_t *classes = v.classes + (size_t)lvl * v.cwords;
  const int nsb1 = v.nsb + 1;
  int blk = pos / CF_RRR_T, b = pos % CF_RRR_T;
  int sbi = blk / CF_RRR_S, a = sbi / CF_RRR_K;
  int r = v.rank_a[(size_t)lvl * v.na + a] + (int)v.rank_d[(size_t)lvl * nsb1 + sbi];
  int obit = v.off_a[(size_t)lvl * v.na + a] + (int)v.off_d[(size_t)lvl * nsb1 + sbi];
  for (int j = sbi * CF_RRR_S; j < blk; ++j) {  // sum classes (and advance the offset bit position) in-superblock
    int cl = cf_rrr_classat(classes, j);
    r += cl;
    obit += v.width[cl];
  }
  if (b > 0) {  // partial popcount inside the target block: decode it once, in registers
    int cl = cf_rrr_classat(classes, blk);
    int off = cf_rrr_readbits(v.offsets, v.offbase[lvl] + obit, v.width[cl]);
    uint32_t word = cf_rrr_decode_word(v.binom, cl, off);
    r += __popc(word & ((1u << b) - 1u));
  }
  return r;
}

// Decode the token at position `i` by walking the wavelet levels. The bit at level `lvl` is read as
// rank1(i+1) - rank1(i) — reusing the RRR decode path instead of a second bit-access path (matches the oracle).
template <int LEVELS>
__device__ __forceinline__ uint32_t cf_rrrw_access_one(const cf_rrrw_view &v, int i, int levels_dyn) {
  int val = 0;
  const int L = (LEVELS > 0) ? LEVELS : levels_dyn;
#pragma unroll
  for (int lvl = 0; lvl < L; ++lvl) {
    int r0 = cf_rrrw_rank1_lvl(v, lvl, i);
    int r1 = cf_rrrw_rank1_lvl(v, lvl, i + 1);
    if (r1 - r0 == 1) {  // bit == 1: descend into the 1-child block
      val = (val << 1) | 1;
      i = v.zeros[lvl] + r0;
    } else {  // bit == 0: rank0(i) = i - rank1(i)
      val = val << 1;
      i = i - r0;
    }
  }
  return (uint32_t)val;
}

// Wavelet rank(c, i) = occurrences of symbol c in [0, i): two RRR rank1 calls per level to narrow [p, q).
template <int LEVELS>
__device__ __forceinline__ uint32_t cf_rrrw_rank_one(const cf_rrrw_view &v, int c, int i, int levels_dyn) {
  const int L = (LEVELS > 0) ? LEVELS : levels_dyn;
  int p = 0, q = i;
#pragma unroll
  for (int lvl = 0; lvl < L; ++lvl) {
    int bitc = (c >> (L - 1 - lvl)) & 1;
    int rp = cf_rrrw_rank1_lvl(v, lvl, p), rq = cf_rrrw_rank1_lvl(v, lvl, q);
    if (bitc) { p = v.zeros[lvl] + rp; q = v.zeros[lvl] + rq; }
    else      { p = p - rp;            q = q - rq; }
  }
  return (uint32_t)(q - p);
}

#ifdef __cplusplus
extern "C" {
#endif
// experimental RRR-wavelet access/rank (not in the public header yet — a benchmark-driven design decision, like
// cf_rrr_rank_async / cf_rank2_async). The view is passed by value; all its pointers are device pointers.
cf_status cf_rrrw_access_async(cf_rrrw_view v, const uint32_t *positions, uint32_t *out, size_t count, void *stream);
cf_status cf_rrrw_rank_async(cf_rrrw_view v, const uint32_t *symbols, const uint32_t *positions, uint32_t *out,
                             size_t count, void *stream);
#ifdef __cplusplus
}
#endif

#endif // CHROMOFOLD_DETAIL_RRR_WAVELET_DEVICE_CUH
