// fused_kv_attention.cu — the KV-path fusion (constitution P3, the second large-intermediate op M6 pointed to).
// K,V are KIVI per-axis-quantized (K per-channel, V per-token) and block-Huffman entropy-coded; causal-windowed
// attention decodes + dequantizes each attended K/V row INSIDE the kernel, so the dense dequantized K/V tiles
// never materialise in VRAM — only the compressed KV store is resident, and only the WINDOWED positions are ever
// decoded (the light/sparse-consumer branch of the thesis). This is the long-context KV-cache memory win, on the
// path to M9. The fused kernel and a decode-then-dense reference use identical float ops in the same order, so
// they agree BIT-FOR-BIT (fusion is numerically free).

#include "chromofold/chromofold.h"
#include "chromofold/detail/block_huffman_device.cuh"

#include <cuda_runtime.h>
#include <math.h>

// advance to the block-Huffman bit position of flat element `flat` (its block start + a mid-block skip decode).
__device__ __forceinline__ int cf_kv_row_pos(const uint32_t *words, const int32_t *boff, const int32_t *lut,
                                             int maxlen, int block, long flat) {
  int bs = (int)(flat / block);
  int pos = boff[bs];
  int skip = (int)(flat - (long)bs * block);
  for (int i = 0; i < skip; ++i) { int sl = cf_bh_decode_at(words, lut, maxlen, pos); pos += sl >> 8; }
  return pos;
}

// FUSED: one thread per query i. Two passes over its causal window [max(0,i-W+1), i], decoding K then V inline.
__global__ void cf_kv_attn_fused_kernel(const uint32_t *kw, const int32_t *kb, const int32_t *kl, int kmax,
                                        const uint32_t *vw, const int32_t *vb, const int32_t *vl, int vmax,
                                        const float *kscale, const float *vscale, const float *Q, float *out,
                                        int seq, int dim, int nq, int window, int block, int zero, float sscale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nq) return;
  float qi[CF_KV_MAX_HEAD_DIM];
  for (int c = 0; c < dim; ++c) qi[c] = Q[(long)i * dim + c];
  int lo = i - window + 1; if (lo < 0) lo = 0;
  int nk = i - lo + 1;
  float sc[CF_KV_MAX_WINDOW];
  float mx = -1e30f;
  for (int jj = 0; jj < nk; ++jj) {                        // pass 1: scores = Q·dequant(K[j]) (decode K inline)
    int j = lo + jj;
    int pos = cf_kv_row_pos(kw, kb, kl, kmax, block, (long)j * dim);
    float s = 0.f;
    for (int c = 0; c < dim; ++c) {
      int sl = cf_bh_decode_at(kw, kl, kmax, pos); int q = sl & 0xFF; pos += sl >> 8;
      s += qi[c] * ((float)(q - zero) * kscale[c]);        // per-channel dequant
    }
    s *= sscale;
    sc[jj] = s;
    if (s > mx) mx = s;
  }
  float sum = 0.f;
  for (int jj = 0; jj < nk; ++jj) { sc[jj] = expf(sc[jj] - mx); sum += sc[jj]; }
  float inv = 1.f / sum;
  float acc[CF_KV_MAX_HEAD_DIM];
  for (int c = 0; c < dim; ++c) acc[c] = 0.f;
  for (int jj = 0; jj < nk; ++jj) {                        // pass 2: out = Σ softmax · dequant(V[j]) (decode V inline)
    int j = lo + jj;
    float wj = sc[jj] * inv, vs = vscale[j];               // per-token dequant scale
    int pos = cf_kv_row_pos(vw, vb, vl, vmax, block, (long)j * dim);
    for (int c = 0; c < dim; ++c) {
      int sl = cf_bh_decode_at(vw, vl, vmax, pos); int q = sl & 0xFF; pos += sl >> 8;
      acc[c] += wj * ((float)(q - zero) * vs);
    }
  }
  for (int c = 0; c < dim; ++c) out[(long)i * dim + c] = acc[c];
}

// REFERENCE stage 1: decode the whole KV store to dense fp32 tiles (the materialised intermediate the fused path
// avoids). One thread per row. K per-channel, V per-token dequant.
__global__ void cf_kv_decode_dense_kernel(const uint32_t *words, const int32_t *boff, const int32_t *lut, int maxlen,
                                          int block, int seq, int dim, int zero, const float *scale, int per_token,
                                          float *dense) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= seq) return;
  int pos = cf_kv_row_pos(words, boff, lut, maxlen, block, (long)j * dim);
  float srow = per_token ? scale[j] : 0.f;
  for (int c = 0; c < dim; ++c) {
    int sl = cf_bh_decode_at(words, lut, maxlen, pos); int q = sl & 0xFF; pos += sl >> 8;
    float s = per_token ? srow : scale[c];
    dense[(long)j * dim + c] = (float)(q - zero) * s;
  }
}

// REFERENCE stage 2: the SAME windowed attention over the dense tiles -> bit-identical to the fused path.
__global__ void cf_kv_attn_dense_kernel(const float *Kd, const float *Vd, const float *Q, float *out, int seq,
                                        int dim, int nq, int window, float sscale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nq) return;
  float qi[CF_KV_MAX_HEAD_DIM];
  for (int c = 0; c < dim; ++c) qi[c] = Q[(long)i * dim + c];
  int lo = i - window + 1; if (lo < 0) lo = 0;
  int nk = i - lo + 1;
  float sc[CF_KV_MAX_WINDOW];
  float mx = -1e30f;
  for (int jj = 0; jj < nk; ++jj) {
    int j = lo + jj;
    float s = 0.f;
    for (int c = 0; c < dim; ++c) s += qi[c] * Kd[(long)j * dim + c];
    s *= sscale;
    sc[jj] = s;
    if (s > mx) mx = s;
  }
  float sum = 0.f;
  for (int jj = 0; jj < nk; ++jj) { sc[jj] = expf(sc[jj] - mx); sum += sc[jj]; }
  float inv = 1.f / sum;
  float acc[CF_KV_MAX_HEAD_DIM];
  for (int c = 0; c < dim; ++c) acc[c] = 0.f;
  for (int jj = 0; jj < nk; ++jj) {
    int j = lo + jj;
    float wj = sc[jj] * inv;
    for (int c = 0; c < dim; ++c) acc[c] += wj * Vd[(long)j * dim + c];
  }
  for (int c = 0; c < dim; ++c) out[(long)i * dim + c] = acc[c];
}

static cf_status cf_validate_kv_args(const uint32_t *kw, const int32_t *kb, const int32_t *kl, int kmax,
                                     const uint32_t *vw, const int32_t *vb, const int32_t *vl, int vmax,
                                     const float *kscale, const float *vscale, const float *Q, const float *out,
                                     int seq, int dim, int nq, int window, int block, float sscale) {
  if (!kw || !kb || !kl || !vw || !vb || !vl || !kscale || !vscale || !Q || !out)
    return CF_ERR_INVALID_ARGUMENT;
  if (seq <= 0 || dim <= 0 || nq < 0 || nq > seq || window <= 0 || block <= 0 || kmax <= 0 || vmax <= 0)
    return CF_ERR_INVALID_ARGUMENT;
  if (!isfinite(sscale)) return CF_ERR_INVALID_ARGUMENT;
  if (dim > CF_KV_MAX_HEAD_DIM || window > CF_KV_MAX_WINDOW) return CF_ERR_UNSUPPORTED;
  return CF_OK;
}

extern "C" cf_status cf_kv_attn_fused_async(const uint32_t *kw, const int32_t *kb, const int32_t *kl, int kmax,
                                            const uint32_t *vw, const int32_t *vb, const int32_t *vl, int vmax,
                                            const float *kscale, const float *vscale, const float *Q, float *out,
                                            int seq, int dim, int nq, int window, int block, int zero, float sscale,
                                            void *stream) {
  cf_status valid = cf_validate_kv_args(kw, kb, kl, kmax, vw, vb, vl, vmax, kscale, vscale, Q, out,
                                        seq, dim, nq, window, block, sscale);
  if (valid != CF_OK) return valid;
  if (nq == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int t = 64;
  cf_kv_attn_fused_kernel<<<(nq + t - 1) / t, t, 0, s>>>(kw, kb, kl, kmax, vw, vb, vl, vmax, kscale, vscale, Q, out,
                                                         seq, dim, nq, window, block, zero, sscale);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}

// Reference path: materialise dense K,V (into caller-provided scratch Kd,Vd), then the same windowed attention.
extern "C" cf_status cf_kv_attn_dense_async(const uint32_t *kw, const int32_t *kb, const int32_t *kl, int kmax,
                                            const uint32_t *vw, const int32_t *vb, const int32_t *vl, int vmax,
                                            const float *kscale, const float *vscale, const float *Q, float *Kd,
                                            float *Vd, float *out, int seq, int dim, int nq, int window, int block,
                                            int zero, float sscale, void *stream) {
  cf_status valid = cf_validate_kv_args(kw, kb, kl, kmax, vw, vb, vl, vmax, kscale, vscale, Q, out,
                                        seq, dim, nq, window, block, sscale);
  if (valid != CF_OK) return valid;
  if (!Kd || !Vd) return CF_ERR_INVALID_ARGUMENT;
  if (nq == 0) return CF_OK;
  cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
  const int t = 64;
  cf_kv_decode_dense_kernel<<<(seq + t - 1) / t, t, 0, s>>>(kw, kb, kl, kmax, block, seq, dim, zero, kscale, 0, Kd);
  cf_kv_decode_dense_kernel<<<(seq + t - 1) / t, t, 0, s>>>(vw, vb, vl, vmax, block, seq, dim, zero, vscale, 1, Vd);
  cf_kv_attn_dense_kernel<<<(nq + t - 1) / t, t, 0, s>>>(Kd, Vd, Q, out, seq, dim, nq, window, sscale);
  return (cudaPeekAtLastError() == cudaSuccess) ? CF_OK : CF_ERR_CUDA;
}
