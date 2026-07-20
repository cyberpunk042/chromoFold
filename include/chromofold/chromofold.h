/* chromofold.h — the stable C ABI surface for the ChromoFold native engine.
 *
 * C surface, C++ implementation (constitution P8). The query path is device-native (P5): it consumes and
 * returns device pointers and runs asynchronously on the caller's stream — no host allocation, no implicit
 * synchronization, no PCIe round trip.
 *
 * v0 covers packed-wavelet access/rank, fused embedding gather, and the M6/M9 compressed-KV attention seam.
 * Superblock width is fixed at CF_WAVELET_SB = 8 words, matching the frozen Warp reference.
 */
#ifndef CHROMOFOLD_H
#define CHROMOFOLD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHROMOFOLD_ABI_VERSION 0
#define CF_WAVELET_SB 8 /* 32-bit words per superblock */
#define CF_KV_MAX_HEAD_DIM 128
#define CF_KV_MAX_WINDOW 512

typedef enum cf_status {
  CF_OK = 0,
  CF_ERR_INVALID_ARGUMENT = 1,
  CF_ERR_UNSUPPORTED = 2,
  CF_ERR_CUDA = 3
} cf_status;

/* An immutable, device-resident packed-wavelet index. All pointers are DEVICE pointers.
 * Layout (row-major per level):
 *   bitplanes   : levels * nwords            uint32, bit i of level l = (bitplanes[l*nwords + i/32] >> (i%32)) & 1
 *   superblocks : levels * (nblocks + 1)      int32, cumulative popcount every CF_WAVELET_SB words
 *   zero_counts : levels                      int32, number of 0-bits per level
 */
typedef struct cf_wavelet_view {
  const uint32_t *bitplanes;
  const int32_t *superblocks;
  const int32_t *zero_counts;
  uint64_t token_count; /* n */
  uint32_t levels;      /* ceil(log2(vocab)) */
  uint32_t nwords;      /* (n + 31) / 32 */
  uint32_t nblocks;     /* superblock count; superblocks[] has nblocks + 1 entries per level */
} cf_wavelet_view;

/* Device-native batched access: decode token IDs at `device_positions` into `device_output`.
 * Both arrays are device pointers of length `count`. Runs on `stream`; does NOT synchronize or allocate.
 * `stream` is a cudaStream_t passed as void* to keep this header CUDA-free. Pass NULL for the default stream. */
cf_status cf_access_async(cf_wavelet_view index, const uint32_t *device_positions,
                          uint32_t *device_output, size_t count, void *stream);

/* Device-native batched rank: for each query, occurrences of `device_symbols[t]` in the first
 * `device_positions[t]` tokens (the FM-index primitive). All arrays are device pointers of length `count`. */
cf_status cf_rank_async(cf_wavelet_view index, const uint32_t *device_symbols,
                        const uint32_t *device_positions, uint32_t *device_output, size_t count, void *stream);

/* Fused decode-and-consume (constitution P3): for each position, decode its token id and IMMEDIATELY gather that
 * token's embedding row — a full decompressed token buffer never exists. `embeddings` is [vocab, dim] row-major
 * device memory; `out` is [count, dim] device memory. All pointers are device pointers; runs on `stream`. */
cf_status cf_embedding_gather_async(cf_wavelet_view index, const float *embeddings, uint32_t dim,
                                    const uint32_t *device_positions, float *out, size_t count, void *stream);

/* Fused compressed-KV attention (M6/M9 integration seam).
 *
 * K and V are block-Huffman-coded int values held in device memory:
 *   kw/vw      packed code words
 *   kb/vb      bit offsets, one per coded block
 *   kl/vl      decode lookup tables
 *   kscale     `dim` per-channel K dequantization scales
 *   vscale     `seq` per-token V dequantization scales
 *   Q          [nq, dim] row-major query matrix
 *   out        [nq, dim] row-major output
 *
 * The kernel performs causal windowed attention over query rows 0..nq-1 and KV rows 0..seq-1. It decodes and
 * dequantizes each attended K/V value inside the consumer kernel, so no dense KV buffer is materialized.
 * All pointers are DEVICE pointers. The function allocates nothing, does not synchronize, and launches on `stream`.
 * Current v0 bounds: 0 < dim <= CF_KV_MAX_HEAD_DIM, 0 < window <= CF_KV_MAX_WINDOW, 0 <= nq <= seq, block > 0.
 */
cf_status cf_kv_attn_fused_async(const uint32_t *kw, const int32_t *kb, const int32_t *kl, int kmax,
                                 const uint32_t *vw, const int32_t *vb, const int32_t *vl, int vmax,
                                 const float *kscale, const float *vscale, const float *Q, float *out,
                                 int seq, int dim, int nq, int window, int block, int zero, float sscale,
                                 void *stream);

/* Dense-reference KV path used for verification and integration baselines.
 * It first decodes K and V into caller-provided [seq, dim] fp32 scratch buffers `Kd` and `Vd`, then executes the
 * same causal windowed attention. This API exists to prove bit-identical behavior and quantify the memory avoided
 * by cf_kv_attn_fused_async; it is not the preferred serving path.
 * All pointers are DEVICE pointers. The function allocates nothing, does not synchronize, and launches on `stream`.
 */
cf_status cf_kv_attn_dense_async(const uint32_t *kw, const int32_t *kb, const int32_t *kl, int kmax,
                                 const uint32_t *vw, const int32_t *vb, const int32_t *vl, int vmax,
                                 const float *kscale, const float *vscale, const float *Q,
                                 float *Kd, float *Vd, float *out,
                                 int seq, int dim, int nq, int window, int block, int zero, float sscale,
                                 void *stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHROMOFOLD_H */
