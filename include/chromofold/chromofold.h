/* chromofold.h — the stable C ABI surface for the ChromoFold native engine.
 *
 * C surface, C++ implementation (constitution P8). The query path is device-native (P5): it consumes and
 * returns device pointers and runs asynchronously on the caller's stream — no host allocation, no implicit
 * synchronization, no PCIe round trip.
 *
 * v0 covers the M1 milestone: packed-wavelet `access`. Superblock width is fixed at CF_WAVELET_SB = 8 words,
 * matching the frozen Warp reference.
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CHROMOFOLD_H */
