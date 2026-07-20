// block_huffman_device.cuh — device-side canonical-Huffman LUT decode (DFloat11-style), shared by the fused
// decode-and-consume kernels (constitution P3). Read `maxlen` bits MSB-first at a bit position -> one table hit
// gives (symbol | code_length << 8), so a thread can decode a value stream inline — inside a GEMM, a gather, a
// dequant — with no separate decompression pass. Faithful port of warp_compress.gpu_block_huffman._decode_k.
#ifndef CHROMOFOLD_DETAIL_BLOCK_HUFFMAN_DEVICE_CUH
#define CHROMOFOLD_DETAIL_BLOCK_HUFFMAN_DEVICE_CUH

#include <cstdint>

// Table-decode one canonical-Huffman code at bit position `pos` (MSB-first). Returns symbol | (code_length << 8).
// `lut` has 2^maxlen entries; codes are length-limited to maxlen at build time so one read always resolves.
__device__ __forceinline__ int cf_bh_decode_at(const uint32_t *words, const int *lut, int maxlen, int pos) {
  int look = 0;
  for (int k = 0; k < maxlen; ++k) {
    int wpos = pos + k;
    look = (look << 1) | (int)((words[wpos >> 5] >> (31 - (wpos & 31))) & 1u);
  }
  return lut[look];
}

#endif // CHROMOFOLD_DETAIL_BLOCK_HUFFMAN_DEVICE_CUH
