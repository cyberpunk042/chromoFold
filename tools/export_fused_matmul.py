"""export_fused_matmul.py — freeze an int4 block-Huffman weight tensor + input + golden fused matmul (.cffw), M6.

The thesis op (constitution P3, roadmap M6 re-target): decode the compressed weights INSIDE the GEMM so the
dequantized (M×K) matrix is NEVER materialised in VRAM — only the compressed store is resident during compute.
This is the LARGE-intermediate fusion the M6 embedding-gather experiment pointed to (there the avoided
intermediate — token ids — was tiny, so fusion lost; here it is the whole dequantized weight matrix, so fusion
wins the memory it was designed to win). Freezes the block-Huffman store (words / block_off / LUT + scale/zero),
the input x, and the prototype's fused output as the golden. The native kernel (src/cuda/fused_matmul.cu) must
reproduce a decode-then-dense GEMM bit-for-bit (fusion is numerically free) and hold only the compressed store.
Depends on warp_compress. Run:

    python tools/export_fused_matmul.py out.cffw --m 2048 --k 2048 --batch 8 --bits 4 --block 64

Binary layout v1 (little-endian):
    magic "CFFW" | u32 version=1 | u32 M | u32 K | u32 B | u32 bits | u32 block | u32 maxlen | u32 zero
                 | u32 nwords | u32 nblocks | u32 lutlen | f32 scale | u64 resident_bytes | u64 dense_bytes
    words[nwords] u32 | block_off[nblocks] i32 | lut[lutlen] i32 | x[B*K] f32 | y_golden[B*M] f32
"""
import argparse
import os
import struct

import numpy as np
import sys

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.gpu_fused_matmul import FusedDecodeMatmul  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    W = (rng.standard_normal((a.m, a.k)) / np.sqrt(a.k)).astype(np.float32)
    x = (rng.standard_normal((a.batch, a.k)) / np.sqrt(a.k)).astype(np.float32)

    fm = FusedDecodeMatmul(W, bits=a.bits, block=a.block)
    y_golden = fm.matmul(x).astype(np.float32)                 # the prototype fused output (decode-in-GEMM)

    params, arr = fm.store.wm.to_host()                        # BlockHuffmanArray: words / block_off / lut
    words = np.ascontiguousarray(arr["words"], np.uint32)
    block_off = np.ascontiguousarray(arr["block_off"], np.int32)
    lut = np.ascontiguousarray(arr["lut"], np.int32)
    maxlen, block = int(params["maxlen"]), int(params["block"])
    scale, zero = float(fm.store.scale), int(fm.store._zero)
    resident, dense = fm.resident_bytes(), fm.dense_bytes()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFFW")
        f.write(struct.pack("<IIIIIIIIIIIfQQ", 1, a.m, a.k, a.batch, a.bits, block, maxlen, zero,
                            words.shape[0], block_off.shape[0], lut.shape[0], scale, resident, dense))
        for x_arr in (words, block_off, lut, np.ascontiguousarray(x.ravel(), np.float32),
                      np.ascontiguousarray(y_golden.ravel(), np.float32)):
            f.write(x_arr.tobytes())

    print(f"wrote {out}")
    print(f"  W {a.m}×{a.k} int{a.bits} block-Huffman (block={block}, maxlen={maxlen})   x {a.batch}×{a.k}")
    print(f"  resident (compressed store) {resident / 1e6:.2f} MB   vs   dense dequantized W {dense / 1e6:.2f} MB"
          f"   => {dense / resident:.1f}× less VRAM held during compute")
    print(f"  scale={scale:.6g}  zero={zero}   golden fused matmul frozen ✓")


if __name__ == "__main__":
    main()
