"""export_kv_attention.py — freeze an entropy-coded KV cache + query + golden windowed attention (.cfkv), M6/M9.

The KV cache is the long-context memory bottleneck. This is the KV-path instance of the P3 fusion thesis (the
second large-intermediate op the M6 fused-matmul pointed to): store K,V **KIVI per-axis-quantized** (K per-channel,
V per-token — arXiv 2402.02750) then **block-Huffman entropy-coded**, and compute causal-windowed attention by
decoding + dequantizing each attended K/V row **inside** the attention kernel, so the dense dequantized K/V tiles
NEVER materialise in VRAM — only the compressed KV store is resident, and (sparse consumer) only the *windowed*
positions are ever decoded. Freezes both stores + per-axis scales + Q + a golden windowed-attention output. The
native kernel (src/cuda/fused_kv_attention.cu) must reproduce a decode-then-dense attention bit-for-bit. Depends on
warp_compress (BlockHuffmanArray). Run:

    python tools/export_kv_attention.py out.cfkv --seq 4096 --dim 64 --window 256 --bits 4 --block 64

Binary layout v1 (little-endian):
    magic "CFKV" | u32 version=1 | u32 seq | u32 dim | u32 nq | u32 window | u32 bits | u32 block | u32 zero
                 | u32 kmaxlen | u32 knwords | u32 knblocks | u32 klutlen
                 | u32 vmaxlen | u32 vnwords | u32 vnblocks | u32 vlutlen
                 | f32 softmax_scale | u64 resident_bytes | u64 dense_bytes
    kwords[knwords] u32 | kboff[knblocks] i32 | klut[klutlen] i32
    vwords[vnwords] u32 | vboff[vnblocks] i32 | vlut[vlutlen] i32
    kscale[dim] f32 | vscale[seq] f32 | Q[nq*dim] f32 | golden[nq*dim] f32
"""
import argparse
import math
import os
import struct

import numpy as np
import sys

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.gpu_block_huffman import BlockHuffmanArray  # noqa: E402


def quant_axis(T, bits, reduce_axis):
    """KIVI per-axis quantization: one scale per slice, shared across reduce_axis. Returns (q in [0,2lim], scale)."""
    lim = (1 << (bits - 1)) - 1
    scale = np.abs(T).max(axis=reduce_axis, keepdims=True) / lim + 1e-12
    q = (np.clip(np.round(T / scale), -lim, lim).astype(np.int64) + lim)
    return q, scale.astype(np.float32), lim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--nq", type=int, default=0, help="number of query rows (default seq)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    nq = a.nq or a.seq
    rng = np.random.default_rng(a.seed)

    K = (rng.standard_normal((a.seq, a.dim)) / np.sqrt(a.dim)).astype(np.float32)
    V = (rng.standard_normal((a.seq, a.dim)) / np.sqrt(a.dim)).astype(np.float32)
    Q = (rng.standard_normal((nq, a.dim)) / np.sqrt(a.dim)).astype(np.float32)

    Kq, Kscale, lim = quant_axis(K, a.bits, 0)      # per-channel (reduce over seq) -> scale [1, dim]
    Vq, Vscale, _ = quant_axis(V, a.bits, 1)        # per-token   (reduce over dim) -> scale [seq, 1]
    Kscale = Kscale.reshape(a.dim).astype(np.float32)
    Vscale = Vscale.reshape(a.seq).astype(np.float32)
    zero = lim

    ks = BlockHuffmanArray(Kq.ravel(), block=a.block)
    vs = BlockHuffmanArray(Vq.ravel(), block=a.block)
    kp, ka = ks.to_host()
    vp, va = vs.to_host()

    # golden: causal-windowed attention over the DEQUANTIZED KV (what the compressed path must reproduce).
    Kdq = (Kq - zero).astype(np.float32) * Kscale[None, :]
    Vdq = (Vq - zero).astype(np.float32) * Vscale[:, None]
    scale = 1.0 / math.sqrt(a.dim)
    out = np.zeros((nq, a.dim), np.float32)
    for i in range(nq):
        lo = max(0, i - a.window + 1)
        sc = (Q[i] @ Kdq[lo:i + 1].T) * scale
        sc = sc - sc.max()
        w = np.exp(sc)
        w /= w.sum()
        out[i] = (w[:, None] * Vdq[lo:i + 1]).sum(0)
    out = out.astype(np.float32)

    kwords = np.ascontiguousarray(ka["words"], np.uint32)
    kboff = np.ascontiguousarray(ka["block_off"], np.int32)
    klut = np.ascontiguousarray(ka["lut"], np.int32)
    vwords = np.ascontiguousarray(va["words"], np.uint32)
    vboff = np.ascontiguousarray(va["block_off"], np.int32)
    vlut = np.ascontiguousarray(va["lut"], np.int32)

    resident = (kwords.nbytes + kboff.nbytes + klut.nbytes + vwords.nbytes + vboff.nbytes + vlut.nbytes
                + Kscale.nbytes + Vscale.nbytes)
    dense = a.seq * a.dim * 4 * 2                    # dense dequantized K + V (fp32), the avoided intermediate

    o = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    with open(o, "wb") as f:
        f.write(b"CFKV")
        f.write(struct.pack("<IIIIIIIIIIIIIIIIfQQ", 1, a.seq, a.dim, nq, a.window, a.bits, a.block, zero,
                            int(kp["maxlen"]), kwords.shape[0], kboff.shape[0], klut.shape[0],
                            int(vp["maxlen"]), vwords.shape[0], vboff.shape[0], vlut.shape[0],
                            scale, resident, dense))
        for arr in (kwords, kboff, klut, vwords, vboff, vlut, Kscale, Vscale,
                    np.ascontiguousarray(Q.ravel(), np.float32), np.ascontiguousarray(out.ravel(), np.float32)):
            f.write(arr.tobytes())

    print(f"wrote {o}")
    print(f"  K,V {a.seq}×{a.dim} int{a.bits} KIVI (K per-channel, V per-token) block-Huffman   window={a.window}  nq={nq}")
    print(f"  resident (compressed KV) {resident / 1e6:.2f} MB   vs   dense dequantized K+V {dense / 1e6:.2f} MB"
          f"   => {dense / resident:.1f}× less VRAM")
    print(f"  golden windowed attention frozen ✓")


if __name__ == "__main__":
    main()
