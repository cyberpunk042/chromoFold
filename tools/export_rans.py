"""export_rans.py — freeze a block-rANS value array + golden decode + a Huffman size comparison (.cfrs), M4.

rANS (range Asymmetric Numeral Systems, Duda 2013 — the coder in zstd-FSE / nvCOMP gANS) approaches H0 with no
per-symbol overhead, unlike Huffman (up to ~1 bit/symbol). rANS decode is sequential (LIFO), so — as for the
block-Huffman coder — the stream is cut into independent fixed-count blocks: whole-tensor decode stays one GPU
thread per block, random access still works within a block. This freezes the per-block rANS streams + the shared
normalized frequency table, the golden decoded values, and (for the honest crossover) the block-Huffman b/val on
the same data. The native kernel (src/cuda/rans.cu) must reproduce the golden decode bit-for-bit. Depends on
warp_compress. Run:

    python tools/export_rans.py out.cfrs --stream skewed --block 1024

Binary layout v1 (little-endian):
    magic "CFRS" | u32 version=1 | u64 n | u32 block | u32 V | u32 M | u32 nblocks | u32 data_bytes | u32 slotlen
                 | f32 h0 | u64 rans_bits | u64 huff_bits
    data[data_bytes] u8 | byte_off[nblocks] i32 | state0[nblocks] u32 | slot2sym[slotlen] i32
    freq[V] i32 | cum[V] i32 | golden[n] i32
"""
import argparse
import math
import os
import struct

import numpy as np
import sys

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.gpu_block_huffman import BlockHuffmanArray  # noqa: E402
from warp_compress.gpu_rans import BlockRANSArray, _M  # noqa: E402


def make_stream(kind, n, rng):
    if kind == "peaky":       # peaky int4 (H0 ~ 3): Huffman already near-optimal
        return np.clip(np.round(rng.standard_normal(n) * 2), -7, 7).astype(np.int64) + 7
    if kind == "skewed":      # very skewed (H0 < 1): Huffman's per-symbol overhead bites, rANS should win
        return np.where(rng.random(n) < 0.05, rng.integers(0, 15, n), 7).astype(np.int64)
    raise SystemExit(f"unknown stream {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--stream", choices=["peaky", "skewed"], default="skewed")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    vals = make_stream(a.stream, a.n, rng)
    ra = BlockRANSArray(vals, block=a.block)
    bh = BlockHuffmanArray(vals, block=a.block)
    golden = ra.decode()
    assert np.array_equal(golden, vals), "prototype rANS decode disagrees with the source"

    params, arr = ra.to_host()
    data = np.ascontiguousarray(arr["data"], np.uint8)
    byte_off = np.ascontiguousarray(arr["byte_off"], np.int32)
    state0 = np.ascontiguousarray(arr["state0"], np.uint32)
    slot2sym = np.ascontiguousarray(arr["slot2sym"], np.int32)
    freq = np.ascontiguousarray(arr["freq_a"], np.int32)
    cum = np.ascontiguousarray(arr["cum_a"], np.int32)
    V = freq.shape[0]
    nblocks = byte_off.shape[0]

    _, c = np.unique(vals, return_counts=True)
    p = c / c.sum()
    h0 = float(-(p * np.log2(p)).sum())
    rans_bits, huff_bits = ra.size_bits(), bh.size_bits()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFRS")
        f.write(struct.pack("<IQIIIIIIfQQ", 1, a.n, a.block, V, int(_M), nblocks, data.shape[0], slot2sym.shape[0],
                            h0, rans_bits, huff_bits))
        for x in (data, byte_off, state0, slot2sym, freq, cum, golden.astype(np.int32)):
            f.write(np.ascontiguousarray(x).tobytes())

    print(f"wrote {out}")
    print(f"  stream={a.stream}  n={a.n:,}  block={a.block}  V={V}  H0={h0:.3f} b/val")
    print(f"  rANS {rans_bits / a.n:.3f} b/val   vs   Huffman {huff_bits / a.n:.3f} b/val   winner="
          f"{'rANS' if rans_bits < huff_bits else 'Huffman'}   golden decode verified ✓")


if __name__ == "__main__":
    main()
