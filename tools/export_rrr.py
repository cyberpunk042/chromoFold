"""export_rrr.py — freeze a Warp RRR bitvector + golden rank1 as a portable binary (.cfrr) for the M4 port.

RRR (Raman-Raman-Rao) stores each T-bit block as (class = popcount, offset = enumerative rank among T-bit words
with that popcount). Skewed planes cost far below 1 bit/bit while rank stays O(1): a superblock sample + a short
in-block class scan + one combinatorial block decode. The native CUDA kernel must reproduce this file's golden
rank1 bit-for-bit. Depends on warp_compress.gpu_rrr. Run:

    python tools/export_rrr.py out.cfrr --n 2000000 --density 0.03 --queries 100000

Binary layout v1 (little-endian):
    magic "CFRR" | u32 version=1 | u64 n | u32 T | u32 SB | u32 nblocks | u32 nsb | u32 cwords | u32 owords
                 | u32 nqueries | f32 density | u64 class_bits | u64 offset_bits
    classes[cwords] u32 | offsets[owords] u32 | sbrank[nsb+1] i32 | sboff[nsb+1] i32
    positions[nqueries] u32 | golden_rank1[nqueries] u32
`class_bits`/`offset_bits` are the stored stream lengths; the RRR footprint is
class_bits + offset_bits + (nsb+1)*8 bytes * 8 (two int32 superblock samples).
"""
import argparse
import os
import struct

import numpy as np
import sys

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.gpu_rrr import S as SB, T, GPURRR, rrr_encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--density", type=float, default=0.03)
    ap.add_argument("--queries", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    bits = (rng.random(a.n) < a.density).astype(np.uint8)
    e = rrr_encode(bits)
    gr = GPURRR(bits)
    pos = rng.integers(0, a.n + 1, a.queries).astype(np.uint32)
    golden = gr.rank1(pos.astype(np.int32)).astype(np.uint32)
    cum = np.concatenate([[0], np.cumsum(bits)]).astype(np.int64)
    assert np.array_equal(golden.astype(np.int64), cum[pos]), "prototype rank1 disagrees with naive count"

    cpk = np.ascontiguousarray(e["cpk"], np.uint32)
    opk = np.ascontiguousarray(e["opk"], np.uint32)
    sbrank = np.ascontiguousarray(e["sbrank"], np.int32)   # len nsb+1
    sboff = np.ascontiguousarray(e["sboff"], np.int32)
    nblocks, nsb = int(e["nblocks"]), int(e["nsb"])
    class_bits, offset_bits = int(e["class_bits"]), int(e["offset_bits"])
    density = float(bits.mean())

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFRR")
        f.write(struct.pack("<IQIIIIIIIfQQ", 1, a.n, T, SB, nblocks, nsb, cpk.size, opk.size, a.queries,
                            density, class_bits, offset_bits))
        for arr in (cpk, opk, sbrank, sboff, pos, golden):
            f.write(arr.tobytes())

    rrr_bits = class_bits + offset_bits + (nsb + 1) * 8 * 8
    print(f"wrote {out}")
    print(f"  n={a.n:,}  density={density:.4f}  T={T}  SB={SB}  nblocks={nblocks:,}  nsb={nsb:,}")
    print(f"  RRR footprint {rrr_bits / a.n:.3f} bits/bit  (packed = 1.000)  golden verified ✓")


if __name__ == "__main__":
    main()
