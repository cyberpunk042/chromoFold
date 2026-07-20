"""export_rrr_wavelet.py — freeze an RRR-backed wavelet matrix + golden access/rank as a portable binary (.cfrw).

The packed wavelet (`gpu_wavelet.GPUWavelet`) stores every level as n raw bits. `RRRWaveletGPU` stores every level
as an **RRR** bitvector instead (`gpu_rrr`) with two-level superblock samples, so the resident index shrinks toward
the sequence's entropy while `access`/`rank` still run on the GPU. Because the FM-index indexes the **BWT** — whose
bitplanes are skewed by construction — RRR wins there, so this exporter builds a structured (Markov) stream, BWTs
it, and freezes the RRR-wavelet of that BWT. The native CUDA kernel (src/cuda/rrr_wavelet.cu) must reproduce this
file's golden access AND rank bit-for-bit (milestone M4, wavelet wiring). Depends on warp_compress. Run:

    python tools/export_rrr_wavelet.py out.cfrw --n 2000000 --vocab 64 --queries 100000 --rank 100000

Binary layout v1 (little-endian):
    magic "CFRW" | u32 version=1 | u64 n | u32 bits | u32 vocab | u32 nblocks | u32 nsb | u32 cwords | u32 na
                 | u32 owords | u32 nqueries | u32 nrank | u64 rrr_bytes | u64 packed_bytes | f32 h0
    classes[bits*cwords] u32 | offsets[owords] u32
    rank_a[bits*na] i32 | rank_d[bits*(nsb+1)] u16 | off_a[bits*na] i32 | off_d[bits*(nsb+1)] u16
    offbase[bits] i32 | zeros[bits] i32
    pos[nqueries] u32 | access_golden[nqueries] u32
    rank_c[nrank] u32 | rank_i[nrank] u32 | rank_golden[nrank] u32

`width`/`binom` are constants (not stored — the loader rebuilds them). `rrr_bytes`/`packed_bytes` let the benchmark
report the entropy-size win without recomputation; `h0` is the BWT's order-0 entropy (the frontier RRR reaches for).
"""
import argparse
import math
import os
import struct

import numpy as np
import sys

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.fm_index import suffix_array  # noqa: E402
from warp_compress.gpu_rrr_wavelet import RRRWaveletGPU  # noqa: E402
from warp_compress.gpu_wavelet import GPUWavelet  # noqa: E402


def markov_stream(n, vocab, alpha, rng):
    """A structured order-1 Markov stream: its BWT has skewed bitplanes -> RRR beats packed. The FM-index's world."""
    trans = rng.dirichlet(np.ones(vocab) * alpha, size=vocab)
    seq = np.empty(n, np.int64)
    seq[0] = 0
    for i in range(1, n):
        seq[i] = rng.choice(vocab, p=trans[seq[i - 1]])
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration; lower = more skewed = RRR wins")
    ap.add_argument("--queries", type=int, default=100_000)
    ap.add_argument("--rank", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    seq = markov_stream(a.n, a.vocab, a.alpha, rng)
    # index the BWT of the stream (like the FM-index): append a unique sentinel 0, shift symbols up by 1.
    s = np.concatenate([seq + 1, [0]])
    bwt = s[(suffix_array(s) - 1) % s.shape[0]].astype(np.int64)
    m = int(bwt.shape[0])

    rrw = RRRWaveletGPU(bwt)
    pk = GPUWavelet(bwt)
    bits = rrw.bits

    # goldens straight from the prototype (the correctness oracle), then cross-checked against a naive count.
    pos = rng.integers(0, m, a.queries).astype(np.uint32)
    acc_golden = rrw.access(pos.astype(np.int32)).astype(np.uint32)
    assert np.array_equal(acc_golden, bwt[pos]), "prototype access disagrees with the raw BWT"

    sigma = int(bwt.max()) + 1
    rank_c = rng.integers(0, sigma, a.rank).astype(np.uint32)
    rank_i = rng.integers(0, m + 1, a.rank).astype(np.uint32)
    rank_golden = rrw.rank(rank_c.astype(np.int32), rank_i.astype(np.int32)).astype(np.uint32)
    # spot-check the prototype rank against a naive prefix count on a small sample (full count is O(n·rank))
    for j in rng.integers(0, a.rank, 32):
        naive = int(np.count_nonzero(bwt[: rank_i[j]] == rank_c[j]))
        assert rank_golden[j] == naive, f"prototype rank disagrees with naive count at {j}"

    _, arr = rrw.to_host()
    classes = np.ascontiguousarray(arr["classes"], np.uint32)   # (bits, cwords)
    offsets = np.ascontiguousarray(arr["offsets"], np.uint32)   # flat
    rank_a = np.ascontiguousarray(arr["rank_a"], np.int32)      # (bits, na)
    rank_d = np.ascontiguousarray(arr["rank_d"], np.uint16)     # (bits, nsb+1)
    off_a = np.ascontiguousarray(arr["off_a"], np.int32)
    off_d = np.ascontiguousarray(arr["off_d"], np.uint16)
    offbase = np.ascontiguousarray(arr["offbase"], np.int32)    # (bits,) already in BITS (word base * 32)
    zeros = np.ascontiguousarray(arr["zeros"], np.int32)        # (bits,)

    cwords, na, nsb1 = classes.shape[1], rank_a.shape[1], rank_d.shape[1]
    nblocks = (m + 15 - 1) // 15  # T = 15
    rrr_bytes, packed_bytes = rrw.index_bytes(), pk.index_bytes()
    h0 = -sum((c / m) * math.log2(c / m) for c in np.bincount(bwt) if c)

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFRW")
        f.write(struct.pack("<IQIIIIIIIIIQQf", 1, m, bits, a.vocab, nblocks, nsb1 - 1, cwords, na,
                            offsets.shape[0], a.queries, a.rank, rrr_bytes, packed_bytes, h0))
        for arrx in (classes, offsets, rank_a, rank_d, off_a, off_d, offbase, zeros,
                     pos, acc_golden, rank_c, rank_i, rank_golden):
            f.write(np.ascontiguousarray(arrx).tobytes())

    print(f"wrote {out}")
    print(f"  BWT length={m:,}  vocab={a.vocab}  bits={bits}  nblocks={nblocks:,}  nsb={nsb1 - 1:,}  na={na:,}")
    print(f"  RRR wavelet {rrr_bytes / 1e6:.2f} MB ({rrr_bytes * 8 / m:.2f} b/tok)  vs  packed "
          f"{packed_bytes / 1e6:.2f} MB ({packed_bytes * 8 / m:.2f} b/tok)  => {packed_bytes / rrr_bytes:.2f}× smaller")
    print(f"  H0(BWT)={h0:.2f} b/tok   access + rank golden verified vs naive ✓")


if __name__ == "__main__":
    main()
