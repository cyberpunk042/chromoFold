"""export_sparse_gather.py — freeze an RRR-wavelet token stream + embedding table + golden sparse gather (.cfsg).

The light/sparse-consumer branch of the P3 fusion thesis, and a direct demonstration of P2 (navigable while
compressed): when a consumer touches only K of N positions (sparse/windowed/retrieval attention, KV-page
selection, FM-candidate retrieval), decode ONLY those K token ids from the compressed RRR-wavelet and gather their
embedding rows — never decompressing the whole sequence. The alternative ("decompress-all") reconstructs every
token id into an N-length buffer, then gathers K rows. This freezes the RRR-wavelet of a structured token stream,
an embedding table, and a golden set of (position, embedding) pairs. The native kernels (src/cuda/sparse_gather.cu)
must reproduce the golden gather bit-for-bit, and the benchmark maps where sparse random access beats
decompress-all. Depends on warp_compress. Run:

    python tools/export_sparse_gather.py out.cfsg --n 1000000 --vocab 256 --dim 64 --positions 50000

Binary layout v1 (little-endian):
    magic "CFSG" | u32 version=1 | u64 n | u32 bits | u32 vocab | u32 emb_rows | u32 dim | u32 cwords | u32 nsb
                 | u32 na | u32 owords | u32 npos | u64 rrr_bytes
    classes[bits*cwords] u32 | offsets[owords] u32
    rank_a[bits*na] i32 | rank_d[bits*(nsb+1)] u16 | off_a[bits*na] i32 | off_d[bits*(nsb+1)] u16
    offbase[bits] i32 | zeros[bits] i32
    embedding[emb_rows*dim] f32 | positions[npos] u32 | golden[npos*dim] f32
"""
import argparse
import os
import struct

import numpy as np
import sys

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.gpu_rrr_wavelet import RRRWaveletGPU  # noqa: E402


def markov_stream(n, vocab, alpha, rng):
    trans = rng.dirichlet(np.ones(vocab) * alpha, size=vocab)
    seq = np.empty(n, np.int64)
    seq[0] = 0
    for i in range(1, n):
        seq[i] = rng.choice(vocab, p=trans[seq[i - 1]])
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--positions", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    seq = markov_stream(a.n, a.vocab, a.alpha, rng)     # index the RAW token stream (no BWT): access(pos)=token id
    rrw = RRRWaveletGPU(seq)
    bits = rrw.bits
    emb_rows = 1 << bits                                 # access() can return any value in [0, 2^bits)
    embedding = (rng.standard_normal((emb_rows, a.dim)) / np.sqrt(a.dim)).astype(np.float32)

    pos = rng.integers(0, a.n, a.positions).astype(np.uint32)
    ids = rrw.access(pos.astype(np.int32))
    assert np.array_equal(ids, seq[pos]), "RRR-wavelet access disagrees with the token stream"
    golden = embedding[ids].astype(np.float32)          # decode + gather = the reference

    _, arr = rrw.to_host()
    classes = np.ascontiguousarray(arr["classes"], np.uint32)
    offsets = np.ascontiguousarray(arr["offsets"], np.uint32)
    rank_a = np.ascontiguousarray(arr["rank_a"], np.int32)
    rank_d = np.ascontiguousarray(arr["rank_d"], np.uint16)
    off_a = np.ascontiguousarray(arr["off_a"], np.int32)
    off_d = np.ascontiguousarray(arr["off_d"], np.uint16)
    offbase = np.ascontiguousarray(arr["offbase"], np.int32)
    zeros = np.ascontiguousarray(arr["zeros"], np.int32)
    cwords, na, nsb1 = classes.shape[1], rank_a.shape[1], rank_d.shape[1]

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFSG")
        f.write(struct.pack("<IQIIIIIIIIIQ", 1, a.n, bits, a.vocab, emb_rows, a.dim, cwords, nsb1 - 1, na,
                            offsets.shape[0], a.positions, rrw.index_bytes()))
        for x in (classes, offsets, rank_a, rank_d, off_a, off_d, offbase, zeros,
                  np.ascontiguousarray(embedding.ravel(), np.float32), pos,
                  np.ascontiguousarray(golden.ravel(), np.float32)):
            f.write(x.tobytes())

    print(f"wrote {out}")
    print(f"  n={a.n:,}  vocab={a.vocab}  bits={bits}  dim={a.dim}  positions={a.positions:,}")
    print(f"  RRR-wavelet {rrw.index_bytes() / 1e6:.2f} MB ({rrw.index_bytes() * 8 / a.n:.2f} b/tok)  "
          f"embedding {embedding.nbytes / 1e6:.2f} MB   golden sparse gather verified ✓")


if __name__ == "__main__":
    main()
