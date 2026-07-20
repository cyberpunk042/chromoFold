"""export_reference.py — freeze a Warp GPUWavelet index + golden access output as a portable binary.

This is M0/M1's reference oracle: the native CUDA `access` kernel must reproduce this file's golden output
bit-for-bit. Depends on the measured prototype (warp-solar-system-shaders/warp_compress). Run:

    python tools/export_reference.py [out.cfwv] [--n N] [--vocab V] [--queries Q]

Binary layout v2 (little-endian):
    magic "CFWV" | u32 version=2 | u32 levels | u64 n | u32 nwords | u32 nblocks | u32 nqueries
                 | u32 token_bytes | u32 vocab
    words[levels*nwords] u32 | superblocks[levels*(nblocks+1)] i32 | zeros[levels] i32
    positions[nqueries] u32 | golden_tokens[nqueries] u32 | raw_tokens[n * token_bytes]
`nblocks` is the superblock count; the sb table has nblocks+1 entries per level. SB (words/superblock) = 8.
`raw_tokens` is the minimal-width packed token stream (the raw-GPU-gather baseline for Experiment A):
token_bytes = 1 if levels<=8, 2 if levels<=16, else 4.
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.gpu_wavelet import SB, GPUWavelet  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "benchmarks",
                                                           "reference.cfwv"))
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--queries", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    p = 1.0 / np.arange(1, a.vocab + 1); p /= p.sum()                 # Zipfian, like the prototype demo
    seq = rng.choice(a.vocab, size=a.n, p=p).astype(np.int64)
    gw = GPUWavelet(seq)
    words = gw.words.numpy().astype(np.uint32)                       # [levels, nwords]
    sb = gw.sb.numpy().astype(np.int32)                             # [levels, nblocks+1]
    zeros = gw.zeros.numpy().astype(np.int32)                       # [levels]
    levels, nwords = words.shape
    nblocks = sb.shape[1] - 1

    pos = rng.integers(0, a.n, a.queries).astype(np.uint32)
    golden = gw.access(pos.astype(np.int32)).astype(np.uint32)
    assert np.array_equal(golden, seq[pos]), "prototype access disagrees with the sequence"
    assert SB == 8, f"exporter assumes SB=8, prototype has SB={SB}"

    token_bytes = 1 if levels <= 8 else (2 if levels <= 16 else 4)
    raw_dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[token_bytes]
    raw = seq.astype(raw_dtype)                                      # minimal-width raw token stream (gather baseline)

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFWV")
        f.write(struct.pack("<IIQIIIII", 2, levels, a.n, nwords, nblocks, a.queries, token_bytes, a.vocab))
        f.write(np.ascontiguousarray(words).tobytes())
        f.write(np.ascontiguousarray(sb).tobytes())
        f.write(np.ascontiguousarray(zeros).tobytes())
        f.write(np.ascontiguousarray(pos).tobytes())
        f.write(np.ascontiguousarray(golden).tobytes())
        f.write(np.ascontiguousarray(raw).tobytes())

    idx_b = words.nbytes + sb.nbytes + zeros.nbytes
    print(f"wrote {out}")
    print(f"  n={a.n:,}  vocab={a.vocab}  levels={levels}  nwords={nwords:,}  nblocks={nblocks:,}  "
          f"queries={a.queries:,}")
    print(f"  wavelet index {idx_b/1e6:.2f} MB ({idx_b*8/a.n:.2f} bits/tok)  raw tokens {token_bytes} B/tok  "
          f"golden verified ✓")


if __name__ == "__main__":
    main()
