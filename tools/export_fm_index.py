"""export_fm_index.py — freeze an RRR-backed FM-index + golden count/locate as a portable binary (.cffm) for M7.

FM-index backward search over the BWT is the search half of the thesis: `count` (substring occurrences) and
`locate` (their text positions) run GPU-resident over the SAME compact, RRR-compressed, GPU-searchable index
built in M4 (`RRRWaveletGPU`). This freezes:
  - the RRR-wavelet of the BWT (identical arrays to .cfrw),
  - the FM `C[]` table (cumulative symbol counts),
  - a succinct sampled suffix array for locate (a packed mark plane over SA positions ≡ 0 mod sa_sample, its
    superblock directory ranked like a wavelet plane, and the sampled SA values),
  - golden `count` and `locate` computed by NAIVE substring matching (ground truth — stronger than a prototype
    match; the prototype's RRR FM-index only exposes count, so native RRR-backed locate is verified vs truth).

The stream is a structured order-1 Markov chain, BWT'd — the skewed regime where the RRR index is entropy-sized.
Depends on warp_compress. The native CUDA kernels (src/cuda/fm_search.cu) must reproduce count AND locate
bit-for-bit. Run:

    python tools/export_fm_index.py out.cffm --n 2000000 --vocab 64 --patterns 512 --plen 4 --sa-sample 16

Binary layout v1 (little-endian):
    magic "CFFM" | u32 version=1 | u64 n | u32 bits | u32 vocab | u32 sigma | u32 nblocks | u32 nsb | u32 cwords
                 | u32 na | u32 owords | u32 sa_sample | u32 mwords_len | u32 msb_len | u32 nsval | u32 npat
                 | u32 patflat | u32 nloc | u64 rrr_bytes
    classes[bits*cwords] u32 | offsets[owords] u32
    rank_a[bits*na] i32 | rank_d[bits*(nsb+1)] u16 | off_a[bits*na] i32 | off_d[bits*(nsb+1)] u16
    offbase[bits] i32 | zeros[bits] i32
    C[sigma] i32 | mwords[mwords_len] u32 | msb[msb_len] i32 | sval[nsval] i32
    pat[patflat] i32 | pstart[npat] i32 | plen[npat] i32
    count_golden[npat] u32 | locoff[npat+1] i32 | locpos[nloc] i32
(`pat` symbols are already shifted +1 into the sentinel'd alphabet, as the backward search consumes them.)
"""
import argparse
import os
import struct

import numpy as np
import sys

sys.path.insert(0, os.path.expanduser("~/warp-solar-system-shaders"))
from warp_compress.fm_index import suffix_array  # noqa: E402
from warp_compress.gpu_fm_index import _pack_plane  # noqa: E402
from warp_compress.gpu_rrr_wavelet import RRRWaveletGPU  # noqa: E402


def markov_stream(n, vocab, alpha, rng):
    trans = rng.dirichlet(np.ones(vocab) * alpha, size=vocab)
    seq = np.empty(n, np.int64)
    seq[0] = 0
    for i in range(1, n):
        seq[i] = rng.choice(vocab, p=trans[seq[i - 1]])
    return seq


def naive_match(seq, pat):
    """Positions in `seq` where the (original-alphabet) pattern occurs, by vectorized rolling comparison."""
    m = len(pat)
    if m == 0 or m > seq.shape[0]:
        return np.zeros(0, np.int64)
    mask = np.ones(seq.shape[0] - m + 1, bool)
    for j in range(m):
        mask &= seq[j: seq.shape[0] - m + 1 + j] == pat[j]
    return np.flatnonzero(mask).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--patterns", type=int, default=512)
    ap.add_argument("--plen", type=int, default=4, help="length of the occurring patterns (locate scans them all)")
    ap.add_argument("--sa-sample", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    seq = markov_stream(a.n, a.vocab, a.alpha, rng)
    s = np.concatenate([seq + 1, [0]])
    n = int(s.shape[0])
    sa = suffix_array(s)
    bwt = s[(sa - 1) % n].astype(np.int64)
    sigma = int(s.max()) + 1
    bits = max(1, (sigma - 1).bit_length())

    wm = RRRWaveletGPU(bwt, bits=bits)
    _, arr = wm.to_host()
    C = np.concatenate([[0], np.cumsum(np.bincount(bwt, minlength=sigma))])[:sigma].astype(np.int32)

    # succinct sampled suffix array for locate: mark SA positions whose value ≡ 0 mod sa_sample, rank the marks.
    marked = (sa % a.sa_sample == 0).astype(np.uint8)
    mwords, msb = _pack_plane(marked)
    mwords, msb = mwords[0].astype(np.uint32), msb[0].astype(np.int32)
    sval = sa[marked == 1].astype(np.int32)

    # patterns: `plen`-grams sampled from the stream (all occur) + a few random long ones (zero-count checks).
    starts_pos = rng.integers(0, a.n - a.plen, a.patterns)
    pats = [[int(x) for x in seq[p:p + a.plen]] for p in starts_pos]
    for _ in range(16):
        pats.append([int(x) for x in rng.integers(0, a.vocab, 10)])  # length-10 random -> almost surely count 0

    flat, pstart, plen, count_g, loc_lists = [], [], [], [], []
    for p in pats:
        pstart.append(len(flat))
        plen.append(len(p))
        flat.extend(x + 1 for x in p)                       # shift into the sentinel'd alphabet
        pos = naive_match(seq, np.asarray(p, np.int64))
        count_g.append(pos.shape[0])
        loc_lists.append(np.sort(pos))
    locoff = np.concatenate([[0], np.cumsum([len(x) for x in loc_lists])]).astype(np.int32)
    locpos = (np.concatenate(loc_lists) if loc_lists else np.zeros(0, np.int64)).astype(np.int32)

    classes = np.ascontiguousarray(arr["classes"], np.uint32)
    offsets = np.ascontiguousarray(arr["offsets"], np.uint32)
    rank_a = np.ascontiguousarray(arr["rank_a"], np.int32)
    rank_d = np.ascontiguousarray(arr["rank_d"], np.uint16)
    off_a = np.ascontiguousarray(arr["off_a"], np.int32)
    off_d = np.ascontiguousarray(arr["off_d"], np.uint16)
    offbase = np.ascontiguousarray(arr["offbase"], np.int32)
    zeros = np.ascontiguousarray(arr["zeros"], np.int32)
    cwords, na, nsb1 = classes.shape[1], rank_a.shape[1], rank_d.shape[1]
    nblocks = (n + 15 - 1) // 15

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"CFFM")
        f.write(struct.pack("<IQIIIIIIIIIIIIIIIQ", 1, n, bits, a.vocab, sigma, nblocks, nsb1 - 1, cwords, na,
                            offsets.shape[0], a.sa_sample, mwords.shape[0], msb.shape[0], sval.shape[0],
                            len(pats), len(flat), locpos.shape[0], wm.index_bytes()))
        for x in (classes, offsets, rank_a, rank_d, off_a, off_d, offbase, zeros, C, mwords, msb, sval,
                  np.asarray(flat, np.int32), np.asarray(pstart, np.int32), np.asarray(plen, np.int32),
                  np.asarray(count_g, np.uint32), locoff, locpos):
            f.write(np.ascontiguousarray(x).tobytes())

    tot = int(locpos.shape[0])
    print(f"wrote {out}")
    print(f"  BWT length={n:,}  vocab={a.vocab}  bits={bits}  sigma={sigma}  sa_sample={a.sa_sample}")
    print(f"  patterns={len(pats)} (plen={a.plen} + 16 zero-count)  total occurrences (locate)={tot:,}")
    print(f"  RRR-FM index {wm.index_bytes() / 1e6:.2f} MB ({wm.index_bytes() * 8 / n:.2f} b/tok)"
          f"  + SA sample {(mwords.nbytes + msb.nbytes + sval.nbytes) / 1e6:.2f} MB")
    print(f"  golden count + locate computed by naive matching (ground truth) ✓")


if __name__ == "__main__":
    main()
