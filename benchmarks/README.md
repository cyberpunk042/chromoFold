# Benchmarks

## Build & run (M1: CUDA C++ `access`)

```sh
make reference     # freeze a Warp GPUWavelet index + golden access output -> benchmarks/reference.cfwv
make bench         # build the CUDA kernel and verify it bit-for-bit against the golden, with timing
```

Requires: a CUDA toolkit (tested nvcc 12.6), an NVIDIA GPU (default build `ARCH=sm_75`), and the Warp prototype
in `~/warp-solar-system-shaders` (for `make reference`). Override the arch with `make ARCH=sm_80 bench`.

## M1 result — RTX 2080 Ti (sm_75), CUDA 12.6

```
index       : n=1,000,000  levels=8  nwords=31,250  nblocks=3,907  resident 1.13 MB (9.0 bits/tok)
queries     : 100,000
correctness : BIT-IDENTICAL ✓  (0 / 100,000 mismatches vs the frozen Warp golden)
kernel-only : median 0.081 ms  p95 0.094 ms  = 1233 M access/s  (0.81 ns/access)
round-trip  : median 0.273 ms  p95 0.319 ms  =  367 M access/s  (2.73 ns/access)
```

**Reading it.** The kernel-only number matches the Warp reference kernel (~1.23 B/s) — same algorithm, same
hardware, now in native CUDA C++ behind the device-native `cf_access_async` (no host allocation, no implicit
sync). The 3.4× gap to the round-trip layer is pure H2D/D2H transfer — exactly the tax the production
device-native API (device pointers on the caller's stream) is designed to remove (constitution P5). Reporting
the two layers separately is required (P7); a single user-facing number would hide where the cost is.

## M2 result — Experiment A: the price of addressability (RTX 2080 Ti)

`make experiment-a` — raw GPU gather (uint8/16/32) vs wavelet `access`, kernel-only median over 30 reps,
across six vocabularies and three query patterns.

```
  vocab  bits raw B/tk  wav b/tk  pattern       gather ns  wavelet ns   ×cost
    256     8        1      9.00   uniform            0.12       0.74     6.3×
                                   sorted             0.10       0.28     2.8×
                                   contiguous         0.10       0.25     2.6×
 131072    17        4     19.13   uniform            0.11       1.47    13.4×
                                   sorted             0.10       0.67     6.6×
```

**Reading it.** Raw gather is the floor — one memory load, ~0.1 ns, flat across vocab, but no rank, no search,
no entropy coding. The wavelet costs 2–13× more per access (it walks `log₂(V)` levels), and **sorted/contiguous
queries roughly halve that** (coalescing + rank-directory cache reuse — the case for warp-cooperative queries).
On footprint the wavelet is ~`log₂(V)` bits/token, so it *beats* raw where the type wastes bits (V=128K: 19 vs
uint32 32) and is slightly larger on byte-aligned vocab. The honest conclusion (P2/P10): **raw gather wins for
dense small-vocab uniform reads; the wavelet's premium buys rank/select/FM-search and large-vocab footprint** —
the win is capabilities and sparse/search access, not per-element gather speed.

## M3 result — native `rank` + two-level rank directory (`make rank`)

Native CUDA `rank(c, i)` verified bit-for-bit vs the Warp golden, then two rank-directory designs measured
head-to-head — the architecture §6 / brief §5.2 question: does a finer directory's extra memory buy latency?

```
  vocab  bits  correct   linear ns   2-level ns   speedup   lin dir   2lvl dir
    256     8      ✓        0.73        0.45        1.63×     0.50 B/w  2.06 B/w
  32768    15      ✓        1.46        0.80        1.82×     0.50 B/w  2.06 B/w
```

- **linear** (M1 layout): superblock every 8 words + scan ≤7 words + one partial popcount.
- **two-level**: coarse int32 superblock (every 64 words) + a per-word uint16 within-superblock prefix → **zero
  word scan**.

**Decision: adopt two-level.** It is 1.6–1.8× faster (the win grows with level count), and although its directory
is 4× larger it is only ~1.35× the *total* index (the bitplane dominates at 4 B/word) — a clear trade for a
query-bound workload, and a per-deployment knob. A `nwords+1` fine-array sentinel is required (a query at pos=n
hits `word == nwords`).

## M6 result — Experiment D: fused decode+embedding-gather (`make fused`)

Tests the P3 thesis (decode inside the consumer, no intermediate token buffer) against unfused
decode-then-gather. V=32768, 100K queries, fp32 embedding.

```
    dim  unfused     fused-blk   fused-tpq   correctness
     64     427 M/s     151 M/s     104 M/s   BIT-IDENTICAL ✓
    768     100 M/s      23 M/s       2 M/s   BIT-IDENTICAL ✓
```

**Honest result: unfused wins here, at every dim.** The wavelet decode is sequential (its best mapping is one
thread/query — fully parallel), but the embedding-row copy wants block/query coalescing — **opposite ideal
mappings**, which the unfused two-kernel path gives each for free. Fusion's block/query variant serializes the
decode (one active thread/block); the thread/query variant strides the copy (uncoalesced) and collapses at large
dim. And the avoided intermediate is tiny (token ids, 4 B), so there's no memory win either.

**The lesson (the actual contribution): fuse by how expensive the avoided intermediate is.** P3 pays when the
intermediate is LARGE — the prototype's decode-in-matmul avoids the full dequantized weight matrix (10.6× less
VRAM) — or the consumer is LIGHT/SPARSE (KV-page selection, sparse gather). Dense embedding gather is the wrong
showcase. This is the constitution's P7 in action: the negative result is reported, and it makes the thesis
sharper and more useful.

## Files
- `reference.cfwv`, `refs/*.cfwv` — frozen Warp indexes + golden queries + raw tokens (git-ignored; regenerate
  with `make reference` / `make experiment-a`).
- `reference_io.h` — the `.cfwv` v2 loader, shared by the benchmarks.
- `gpu_access.cu` — M1: loads the reference, runs `cf_access_async`, verifies bit-identity, times both layers.
- `frontier.cu` — M2: raw gather vs wavelet access across vocab and pattern (Experiment A).
- `rank_bench.cu` — M3: native rank vs golden + linear vs two-level rank directory.
- `fused_embedding.cu` — M6: fused decode+gather (two mappings) vs unfused (Experiment D).
