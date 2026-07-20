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

## Files
- `reference.cfwv`, `refs/*.cfwv` — frozen Warp indexes + golden queries + raw tokens (git-ignored; regenerate
  with `make reference` / `make experiment-a`).
- `reference_io.h` — the `.cfwv` v2 loader, shared by both benchmarks.
- `gpu_access.cu` — M1: loads the reference, runs `cf_access_async`, verifies bit-identity, times both layers.
- `frontier.cu` — M2: raw gather vs wavelet access across vocab and pattern (Experiment A).
