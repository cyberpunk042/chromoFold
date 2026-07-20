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

## Files
- `reference.cfwv` — frozen Warp index + golden queries (git-ignored; regenerate with `make reference`).
- `gpu_access.cu` — loads the reference, runs `cf_access_async`, verifies bit-identity, times both layers.
