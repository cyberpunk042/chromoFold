# M11 Native KV Evidence and Crossover Campaign

## What changed

M11 replaced the scalar compressed paged-attention path with a warp-cooperative fixed-width path. One warp owns each query/head pair, head dimensions are distributed across lanes, per-lane query and accumulation state remains in registers, and QK reduction uses warp shuffles.

On the measured RTX 2080 Ti workload, this moved the compressed decoder from roughly 243 ms to 2.53 ms at N=4096: approximately 122 times faster than the original implementation.

## What that does not mean

The 122× result compares the current ChromoFold kernel with an older ChromoFold kernel. It is not a comparison with dense attention.

The fair dense baseline was rebuilt with the same warp decomposition and online-softmax structure. On the measured Qwen2.5-0.5B-shaped workload:

| Workload | Compressed | Dense f16 | Result |
| --- | ---: | ---: | --- |
| N=512 decode | 388 µs | 310 µs | compressed 1.25× slower |
| N=4096 decode | 2528 µs | 1912 µs | compressed 1.32× slower |
| N=4096 batch, Qn=256 | 6185 µs | 4537 µs | compressed 1.36× slower |

The equal-or-better-latency ship criterion is therefore not met on this workload.

## Why the next step changed

Before the warp rewrite, Qn=1 and Qn=256 had nearly equal latency. That indicated a per-token work chain rather than a bandwidth or occupancy bottleneck. Shared-memory staging was therefore not the correct immediate lever.

After the rewrite, the remaining cost is the fixed-width int4 decode arithmetic that dense f16 does not pay. The next scientific question is no longer simply “optimize the kernel.” It is:

> At what workload shape, if any, do the smaller compressed memory reads outweigh the decode arithmetic?

## Crossover campaign

`product/kv-crossover-campaign.json` defines the controlled matrix across:

- context length;
- head dimension;
- query count;
- KV-head count;
- code width.

The campaign requires repeated measurements, bounded coefficient of variation, a structurally fair dense baseline, correctness checks, and memory measurements.

Generate the complete plan:

```bash
python3 tools/kv_crossover_campaign.py plan --output kv-crossover-plan.json
```

Evaluate completed results:

```bash
python3 tools/kv_crossover_campaign.py evaluate kv-crossover-results.json --output kv-crossover-analysis.json
```

Possible outcomes are:

- `NO_CROSSOVER`;
- `PARITY`;
- `WIN`;
- `INCOMPLETE`.

A negative result remains a valid campaign result and must not be hidden.

## Remaining product boundary

Even a kernel-level `WIN` does not establish an end-to-end serving win. The current live llama.cpp path still owns and allocates its dense KV cache. Runtime replacement and exact-artifact qualification remain separate milestones.
