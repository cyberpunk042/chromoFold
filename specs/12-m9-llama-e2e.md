# M9 pinned llama.cpp workload proof

## Purpose

This round freezes the external runtime and makes every future real-model claim reproducible. The pinned llama.cpp revision is `76f46ad29d61fd8c1401e8221842934bf62a6064`.

The harness separates three facts that must not be conflated:

1. the pinned checkout and integration anchors are intact;
2. the ChromoFold-enabled binary builds and runs;
3. real-model evidence proves compressed attention was actually exercised without dense fallback.

## Reproduction

```bash
make -f m9-llama-e2e.mk llama-fetch
make -f m9-llama-e2e.mk llama-patch-check
make -f m9-llama-e2e.mk llama-build-dense
make -f m9-llama-e2e.mk llama-build-chromofold
make -f m9-llama-e2e.mk llama-dense-evidence MODEL=/models/model.gguf
make -f m9-llama-e2e.mk llama-chromofold-evidence MODEL=/models/model.gguf
make -f m9-llama-e2e.mk llama-capacity MODEL=/models/model.gguf
```

## Evidence gates

A ChromoFold result is rejected unless all of the following are present:

- exact llama.cpp and ChromoFold commit IDs;
- model SHA-256;
- finite outputs;
- successful prefill and decode;
- positive compressed-attention launch count;
- positive sealed-value consumption;
- zero dense fallback launches;
- measured peak VRAM and KV bytes;
- at least 99% top-1 token agreement;
- a completed context-capacity case.

The validator intentionally rejects a successful process that does not emit backend counters. Text generation alone is not proof that the compressed path ran.

## Paired protocol

Dense and ChromoFold cases must use the same:

- pinned llama.cpp commit;
- model bytes;
- prompt;
- seed and temperature;
- generated-token count;
- context size;
- CUDA device and launch environment.

Capacity sweeps stop at the first failed context for each backend and report the largest completed context, not merely the largest successful allocation.

## CI boundary

Hosted CI validates scripts, the exact upstream checkout, required source anchors, and false-claim rejection. It does not have a GPU or model and therefore cannot produce M9 workload evidence.

The self-hosted workflow requires labels:

```text
self-hosted, linux, x64, cuda, chromofold-gpu
```

and an explicit absolute GGUF path. Results are uploaded as artifacts and remain non-claims until `validate_evidence.py --require-claim` succeeds.

## Current non-claim

This PR establishes the immutable runtime pin and evidence machinery. The actual upstream source patch and real-model artifact still require execution and review on the GPU runner. No context multiplier, latency improvement, or model-quality statement follows from hosted CI alone.
