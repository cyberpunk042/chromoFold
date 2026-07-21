# M9 hardware end-to-end proof

This round converts the previous compile-time and contract-level integration into a reproducible hardware experiment for the pinned llama.cpp revision and a real GGUF model.

## Proof boundary

Hosted CI validates scripts, schemas, workflow syntax, false-claim rejection, and absence of committed model files. It does not claim successful GPU execution. End-to-end proof exists only after the self-hosted NVIDIA workflow uploads a strict evidence-v2 artifact that passes `validate_evidence_v2.py`.

## Required real execution path

```text
real GGUF model
  -> pinned llama.cpp CUDA graph
  -> real K/V device tensor capture
  -> page-aware prefill splitting
  -> device active-page append
  -> transactional GPU seal
  -> descriptor publication
  -> compressed attention over sealed history and active tail
  -> llama.cpp output tensor
  -> token and logit capture
```

The ChromoFold backend must fail rather than execute dense attention when an operation is unsupported or a compressed dispatch fails.

## Reproduction

```bash
make -f m9-llama-e2e.mk llama-fetch
make -f m9-llama-runtime.mk llama-runtime-apply
make -f m9-llama-graph.mk llama-graph-apply
make -f m9-hardware-e2e.mk hardware-proof-build CUDA_ARCH=80
make -f m9-hardware-e2e.mk hardware-proof-pair \
  MODEL=/models/model.gguf \
  PROMPT=integrations/llama.cpp/e2e/prompts/long-context.txt \
  CUDA_ARCH=80
make -f m9-hardware-e2e.mk hardware-proof-capacity \
  MODEL=/models/model.gguf \
  CUDA_ARCH=80
```

## Evidence v2

A valid bundle records:

- GPU, compute capability, driver, CUDA toolkit, OS, and compiler;
- exact llama.cpp and ChromoFold commits;
- GGUF SHA-256 and size;
- model topology and runtime arguments;
- device append, seal, publication, attention, fallback, and CUDA counters;
- measured process and ChromoFold-owned VRAM;
- prefill, decode, seal, and attention timings;
- token agreement and first divergence;
- logit errors and cosine similarity;
- Compute Sanitizer outcomes;
- dense and ChromoFold context-capacity sweeps.

The strict validator rejects counter-only claims, theoretical-only memory values, missing model hashes, non-finite results, dense fallback, CUDA errors, missing sanitizer results, or evidence where seals and descriptor publications do not reconcile.

## Reproducibility artifact

The self-hosted workflow uploads the complete `build/m9-hardware-e2e` directory, including commands, logs, runtime JSON, dense and ChromoFold run captures, pair comparison, capacity sweep, and final evidence document.

Models are runner-local inputs and must never be committed to the repository.

## Numerical interpretation

Two comparisons are needed:

1. original dense K/V versus ChromoFold, which includes expected int4 quantization error;
2. dequantized compressed K/V versus the compressed attention kernel, which isolates layout and kernel correctness.

The initial proof is correctness-first. It may show worse throughput than dense llama.cpp. Performance claims require measured evidence and are not inferred from compression ratio.

## Completion gate

The hardware milestone is complete only when the artifact proves all of the following:

```text
device_append_calls > 0
seal_successes > 0
descriptor_publications == seal_successes
compressed_attention_launches > 0
sealed_pages_consumed > 0
sealed_values_consumed > 0
active_values_consumed > 0
dense_fallback_launches == 0
cuda_errors == 0
all outputs finite
measured VRAM present
Compute Sanitizer passed
```
