# M9 llama.cpp graph wiring

This round binds the pinned llama.cpp execution graph to the ChromoFold runtime contract without weakening the no-fallback evidence policy.

## Implemented

- graph options and model-topology contracts;
- runtime ownership bound to llama context lifetime;
- graph K/V append entry point;
- compressed-attention dispatch accounting;
- explicit dense-dispatch violation path;
- deterministic pinned-checkout mutation;
- generated CLI, context-owner, graph-hook, and CMake files;
- idempotent graph mutation and hash verification;
- compile seam and hosted CI.

## Initial support envelope

- CUDA required;
- causal non-recurrent models;
- uniform KV geometry;
- head dimension at most 128;
- MHA or integer-ratio GQA;
- page sizes 64, 128, or 256;
- linear sequence 0;
- no dense fallback after ChromoFold selection.

## Required upstream call sites

The generated pinned integration exposes explicit hooks for:

1. CLI parsing and validation;
2. context creation after model topology is known;
3. K/V append after each supported layer produces new cache rows;
4. compressed attention dispatch;
5. dense-dispatch rejection;
6. clear and sequence-0 removal;
7. evidence emission during normal shutdown and failures.

## Proof gate

A real-model artifact is valid only when the execution path itself reports:

```text
graph_append_calls > 0
graph_attention_dispatches > 0
compressed_attention_launches > 0
sealed_values_consumed > 0
graph_dense_dispatches == 0
dense_fallback_launches == 0
cuda_errors == 0
```

Compilation, CLI help text, or successful token generation alone are not sufficient evidence.

## Remaining hardware work

The graph contract currently reuses the contiguous FP32 bridge. Device-native FP16/BF16 layout conversion, device-resident active pages, GPU sealing, and the final pinned llama.cpp call-site implementation must be validated on a CUDA runner before claiming end-to-end inference.
