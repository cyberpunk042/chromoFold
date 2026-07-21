# M9 pinned llama.cpp runtime integration

## Purpose

This round turns the existing cache adapter and evidence harness into a strict runtime integration boundary for the pinned llama.cpp revision `76f46ad29d61fd8c1401e8221842934bf62a6064`.

It does not claim a completed production backend yet. It establishes the code, patching, build, counter, and evidence contracts that the final upstream wiring must satisfy.

## Runtime invariants

When the selected backend is ChromoFold:

- CUDA is mandatory;
- dense fallback is disabled;
- K/V tensor topology must match the configured cache;
- the initial bridge accepts contiguous fp32 K/V tensors only;
- every compressed launch is counted;
- sealed and active values consumed are counted separately;
- CUDA failures and rejected operations are retained in evidence;
- evidence is incomplete until compressed launches and sealed-value consumption are both positive;
- evidence is invalid when any dense fallback launch occurs.

## Pinned patch model

The runtime patcher operates only on the exact pinned llama.cpp commit. It:

1. validates required upstream files;
2. copies the ChromoFold runtime into an isolated overlay;
3. creates a dedicated `chromofold-runtime` CMake target;
4. installs a top-level CMake hook exactly once;
5. creates version-sensitive runtime hook files;
6. records hashes for generated files;
7. verifies idempotence and generated-file drift.

The generated checkout contains `.chromofold-runtime-state.json`, which is evidence of the exact patch state and is not a release artifact.

## Build boundary

`GGML_CHROMOFOLD` is disabled by default. Enabling it without `GGML_CUDA` must fail configuration.

The runtime library currently compiles:

- the llama adapter;
- the strict runtime bridge;
- the compressed cache manager;
- deterministic page encoding;
- CUDA page ownership;
- paged CUDA attention.

## Remaining upstream wiring

The generated hook files are intentionally narrow. The next implementation work inside the same branch must connect them to:

- CLI registration in `common/arg.cpp`;
- context ownership in `src/llama-context.cpp`;
- K/V production and append points;
- CUDA attention dispatch;
- cache clear and sequence operation forwarding;
- evidence emission at process shutdown.

No real-model claim is allowed until these hooks are connected and the PR #9 evidence validator accepts a hardware artifact.

## Validation

```bash
make -f m9-llama-runtime.mk llama-runtime-check
make -f m9-llama-runtime.mk llama-runtime-apply
make -f m9-llama-runtime.mk llama-runtime-verify
make -f m9-llama-runtime.mk llama-runtime-compile
```

Hosted CI also proves that applying the runtime mutation twice does not duplicate the CMake hook and that ChromoFold configuration without CUDA is rejected.
