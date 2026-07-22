# Downstream llama.cpp patch — layer 1: the dense↔ChromoFold e2e evidence hook

The maintained downstream llama.cpp patch is what turns the ChromoFold KV integration from a build seam into a
runnable end-to-end comparison (`m9-llama-e2e.mk` / `integrations/llama.cpp/e2e/run_pair.py`). This is **layer 1**
of that patch: it makes the e2e **runnable** and makes it emit **honest** evidence. It deliberately does *not* yet
serve attention from a compressed KV cache — that is layer 2 (see "What's left").

## What layer 1 does

`run_pair.py` drives `llama-cli` via environment variables (`CHROMOFOLD_KV_BACKEND`, `CHROMOFOLD_PAGE_SIZE`,
`CHROMOFOLD_EVIDENCE_PATH`) and expects the binary to write a `*.runtime.json`. Layer 1 supplies that:

- **[`chromofold-cli-evidence.cpp`](chromofold-cli-evidence.cpp)** — a small hook that, on exit, reads the env
  vars and writes the evidence fields only the binary knows: `correctness.finite` (did generation complete) and
  honest `counters`. For the **dense** backend the counters are zero; for **chromofold** it records a
  **`dense_fallback_launches: 1`** — a *loud, in-evidence* fallback (never silent), because this layer runs the
  normal (dense) attention. Generation is therefore numerically exact, and the chromofold evidence is valid at the
  baseline but honestly **fails `--require-claim`** (which needs `compressed_attention_launches > 0`, sealed pages
  consumed, and zero dense fallback).
- **Peak VRAM is measured externally** by `run_pair.py` (it samples the device with `nvidia-smi` while the child
  runs) — a real measurement the binary can't take post-hoc, since the model is freed before it exits.

## The edits (applied to the pinned llama.cpp tree, commit `76f46ad2…`)

1. Add [`chromofold-cli-evidence.cpp`](chromofold-cli-evidence.cpp) to `tools/cli/`.
2. `tools/cli/main.cpp` — call the hook after the CLI returns:
   ```cpp
   int llama_cli(int argc, char ** argv);
   void chromofold_write_cli_evidence(int rc);   // added
   int main(int argc, char ** argv) {
       int rc = llama_cli(argc, argv);
       chromofold_write_cli_evidence(rc);        // added
       return rc;
   }
   ```
3. `tools/cli/CMakeLists.txt` — add the source to the executable:
   ```cmake
   add_executable(${TARGET} main.cpp chromofold-cli-evidence.cpp)
   ```
4. `integrations/llama.cpp/e2e/run_pair.py` (in this repo, already committed) — external peak-VRAM sampling + a
   non-interactive completion flag on the `llama-cli` command.

## Build & run

```sh
cd build/llama.cpp-m11
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
cmake --build build --target llama-cli -j

# from the repo root:
PIN=76f46ad29d61fd8c1401e8221842934bf62a6064
python3 integrations/llama.cpp/e2e/run_pair.py --llama-cli build/llama.cpp-m11/build/bin/llama-cli \
  --model /path/model.gguf --output build/e2e/dense.json      --backend dense      --context 8192 --llama-commit $PIN
python3 integrations/llama.cpp/e2e/validate_evidence.py build/e2e/dense.json                 # expect: valid
python3 integrations/llama.cpp/e2e/run_pair.py --llama-cli build/llama.cpp-m11/build/bin/llama-cli \
  --model /path/model.gguf --output build/e2e/chromofold.json --backend chromofold --context 8192 --llama-commit $PIN
python3 integrations/llama.cpp/e2e/validate_evidence.py build/e2e/chromofold.json            # expect: valid (baseline)
python3 integrations/llama.cpp/e2e/validate_evidence.py build/e2e/chromofold.json --require-claim  # expect: HONEST FAIL
```

## Status of this work

- ✅ Written; **`llama-cli` builds with the hook** (verified).
- ⚠️ Interactive `llama-cli` did not run to completion in this WSL sandbox (instruct-model conversation auto-mode +
  GPU-CLI instability). Independent of the patch; the `llama-cli` commands above should complete on a normal box.
- ✅ **Runnable here via the server path.** [`../e2e/run_pair_server.py`](../e2e/run_pair_server.py) drives the same
  evidence contract through `llama-server` (which serves non-interactive `/completion` and works in this sandbox).
  Verified on the RTX 2080 Ti: the **dense** e2e is `valid`, the **chromofold** e2e is `valid` at the baseline with
  a recorded `dense_fallback_launches: 1`, and `--require-claim` is **honestly rejected** ("compressed attention
  was not exercised"). Both backends generated real tokens with real peak VRAM (~2.3 GB) sampled externally.

  This also gives layer 2 a **verification path that works here**: run the server with `--parallel 1` (single
  sequence — within the supported scope) and, once the compressed backend is wired in, `run_pair_server.py` can
  check `--require-claim` on this GPU.

## What's left (layer 2 — the real claim)

To make the chromofold evidence pass `--require-claim`, the patch must **serve attention from the compressed KV
cache** and route llama's attention through it with **≥99% token parity** to dense and **zero dense fallback**,
for the supported single-sequence, causal, f32 scope (`initial_support` in `runtime/patch_manifest.json`).

**The hard part of that is already implemented and verified.** The compressed paged-attention kernel
(`cf_kv_paged_attention_async`, `src/cuda/paged_kv_attention.cu`) decodes history from compressed pages inside the
kernel and computes attention; `make -f m9-gpu.mk gpu-correctness` checks its output against a dense CPU reference
and passes on this GPU with **max abs error 5.96e-07** (mse 2.8e-14, 128 tokens, head_dim 64) — i.e. compressed
serving is numerically equal to dense within float rounding. The `paged-kv-seam` and `llama-adapter-seam` ABI
tests also pass.

So layer 2 is **not** unwritten compressed-attention math — it is the **graph wiring**: threading
`cf_kv_paged_attention_async` into llama's attention path in place of the dense KV read (single-sequence first),
and recording the real counters.

### The interception seam is found, installed, and mapped (verified on the real model)

Rather than patch llama's graph builder, layer 2 hooks the **ggml graph eval callback** (`cparams.cb_eval`) — a
single, clean, per-node interception point. [`chromofold-kv-callback.{h,cpp}`](chromofold-kv-callback.cpp) is
installed **env-gated** from `common_context_params_to_llama` (one edit in `common/common.cpp`) when
`CHROMOFOLD_KV_BACKEND=chromofold`; with `CHROMOFOLD_KV_MAP_PATH` set it maps the attention/KV nodes and otherwise
never alters the graph (unset env ⇒ llama runs exactly as before). **Verified on `qwen2.5-0.5b-instruct-q2_k` via
`run_pair_server.py`:** the callback fires and yields the concrete hook spec —

| per-layer tensor | shape | dtype | role |
|---|---|---|---|
| `Kcur-L (view)`, `Vcur-L (view)` | `[128, T]` (head_dim 64 × kv_heads 2) | **f32** | **append** into the ChromoFold adapter |
| `Qcur-L (view) (permuted)` | | f32 | query for paged attention |
| `kqv_out-L` | `[896, T]` (14 query heads × 64) | f32 | attention output — **replace** target |

Model config: 24 layers, 2 KV heads, 14 query heads (GQA group 7), head_dim 64, **scalar f32** — all inside the
`initial_support` scope (`patch_manifest.json`). So the two remaining coding increments are now fully specified and
verifiable here:
1. **Append** — on `Kcur-L/Vcur-L (view)` (`ask==false`), `ggml_backend_tensor_get` the f32 rows and
   `cf_llama_kv_append(...)` into the adapter → real compressed pages + real compressed-vs-dense KV bytes in the
   evidence (the memory win, on a real model).
2. **Replace** — on `kqv_out-L`, run `cf_kv_paged_attention_async(Qcur, pages)` and overwrite the output buffer;
   check token parity via `run_pair_server.py --parallel 1` → this is what flips `--require-claim` to a real pass.

## Layer 2 — append: **landed and verified on the live model** (2026-07-22)

The append increment is implemented and runs inside a real `llama-server`. The ChromoFold compressed-KV
engine is now **compiled and linked into `llama-server`** and the graph-eval callback drives it.

**Build integration** (reproducible):
1. `bash integrations/llama.cpp/runtime/build_kv_lib.sh` — precompiles the engine (`chromofold_kv_adapter`,
   `compressed_kv_cache.cu`, `kv_cuda_owner.cu`, `paged_kv_attention.cu`, `kv_gpu_fixture`) into a **`-fPIC`
   static lib** with nvcc, so the CUDA language never has to be enabled in llama's CMake scope (and PIC is
   required because `llama-common` is a shared lib).
2. `common/CMakeLists.txt` — a `GGML_CHROMOFOLD`-guarded block links that `.a` into `llama-common`, adds the
   repo `include/` + `integrations/llama.cpp/` include dirs, and defines `GGML_CHROMOFOLD`. (The auto-generated
   `cmake/chromofold-runtime.cmake` from `apply_runtime_patch.py` is **disabled** — it tried to build the engine
   inside CMake and hit `CUDA::cudart` scope errors; the prebuilt-lib path above replaces it.)
3. `cmake -S . -B build -DGGML_CHROMOFOLD=ON && cmake --build build --target llama-server`.

**What the callback does** (append mode, `CHROMOFOLD_KV_BACKEND=chromofold`): on each per-layer `Kcur-L` (ROPE)
and `Vcur-L` (RESHAPE) node — `[head_dim, kv_heads, T]` f32, contiguous — it `ggml_backend_tensor_get`s the rows
to host, pairs K with V per layer, splits the `head_dim×kv_heads` width into per-kv-head `[T, head_dim]`, and
`cf_llama_kv_append(...)`s into a lazily-created adapter (config inferred from the observed shape; append-only so
`gqa=1`). Adapter stats are written to `CHROMOFOLD_KV_STATS_PATH` after every append. It is a **pure shadow** —
it only *reads* graph tensors, never writes them.

**Verified on `qwen2.5-0.5b-instruct-q2_k` (RTX 2080 Ti, `--parallel 1 --ctx-size 512`, page_size 32):**

| check | result |
|---|---|
| generation vs dense (same prompt/seed) | **token-parity IDENTICAL** — 64/64 tokens byte-for-byte (callback is non-perturbing) |
| append errors / non-finite extractions | **0 / 0** |
| sealed pages / active | **96 sealed, 0 active** (24 layers × 2 kv-heads × 2 pages of 32) |
| compressed size | **270 336 B pages + 9 216 B descriptors = 279 KB** |
| vs dense f32 (`3072·64·2·4`) | **5.6×** smaller |
| vs dense f16 (llama's usual KV) | **2.8×** smaller |

Evidence: [`integrations/llama.cpp/runtime/append_live_kv_stats.json`](append_live_kv_stats.json).

**Honest scope — what this does *not* yet prove:**
- **Extraction-layout round-trip.** The finite check rules out NaN/inf, and byte-identical generation proves the
  shadow doesn't corrupt the graph — but neither proves the *extracted* K/V equal llama's true K/V bit-for-bit (a
  wrong-but-finite permutation would still compress). The rigorous gate is the **replace** step (route attention
  through the compressed pages and compare `kqv_out` to dense); the encode/decode itself is already proven by
  `m9-gpu.mk gpu-correctness` (5.96e-07). Until then this is honestly a **memory measurement over live-but-
  unverified-layout K/V**, not a verified-correct KV compression.
- **Prompt-prefill positions.** This run appended exactly the 64 decode steps per (layer, kv-head); the short
  prompt-prefill batch was not captured (a first-batch pairing detail to chase). Noted, not hidden.

So `--require-claim` still **honestly fails** (compressed attention not yet exercised; that's the replace step).
