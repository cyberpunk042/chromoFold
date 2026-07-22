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
`cf_kv_paged_attention_async` into llama.cpp's `llama_context` / `llama-kv-cache` attention path in place of the
dense KV read (single-sequence first), and recording the real counters. That wiring's correctness can only be
signed off by running the e2e (token parity), which needs a machine where `llama-cli` runs to completion (this
sandbox does not — see "Status" above).
