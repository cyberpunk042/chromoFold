# AGENTS.md

Vendor-neutral agent onboarding for **chromoFold**. The full guide is [`CLAUDE.md`](CLAUDE.md); this file is the
short, tool-agnostic entry point (per the AGENTS.md convention).

## Project in one paragraph

chromoFold is a **GPU-resident, random-access, searchable succinct-data-structure runtime** for LLM data — the
native **C++20 / CUDA C++** engine. It keeps token/KV/adapter data compressed *and* navigable in VRAM, and decodes
only inside the consuming kernel (no full decompressed buffer). Its measured Python/Warp prototype lives at
`~/warp-solar-system-shaders/warp_compress` and is the correctness oracle. Governing spec: [`specs/`](specs/).

## Rules of engagement

1. Obey the **constitution** ([`specs/00-constitution.md`](specs/00-constitution.md), P1–P10). A change that
   violates a principle is out of scope until the principle is amended.
2. **Measured, not asserted** (P7): every performance claim ships with its reproducibility envelope and an honest
   baseline. **Report negative results plainly** — they are as valuable as wins.
3. **Verify before you time:** every new kernel must be bit-for-bit identical to a frozen Warp golden (M0) before
   any benchmark number is trusted.
4. **Device-native** (P5): query APIs take/return device pointers on the caller's stream — no host copy/alloc/sync.

## Build

```sh
make all
make PYTHON=~/warp-solar-system-shaders/.venv/bin/python {bench|experiment-a|rank|rrr|fused}
```

Needs `nvcc` (tested 12.6) and an NVIDIA GPU (default `ARCH=sm_75`; override for other GPUs). The reference-
generation targets import `warp_compress` from the prototype repo — hence its venv python.

## Status & next steps

See the live board in [`specs/03-roadmap.md`](specs/03-roadmap.md). Done: M0/M1/M3/M4 (+ M2 frontier, M6 fused
with an honest boundary). Next: wire RRR under the wavelet levels, then M5 (C++ builder + AVX CPU backend),
M7 (FM backward-search), pybind11.

## Gotchas

See the "Gotchas" section of [`CLAUDE.md`](CLAUDE.md) (venv python; `git add -f Makefile`; format-version bumps
invalidate old `.cfwv`/`.cfrr`; no `CK(...)` inside typed lambdas; `nwords+1` fine-array sentinel).
