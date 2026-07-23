# chromoFold

**A way to keep an AI model's data *compressed in GPU memory* while still being able to read, search, and use it
directly — decompressing only the handful of bytes a computation actually touches, never the whole thing.**

New here? This page explains the whole idea in plain language. For definitions of any unfamiliar word (wavelet
tree, BWT, FM-index, rANS…), keep [`docs/GLOSSARY.md`](docs/GLOSSARY.md) open in another tab — every term is
defined there with an analogy.

---

## The one-sentence version

> Think of a **ZIP file you can search and read from without unzipping it** — and that lives on the GPU next to
> the model that needs it.

Normally, to use compressed data you first decompress it all into memory, then work with it. chromoFold skips the
"decompress it all" step: the data stays small, and each GPU computation decodes *just the pieces it needs, at the
moment it needs them.*

---

## Why this matters (the problem)

Modern AI models are bottlenecked by **memory, not math.** A GPU can do trillions of arithmetic operations per
second, but it has a fixed, small amount of fast memory (VRAM — e.g. 11–80 GB). What fills that memory up?

- **The KV cache** — the model's running "memory" of the conversation so far. It grows with every token and
  dominates VRAM in long-context serving. (Don't know what a KV cache is? → [glossary](docs/GLOSSARY.md).)
- **Weights** — the model's parameters. Bigger models = more weights = more VRAM.
- **Token streams, prompt caches, adapter libraries** — lots of repetitive data held in memory.

When you run out of VRAM you must use a smaller model, a shorter context, or a smaller batch. **Memory is the
wall.** Meanwhile the GPU's math units often sit idle waiting for memory. So there's a trade available:

> **Spend the GPU's spare compute to buy back its scarce memory.** Compress the data; pay a little extra math to
> decode it on the fly. That is chromoFold's entire thesis, and the project's constitution calls it
> **"compute-for-memory."**

---

## The key idea (and why it's not "just compression")

Ordinary compression (ZIP, gzip) has two problems for this use case:

1. **You have to decompress the whole thing to use any of it.** That defeats the purpose — the decompressed copy
   is exactly the big buffer you were trying to avoid.
2. **You can't search or index it while it's compressed.**

chromoFold uses **succinct data structures** — compressed representations that stay *usable while compressed*.
Three properties make it work (these are principles **P2** and **P3** in the [constitution](specs/00-constitution.md)):

- **Navigable while compressed (P2).** You can jump to position *i* (`access`), count how many times a symbol
  appears before position *i* (`rank`), and even do full substring search (`count`/`locate`) — all *directly on
  the compressed bytes*, in `O(1)` or `O(log n)` time. No decompression pass.
- **Decode only inside the consumer (P3).** When a GPU kernel (say, a matrix multiply or an attention step) needs
  a value, it decodes that one value *in its own registers*, uses it, and moves on. A fully-decompressed buffer
  **never exists in memory.** This is called **fusion**.
- **Lossless over the chosen precision (P4).** Compression here is *exactly* reversible. If the model already
  quantized its weights to 4-bit, chromoFold stores those 4-bit values losslessly — accuracy is the quantizer's
  job, not ours.

The payoff is measured not as a compression ratio but as a **workload win**: fit a longer context, a bigger
batch, or more adapters in the same GPU memory, at equal-or-better speed.

**A guided tour of how all the pieces fit together is in [`docs/CONCEPTS.md`](docs/CONCEPTS.md).**

---

## What's built, and what it proved

This repo is the **native C++20 / CUDA engine**. Every result below is **verified bit-for-bit** against an
independent reference (a "golden" file) before any speed is reported — that's principle **P0**: *a new kernel is
not trusted until it reproduces the reference exactly.* Full explanations of each line, in plain language, are in
[`docs/RESULTS.md`](docs/RESULTS.md). Measured on an NVIDIA RTX 2080 Ti (2018-era gaming GPU).

| What | Plain-language result |
|---|---|
| **Read a compressed token** (`access`) | Jump to any position in a compressed sequence in ~5 nanoseconds, exact. |
| **Entropy-sized & searchable index** (RRR wavelet) | A searchable index that is **smaller than the raw data** (5.8 bits/token, below the information-theoretic floor of a naive layout) yet still supports instant random access. |
| **Two entropy coders, honestly compared** (Huffman vs rANS) | rANS packs skewed data **2.2× smaller** than Huffman in its sweet spot; Huffman wins elsewhere. We show the exact crossover instead of cherry-picking. |
| **Substring search** (FM-index `count`/`locate`) | Find every occurrence of a pattern, and *where* it is, running entirely on the compressed data in VRAM — ~19 million pattern-searches/second. |
| **Fused decode-in-matmul** | Multiply by a compressed weight matrix that is **never** decompressed into memory: **10.6× less VRAM held during the compute**, and *faster* at large sizes. |
| **Fused KV-cache attention** | Attention over a compressed KV cache: **7.7–8.3× less KV memory**, letting you hold a much longer context. |
| **Sparse gather** | When you only need a few positions out of millions, reading them directly is **up to ~18× faster** than decompressing everything first. |
| **Cross-request dedup** (delta clusters) | 256 near-identical requests sharing one prompt cost **25× less memory** than storing 256 copies; adding a conversation turn is **63× cheaper** than duplicating. |
| **Build the index on the GPU** | Constructing the searchable index (suffix array) runs **21–23× faster on the GPU** than on the CPU, bit-identical. |
| **Self-hosted build** | The whole searchable index builds in plain C++ with **no Python dependency**, and the GPU answers queries bit-identically to a CPU reference. |

**We also report our failures.** The first attempt at fusion (on an embedding lookup) *lost* — the thing it
avoided building was too small to matter. That negative result is kept in [`docs/RESULTS.md`](docs/RESULTS.md) and
[the roadmap](specs/03-roadmap.md), because it taught us the rule: *fuse only when the avoided intermediate is
large, or the consumer is sparse.* Honesty about negatives is the project's brand (principle **P7**).

### Beyond the engine: runtime, packaging, and release qualification

The table above is the **engine** (milestones M0–M8). On top of it the repository also has the layers that turn it
into a real serving runtime — that's the spec's **M9** ("integrate with one real inference path") and the
productionization work beyond it:

- **Packaging** (`packaging/`): `libchromofold.so` + a stable C ABI + a capability descriptor + CPU-only
  conformance seams, for a native runtime to link.
- **Runtime integration** (`integrations/llama.cpp/`): a real llama.cpp **compressed paged-KV backend**
  (`--kv-cache-backend chromofold`, honest-degrade — no silent fallback) plus a milestone stack (multi-GPU,
  scheduler, security, disaggregated, distributed, production server), each with a machine-checkable evidence
  schema.
- **Release qualification** (`rc1-*.mk`, `tools/chromofold_*`): a serving adapter + hardware harness that runs
  nine failure/lifecycle scenarios and produces a signed **PASS / FAIL / INCOMPLETE** evidence artifact — a
  release is gated on a validated PASS, never on a green build.

These are surveyed, with an honest note on what has and hasn't been independently verified, in
**[`docs/INTEGRATION.md`](docs/INTEGRATION.md)**.

---

## Quickstart

You need an NVIDIA GPU, the CUDA toolkit (`nvcc`, tested 12.6), and a C++ compiler. Some benchmarks regenerate
their reference file from the sibling Python prototype; those need its virtual environment's Python (see below).

```sh
make all            # build every benchmark

# Each of these builds a golden reference, runs the kernel, and prints "BIT-IDENTICAL ✓" plus the numbers.
# Targets that regenerate a reference need the prototype's Python — pass it via PYTHON=...
PY=~/warp-solar-system-shaders/.venv/bin/python

make PYTHON=$PY rrr-wavelet    # searchable, entropy-sized index: read + count on compressed data
make PYTHON=$PY rans           # entropy coders compared: rANS vs Huffman, the honest crossover
make PYTHON=$PY fm-search      # substring search (count + locate) on the compressed index
make PYTHON=$PY fused-matmul   # multiply by a compressed weight matrix, never decompressing it
make PYTHON=$PY kv-attention   # attention over a compressed KV cache
make PYTHON=$PY sparse-gather  # read a few positions vs decompress-everything
make delta                     # cross-request dedup (this one needs no Python — pure numpy reference)
make suffix-array              # build the search index on the GPU vs CPU (no Python)
make build-index               # build the whole index in C++ (no Python), verify GPU == CPU
```

Each command prints a small table you can read top to bottom; [`docs/RESULTS.md`](docs/RESULTS.md) walks through
exactly what each column means.

> **Note on the reference files.** The `benchmarks/refs/*.cf*` "golden" files are *generated*, not committed (they
> are git-ignored and regenerate on demand). A fresh checkout builds them the first time you run a target.

---

## How to read this repository

Start at the top and go as deep as you like:

| If you want to… | Read |
|---|---|
| Understand the idea in plain English | **this README** |
| Look up any unfamiliar term | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) |
| Understand how the pieces fit together | [`docs/CONCEPTS.md`](docs/CONCEPTS.md) |
| See every result explained, with caveats | [`docs/RESULTS.md`](docs/RESULTS.md) |
| See the guiding principles (the "rules") | [`specs/00-constitution.md`](specs/00-constitution.md) |
| See the milestone-by-milestone log + numbers | [`specs/03-roadmap.md`](specs/03-roadmap.md) |
| See the strategic direction (where it competes, where it shouldn't) | [`docs/SEARCHABLE_WORKLOADS.md`](docs/SEARCHABLE_WORKLOADS.md) + visual memos in [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) |
| Build / current status / gotchas | [`CLAUDE.md`](CLAUDE.md) |
| Understand the runtime / packaging / release-qualification layers | [`docs/INTEGRATION.md`](docs/INTEGRATION.md) |
| Use it as a C library | [`packaging/README.md`](packaging/README.md) |

### Where the code lives

```
include/chromofold/            the stable C interface (the "contract" other programs call)
  chromofold.h                   access / rank / embedding-gather
  detail/*.cuh, *.hpp            the reusable GPU + CPU building blocks (well-commented, header-only)
src/cuda/*.cu                   the CUDA kernels — one file per capability (access, rank, rrr, fm_search, …)
benchmarks/*.cu                 each capability's verify-and-measure harness ("is it exact? how fast?")
tools/*.py, build_index.cpp     build the golden reference files (Python prototype exports; native C++ builder)
specs/                          the design documents (constitution → spec → architecture → roadmap → benchmarks)
packaging/                      the shared-library (.so) + C-ABI conformance harness
docs/                           these plain-language guides
```

---

## The honest-science rules (very short)

This project is run like a lab notebook. The full list is the [constitution](specs/00-constitution.md); the spirit:

1. **Measured, not asserted.** Every number carries the hardware/settings it was measured on and an honest
   baseline to compare against.
2. **Report negatives.** When something loses, we say so and extract the lesson.
3. **Bit-identical first.** Correctness (exact match to an independent reference) is proven *before* any speed
   claim is made.
4. **The proof is a workload, not a ratio.** The finish line (milestone M9) is a real model fitting more in the
   same GPU, not a compression number.

---

## Relationship to the research prototype

The ideas were first validated in a Python/[Warp](https://github.com/NVIDIA/warp) research prototype (`warp_compress`,
341 tests) that lives in a sibling repo. That prototype is the **correctness oracle** (its outputs are the goldens
this engine must match) and the **performance floor** (the numbers this engine must meet or beat). How the two
stay in sync is documented in [`docs/PROJECT_SYNC.md`](docs/PROJECT_SYNC.md) and
[`specs/05-porting-map.md`](specs/05-porting-map.md).

## Name

"chromoFold" is a nod to genomics — the same succinct-index machinery (BWT, FM-index, wavelet trees) that lets DNA
aligners search a whole genome in a compact index is repurposed here for LLM data. It is **not** literal
DNA-folding and **not** a general-purpose file compressor.

## License

See [`LICENSE`](LICENSE).
