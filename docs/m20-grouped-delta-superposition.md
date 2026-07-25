# M20 — Grouped delta superposition: folding similar tensors into shared-reference atoms

> **Status: DESIGN (proposal).** No kernels, no `.mk`, no measured results yet — this doc pins the transform so it can be built + measured deliberately (P7, P10). It proposes a *new compressed representation*, so if accepted it also warrants a `specs/NN-*.md` entry + a roadmap-board update (the formal `specs/03-roadmap.md` still ends at M11; the `docs/mNN` series runs to M19 — this extends the `docs/mNN` forward track as **M20**).
> **Thesis fit:** a pure P1 move — spend GPU matrix FLOPs to buy back VRAM by exploiting *redundancy across a group of similar tensors* that today are each stored independently.

## The idea (operator, verbatim — sacrosanct)

> "the idea is that if you can have similar things and group them and rather them being defined by a delta from the root or from a third state of it you then having your delta, then you can overlap them into a single atom like shape and play with the matrix to compact them into this new model of fit the most inside GPU memory. trying to find the sweet spot for how much to rely on Processor VS Memory and tried to regain other performances by doing so."
>
> "An even smarter transformation, we keep and keep on ravaging with our matrix operations"

Nothing below overrides those words; the sections that follow are the engineering reading of them, part by part.

## Reading the idea, part by part

| Operator's phrase | Engineering reading |
|---|---|
| "similar things and group them" | **Cluster similar tensors** (weight tiles across layers/experts, LoRA adapters in a library, KV pages) into a group `G = {T₁…Tₙ}` by a similarity metric. Not sequences (that is M8) — *tensors*. |
| "a delta from the root **or from a third state of it**" | Pick a **reference** the group is expressed against. Two modes: **root** = a designated member (or canonical base), like M8's base; **"third state"** = a *synthesized* reference that is **no single member** — a centroid/medoid computed to minimize total residual. The "third state" is the novel bit (M8 uses a designated base, never a computed centroid). |
| "having your delta" | Each member becomes `Tᵢ = R + Δᵢ` — the shared reference `R` plus a per-member residual `Δᵢ`. Similar members ⇒ small, structured `Δᵢ`. |
| "overlap them into a single atom like shape" | **Superimpose** the residual set `{Δᵢ}` into one compact combined structure — the **"atom"** — instead of `n` independent residual blobs. Superposition, not concatenation. |
| "play with the matrix to compact them into this new model" | Treat the stacked residuals `Δ = [Δ₁…Δₙ]` as a matrix and **factor it** — low-rank `Δ ≈ U·Vᵀ`, and/or a **shared basis / dictionary** the members index into. The "atom" is `R` + the factors, not the raw residuals. |
| "fit the most inside GPU memory" | The objective: **maximize resident model size / batch / context** at fixed VRAM (P10 — a workload, not a ratio). |
| "the sweet spot … Processor VS Memory" | The **P1 dial**: how much decode/reconstruct FLOP to spend per byte saved. A group with a rank-`r` factorization pays `O(r)` reconstruct FLOPs per element; `r` is the tunable. |
| "regain other performances" | Beyond capacity: less VRAM **bandwidth** (read `R` once for the group; residual factors are tiny), better cache residency, larger batch/longer context (the P10 win). |
| "keep on ravaging with our matrix operations" | The reconstruct + the consuming op (GEMM / attention) **fuse** — the member is rebuilt `R + UᵢVᵀ` *inside* the consumer kernel (P3), never materialized. Matrix ops are the vehicle end-to-end: factor to store, multiply to consume. |

## Where this sits vs what ChromoFold already has

Grounded against the current object model (`specs/01-spec.md` §2.1) + roadmap:

| Existing mechanism | What it does | Why M20 is not it |
|---|---|---|
| **M8 reference-delta cluster** (`delta_apply.cu`, `cf_delta_fetch`) | base + per-member **sparse `(pos,val)` deltas** over **1-D token/id sequences** (shared prompt prefix, adapter *library as sequences*, conversation turns) | M20 groups **tensors** (2-D weight/KV matrices), against a **computed centroid** ("third state"), and **factors** the residual set — not sparse-point overrides on a designated base sequence. M8 is the nearest cousin + the reuse target for the reference-delta *plumbing*. |
| **M6 block-Huffman decode-in-GEMM** (`cf_fused_matmul_async`, `cf_bh_decode_at`) | fold **one** weight matrix: int4 block-Huffman decoded in-register during matmul, no dense `W` in VRAM | M20 is a layer **above** a single tensor's codec: it removes redundancy **across a group** first, then each of `R`/`U`/`V` can still be block-Huffman-folded by M6. Composes, doesn't replace. |
| **m13 adaptive compression** | **per-payload** codec/precision selection (int2/4/8/raw) + persistent page store | per-tensor, independent. M20 is explicitly **cross-tensor** — its whole value is the correlation m13 leaves on the table. |
| **RRR / rANS / FM-index** | entropy + addressing over a single stream | unchanged; they entropy-code M20's `R` and residual factors like any other stream (P2 addressing preserved). |

**Novelty confirmed:** grouping *tensors*, a *computed-centroid* reference, residual *superposition*, and *cross-tensor low-rank/shared-basis* are all absent today (a vector quantizer is named future work in M9). M20 is new representation ground.

## The transform (build path — P9 "build ≠ query")

Construction is offline / CPU / non-serving accelerator (P9); the serving GPU carries only the compact immutable atom.

1. **Group.** Cluster candidate tensors by similarity (cosine / low residual-norm after alignment). Groups are small (2–64), same shape + same upstream quantizer.
2. **Reference `R`.**
   - *root mode* — designate a member (or a canonical base) as `R`.
   - *third-state mode* — compute a centroid/medoid `R` minimizing `Σᵢ‖Tᵢ − R‖` (then snap `R` onto the quant grid so it is a legal quantized tensor — see P4 below).
3. **Residuals.** `Δᵢ = Tᵢ − R`, exact over the quantizer's grid (integer residuals when the members share the grid).
4. **Superpose + factor (the "atom").** Stack `Δ = [Δ₁…Δₙ]`; compact it with matrix operations:
   - **low-rank** `Δ ≈ U Vᵀ` (rank `r` = the P1 dial), and/or
   - **shared dictionary** `Δᵢ = Σ B·cᵢ` (a small basis `B` + per-member sparse codes `cᵢ`).
   The **atom** = `{R, U, V}` (or `{R, B, {cᵢ}}`) + the mapping member→slice. `R` is read once per group.
5. **Entropy + address.** `R`, `U`, `V`/`B`, `cᵢ` are each handed to the existing codecs (block-Huffman / rANS / RRR) so the atom stays **navigable while compressed** (P2) and every member remains O(1)-addressable via `(group, member)`.

## The query path (decode-in-consumer — P3)

The member is **never** reconstructed into VRAM. Inside the consuming kernel (GEMM or attention):

- for the tile the thread needs, decode the corresponding slice of `R` (via `cf_bh_decode_at`, exactly as M6 does today) **and** the low-rank contribution `UᵢVᵀ` for that tile, add them **in registers**, then multiply-accumulate.
- reconstruct cost per element ≈ M6's decode + `O(r)` FMAs — the "processor vs memory" trade made explicit and tunable by `r`.

This is the "keep ravaging with matrix operations" end state: the reconstruct is itself a small matmul folded into the big one.

## Constitution compliance (the gates this must pass)

| Principle | How M20 satisfies it — or the caveat |
|---|---|
| **P1 compute-for-memory** | Core motivation; the win is cross-tensor VRAM removal, `r` is the compute/memory dial. |
| **P2 navigable while compressed** | Every member stays O(1)-addressable by `(group, member)`; `access`/slice-decode reconstruct any tile without decompressing the group. |
| **P3 decode only inside the consumer** | Reconstruct `R + UᵢVᵀ` in-register inside GEMM/attention; no dense member in global memory. |
| **P4 lossless over the chosen quantization — LOAD-BEARING CAVEAT** | ChromoFold's layer must stay **bit-exact over the quantizer**. So the split is: the **exact** path — `R` snapped to the grid + **exact integer residuals**, entropy-coded losslessly — is ChromoFold's (a redundancy re-encoding, provably lossless). Any **lossy** step (low-rank *truncation* `r < full`, centroid rounding that drops residual bits, a vector quantizer over the codes) is **NOT ChromoFold's to hide**: it must be a **named layer above** with its own error budget + measured accuracy, exactly as ChromoFold "composes on top of GPTQ/AWQ/KIVI." Two products, clearly labelled: **M20-lossless** (exact residuals, guaranteed) and **M20-approx** (rank-truncated, an owned lossy layer measured separately). Do not conflate. |
| **P5 device-native** | Query APIs take/return device pointers, run on the caller's stream (as M6/M8 already do). |
| **P6 layered** | New representation in C++ core + CUDA hot path; construction may use CPU/Python research first (factorization is a research problem — prototype in the Warp sibling first, per PROJECT_SYNC). |
| **P7 measured, not asserted** | This doc claims **nothing measured**. Acceptance requires the envelope + honest baselines + **reporting negatives** (if grouping wins nothing on real weights, say so — M6's "fusion lost" precedent). |
| **P8 the C ABI is the contract** | A future `cf_grouped_matmul_async` / `cf_group_reconstruct_async` C ABI entry, versioned, is the interop surface (mirrors the M6/M8 entry style). |
| **P9 build ≠ query** | Clustering + factorization are offline/expensive; the serving GPU holds only the immutable atom. Append-heavy = the M8 reference-delta path already blessed for this. |
| **P10 the proof is a workload** | Success = a real model fitting a **bigger batch / longer context / more experts resident** in the same VRAM at equal-or-better latency — not a compression ratio. |

## CI boundary (what a PR can prove without a GPU)

Following the m12–m19 pattern (PR CI proves the CPU contract; GPU proof is manual `workflow_dispatch`):

- **CPU-provable in CI:** the construction math (cluster → centroid → residual → factor → re-encode) and the **lossless round-trip** — `reconstruct(atom) == original` bit-for-bit for M20-lossless on CPU fixtures; the ABI shape; the addressing (member `(group,i)` resolves). This is the correctness floor and needs no device.
- **GPU / hardware-gated (not claimed until measured):** the fused in-register reconstruct-in-GEMM kernel; the real VRAM + tok/s envelope on device; the P10 workload proof. No blind CUDA is merged as "verified."

## Hardware completion boundary — this milestone does NOT claim, until measured on device

- that grouped superposition **saves net VRAM on real model weights** (it may not — real weight tiles across layers may be less correlated than adapters; **report the negative** if so, per P7);
- any tok/s or latency number for the fused reconstruct path;
- a "sweet spot" `r` — the compute/memory dial is only meaningful once measured against a real workload's latency budget (P10).

The **decisive experiment** (P10): take a model where a group is genuinely correlated (a LoRA adapter library, or MoE experts, or repeated-structure layers), and show a **larger resident footprint at equal accuracy + equal-or-better latency** vs storing each member with M6 alone.

## Reproduction

_Design stage — no reproduction yet._ The first buildable step is a **Warp-sibling prototype** (`~/warp-solar-system-shaders/warp_compress`, the correctness oracle per PROJECT_SYNC): implement construction + lossless round-trip + a rank-`r` reconstruct on CPU, measure residual compressibility on a real adapter library, and **report whether the correlation is there** before any CUDA is written. That negative-or-positive is the gate for an M20 `.mk` + kernels.

## Cross-references

- `specs/00-constitution.md` — P1–P10 (P4 is the load-bearing gate here).
- `src/cuda/fused_matmul.cu` + `include/chromofold/detail/block_huffman_device.cuh` — the M6 decode-in-GEMM this composes with (`R`/`U`/`V` are each still block-Huffman-foldable).
- `delta_apply.cu` / `cf_delta_fetch` (M8) — the reference-delta cousin (sequences); the reference-delta plumbing to generalize from 1-D to tensors.
- `docs/m13-adaptive-compression.md` — the per-tensor codec selection M20 sits cross-tensor above.
- `docs/PROJECT_SYNC.md` — the Warp prototype is the correctness oracle; M20 construction prototypes there first.
- **Downstream consumer:** `sovereign-os` SDD-401/402 (the GPU-fold hotswap + the weight decode-in-GEMM C-ABI export contract) — a `cf_grouped_matmul_async` export would extend that contract once M20 proves out.
