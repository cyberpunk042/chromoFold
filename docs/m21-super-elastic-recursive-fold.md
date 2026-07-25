# M21 — Super-elastic fold: stacked residual layers as an elastic accuracy dial

> Companion visual: **[.cfold vs .secfold spec sheet](https://claude.ai/code/artifact/1bacb7ef-4151-4262-9099-497fa4dbc8b2)** — this scheme (`.secfold`) beside the base `.cfold` fold it extends, with the measured error-vs-depth curve. Indexed in [`ARTIFACTS.md`](ARTIFACTS.md).

> **Status: DESIGN, prototype-backed.** No native kernels yet, but the transform is **implemented +
> measured** in the Warp sibling (`warp_compress/super_elastic.py` + `bench_super_elastic.py`), so the
> numbers below are real, not asserted (P7). Extends **M20** (grouped delta superposition).
> **Thesis fit:** pure P1/P3 — spend more decode matrix-ops (stacked in-consumer reconstruction) to buy
> a tighter squeeze / higher accuracy of the same folded representation.

## The idea (operator, verbatim — sacrosanct)

> "the super-elastic mode that skeeze even more in the folding, more layer of matrix operation to solve
> the puzzle to resolve our own representation of the data."

## What it is

M20 folds a group once: a reference (root / "third state" centroid) + a single rank-``r`` factored,
int8-quantized residual "atom". **Super-elastic folds the leftover again, and again.** Each layer takes
the residual the previous layers could not represent, factors it (rank-``r``), int8-quantizes the factors
with a **fresh per-matrix scale**, and subtracts its reconstruction; the next layer works on the smaller
leftover. Decode "solves the puzzle" = reference + the **sum of every layer's reconstruction**. The layer
count ``L`` is the **elastic depth dial**.

This is residual (multi-stage) quantization applied to the grouped-delta atom — the compressed-domain
analogue of RVQ, but over the shared-reference residual and consumed in-kernel (P3): the member is never
materialized; each layer's `R + U_l V_l` accumulates in registers.

## The measured result (Warp prototype, seed 20260725, a rank-4 group)

**Two honest faces — reported both ways (P7):**

**(a) At matched total rank, more layers is ~a wash.** Splitting one rank-``L·r`` fold into ``L`` rank-``r``
layers re-pays a per-matrix scale each and the factorization is no tighter — bytes rise slightly, error
falls slightly; net neutral. *More layers is not free lunch*, and the design says so.

**(b) On the accuracy axis, it is decisive.** Because each layer re-quantizes its leftover with a fresh
scale, stacking drives error toward zero where a single low-rank fold **plateaus**:

| approach | mean-abs error trajectory |
|---|---|
| **super-elastic** (stack rank-4 layers) | L1 0.512 → L2 0.293 → L3 0.069 → L4 0.005 → **L6 0.000 (bit-exact)** |
| single fold (add rank instead) | r4 0.512 → r8 0.418 → **r16 0.239 (plateaus)** |

Adding rank past the true structural rank cannot fix the int8 **quantization-error floor**; adding
**residual layers can**. So super-elastic reaches accuracy — and, with a final exact residual, **bit-exact
losslessness** — that one bigger fold fundamentally cannot. A lossless L=3 stack (+ exact final residual)
was still **1.45×** vs independent storage.

**Consequence for ChromoFold:** super-elastic is the *elastic accuracy dial* for the folded weight/KV
representation — one knob spanning cheap-lossy → near-lossless → bit-exact, at monotonically rising decode
cost. It composes on top of M20 (which chooses the reference + first factored atom) and M6 (block-Huffman
decode-in-GEMM, which can still entropy-code every layer's factors).

## The transform (build path — P9 build ≠ query)

Offline / CPU: `R = reference(group)`; `resid = group - R`; then for `l in 1..L`: SVD(resid) → rank-``r``,
fold `U_l·S_l` and `V_l` to int8 (fresh per-matrix scales), append the layer, `resid -= reconstruct(layer)`.
Optionally keep the exact integer `resid` after the last layer for the lossless product. The serving GPU
carries the immutable stack: `R` + `L` × (int8 `U_l`, int8 `V_l`, two scales) [+ exact final residual].

## The query path (decode-in-consumer — P3)

Inside the consuming kernel, for the needed tile: decode `R`'s slice (as M6 does today) and accumulate
each layer's `(U_l·s) @ (V_l·s)` contribution in registers, then multiply-accumulate. Decode cost scales
with ``L`` — the explicit compute-for-accuracy trade (P1). No dense member ever exists in global memory.

## Constitution compliance

| Principle | How M21 satisfies it |
|---|---|
| **P1 compute-for-memory** | ``L`` (and ``r``) is the compute↔accuracy/size dial made explicit. |
| **P2 navigable while compressed** | Each member stays O(1)-addressable; a tile is reconstructed from the reference slice + the per-layer factor slices without decompressing the group. |
| **P3 decode only inside the consumer** | Layers accumulate in registers inside GEMM/attention; nothing materialized. |
| **P4 lossless over the quantizer** | Each layer is a *named lossy* step; ``final_residual`` keeps the exact leftover → the whole stack is bit-exact over the quantizer. The lossy/lossless split is explicit, never hidden. |
| **P7 measured, not asserted** | The Warp prototype's numbers are inlined above, including the **negative** (matched-rank wash). |
| **P9 build ≠ query** | SVD + quantization are offline; the GPU holds only the immutable layer stack. |
| **P10 proof is a workload** | The decisive proof (SAIN-gated) is a real model reaching a target accuracy at lower resident bytes via the L-dial than a single fold can, at equal-or-better decode latency. |

## CI boundary (what a PR proves without a GPU)

- **CPU-provable in CI (done in the Warp prototype):** the layer stack's construction, the monotone
  error-vs-depth curve, the bit-exact round-trip with a final residual, and that stacking beats the
  single-fold error plateau — all in `warp_compress` (`tests/test_super_elastic.py`, 5 checks).
- **GPU / hardware-gated (not claimed until measured):** the fused multi-layer accumulate-in-GEMM kernel,
  the real per-``L`` decode-latency curve, and the P10 workload proof on device.

## Hardware completion boundary — this milestone does NOT claim, until measured on device

- any decode-latency number for the L-layer accumulate (the compute cost of the dial);
- that the accuracy-per-byte curve holds on real model weights (the Warp numbers are a rank-4 synthetic;
  a real correlated group — MoE experts / an adapter library — is the follow-on, and the negative-if-not
  discipline applies);
- a recommended default ``L`` — that is a per-workload latency-budget choice, only meaningful once the
  device decode-latency curve is measured (P10).

## Cross-references

- `docs/m20-grouped-delta-superposition.md` — the single-fold atom M21 stacks on (reference + first layer).
- `src/cuda/fused_matmul.cu` + `include/chromofold/detail/block_huffman_device.cuh` — the M6 decode-in-GEMM
  each layer's factors still compose with.
- `specs/00-constitution.md` — P1/P3/P4/P7/P9/P10 (the gates above).
- **Warp prototype (the measured backing):** `warp-solar-system-shaders/warp_compress/super_elastic.py`,
  `bench_super_elastic.py`, `tests/test_super_elastic.py`; the M20 substrate it extends lives in
  `grouped_delta.py` (`quant_factors`) + `bench_grouped_delta_lora.py`.
- **Downstream consumer:** `sovereign-os` SDD-401/402 — a multi-layer variant of the weight-fold C ABI
  — **reconciled 2026-07-23 (P7): SDD-401/402 are not yet written.** sovereign-os's only committed ChromoFold
  binding today is **SDD-400** (`chromofold-compressed-domain-integration`), whose confirmed first lane is
  **FM-index compressed-domain search** (Lane A, opt-in / off by default); the weight-fold is an unbuilt later
  lane. So this consumer is aspirational — the pipe that would eat a `.secfold` does not exist on either side yet.
  (an `L`-layer `cf_grouped_matmul_async`) would extend that export contract once M21 proves out on device.
