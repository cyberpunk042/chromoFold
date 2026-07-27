# M22 — The real-model measurement: folding a production MoE expert bank (NEGATIVE)

> **Status: MEASURED on real hardware.** This is the P10 workload proof that M6/M20/M21 have been
> gated on — run against a **production 744B MoE checkpoint** on an **NVIDIA RTX PRO 6000 Blackwell
> (sm_120, CUDA 13.3)**. The result is **negative on every fold lane**, reported in full per P7.
> `benchmarks/fused_matmul.cu` docstring asked for exactly this: *"a real model (a true expert bank
> from the venv) is the SAIN/venv-gated follow-on."* This is that follow-on.
>
> **Thesis fit:** P7 (measured, not asserted) + P10 (proof is a workload, not a ratio). Nothing here
> changes the constitution; it changes which lanes are worth building.

## What was measured against

| | |
|---|---|
| Model | GLM-5.2, 753.3B params, `glm_moe_dsa`, 78 layers (75 MoE), 256 routed experts/layer, top-8 |
| Checkpoint | Colibri int4 container, **per-row scales** (one f32 per row of 6144), 383.7 GB on NVMe |
| Source cross-check | `zai-org/GLM-5.2-FP8` (e4m3, 128×128 block scales), one 5.36 GB shard pulled for the quantiser study |
| Expert shape | gate/up `[2048, 6144]`, down `[6144, 2048]` → 37.75M weights = 18.87 MB/expert at int4 |
| GPU | RTX PRO 6000 Blackwell Max-Q 96 GB + RTX 5090 32 GB, driver 590.48.01, CUDA 13.3 |
| Host | Ryzen 9 9900X (12c/24t, Zen 5 AVX-512), 249 GB DDR5 |

## Result 1 — decode-in-GEMM is **slower** than dense on Blackwell at MoE shapes

`make fused-matmul` with fixtures regenerated at the real GLM expert shape
(`--m 2048 --k 6144 --block 1024`), `ARCH=sm_120`:

```
  W (int4)        B    resident     dense     less    fused ms  dense ms   fused==dense
  2048×6144      1     3.99 MB  50.33 MB    12.6×       3.098     1.381   BIT-IDENTICAL
  2048×6144      4     3.99 MB  50.33 MB    12.6×       3.829     2.191   BIT-IDENTICAL
  2048×6144     16     3.99 MB  50.33 MB    12.6×       7.281     5.298   BIT-IDENTICAL
  2048×6144     64     3.99 MB  50.33 MB    12.6×      51.878    35.411   BIT-IDENTICAL
```

**The fused path is slower than dense at every batch size tested**, including B=64. `RESULTS.md` M6
reports *"1.3× faster"* at 4096×4096 on an RTX 2080 Ti; **that crossover does not reproduce on sm_120
at MoE expert shapes.** Bit-exactness holds throughout (`rel=0.0e+00`) — fusion remains numerically
free, as documented. Only the speed claim fails.

### Against the kernel it would replace

The consuming engine (Colibri) runs a plain fixed-width int4 matvec. Head-to-head at B=1:

```
per million weights:   ChromoFold fused  246.21 µs
                       Colibri quant_matmul 7.47 µs      -> ChromoFold 33× SLOWER
full GLM expert:       ChromoFold 9.29 ms   vs   Colibri 0.282 ms
```

**Root cause (structural, not a tuning gap):** the consumer already decodes in-GEMM — it just uses
**fixed-width nibbles** (shift + mask, fully parallel per element). ChromoFold uses **variable-length
block-Huffman**, which requires walking a bit-stream. At r=1 there is no arithmetic intensity to hide
that behind. The extra ~1.35× of compression is bought with a decode that cannot parallelise.

This is P1 (compute-for-memory) behaving exactly as designed — the benchmark's own output says
*"the fused path re-decodes W every matmul, so it is slower than a plain GEMM over a cached dense W"*.
The finding is that for **single-use, r=1 MoE decode**, that trade is 33:1 against us.

## Result 2 — M20/M21 grouped-delta returns **0.99×** on a real expert bank

`warp_compress.grouped_delta` over 24 real same-layer experts, several configurations:

```
configuration                                    mn|Δ|   entropy   M20/nat   M20/base
layer 30 gate_proj, experts 0..23 (by index)     1.399     1.18x     1.17x      0.99x
layer 30 down_proj, experts 0..23 (by index)     1.506     1.14x     1.13x      0.99x
layer  7 gate_proj, experts 0..23 (by index)     1.354     1.19x     1.18x      0.99x
layer 19 gate_proj, 24 HOTTEST (co-routed)       1.398     1.18x     1.17x      0.99x
layer 19 gate_proj, 24 COLDEST                   1.398     1.18x     1.17x      0.99x
```

**Grouping adds nothing over coding each expert independently**, and the cause is structural:

- **cross-expert correlation ≈ 0** (`corrcoef +0.0007`); co-routed experts are no more similar than
  hot-vs-cold pairs (`mean|a−b|` 2.022 vs 2.025);
- **no permutation alignment.** MoE experts are permutation-symmetric in the intermediate dim, so
  index-aligned correlation cannot see similarity hidden under a row permutation. Tested with 24-dim
  random-projection row signatures: **mean best-match cosine +0.6373 against a random-vector null of
  +0.6391** — i.e. at the null. There is no alignment to exploit.
- **not low-rank within an expert either:** effective rank for 95% energy is **1227 of 2048**
  (a random gaussian needs ~1945). Structure exists but nowhere near enough; a rank-*r* fold costs
  `2nr` and only pays for `r ≪ n/2`.

Routed experts are *trained to differ* — specialisation is the mechanism. They behave as independent
draws from a shared bell-shaped marginal, which a per-expert coder already captures.

### Why the synthetic benches disagreed

`bench_fold_cost` reports **3.68×** and `bench_model_fold` reports **1.21×** for "the correlated
bank". Both are synthetic and the gap is entirely the assumed similarity:

| bench | construction | M20 lossless |
|---|---|---|
| `bench_fold_cost` | `base ± 2` of 256 | 3.68× |
| `bench_model_fold` | base + rank-6 spread, amp 8 | 1.21× |

A sensitivity sweep on the matched int4 alphabet places the real bank precisely: at the measured
`mean|Δ| = 1.399` of 15 levels, the curve predicts ~1.2× and the real measurement is 1.17×. **The
3.68× headline corresponds to members differing by ±2 of 256 — that is an adapter library or tied
layers, not a trained expert bank.** M20/M21 should be scoped to those explicitly.

## Result 3 — real-weight entropy, and the `block` parameter is not free

Order-0/order-1 entropy over **201M real int4 weights** (4 layers × 2 roles × 2 experts):

```
AGGREGATE: 2.9619 b/w -> 1.350x lossless      (gate/up 1.37-1.39x, down 1.32-1.33x)
H1 = H0 to four decimals -> NO sequential context
```

`H1 == H0` is good news for the codec: a **static canonical Huffman table reaches the floor**, so
`cf_bh_decode_at` needs no adaptive state. It is also the ceiling — 1.35× is near the
information-theoretic limit for lossless on this container.

**Contract consequence for `cf_fused_matmul_async`'s `block` argument.** With one shared `lut` and a
32-bit `block_off` per block:

```
   block       b/w    ratio
      64    3.4619   1.155x
     256    3.0869   1.296x
    1024    2.9931   1.336x
    4096    2.9697   1.347x
```

With **per-block tables** (64 bits each) instead of a shared LUT, `block=64` measures **0.935× — it
expands the data.** The export contract should therefore require `block ≥ 1024` and state explicitly
that **coding granularity (one shared LUT per tensor) is decoupled from addressing granularity
(`block_off`)**. Downstream this is tracked as sovereign-os SDD-402 **Q-402-E**.

## Result 4 — the "10.6×" is against fp32, and the baseline matters

`RESULTS.md` M6 reports 10.6× memory saved: 1.58 MB resident vs 16.78 MB dense for 2048². That dense
figure is **4 bytes/weight = fp32**, and the compressed store is **3.02 bits/weight** — which agrees
closely with the 2.96 b/w measured here on real weights. The claim is sound; the baseline needs
stating, because consumers arrive with **already-quantised** checkpoints:

```
ChromoFold on the ORIGINAL weights:  BF16 1506.6 GB -> 284.4 GB @3.02 b/w
The path the consumer already took:  int4  376.6 GB -> 278.7 GB @2.96 b/w
```

**Both land at ~280 GB.** The 10.6× and the consumer's int4 quantisation are two routes to the same
~3 bits/weight, not two compressions that stack. Against an int4 container the remaining fold is
**1.35×**. Recommend `RESULTS.md` state the baseline inline.

## Consequence for the roadmap

| lane | verdict on a trained MoE expert bank |
|---|---|
| **M6 block-Huffman (entropy)** | **valid** — 1.35× lossless, static table, `block ≥ 1024` |
| **M6 decode-in-GEMM (speed)** | **negative at r=1** — 33× slower than fixed-width int4; the fused/dense crossover does not reproduce on sm_120 |
| **M20 grouped delta** | **negative** — 0.99×; correlation is 0, no permutation alignment |
| **M21 super-elastic** | **negative here** — costs bytes *and* latency; its accuracy dial has no target on this workload |

**Where M20/M21 still belong:** adapter libraries and tied layers — genuinely near-identical tensors,
which is what `bench_fold_cost`'s `noise=2` construction actually models. The docs should say so.

**Where the fold still wins:** *capacity for residency*, not speed. On the measured host, 1.35×
takes a 367 GB expert bank to 272 GB, crossing below the 344 GB of VRAM+RAM — which removes the disk
tier. That is a real workload win and the honest headline for M6.

## Constitution compliance

| Principle | How M22 satisfies it |
|---|---|
| **P7 measured, not asserted** | Every number here is measured on a production checkpoint and real hardware; all four results are negative and reported in full. |
| **P10 proof is a workload** | This *is* the workload proof, run on a real 744B model. It says the fold buys residency, not throughput. |
| **P4 lossless over the quantizer** | Confirmed — fused output stayed `rel=0.0e+00` vs dense at every batch size. |
| **P1 compute-for-memory** | Confirmed and quantified: on r=1 MoE decode the trade is 33:1 against, because variable-length decode cannot parallelise. |

## Reproduce

```sh
# 1. decode-in-GEMM at real MoE shapes (needs warp-lang for the exporter)
python3 tools/export_fused_matmul.py benchmarks/refs/glm_gate_b1.cffw \
        --m 2048 --k 6144 --batch 1 --bits 4 --block 1024
make build/fused_matmul ARCH=sm_120
./build/fused_matmul benchmarks/refs/glm_gate_b1.cffw

# 2. grouped-delta / entropy / permutation / rank on a real expert bank
#    (consumer-side harness, read-only, numpy only)
sovereign-os/scripts/inference/chromofold-fold-bench.py --model <int4-container> --mode all
```

## Cross-references

- `docs/RESULTS.md` — M6 (fused matmul, 10.6× / "1.3× faster") and the fold-cost tables this qualifies.
- `docs/m20-grouped-delta-superposition.md`, `docs/m21-super-elastic-recursive-fold.md` — the lanes measured negative here.
- `docs/PROJECT_SYNC.md` — the two-repos-one-system contract; this is the consumer-side measurement returning.
- **sovereign-os** `docs/evaluations/chromofold-fold-measurement-glm52-2026-07-27.md` (full consumer-side record),
  `scripts/inference/chromofold-fold-bench.py` (harness), SDD-400/401/402 (**Q-402-E** = the `block ≥ 1024` contract fix).
- `specs/00-constitution.md` — P1, P4, P7, P10.
