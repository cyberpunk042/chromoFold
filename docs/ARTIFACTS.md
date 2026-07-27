# Artifacts

Published visual artifacts — strategic memos, explainers, evidence dashboards — rendered as self-contained pages on
claude.ai and referenced here so the repo has a stable index of them.

**These are companion pieces, not the source of truth.** Measured benchmark claims live in `product/*.json` with
their evidence scope (see [`WEBSITE.md`](WEBSITE.md)); the milestone record lives in
[`../specs/03-roadmap.md`](../specs/03-roadmap.md). Artifacts here *synthesize and present* that evidence — every
number in one traces back to a committed measurement.

**Access.** Each link is a private claude.ai artifact by default; viewers need it shared from the artifact's own
share menu. Treat the URLs here as pointers, not public assets.

**Convention (adding a new one).** Publish the page, then append a row below with: title · one-line purpose · the
`claude.ai/code/artifact/…` URL · date · the repo doc(s) it draws from. Keep newest first. To revise an existing
artifact, republish to the **same URL** (don't mint a new link) and leave the row unchanged except the date.

---

## Index

### Searchable while compressed — engine report · 2026-07-27
The measured account of the searchable FM-index substrate: the cross-repo integration (safe sovereign Rust → `-sys`
FFI → `libchromofold.so` → GPU, gated `A==B==oracle`), the **CF_RRR_S compute↔memory frontier** (the signature
visual — index b/tok vs search ms, asymmetric: 8× sample rate → −11% size, +56% latency), the honest footprint
(~6.8 b/tok, ~13% over bit-packed raw), and the GPU/CPU latency crossover. Foregrounds **three refuted hypotheses**
as first-class results — host-marshalling bottleneck, WSL2 tax, and the single-descent 2× (caught by the parity gate:
count PASS / locate FAIL, reverted). Honesty-as-method is the editorial hook.
- **Artifact:** https://claude.ai/code/artifact/1b983408-3d5e-4a9c-99c4-69fd89b97894
- **Draws from:** [`SOVEREIGN_SEARCH_INTEGRATION.md`](SOVEREIGN_SEARCH_INTEGRATION.md),
  [`SEARCHABLE_WORKLOADS.md`](SEARCHABLE_WORKLOADS.md); `../packaging/{functional_device.c,launch_latency.cu}`;
  sovereign-os `sovereign-chromofold` examples (`parity_smoke`, `bench_search_device`)
- **Receipts (all committed):** CF_RRR_S sweep 16→128 (7.56 b/tok/1.118 ms → 6.70/1.743), all parity PASS; footprint
  converges 6.80 b/tok at n=1M; crossover GPU 1.8M vs CPU 0.73M pat/s @ batch 4096 (1M); floor decomposition
  launch 0.028 ms / marshalling ~0.2 / kernel ~1.4; device-native vs host 1.07–1.41× (the negative).

### .cfold vs .secfold — weight-fold spec sheet · 2026-07-23
The two tensor-folding schemes side by side: `.cfold` grouped-delta superposition (M20 — group similar tensors
against a root or computed-centroid reference, one rank-`r` factored atom) vs `.secfold` super-elastic fold (M21 —
restack the residual `L` times, the elastic accuracy dial). Signature visual: the measured error-vs-depth curve
(super-elastic 0.512 → bit-exact vs single-fold plateau at 0.239). Honest status foregrounded: both design-stage,
`.secfold` prototype-measured, and the named consumer (sovereign-os SDD-401/402) does not exist yet.
- **Artifact:** https://claude.ai/code/artifact/1bacb7ef-4151-4262-9099-497fa4dbc8b2
- **Draws from:** [`m20-grouped-delta-superposition.md`](m20-grouped-delta-superposition.md),
  [`m21-super-elastic-recursive-fold.md`](m21-super-elastic-recursive-fold.md); reconciled against
  `../../sovereign-os/docs/sdd/400-chromofold-compressed-domain-integration.md`
- **Receipts:** M21 Warp-prototype (seed 20260725, rank-4 group) — error L1 0.512 → L6 0.000 (bit-exact); lossless
  L=3 stack 1.45× vs independent; matched-rank wash (the reported negative).

### ChromoFold and the O(n) wall — strategic positioning memo · 2026-07-23
How the engine relates to the transformer scaling problem: the two moves against an O(n²) curve (shrink the
coefficient vs. bend the exponent), why KV compression is a commoditized constant factor, and the honest
positioning as the **sub-linear searchable substrate** for sparse/retrieval attention — not an attention algorithm.
- **Artifact:** https://claude.ai/code/artifact/9137a101-4076-4afa-a822-ae71ce1c69f2
- **Draws from:** [`SEARCHABLE_WORKLOADS.md`](SEARCHABLE_WORKLOADS.md),
  [`../integrations/llama.cpp/runtime/KV_BACKEND_FINDINGS.md`](../integrations/llama.cpp/runtime/KV_BACKEND_FINDINGS.md),
  [`M11_EVIDENCE_AND_CROSSOVER.md`](M11_EVIDENCE_AND_CROSSOVER.md)
- **Receipts (all committed):** KV capacity 2.67× / latency crossover ~1.04× (and commoditized vs llama q4_0);
  FM-search 2,000,000 tokens → 1.45 MB, O(|pattern|); spec-draft identical accuracy, one index any length; sparse
  gather 17.8× at 0.1% touched.
