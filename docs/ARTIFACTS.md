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
