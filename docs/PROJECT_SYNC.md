# Project Sync — Python prototype ⇄ C++/CUDA native engine

Two repositories, one system. This doc defines how they relate, how they stay in sync, and how their results are
compared. Keep it current when either side moves.

## The two repos

| | **Prototype** | **Native engine** |
|---|---|---|
| Path | `~/warp-solar-system-shaders` | `~/chromoFold` |
| Package | `warp_compress/` (Python + NVIDIA Warp) | `src/cuda/`, `include/`, `benchmarks/` (C++20 / CUDA C++) |
| Role | **research surface + correctness oracle + performance floor** | **production hot path** |
| Coverage | every stratum end-to-end (weights, KV, MoE, LoRA, prompt cache, token index, FM-index, spec-decode) — 341 tests | the ported hot primitives (access, rank, RRR, fused), device-native |
| Speed of iteration | fast (Python) | slower (compile), but device-native + specialized |

The prototype is **not** deprecated. It is where new ideas are validated cheaply and where the honest,
end-to-end measurements live. The native engine ports the *proven, hot* primitives once they are worth the CUDA
cost, and must reproduce the prototype's output bit-for-bit.

## The contract between them

1. **The prototype is the oracle.** Every native kernel is verified against a *frozen Warp golden vector*
   exported from the prototype (`tools/export_reference.py` → `.cfwv`, `tools/export_rrr.py` → `.cfrr`). No
   native timing number is trusted until the output is bit-identical to that golden.
2. **The porting map is the ledger.** [`specs/05-porting-map.md`](../specs/05-porting-map.md) maps each prototype
   module to its native home and records the numbers to match or beat.
3. **Format versions are shared truth.** The `.cfwv`/`.cfrr` binary layouts are the interchange between Python
   (writer) and C++ (reader). Bump the version on any layout change and regenerate.

## Keeping them in sync (checklist when either side changes)

When the **prototype** changes a ported primitive (e.g. the wavelet layout, RRR encoding, superblock scheme):
- [ ] bump the relevant reference format version if the on-wire layout changed;
- [ ] regenerate the golden vectors; re-run the native benches — they must stay bit-identical;
- [ ] update [`specs/05-porting-map.md`](../specs/05-porting-map.md) with any new reference numbers.

When the **native engine** ports or optimizes a primitive:
- [ ] verify bit-for-bit vs the frozen golden **before** timing (M0 discipline);
- [ ] record the result + reproducibility envelope in [`specs/03-roadmap.md`](../specs/03-roadmap.md) and
      `benchmarks/README.md`;
- [ ] add a row to `benchmarks/results.json` (Python-vs-C++, per machine) so the dashboard stays honest;
- [ ] if a result contradicts a prototype claim, **fix the claim in both places** (P7 honesty).

## Comparison methodology (Python/Warp vs C++/CUDA)

Compare **like for like** and report the four timing layers separately (see
[`specs/04-benchmarks.md`](../specs/04-benchmarks.md)):

- **Same input, same golden.** Both implementations run the exact frozen vector; correctness is bit-identity to
  the golden, not to each other's floats.
- **Kernel-only vs round-trip.** The Warp prototype's user-facing path is transfer-bound (~460 M access/s) while
  its kernel is ~1.24 B/s; the native kernel matches the kernel number and the device-native API removes the
  round-trip. Always say which layer a number is.
- **Expect parity, not magic, on ported kernels.** Same algorithm + same GPU ⇒ the CUDA kernel should *match* the
  Warp kernel (M1: 1233 vs ~1240 M access/s). The native engine's wins come from **specialization** (compile-time
  level counts), **directory redesign** (two-level rank, 1.6–1.8×), **entropy coding** (RRR, 2.2–2.8×), the
  **device-native API** (no transfer), and **fusion where the intermediate is large** — not from "C++ is faster".
- **Results are hardware-specific.** Every number carries its GPU/arch/CUDA/commit. This box is 2×RTX 2080 Ti
  (sm_75), CUDA 12.6. Other machines will differ — the *shape* (parity, ratios, scaling) is the invariant, the
  absolute ns are not. `benchmarks/results.json` is structured per machine so other GPUs slot in.

## The comparison dashboard

`benchmarks/results.json` is the machine-readable results ledger (per machine, per milestone, Python vs C++). It
feeds a shared web dashboard (published Artifact) that shows the milestone parity, the frontiers, and the honest
caveats — designed so additional machines and the Python-vs-C++ view extend by adding JSON, not rewriting the
page. Regenerate/extend it whenever a milestone result changes.
