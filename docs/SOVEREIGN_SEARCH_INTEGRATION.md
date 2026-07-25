# ChromoFold ↔ sovereign-os — the FM-search integration (reconciled + verified)

The **one live ChromoFold integration**, reconciled 2026-07-23. Unlike the weight-fold consumer M21 names
(`sovereign-os` SDD-401/402, which [does not exist yet](m21-super-elastic-recursive-fold.md)), the **compressed-
domain search** lane is real on both sides: the ABI matches, and the chromoFold half is verified through the
shared library. This is what *both* projects prioritized — chromoFold's M7 + the spec-draft demo, and sovereign-os
SDD-400 **Lane A (FM-index-search-first)**.

## The ABI is aligned (chromoFold ships = sovereign-os mirrors)

| capability | chromoFold — [`chromofold_search.h`](../include/chromofold/chromofold_search.h) | sovereign-os (SDD-400/500) |
|---|---|---|
| `fm_count` (flagged `sovereign_os_first`) | `cf_fm_count_async(cf_fm_view, …)` | mirrored `cf_fm_count` in its `chromofold_search.h` |
| `fm_ranges` | `cf_fm_ranges_async(…)` — `[lo,hi)` per pattern | mirrored `cf_fm_ranges` |
| `fm_locate` | `cf_fm_locate_async(…)` — text positions | mirrored `cf_fm_locate` |
| capability descriptor | [`packaging/chromofold_capability.json`](../packaging/chromofold_capability.json) — `null_arg_contract`, `conformance_requires_gpu:false` | mirrored as the sovereign `CapabilityDescriptor` |

`packaging/Makefile` exists **for this**: "the `sovereign-chromofold-sys` FFI crate links this `.so`; CI runs
`make conformance` with no GPU."

## Verified live (this session, RTX 2080 Ti)
- **`make -C packaging functional`** → `libchromofold.so` serves `cf_fm_count` + `cf_fm_locate` **bit-identical to
  golden** over `fixtures/tiny.cffm` (n=8193, σ=65, 528 patterns). The engine side of the Lane-A binding works
  end-to-end **through the shared library** — the exact surface sovereign links.
- **`make conformance`** → the no-GPU ABI seam (`null_arg_contract`) sovereign CI runs.

## `predict` = a *derived* capability, resolved on both sides
Neither side has a native `cf_predict` (an early assumption both corrected). The speculative-decode **n-gram draft
model is derived** from `count`/`ranges`/`locate`:
- **chromoFold:** [`tests/spec_draft_demo.cu`](../tests/spec_draft_demo.cu) (`make spec-draft`) is the GPU-side
  proof — FM backward-search → `locate` → argmax next-token, **bit-matching an uncompressed hash n-gram baseline**
  (identical hit-rates), one compressed index for any context length.
- **sovereign-os:** provenance-B (CPU-native Rust `FmIndex`) has its own derived `predict`, verified against a
  reference cross-check (~4000 assertions).
- Both derive from the same FM primitive, so they agree — chromoFold's demo is a natural shared reference/golden.

## The one open gap — sovereign-side, not the engine
sovereign-os's **linked** (C++ engine) backend still returns `NotImplemented` for `count`/`ranges`/`locate`:
host→device marshalling (SDD-400 "step 7") isn't wired. The working backend there today is **provenance-B (CPU
Rust)**. The chromoFold `.so` is ready and verified above — the remaining work is the **sovereign FFI marshalling**,
not the engine.

## Honest bottom line
Search-while-compressed is the through-line of everything this cycle produced (the O(n) memo, the spec-draft demo)
**and** it's the only ChromoFold capability sovereign-os has committed to bind first — because it's the one with no
existing analogue. The ABI is aligned, the engine half is verified through the `.so`, and `predict` reduces to the
FM primitive on both sides. Weight-folding (`.cfold`/`.secfold`) remains a promising, later, design-stage track.

## Reproduce
```sh
# chromoFold side (this repo)
make -C packaging functional     # GPU: cf_fm_count/locate through libchromofold.so vs golden
make -C packaging conformance    # no-GPU: the null_arg ABI seam sovereign CI runs
make spec-draft                  # the derived n-gram predict (FM draft vs hash baseline)
```
sovereign-os side: `docs/sdd/400-chromofold-compressed-domain-integration.md` + the `sovereign-chromofold-sys` crate.
