# ChromoFold Product Platform

ChromoFold is not only a CUDA technique. The product layer answers one operational question:

> Did ChromoFold make this workload better on this machine, and can another person reproduce the result?

## Product layers

1. **Engine** — compressed GPU primitives, fused consumers and stable native interfaces.
2. **Runtime Kit** — llama.cpp integration, serving adapter, telemetry and qualification hooks.
3. **Configure** — hardware inspection, workload recommendation and generated runtime bundles.
4. **Qualify** — baseline comparison, failure scenarios and PASS / FAIL / INCOMPLETE evidence.
5. **Hub data** — profiles, downloads, compatibility and evidence-maturity registries.
6. **Assistant surface** — a future conversational UI over the same machine-readable contracts.

The CLI is intentionally deterministic. A conversational assistant may explain or collect inputs, but it must call these contracts rather than inventing configuration or performance claims.

## Quick start

```bash
python3 tools/chromofold.py inspect
python3 tools/chromofold.py recommend --goal longer-context --model model.gguf --context 65536
python3 tools/chromofold.py configure --profile maximum-context --model model.gguf
./chromofold-bundle/run-chromofold.sh
python3 tools/chromofold.py qualify \
  --mode smoke \
  --release-digest sha256:<digest> \
  --artifact qualification/evidence.json
```

## Commands

### `inspect`

Collects local OS, CPU, RAM, NVIDIA GPU, VRAM, driver, CUDA and NVLink signals. Missing NVIDIA tooling is represented as unavailable; it is not treated as a compatible GPU.

### `recommend`

Selects one of the versioned profiles using the requested goal and an intentionally conservative memory-pressure estimate. Every recommendation:

- explains why the profile was selected;
- lists known risks;
- labels capacity guidance as an estimate;
- requires hardware qualification.

### `configure`

Generates a portable bundle containing:

- `chromofold.json`;
- `run-chromofold.sh`;
- `bundle-manifest.json`.

Generated configuration disables silent fallback. Unsupported ChromoFold execution must fail visibly rather than silently benchmarking a dense path.

### `compare`

Consumes baseline and ChromoFold JSON results and reports per-metric deltas. A result is called a workload win only when measured capacity improves. Operators must ensure that model, hardware, runtime and workload fingerprints match.

### `qualify`

Delegates to the repository-owned RC1 hardware harness. It preserves PASS, FAIL and INCOMPLETE as distinct outcomes.

### `catalog`

Prints the profile and compatibility registries for websites, installers, assistants and external tooling.

## Product profiles

- **safe** — compatibility-first, large active tail, smallest behavior change.
- **balanced** — moderate compressed-page pressure relief.
- **maximum-context** — prioritize KV residency and context capacity.
- **high-concurrency** — prioritize the number of resident requests.
- **shared-prefix** — prioritize common prompt and agent-prefix reuse.

Profiles are recommendations, not claims. They become qualified only after real hardware evidence.

## Public claims policy

The evidence registry distinguishes:

- developer reported;
- CI reproduced;
- maintainer qualified;
- independently reproduced.

Estimated results must be labeled estimates. Measured results must disclose hardware, workload and baseline. A qualified claim requires validated PASS evidence. Negative and regressed metrics remain part of the published record.

## Download model

`product/downloads.json` is the source of truth for a future download center. It lists bundle contents, maturity, requirements and qualification obligations. A release publisher can use this registry to build signed archives, checksums, SBOMs and website cards without duplicating support claims.

## Assistant contract

A future ChromoFold Assistant should:

1. collect the user's goal, model, runtime and hardware;
2. run or import `inspect` output;
3. explain the `recommend` decision and risks;
4. generate files through `configure`;
5. launch or guide baseline and candidate runs;
6. compare measured results;
7. run qualification;
8. publish only evidence-supported wording.

The assistant must never:

- promise a compression ratio as a workload win;
- mark missing evidence as success;
- hide fallback;
- compare unmatched environments;
- claim compatibility outside the registry.

## CI

```bash
make -f product-assistant.mk product-all
```

The hosted workflow tests the recommendation policy, generated bundle safety, comparison semantics and versioned registries without requiring a GPU.
