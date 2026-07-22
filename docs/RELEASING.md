# ChromoFold release candidates

ChromoFold release candidates are built from repository-owned sources and published with enough metadata to verify exactly what was produced.

## Build locally

```bash
python3 tests/test_release_candidate.py
python3 tools/build_release_candidate.py \
  --version v1.0.0-rc1 \
  --output dist/release

cd dist/release
sha256sum --check SHA256SUMS
```

The output contains:

- `chromofold-<version>.tar.gz` — product archive;
- `chromofold-<version>.tar.gz.sha256` — archive checksum sidecar;
- `chromofold-<version>.sbom.spdx.json` — SPDX 2.3 file inventory;
- `chromofold-<version>.provenance.json` — source commit, builder and qualification boundary;
- `SHA256SUMS` — integrity list for every release asset.

## Publish from GitHub Actions

Open **Actions → ChromoFold release candidate → Run workflow**.

1. Enter a prerelease version such as `v1.0.0-rc1`.
2. Leave **publish** disabled for a dry run and downloadable workflow artifact.
3. Inspect the generated archive, SBOM, provenance and checksum verification.
4. Run again with **publish** enabled only when the asset set is correct.

The workflow publishes a GitHub prerelease. It does not call the candidate production-qualified.

## Qualification boundary

The generated provenance starts with:

```json
{
  "qualification": {
    "state": "UNQUALIFIED_CANDIDATE",
    "required_for_production_claim": true,
    "artifact_digest": "sha256:<archive digest>"
  }
}
```

To promote a claim beyond candidate status:

1. run the exact archive and runtime on the target hardware;
2. run the RC1 qualification harness;
3. require a `PASS` result bound to the provenance archive digest;
4. preserve and publish `FAIL` or `INCOMPLETE` results honestly;
5. update `product/release-channel.json` only after the evidence exists.

## Public website

The GitHub Pages site reads `product/release-channel.json`. Updating that registry triggers a site rebuild so the displayed release state remains aligned with the repository contract.
