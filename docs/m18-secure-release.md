# M18 secure release boundary

M18 closes the production trust and release-engineering boundary around ChromoFold.

## Runtime trust

Workers authenticate with mutually trusted certificates. Identity includes role, worker ID, tenant namespace, key generation, validity period, and certificate fingerprint. Authorization is separate and default-deny. Tenant-bound operations reject cross-tenant substitution unless an explicit administrative policy permits it.

Signed page manifests cover tenant, model, page identity, payload length and checksum, lease generation, owner, protocol, codec, and key generation. Replay protection binds peer, nonce, timestamp, and operation. Key rotation allows an explicit overlap window and records stale-key rejection.

## Audit integrity

Security and control-plane events are appended to a hash chain. Operational audit records must not contain prompts or generated text. A release claim requires the chain to verify without missing, reordered, or modified entries.

## Release trust chain

A release bundle must contain:

- a canonical release manifest;
- immutable artifact hashes;
- CycloneDX or SPDX SBOM;
- build provenance tied to commit and toolchain;
- signatures for artifacts, SBOM, provenance, and deployment bundles;
- compatibility metadata for protocol, codec, and checkpoint formats.

`tools/chromofold_release.py verify` rejects missing files, duplicate paths, path traversal, and mismatched hashes. Promotion between nightly, candidate, and stable channels must reuse artifact digests rather than rebuilding.

## Deployment hardening

The M18 container runs as UID/GID 65532, has no login shell, and exposes only explicit writable paths. Kubernetes assets disable service-account token mounting, use least-privilege RBAC, run as non-root, drop all capabilities, enforce a read-only root filesystem, mount TLS material read-only, and start from default-deny networking.

## Production evidence boundary

Hosted pull-request CI proves the deterministic CPU trust contract, schemas, static deployment anchors, SBOM generation, and fabricated-evidence rejection. It does not prove live mTLS, signing-key custody, reproducibility across isolated builders, vulnerability policy, rolling upgrades, or rollback. Those require the manual release-security workflow and external signing material.
