# M16 disaggregated serving

M16 separates prompt prefill from decode and defines a cluster-wide compressed KV transfer contract.

## Roles

- `router`: selects workers using health, VRAM, backlog, and cached-prefix locality.
- `prefill-worker`: computes missing prompt suffixes and publishes immutable compressed-page manifests.
- `decode-worker`: imports validated pages and continues generation without repeating prefill.
- `standalone`: preserves the existing single-process mode.

## Protocol

Protocol version 1 defines `HELLO`, `CAPABILITIES`, `PREFILL_REQUEST`, `PAGE_MANIFEST`, `PAGE_DATA`, `PAGE_ACK`, `LEASE_ACQUIRE`, `LEASE_RENEW`, `LEASE_TRANSFER`, `DECODE_ATTACH`, `CANCEL`, and `ERROR` messages.

Cluster page identity binds model, tokenizer, token sequence, RoPE configuration, compression profile, layer, page index, page size, and codec version. Incompatible identities are never reused.

## Ownership and publication

Pages use generation-bound leases. A manifest is accepted only when its lease generation and token match the authoritative lease. Ownership transfer increments the generation and produces a new token, so stale publishers fail closed.

A transferred page is visible only after authentication, integrity verification, complete-byte validation, and manifest acknowledgement. Partial transfers are rejected.

## Routing and recovery

The reference router prefers healthy loaded workers with available VRAM and low queue pressure. Valid prefix pages reduce the required prefill suffix and are reported as avoided prefill tokens.

Replication is explicit. A failed owner can be replaced only when a valid replica exists. Active work is bounded by configured worker, request, transfer, and pinned-memory limits.

## CI boundary

Pull-request CI compiles the CPU state-machine contract and validates schemas and evidence semantics. The multi-process GGUF workflow is manual-only because it requires a local model, GPU resources, patched llama-server worker roles, failure injection, and strict evidence generation.

M16 does not claim production disaggregated inference until the manual proof demonstrates separate router/prefill/decode processes, real compressed-page transfer, no prefill recomputation on decode, prefix reuse, authenticated and integrity-protected transport, stale-lease rejection, replica recovery, zero contamination, zero dense fallback, and reconciled shutdown references.
