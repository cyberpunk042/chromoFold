# M14 distributed and multi-GPU execution

M14 introduces the control-plane and proof contract for running ChromoFold across multiple CUDA devices.

## Supported modes

- single-device compatibility mode;
- layer-parallel routing;
- tensor-parallel routing;
- CUDA peer transport when the topology permits it;
- explicit host-staged transport only when configured.

## Invariants

- sequence placement is deterministic for a fixed routing seed;
- migration increments a generation and never mutates an in-flight snapshot;
- all batch items are validated before dispatch;
- unknown devices, sequences, or layers fail closed;
- dense KV fallback remains prohibited;
- cross-device contamination must remain zero;
- shutdown reconciles page and snapshot references on every device.

## Persistent cache coordination

M13 page identities remain authoritative. A distributed deployment may load a page on several devices, but shared-index coordination must prevent duplicate semantic records and reject mismatched model, codec, topology, or generation metadata.

## CI

Routine pull-request CI is CPU-only and validates placement, routing, schemas, anchors, and fabricated-evidence rejection. The full multi-GPU server workload is manual-only and requires a self-hosted runner labelled `multi-gpu`.

## Production boundary

This round establishes routing and evidence contracts. A complete production claim still requires real CUDA peer copies, tensor-parallel collectives, mixed-codec attention shards, actual shared-cache page uploads, injected device-failure behavior, and strict passing evidence from at least two NVIDIA devices.
