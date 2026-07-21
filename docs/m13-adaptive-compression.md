# M13 adaptive compression and persistent pages

M13 adds a versioned mixed-precision page format and a persistent compressed-page store.

## Codec selection

Each key and value payload independently selects int2, int4, int8, or raw FP16. Quality-budget mode chooses the lowest-cost candidate whose estimated error satisfies the configured page limit. Fixed-int4 remains available for controlled comparisons. Invalid data and candidates fail closed.

## Recompression

Published pages are immutable. Recompression decodes an existing page into a new candidate, validates and publishes the replacement, and leaves existing snapshots attached to the old generation until their references retire.

## Persistent format

Records contain a magic value, format version, semantic page key, independent K/V codec descriptors, payload lengths, scales, value count, and checksum. Writes use a temporary file followed by replacement. Readers reject invalid magic, unsupported versions, truncated records, and checksum failures.

Persistent identity is bound to the model SHA-256 supplied when the store opens. Production integration must additionally bind llama.cpp compatibility, RoPE parameters, page topology, and source-token identity.

## CI

Routine pull-request CI runs only CPU codec round trips, persistent-cache verification, schemas, semantic anchors, and fabricated-evidence rejection. The six-hour NVIDIA proof remains manual-only through `workflow_dispatch`.

## Hardware completion boundary

The final M13 claim requires actual CUDA int2/int4/int8/FP16 mixed-codec attention, transactional device recompression, a real pinned-server warm restart, cache hits that bypass valid prefix K/V recomputation, corruption rejection, memory-budget enforcement, zero dense fallback, and reconciled shutdown references.
