# M10 shared multi-sequence compressed KV cache

## Purpose

M10 introduces the ownership and lifecycle model required for `llama-server` workloads. The earlier M9 path proved a strict single-sequence compressed data plane. M10 separates immutable compressed history from sequence-local mutable tails and makes sequence operations explicit and auditable.

## Ownership model

A logical sequence contains one layer state per model layer. Each layer owns:

- ordered references to immutable compressed pages;
- a sequence-local active-tail token range;
- an active-tail generation;
- a descriptor generation.

The global page store owns immutable page payloads. Sequence tables hold references. A page is retired when its logical reference count reaches zero and is reclaimed only through the explicit reclamation path.

## Transactional operations

Sequence copy is prepare/publish:

1. validate the source and requested token range;
2. acquire all page references for the candidate destination;
3. build destination layer metadata and a private active-tail snapshot contract;
4. replace the destination only after candidate construction succeeds;
5. release the replaced destination after publication.

Any failed acquisition is rolled back.

Whole-sequence removal releases every page reference. Suffix removal is supported for the active tail. Removal inside an immutable compressed page is rejected until page splitting or decompression-to-active support is implemented.

Context shifts are supported only when immutable page boundaries are respected. The physical compressed payload remains immutable while logical token positions change.

## Speculative transactions

A speculation transaction records the active-tail token count and sequence generation. Commit retains an accepted prefix of provisional tokens. Rollback restores the baseline active-tail length and topology generation. Sealing provisional data into shared immutable history is not enabled by this host core.

## Invariant auditor

The auditor verifies:

- every sequence map key matches its stored sequence ID;
- every sequence has the configured layer count;
- every page reference resolves to a live global page;
- page ranges are ordered and non-overlapping;
- active tails do not overlap sealed history;
- observed logical references equal stored page reference counts;
- retired pages have zero references.

Debug and contract configurations can run the auditor after every mutation.

## llama-server lifecycle

The bridge defines explicit operations for:

- slot creation;
- slot reset and reclamation;
- sequence copy;
- sequence removal;
- context shifting;
- request cancellation;
- batched attention submission;
- shutdown auditing.

Slot reuse never retains a stale sequence mapping.

## Strict boundary

The multi-sequence host ownership and lifecycle layer is implemented in this PR. The CUDA descriptor resolver that maps each batch item to its sequence-specific compressed page table and active device tail remains fail-closed. `cf_multisequence_attention_async` rejects dispatch until that resolver is connected and validated on hardware.

Dense fallback is forbidden. A rejected multi-sequence dispatch must fail the request rather than silently use dense KV.

## Completion gates for the device/server round

A later hardware artifact must prove:

```text
active_sequences_peak > 1
shared_page_references > 0
sequences_copied > 0
batches_dispatched > 0
sequences_per_batch_peak > 1
speculative_transactions_rolled_back > 0
pages_reclaimed > 0
invariant_failures == 0
dense_fallback_launches == 0
```

It must also show no cross-sequence output contamination, successful slot reuse, request cancellation cleanup, Compute Sanitizer success, and zero leaked page references at shutdown.

## Reproduction

```bash
make -f m10-multisequence.mk multisequence-compile
make -f m10-multisequence.mk multisequence-contract
make -f m10-multisequence.mk multisequence-anchor-check
```
