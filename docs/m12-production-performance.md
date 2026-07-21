# M12 production validation and performance

M12 is the first performance-oriented layer after exact pinned llama-server integration. It keeps the M9 correctness kernel as the reference oracle and adds an optimized compressed-attention runtime designed for real concurrent decode.

## Execution modes

- `dense`: pinned llama.cpp baseline.
- `reference`: correctness-first ChromoFold attention.
- `optimized`: warp-cooperative fused int4 attention.

The optimized path performs packed K/V reads, in-register dequantization, query/key dot products, online softmax, and value accumulation without writing dense K/V tensors to global memory.

## Kernel contract

The initial optimized decode kernel uses one warp per query head and one grid item per sequence/layer work item. Compatible sequences are grouped into one launch. GQA maps query heads to their shared KV head explicitly.

The reference path remains mandatory for numerical comparison. A production artifact is invalid unless both paths execute and the committed error thresholds pass.

## Numerical thresholds

The hardware harness must record maximum absolute error, mean absolute error, cosine similarity, token agreement, and the first divergent token. The initial cosine-similarity floor is 0.99. Threshold changes require a source change and review; the workflow must not tune them after observing results.

## CUDA Graphs and overlap

Stable decode shapes may be captured and replayed. Topology mutations invalidate graphs and continue through the non-graph compressed path. Dense fallback remains forbidden.

Asynchronous sealing may overlap attention only for already-published immutable pages. Candidate pages remain invisible until transactional publication completes.

## Benchmark matrix

The self-hosted workflow compares dense, reference, and optimized modes over configurable context and concurrency matrices. Every case binds results to the GGUF SHA-256, upstream commit, ChromoFold commit, GPU, CUDA toolkit, and driver.

## Soak gate

The production gate includes at least 1,000 deterministic HTTP requests with mixed prefixes and slot reuse. It requires stable memory, zero cross-sequence contamination, zero CUDA errors, passing Compute Sanitizer tools, and complete page/snapshot reconciliation at shutdown.

## Non-claim boundary

Hosted CI proves CUDA compilation, contracts, schemas, anchors, and false-evidence rejection. It does not prove performance or production correctness. Those claims remain gated on the self-hosted workflow running the patched pinned server with a compatible runner-local GGUF and producing strict passing M12 evidence.
