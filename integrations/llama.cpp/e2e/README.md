# llama.cpp end-to-end operator guide

This directory owns the reproducible workload proof for ChromoFold M9.

## Files

- `llama-pin.json`: immutable upstream revision and required capabilities.
- `verify_upstream.py`: rejects wrong, dirty, or structurally incompatible checkouts.
- `fetch_and_prepare.py`: fetches the pin and creates an isolated ChromoFold overlay.
- `run_pair.py`: executes one deterministic dense or ChromoFold case.
- `capacity_sweep.py`: finds the largest completed context for each backend.
- `validate_evidence.py`: rejects incomplete and false-positive evidence.
- `evidence.schema.json`: machine-readable evidence shape.

## Required runtime instrumentation

A ChromoFold-enabled llama.cpp binary must write the file named by `CHROMOFOLD_EVIDENCE_PATH`. The runtime JSON must contain:

```json
{
  "correctness": {
    "finite": true,
    "token_match_rate": 1.0,
    "max_logit_error": 0.0
  },
  "memory": {
    "peak_vram_bytes": 0,
    "kv_bytes": 0
  },
  "latency": {
    "prefill_ms": 0.0,
    "decode_ms_per_token": 0.0
  },
  "counters": {
    "compressed_attention_launches": 0,
    "sealed_values_consumed": 0,
    "dense_fallback_launches": 0
  }
}
```

The wrapper does not fabricate missing backend counters. Missing runtime instrumentation therefore fails claim validation.

## Environment contract

- `CHROMOFOLD_KV_BACKEND`: `dense` or `chromofold`.
- `CHROMOFOLD_PAGE_SIZE`: positive page size.
- `CHROMOFOLD_EVIDENCE_PATH`: runtime evidence output path.

The upstream patch may additionally expose CLI flags, but the environment contract remains stable for automation.

## Failure interpretation

- Wrong upstream commit: integration drift, do not patch automatically.
- Dirty checkout: evidence is not reproducible.
- Missing runtime JSON: patched binary did not expose instrumentation.
- Positive dense fallback counter: workload did not remain on ChromoFold.
- Zero sealed values: context never crossed a page boundary or compressed history was not consumed.
- Capacity allocation without decode: not a successful capacity case.
