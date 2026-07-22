# ChromoFold Evidence Workbench

The Evidence Workbench answers the product question:

> Did the candidate improve this exact workload on this exact machine?

It has two implementations over the same contract:

- `site/workbench.html` performs local browser analysis without network requests;
- `tools/evidence_workbench.py` performs deterministic CLI/CI analysis.

## Result contract

Both inputs use `chromofold.evidence-result.v1` and identify:

- role: baseline or candidate;
- model fingerprint;
- runtime fingerprint;
- hardware fingerprint;
- workload fingerprint;
- candidate release digest;
- throughput, latency, peak VRAM, capacity and correctness metrics.

See `product/evidence-result.schema.json` and `examples/evidence/`.

## States

- `PASS`: fingerprints match, correctness passes, a candidate digest is present, and capacity or peak-VRAM improves.
- `FAIL`: fingerprints differ, correctness fails, or no capacity improvement is measured.
- `INCOMPLETE`: comparison inputs are otherwise valid but required publication identity is missing.

A Workbench `PASS` is not maintainer qualification and not independent reproduction. It is a portable, auditable comparison result suitable as an input to the qualification process.

## CLI

```bash
python3 tools/evidence_workbench.py \
  examples/evidence/baseline.json \
  examples/evidence/candidate.json \
  --output evidence-analysis.json \
  --markdown evidence-report.md
```

The command exits `0` only for `PASS`; `FAIL` and `INCOMPLETE` exit `2`, making regressions visible in CI.

## Privacy

The browser implementation uses `File.text()`, in-memory comparison and Blob downloads. It contains no `fetch`, XMLHttpRequest, analytics or upload endpoint. Refreshing the page clears the imported data.

## Promotion path

1. Produce baseline and candidate result files from matched runs.
2. Analyze them in the Workbench.
3. Review every positive and negative delta.
4. Preserve the generated comparison report.
5. Run repository qualification against the exact candidate digest.
6. Publish only wording supported by the resulting evidence level.
