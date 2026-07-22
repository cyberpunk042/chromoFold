# ChromoFold Qualification Sessions

Qualification Sessions extend the single-pair Evidence Workbench into a repeatability-aware workflow.

## Contract

A session uses `chromofold.qualification-session.v1` and contains at least three baseline runs and three candidate runs. Every run remains a `chromofold.evidence-result.v1` document.

All runs on each side must share the same model, runtime, hardware and workload fingerprint. Baseline and candidate fingerprints must also match. The candidate must identify the exact release digest.

## Analysis

The analyzer calculates, for every available metric:

- raw run values;
- median;
- arithmetic mean;
- sample standard deviation;
- coefficient of variation;
- minimum and maximum;
- median percentage delta between baseline and candidate.

The default variability ceiling is 5% coefficient of variation. A metric above that ceiling makes the session `INCOMPLETE` because the result is not stable enough to promote.

A material regression is more than 5% worse latency, more than 5% lower throughput or capacity, or another explicitly lower-is-better metric moving in the wrong direction. Material regressions produce `FAIL` even when memory improves.

A `PASS` requires:

1. matching fingerprints;
2. successful correctness in every run;
3. an exact candidate release digest;
4. no unstable metrics;
5. no material regressions;
6. improved capacity or reduced peak VRAM.

A session `PASS` is portable evidence. It is not maintainer qualification and does not count as independent reproduction.

## CLI

```bash
python3 tools/qualification_session.py \
  examples/evidence/session.json \
  --output qualification-analysis.json \
  --package chromofold-support.zip
```

Exit code `0` means `PASS`. `FAIL` and `INCOMPLETE` return exit code `2`.

## Support package

The ZIP package contains:

- `session.json` — complete repeated-run input;
- `analysis.json` — statistics, gates and analysis digest;
- `REPORT.md` — human-readable summary;
- `manifest.json` — package identity and privacy warning.

Review notes and attachments before sharing. The browser and CLI never upload a package automatically.

## Browser workspace

The public portal route `sessions.html` performs the same core calculations locally. A session is written to browser storage only when the user presses **Save locally**. It can be restored or cleared from the same page.
