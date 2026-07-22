#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import zipfile
from pathlib import Path
from typing import Any

MATCH = ("model", "runtime", "hardware", "workload")
METRICS = ("throughput_tokens_s", "latency_p50_ms", "latency_p95_ms", "peak_vram_mib", "capacity_tokens")
LOWER_IS_BETTER = {"latency_p50_ms", "latency_p95_ms", "peak_vram_mib"}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def validate_run(run: dict[str, Any], role: str) -> None:
    if run.get("schema") != "chromofold.evidence-result.v1" or run.get("role") != role:
        raise ValueError(f"expected {role} chromofold.evidence-result.v1")
    if not run.get("fingerprint") or not run.get("metrics"):
        raise ValueError(f"{role}: fingerprint and metrics required")
    if not bool(run["metrics"].get("correctness")):
        raise ValueError(f"{role}: correctness failed")


def summarize(runs: list[dict[str, Any]], role: str) -> dict[str, Any]:
    if len(runs) < 3:
        raise ValueError(f"{role}: at least three runs required")
    for run in runs:
        validate_run(run, role)
    fingerprint = runs[0]["fingerprint"]
    for run in runs[1:]:
        for key in MATCH:
            if run["fingerprint"].get(key) != fingerprint.get(key):
                raise ValueError(f"{role}: inconsistent {key}")
    metrics: dict[str, Any] = {}
    for name in METRICS:
        values = [float(r["metrics"][name]) for r in runs if name in r["metrics"]]
        if not values:
            continue
        median = statistics.median(values)
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        metrics[name] = {
            "values": values,
            "median": median,
            "mean": mean,
            "stdev": stdev,
            "cv_pct": 0.0 if mean == 0 else abs(stdev / mean) * 100,
            "min": min(values),
            "max": max(values),
        }
    return {"run_count": len(runs), "fingerprint": fingerprint, "metrics": metrics}


def analyze(session: dict[str, Any]) -> dict[str, Any]:
    if session.get("schema") != "chromofold.qualification-session.v1":
        raise ValueError("expected chromofold.qualification-session.v1")
    baseline = summarize(session.get("baseline_runs", []), "baseline")
    candidate = summarize(session.get("candidate_runs", []), "candidate")
    mismatches = [k for k in MATCH if baseline["fingerprint"].get(k) != candidate["fingerprint"].get(k)]
    digest = candidate["fingerprint"].get("release_digest")
    deltas: dict[str, float] = {}
    unstable: list[str] = []
    regressions: list[str] = []
    for name in sorted(set(baseline["metrics"]) & set(candidate["metrics"])):
        b = baseline["metrics"][name]["median"]
        c = candidate["metrics"][name]["median"]
        if b == 0:
            raise ValueError(f"baseline median for {name} cannot be zero")
        delta = (c - b) / b * 100
        deltas[name] = delta
        if max(baseline["metrics"][name]["cv_pct"], candidate["metrics"][name]["cv_pct"]) > 5:
            unstable.append(name)
        if (name in LOWER_IS_BETTER and delta > 5) or (name not in LOWER_IS_BETTER and delta < -5):
            regressions.append(name)
    capacity_win = deltas.get("capacity_tokens", 0) > 0 or deltas.get("peak_vram_mib", 0) < 0
    if mismatches or regressions:
        state = "FAIL"
    elif not digest or unstable:
        state = "INCOMPLETE"
    elif capacity_win:
        state = "PASS"
    else:
        state = "FAIL"
    result = {
        "schema": "chromofold.qualification-analysis.v1",
        "session_id": session.get("session_id"),
        "state": state,
        "baseline": baseline,
        "candidate": candidate,
        "median_deltas_pct": deltas,
        "mismatched_fingerprint_fields": mismatches,
        "unstable_metrics": unstable,
        "regressions": regressions,
        "release_digest_present": bool(digest),
        "boundary": "PASS means repeated-run statistical gates passed for this exact session. It is not maintainer qualification or independent reproduction.",
    }
    result["analysis_digest"] = "sha256:" + hashlib.sha256(canonical(result)).hexdigest()
    return result


def write_package(path: Path, session: dict[str, Any], analysis: dict[str, Any]) -> None:
    manifest = {
        "schema": "chromofold.support-package.v1",
        "session_id": session.get("session_id"),
        "analysis_digest": analysis["analysis_digest"],
        "files": ["session.json", "analysis.json", "REPORT.md"],
        "privacy": "Review notes and attachments before sharing. No package is uploaded automatically.",
    }
    report = f"# ChromoFold qualification session\n\n**State:** {analysis['state']}\n\n**Runs:** {analysis['baseline']['run_count']} baseline / {analysis['candidate']['run_count']} candidate\n\n**Analysis digest:** `{analysis['analysis_digest']}`\n\n## Median deltas\n" + "\n".join(f"- {k}: {v:+.3f}%" for k, v in analysis["median_deltas_pct"].items()) + f"\n\n{analysis['boundary']}\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("session.json", json.dumps(session, indent=2, sort_keys=True))
        zf.writestr("analysis.json", json.dumps(analysis, indent=2, sort_keys=True))
        zf.writestr("REPORT.md", report)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a repeated-run ChromoFold qualification session")
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    session = json.loads(args.session.read_text(encoding="utf-8"))
    result = analyze(session)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.package:
        write_package(args.package, session, result)
    print(text)
    return 0 if result["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
