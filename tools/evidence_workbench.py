#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_METRICS = ("throughput_tokens_s", "latency_p50_ms", "latency_p95_ms", "peak_vram_mib", "correctness")
MATCH_FIELDS = ("model", "runtime", "hardware", "workload")


def load_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "chromofold.evidence-result.v1":
        raise ValueError(f"{path}: unsupported evidence schema")
    if value.get("role") not in {"baseline", "candidate"}:
        raise ValueError(f"{path}: role must be baseline or candidate")
    fingerprint = value.get("fingerprint")
    metrics = value.get("metrics")
    if not isinstance(fingerprint, dict) or not isinstance(metrics, dict):
        raise ValueError(f"{path}: fingerprint and metrics are required objects")
    missing = [name for name in MATCH_FIELDS if not fingerprint.get(name)]
    missing += [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    return value


def pct(candidate: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline metric cannot be zero")
    return round((candidate - baseline) / baseline * 100.0, 3)


def analyze(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if baseline["role"] != "baseline" or candidate["role"] != "candidate":
        raise ValueError("comparison requires baseline then candidate")
    mismatches = [field for field in MATCH_FIELDS if baseline["fingerprint"].get(field) != candidate["fingerprint"].get(field)]
    bm, cm = baseline["metrics"], candidate["metrics"]
    deltas = {
        "throughput_pct": pct(float(cm["throughput_tokens_s"]), float(bm["throughput_tokens_s"])),
        "latency_p50_pct": pct(float(cm["latency_p50_ms"]), float(bm["latency_p50_ms"])),
        "latency_p95_pct": pct(float(cm["latency_p95_ms"]), float(bm["latency_p95_ms"])),
        "peak_vram_pct": pct(float(cm["peak_vram_mib"]), float(bm["peak_vram_mib"])),
    }
    if "capacity_tokens" in bm and "capacity_tokens" in cm:
        deltas["capacity_pct"] = pct(float(cm["capacity_tokens"]), float(bm["capacity_tokens"]))
    gates = {
        "fingerprints_match": not mismatches,
        "candidate_correct": bool(cm["correctness"]),
        "baseline_correct": bool(bm["correctness"]),
        "release_digest_present": bool(candidate["fingerprint"].get("release_digest")),
        "capacity_improved": deltas.get("capacity_pct", 0.0) > 0 or deltas["peak_vram_pct"] < 0,
    }
    if mismatches or not gates["candidate_correct"] or not gates["baseline_correct"]:
        state = "FAIL"
    elif not gates["release_digest_present"]:
        state = "INCOMPLETE"
    elif gates["capacity_improved"]:
        state = "PASS"
    else:
        state = "FAIL"
    canonical = json.dumps({"baseline": baseline, "candidate": candidate}, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "chromofold.evidence-analysis.v1",
        "state": state,
        "mismatched_fingerprint_fields": mismatches,
        "gates": gates,
        "deltas": deltas,
        "baseline": bm,
        "candidate": cm,
        "comparison_digest": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "boundary": "PASS means the portable comparison gates passed. It is not maintainer qualification or independent reproduction.",
    }


def render_markdown(result: dict[str, Any]) -> str:
    d = result["deltas"]
    rows = [
        ("Throughput", d["throughput_pct"], "higher is better"),
        ("Latency p50", d["latency_p50_pct"], "lower is better"),
        ("Latency p95", d["latency_p95_pct"], "lower is better"),
        ("Peak VRAM", d["peak_vram_pct"], "lower is better"),
    ]
    if "capacity_pct" in d:
        rows.append(("Capacity", d["capacity_pct"], "higher is better"))
    table = "\n".join(f"| {name} | {value:+.3f}% | {direction} |" for name, value, direction in rows)
    return f"""# ChromoFold evidence comparison\n\n**State:** `{result['state']}`  \n**Comparison digest:** `{result['comparison_digest']}`\n\n| Metric | Delta | Direction |\n|---|---:|---|\n{table}\n\n## Gates\n\n""" + "\n".join(f"- {'PASS' if ok else 'FAIL'} — `{name}`" for name, ok in result["gates"].items()) + f"\n\n> {result['boundary']}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze portable ChromoFold baseline and candidate results")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = analyze(load_result(args.baseline), load_result(args.candidate))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.markdown:
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    return 0 if result["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
