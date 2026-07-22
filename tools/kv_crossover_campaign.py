#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "product" / "kv-crossover-campaign.json"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def campaign_cases(config: dict[str, Any]) -> list[dict[str, int]]:
    dims = config["dimensions"]
    names = ("context_tokens", "head_dim", "query_count", "kv_heads", "code_bits")
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*(dims[name] for name in names))]


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0:
        return float("inf")
    return statistics.stdev(values) / mean * 100.0


def evaluate_case(case: dict[str, Any], max_cv_pct: float) -> dict[str, Any]:
    required = ("compressed_latency_us", "fair_dense_latency_us", "compressed_peak_vram_mib", "dense_peak_vram_mib", "correctness_max_error")
    missing = [key for key in required if key not in case]
    if missing:
        return {"state": "INCOMPLETE", "reason": f"missing {', '.join(missing)}"}
    compressed = [float(value) for value in case["compressed_latency_us"]]
    dense = [float(value) for value in case["fair_dense_latency_us"]]
    if len(compressed) < 3 or len(dense) < 3:
        return {"state": "INCOMPLETE", "reason": "at least three latency repetitions are required"}
    compressed_cv = coefficient_of_variation(compressed)
    dense_cv = coefficient_of_variation(dense)
    if compressed_cv > max_cv_pct or dense_cv > max_cv_pct:
        return {"state": "INCOMPLETE", "reason": "latency variability exceeds the campaign gate", "compressed_cv_pct": compressed_cv, "dense_cv_pct": dense_cv}
    if float(case["correctness_max_error"]) > float(case.get("correctness_tolerance", 0.0002)):
        return {"state": "INCOMPLETE", "reason": "correctness tolerance failed"}
    compressed_median = statistics.median(compressed)
    dense_median = statistics.median(dense)
    ratio = compressed_median / dense_median
    memory_improved = float(case["compressed_peak_vram_mib"]) < float(case["dense_peak_vram_mib"])
    if ratio <= 0.98 and memory_improved:
        state = "WIN"
    elif ratio <= 1.02 and memory_improved:
        state = "PARITY"
    else:
        state = "NO_CROSSOVER"
    return {
        "state": state,
        "compressed_median_us": compressed_median,
        "dense_median_us": dense_median,
        "compressed_over_dense": ratio,
        "compressed_cv_pct": compressed_cv,
        "dense_cv_pct": dense_cv,
        "memory_improved": memory_improved,
    }


def evaluate(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "chromofold.kv-crossover-results.v1":
        raise ValueError("expected chromofold.kv-crossover-results.v1")
    results = []
    for case in payload.get("cases", []):
        identity = {key: case.get(key) for key in config["dimensions"]}
        results.append({"case": identity, **evaluate_case(case, float(config["maximum_cv_pct"]))})
    states = [item["state"] for item in results]
    overall = "WIN" if "WIN" in states else "PARITY" if "PARITY" in states else "INCOMPLETE" if not results or all(state == "INCOMPLETE" for state in states) else "NO_CROSSOVER"
    canonical = json.dumps(results, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "chromofold.kv-crossover-analysis.v1",
        "state": overall,
        "cases": results,
        "counts": {state: states.count(state) for state in config["publication_states"]},
        "analysis_digest": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        "boundary": config["boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or evaluate the ChromoFold KV latency crossover campaign")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=Path)
    analyze = sub.add_parser("evaluate")
    analyze.add_argument("results", type=Path)
    analyze.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_json(CONFIG)
    if args.command == "plan":
        payload = {"schema": "chromofold.kv-crossover-plan.v1", "campaign": config, "cases": campaign_cases(config)}
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        payload = evaluate(load_json(args.results), config)
        text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
