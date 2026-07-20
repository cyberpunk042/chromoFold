#!/usr/bin/env python3
"""Validate and summarize a ChromoFold M9 benchmark result.

This intentionally uses only the Python standard library so it can run in CI and
on benchmark hosts without installing packages. Structural validation is strict
for the ship-critical fields; the JSON Schema remains the canonical full shape.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODES = {"context-capacity", "batch-capacity", "latency-sweep", "correctness"}


class ValidationError(ValueError):
    pass


def require(mapping: dict[str, Any], key: str, expected: type, path: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"{path}.{key}: missing required field")
    value = mapping[key]
    if expected is int and isinstance(value, bool):
        raise ValidationError(f"{path}.{key}: expected integer")
    if not isinstance(value, expected):
        raise ValidationError(f"{path}.{key}: expected {expected.__name__}")
    return value


def finite_nonnegative(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path}: expected number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValidationError(f"{path}: expected finite non-negative number")
    return number


def validate_measurement(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected object")
    require(value, "completed", bool, path)
    for key in ("peak_vram_bytes", "resident_kv_bytes"):
        number = require(value, key, int, path)
        if number < 0:
            raise ValidationError(f"{path}.{key}: must be non-negative")
    for key in (
        "ttft_ms",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "decode_latency_p50_ms",
        "decode_latency_p95_ms",
    ):
        finite_nonnegative(require(value, key, (int, float), path), f"{path}.{key}")
    if value["decode_latency_p95_ms"] < value["decode_latency_p50_ms"]:
        raise ValidationError(f"{path}: p95 latency is lower than p50")
    return value


def validate_document(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise ValidationError("root: expected object")
    if require(doc, "schema_version", int, "root") != 1:
        raise ValidationError("root.schema_version: unsupported version")

    run = require(doc, "run", dict, "root")
    mode = require(run, "mode", str, "root.run")
    if mode not in MODES:
        raise ValidationError(f"root.run.mode: unsupported mode {mode!r}")

    software = require(doc, "software", dict, "root")
    if not SHA40.fullmatch(require(software, "chromofold_commit", str, "root.software")):
        raise ValidationError("root.software.chromofold_commit: expected 40 lowercase hex characters")
    if require(software, "runtime", str, "root.software") != "llama.cpp":
        raise ValidationError("root.software.runtime: M9 v1 requires llama.cpp")
    if not SHA40.fullmatch(require(software, "runtime_commit", str, "root.software")):
        raise ValidationError("root.software.runtime_commit: expected 40 lowercase hex characters")

    workload = require(doc, "workload", dict, "root")
    if not SHA256.fullmatch(require(workload, "model_sha256", str, "root.workload")):
        raise ValidationError("root.workload.model_sha256: expected 64 lowercase hex characters")
    for key in ("context_tokens", "batch_size", "prompt_tokens", "generated_tokens"):
        if require(workload, key, int, "root.workload") <= 0:
            raise ValidationError(f"root.workload.{key}: must be positive")

    baseline = validate_measurement(require(doc, "baseline", dict, "root"), "root.baseline")
    chromofold = validate_measurement(require(doc, "chromofold", dict, "root"), "root.chromofold")

    correctness = require(doc, "correctness", dict, "root")
    require(correctness, "generation_completed", bool, "root.correctness")
    quality = require(correctness, "quality_check", str, "root.correctness")
    if quality not in {"pass", "fail", "not-run"}:
        raise ValidationError("root.correctness.quality_check: invalid value")
    require(correctness, "historical_dense_kv_materialized", bool, "root.correctness")

    if chromofold["completed"] and correctness["historical_dense_kv_materialized"]:
        raise ValidationError(
            "root.correctness.historical_dense_kv_materialized: must be false for a ChromoFold workload claim"
        )

    return doc


def claim_summary(doc: dict[str, Any]) -> tuple[bool, list[str]]:
    base = doc["baseline"]
    cf = doc["chromofold"]
    correct = doc["correctness"]
    reasons: list[str] = []

    if not cf["completed"] or not correct["generation_completed"]:
        reasons.append("ChromoFold generation did not complete")
    if correct["quality_check"] != "pass":
        reasons.append("quality check did not pass")
    if correct["historical_dense_kv_materialized"]:
        reasons.append("historical dense KV was materialized")

    if base["completed"]:
        if cf["peak_vram_bytes"] >= base["peak_vram_bytes"]:
            reasons.append("peak VRAM was not lower than the completed baseline")
        if cf["resident_kv_bytes"] >= base["resident_kv_bytes"]:
            reasons.append("resident KV bytes were not lower than the completed baseline")
    else:
        # A baseline OOM with a completed ChromoFold run is the capacity proof.
        if not base.get("oom", False):
            reasons.append("baseline neither completed nor reported OOM")

    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--require-claim", action="store_true", help="fail unless the M9 workload claim is supported")
    args = parser.parse_args()

    try:
        with args.result.open("r", encoding="utf-8") as handle:
            doc = validate_document(json.load(handle))
        supported, reasons = claim_summary(doc)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2

    base = doc["baseline"]
    cf = doc["chromofold"]
    ratio = (base["resident_kv_bytes"] / cf["resident_kv_bytes"]) if cf["resident_kv_bytes"] else math.inf
    print("VALID M9 RESULT")
    print(f"mode: {doc['run']['mode']}")
    print(f"context: {doc['workload']['context_tokens']} tokens")
    print(f"batch: {doc['workload']['batch_size']}")
    print(f"resident KV reduction: {ratio:.2f}x")
    print(f"claim supported: {'yes' if supported else 'no'}")
    for reason in reasons:
        print(f"- {reason}")

    return 0 if supported or not args.require_claim else 3


if __name__ == "__main__":
    raise SystemExit(main())
