#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover
    raise SystemExit("jsonschema is required: python -m pip install jsonschema") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def validate_semantics(evidence: dict[str, Any]) -> None:
    runtime = evidence["runtime"]
    counters = evidence["counters"]
    memory = evidence["memory"]
    numerics = evidence["numerics"]
    sanitizer = evidence["sanitizer"]

    require(runtime["backend"] == "chromofold", "proof evidence must use the chromofold backend")
    require(runtime["completed"] is True and runtime["exit_code"] == 0, "inference did not complete successfully")
    require(counters["device_append_calls"] > 0, "no real device append was recorded")
    require(counters["seal_successes"] > 0, "no page was sealed")
    require(counters["descriptor_publications"] == counters["seal_successes"], "every successful seal must publish exactly one descriptor")
    require(counters["compressed_attention_launches"] > 0, "compressed attention never launched")
    require(counters["sealed_pages_consumed"] > 0, "compressed attention did not consume a sealed page")
    require(counters["sealed_values_consumed"] > 0, "compressed attention did not consume sealed values")
    require(counters["active_values_consumed"] > 0, "compressed attention did not consume the active tail")
    require(counters["dense_fallback_launches"] == 0, "dense fallback occurred")
    require(counters["cuda_errors"] == 0, "CUDA errors were recorded")
    require(memory["measured"] is True, "memory values must be measured")
    require(memory["peak_process_vram_bytes"] >= memory["peak_chromofold_owned_bytes"], "owned bytes exceed process VRAM")
    require(numerics["all_finite"] is True, "non-finite output was observed")
    for key in ("max_abs_error", "mean_abs_error", "top1_agreement"):
        require(math.isfinite(float(numerics[key])), f"numerics.{key} must be finite")
    require(numerics["mean_abs_error"] <= numerics["max_abs_error"], "mean absolute error exceeds maximum absolute error")
    require(all(value == "passed" for value in sanitizer.values()), "all Compute Sanitizer tools must pass")

    capacity = evidence["capacity"]
    require(capacity["results"], "capacity sweep must contain results")
    require(capacity["chromofold_max_context"] > 0, "no successful ChromoFold capacity point")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ChromoFold M9 hardware evidence v2")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("evidence-v2.schema.json"))
    args = parser.parse_args()

    evidence = load(args.evidence)
    schema = load(args.schema)
    jsonschema.Draft202012Validator(schema).validate(evidence)
    validate_semantics(evidence)
    print(json.dumps({
        "valid": True,
        "model_sha256": evidence["model"]["sha256"],
        "llama_commit": evidence["runtime"]["llama_commit"],
        "seal_successes": evidence["counters"]["seal_successes"],
        "compressed_attention_launches": evidence["counters"]["compressed_attention_launches"],
        "chromofold_max_context": evidence["capacity"]["chromofold_max_context"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
