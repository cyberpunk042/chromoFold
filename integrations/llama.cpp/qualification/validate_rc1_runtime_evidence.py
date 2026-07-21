#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = Path(__file__).with_name("rc1-runtime-harness.schema.json")
REQUIRED_SCENARIOS = {
    "cache-cold",
    "cache-warm",
    "multi-tenant-overload",
    "cancellation",
    "decode-worker-failure",
    "network-fault",
    "certificate-rotation",
    "rolling-upgrade",
    "rollback",
}


def validate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(data, schema)

    decision = data["decision"]
    if decision == "PASS":
        if data.get("qualified_digest") != data.get("promoted_digest"):
            raise ValueError("PASS requires identical qualified and promoted digests")
        if not data.get("gpu", {}).get("devices"):
            raise ValueError("PASS requires observed GPU devices")
        telemetry = data.get("telemetry", {})
        for key in ("end_to_end_correlated", "metrics_scraped", "mtls_observed", "audit_chain_verified"):
            if telemetry.get(key) is not True:
                raise ValueError(f"PASS requires telemetry.{key}=true")
        if telemetry.get("prompt_content_leaks") != 0:
            raise ValueError("PASS requires zero prompt-content leaks")
        quality = data.get("quality", {})
        for key in ("correctness_failures", "cuda_errors", "cross_tenant_contamination", "dense_fallback_launches"):
            if quality.get(key) != 0:
                raise ValueError(f"PASS requires quality.{key}=0")
        if data.get("scenario_execution_complete") is not True or data.get("scenario_success") is not True:
            raise ValueError("PASS requires all scenarios executed and successful")
        scenarios = set(data.get("scenarios", {}))
        missing = REQUIRED_SCENARIOS - scenarios
        if missing:
            raise ValueError(f"PASS missing scenarios: {sorted(missing)}")
        if data.get("incident_bundle_redacted") is not True:
            raise ValueError("PASS requires a verified redacted incident bundle")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_rc1_runtime_evidence.py evidence.json")
    try:
        validate(Path(sys.argv[1]))
    except Exception as exc:
        print(f"RC1 runtime evidence INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print("RC1 runtime evidence VALID")
