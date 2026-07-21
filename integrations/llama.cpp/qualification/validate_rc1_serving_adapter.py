#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("rc1-serving-adapter.schema.json"))
    parser.add_argument("--strict-ready", action="store_true")
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(snapshot)

    errors: list[str] = []
    if args.strict_ready:
        if not snapshot["qualification_mode"]:
            errors.append("qualification mode is disabled")
        if not snapshot["workers"]:
            errors.append("worker inventory is empty")
        if not snapshot["mtls_verified"]:
            errors.append("mTLS is not verified")
        if not snapshot["audit_chain_verified"]:
            errors.append("audit chain is not verified")
        if not snapshot["telemetry_correlated"]:
            errors.append("telemetry is not correlated")
        for field in ("cuda_errors", "correctness_failures", "cross_tenant_violations", "dense_fallback_launches"):
            if snapshot[field] != 0:
                errors.append(f"{field} must be zero")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    print("RC1 serving adapter snapshot: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
