#!/usr/bin/env python3
import json
import pathlib
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_m19_evidence.py evidence.json", file=sys.stderr)
        return 2
    try:
        data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
        release = data["release"]
        telemetry = data["telemetry"]
        slo = data["slo"]
        correctness = data["correctness"]
        operations = data["operations"]
        promotion = data["promotion"]
        shutdown = data["shutdown"]
        require(release["qualified_digest"] == release["promoted_digest"], "promotion rebuilt the artifact")
        require(telemetry["end_to_end_traces"] > 0 and telemetry["correlated_records"] is True, "trace correlation missing")
        require(telemetry["prompt_content_leaks"] == 0 and telemetry["bounded_cardinality"] is True, "telemetry privacy/cardinality failed")
        require(telemetry["incident_bundle_redacted"] is True, "incident bundle was not redacted")
        require(slo["ttft_measured"] is True and slo["token_latency_measured"] is True, "latency evidence incomplete")
        require(slo["regressions_within_policy"] is True and slo["error_budget_exhausted"] is False, "release SLO gate failed")
        require(all(correctness[key] == 0 for key in ["failures", "cross_tenant_contamination", "cuda_errors"]), "correctness failure")
        require(correctness["memory_reconciled"] is True, "memory accounting did not reconcile")
        required_ops = ["cache_cold", "cache_warm", "multi_tenant_overload", "cancellation_cleanup", "worker_recovery", "network_fault_recovery", "certificate_rotation", "rolling_upgrade", "rollback", "autoscaling_response"]
        require(all(operations[key] is True for key in required_ops), "operational qualification incomplete")
        require(operations["steady_state_growth_bytes"] == 0, "steady-state resource growth detected")
        require(promotion["same_digest"] is True and promotion["qualification_approved"] is True, "promotion not approved")
        require(all(shutdown[key] == 0 for key in ["page_refs", "snapshot_refs", "lease_refs", "request_refs"]), "references leaked")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"invalid M19 evidence: {error}", file=sys.stderr)
        return 1
    print("M19 evidence valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
