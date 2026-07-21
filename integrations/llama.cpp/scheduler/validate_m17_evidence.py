#!/usr/bin/env python3
import json
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    require(len(sys.argv) == 2, "usage: validate_m17_evidence.py evidence.json")
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        data = json.load(handle)
    tenants = data.get("tenants", {})
    scheduling = data.get("scheduling", {})
    resilience = data.get("resilience", {})
    autoscaling = data.get("autoscaling", {})
    isolation = data.get("isolation", {})
    shutdown = data.get("shutdown", {})
    require(tenants.get("count", 0) >= 2, "at least two tenants required")
    require(tenants.get("quotas_enforced") is True, "quotas not enforced")
    require(tenants.get("fairness_demonstrated") is True, "fairness not demonstrated")
    for key in ("deadline_schedules", "continuous_batches", "preemptions", "checkpoint_resumes", "cancellations"):
        require(scheduling.get(key, 0) > 0, f"missing scheduling proof: {key}")
    require(resilience.get("circuit_opened") is True and resilience.get("circuit_recovered") is True, "circuit breaker proof missing")
    require(resilience.get("safe_drains", 0) > 0 and resilience.get("rolling_compatibility") is True, "drain or upgrade proof missing")
    for key in ("scale_up_plans", "heterogeneous_routes", "model_placement_decisions"):
        require(autoscaling.get(key, 0) > 0, f"missing autoscaling proof: {key}")
    require(isolation.get("cross_tenant_reuse_rejections", 0) > 0, "tenant reuse rejection missing")
    require(isolation.get("cross_tenant_contamination") == 0, "cross-tenant contamination detected")
    require(isolation.get("dense_fallback_launches") == 0, "dense fallback detected")
    require(shutdown.get("active_refs") == 0, "active references remain")
    require(shutdown.get("usage_idempotent") is True, "usage accounting is not idempotent")
    print("M17 evidence accepted")


if __name__ == "__main__":
    main()
