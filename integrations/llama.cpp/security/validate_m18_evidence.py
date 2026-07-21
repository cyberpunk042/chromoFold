#!/usr/bin/env python3
import json
import pathlib
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_m18_evidence.py evidence.json", file=sys.stderr)
        return 2
    try:
        evidence = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
        require(isinstance(evidence, dict), "evidence must be an object")
        transport = evidence["transport"]
        authorization = evidence["authorization"]
        integrity = evidence["integrity"]
        supply = evidence["supply_chain"]
        upgrade = evidence["upgrade"]
        isolation = evidence["isolation"]
        shutdown = evidence["shutdown"]
        require(transport["mtls_connections"] > 0, "mTLS was not exercised")
        require(transport["unauthenticated_rejections"] > 0, "unauthenticated peers were not rejected")
        require(transport["rotation_without_outage"] is True, "key rotation was not proven")
        require(authorization["grants"] > 0 and authorization["denials"] > 0, "authorization paths incomplete")
        require(authorization["default_deny"] is True, "authorization is not default deny")
        require(integrity["manifests_verified"] > 0, "no manifest was verified")
        require(integrity["tampered_manifests_rejected"] > 0, "tampering was not rejected")
        require(integrity["replays_rejected"] > 0, "replay rejection was not proven")
        require(integrity["audit_chain_verified"] is True, "audit chain not verified")
        require(all(supply[name] is True for name in ["reproducible_build", "sbom_generated", "provenance_generated", "images_by_digest"]), "supply-chain evidence incomplete")
        require(supply["critical_vulnerabilities"] == 0, "critical vulnerabilities remain")
        require(all(upgrade[name] is True for name in ["rolling_upgrade_completed", "incompatible_upgrade_rejected", "rollback_completed"]), "upgrade lifecycle incomplete")
        require(isolation["cross_tenant_substitutions_rejected"] > 0, "tenant substitution was not tested")
        require(isolation["cross_tenant_contamination"] == 0, "cross-tenant contamination detected")
        require(shutdown["active_page_refs"] == 0 and shutdown["active_snapshot_refs"] == 0, "references leaked at shutdown")
        require(shutdown["dense_fallback_launches"] == 0, "dense fallback was used")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"invalid M18 evidence: {error}", file=sys.stderr)
        return 1
    print("M18 evidence valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
