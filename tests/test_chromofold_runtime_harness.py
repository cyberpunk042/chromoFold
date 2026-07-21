#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("harness", ROOT / "tools/chromofold_runtime_harness.py")
assert SPEC and SPEC.loader
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


def test_redaction() -> None:
    value = {
        "prompt": "secret prompt",
        "nested": {"api_key": "abc", "safe": "Authorization: Bearer token-value"},
    }
    out = harness.redact(value)
    assert out["prompt"] == "[REDACTED]"
    assert out["nested"]["api_key"] == "[REDACTED]"
    assert "token-value" not in out["nested"]["safe"]


def test_incident_bundle_is_redacted() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "runtime.log").write_text("password=secret\nAuthorization: Bearer abc", encoding="utf-8")
        bundle = root / "incident.tar.gz"
        harness.create_incident_bundle(bundle, {"prompt": "hello", "safe": "ok"}, root)
        with tarfile.open(bundle, "r:gz") as archive:
            data = "\n".join(
                archive.extractfile(member).read().decode("utf-8", "replace")
                for member in archive.getmembers()
                if member.isfile()
            )
        assert "hello" not in data
        assert "secret" not in data
        assert "Bearer abc" not in data
        assert "[REDACTED]" in data


def test_required_scenarios_are_stable() -> None:
    names = [name for name, _ in harness.required_scenarios()]
    assert names == [
        "cache-cold",
        "cache-warm",
        "multi-tenant-overload",
        "cancellation",
        "decode-worker-failure",
        "network-fault",
        "certificate-rotation",
        "rolling-upgrade",
        "rollback",
    ]


def test_incomplete_decision_fixture() -> None:
    fixture = {
        "schema": "chromofold.rc1.runtime-evidence.v1",
        "decision": "INCOMPLETE",
        "reason": "runtime not ready",
    }
    assert json.loads(json.dumps(fixture))["decision"] == "INCOMPLETE"


if __name__ == "__main__":
    test_redaction()
    test_incident_bundle_is_redacted()
    test_required_scenarios_are_stable()
    test_incomplete_decision_fixture()
    print("RC1 runtime harness tests: PASS")
