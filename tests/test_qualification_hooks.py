#!/usr/bin/env python3
"""Contract tests for the RC1 qualification scenario hooks.

Verifies each hook (integrations/llama.cpp/qualification/hooks/<scenario>) is executable, emits one valid JSON
result with the required fields, uses a legal status, and — critically for the honesty contract — does NOT
fabricate a PASS on a machine with no serving runtime. Also checks the hook set matches the adapter's SCENARIOS.
"""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "integrations" / "llama.cpp" / "qualification" / "hooks"
REQUIRED = {"status", "requests_started", "requests_completed", "requests_failed",
            "recovery_ms", "references_reconciled", "observations"}


def adapter_scenarios() -> set[str]:
    src = (ROOT / "tools" / "chromofold_qualification_adapter.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "SCENARIOS" for t in node.targets):
            return {e.value for e in ast.walk(node.value) if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    raise AssertionError("SCENARIOS not found in adapter")


def test_hook_set_matches_scenarios() -> None:
    scenarios = adapter_scenarios()
    present = {p.name for p in HOOKS.iterdir() if p.is_file() and not p.name.startswith(("_", "."))
               and p.suffix != ".md"}
    assert scenarios <= present, f"missing hooks for scenarios: {scenarios - present}"


def test_hooks_are_executable_and_contract_valid() -> None:
    # No serving runtime is reachable in the test environment, so every hook must return INCOMPLETE (or FAIL),
    # never a fabricated PASS.
    env = dict(os.environ)
    env.pop("CHROMOFOLD_RUNTIME_ENDPOINT", None)
    env["CHROMOFOLD_RUNTIME_ENDPOINT"] = "http://127.0.0.1:9"  # nothing listens here
    for name in sorted(adapter_scenarios()):
        hook = HOOKS / name
        assert hook.is_file() and os.access(hook, os.X_OK), f"{name} missing or not executable"
        proc = subprocess.run([str(hook)], input=json.dumps({"name": name, "mode": "smoke"}),
                              text=True, capture_output=True, env=env, timeout=60)
        result = json.loads(proc.stdout)
        assert isinstance(result, dict), f"{name} did not emit a JSON object"
        assert REQUIRED <= set(result), f"{name} missing keys: {REQUIRED - set(result)}"
        assert result["status"] in {"PASS", "INCOMPLETE", "FAIL"}, f"{name} illegal status {result['status']}"
        assert result["status"] != "PASS", f"{name} fabricated a PASS with no serving runtime"
        if result["status"] == "PASS":
            assert proc.returncode == 0, f"{name} PASS must exit 0"


if __name__ == "__main__":
    test_hook_set_matches_scenarios()
    test_hooks_are_executable_and_contract_valid()
    print("RC1 qualification hooks tests: PASS")
