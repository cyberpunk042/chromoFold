#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qualification_adapter", ROOT / "tools/chromofold_qualification_adapter.py")
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def make_state(root: Path, *, enabled: bool = True, hooks: bool = False):
    snapshot = root / "snapshot.json"
    snapshot.write_text(json.dumps({
        "workers": [{"id": "decode-0", "role": "decode", "healthy": True}],
        "audit_chain_verified": True,
        "mtls_verified": True,
        "telemetry_correlated": True,
    }), encoding="utf-8")
    hooks_dir = root / "hooks" if hooks else None
    if hooks_dir: hooks_dir.mkdir()
    return adapter.State(enabled, "admin-token", "sha256:abc", snapshot, hooks_dir)


def test_snapshot_is_observed_and_pinned() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = make_state(Path(temp))
        snap = state.snapshot()
        assert snap["qualification_mode"] is True
        assert snap["release_digest"] == "sha256:abc"
        assert snap["workers"][0]["role"] == "decode"
        assert snap["audit_chain_verified"] is True


def test_controls_disabled_by_default() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = make_state(Path(temp), enabled=False)
        assert state.snapshot()["qualification_mode"] is False


def test_missing_hook_is_incomplete() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = make_state(Path(temp))
        result = state.run_scenario("decode-worker-failure", {})
        assert result["status"] == "INCOMPLETE"
        assert result["references_reconciled"] is False


def test_real_hook_result_is_preserved() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        state = make_state(root, hooks=True)
        hook = state.hooks_dir / "cache-cold"
        hook.write_text("#!/bin/sh\nprintf '%s' '{\"status\":\"PASS\",\"requests_started\":4,\"requests_completed\":4,\"requests_failed\":0,\"references_reconciled\":true}'\n", encoding="utf-8")
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
        result = state.run_scenario("cache-cold", {})
        assert result["status"] == "PASS"
        assert result["requests_completed"] == 4
        assert result["references_reconciled"] is True


def test_unsupported_scenario_rejected() -> None:
    with tempfile.TemporaryDirectory() as temp:
        state = make_state(Path(temp))
        try:
            state.run_scenario("fabricated-success", {})
        except ValueError as exc:
            assert "unsupported scenario" in str(exc)
        else:
            raise AssertionError("unsupported scenario accepted")


if __name__ == "__main__":
    test_snapshot_is_observed_and_pinned()
    test_controls_disabled_by_default()
    test_missing_hook_is_incomplete()
    test_real_hook_result_is_preserved()
    test_unsupported_scenario_rejected()
    print("RC1 qualification adapter tests: PASS")
