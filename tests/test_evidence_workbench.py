#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "tools/evidence_workbench.py"
    spec = importlib.util.spec_from_file_location("evidence_workbench", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_passes_portable_gates() -> None:
    module = load_module()
    baseline = module.load_result(ROOT / "examples/evidence/baseline.json")
    candidate = module.load_result(ROOT / "examples/evidence/candidate.json")
    result = module.analyze(baseline, candidate)
    assert result["state"] == "PASS"
    assert result["deltas"]["peak_vram_pct"] < 0
    assert result["deltas"]["capacity_pct"] > 0
    assert result["comparison_digest"].startswith("sha256:")
    assert "not maintainer qualification" in result["boundary"]


def test_mismatched_environment_fails_closed() -> None:
    module = load_module()
    baseline = json.loads((ROOT / "examples/evidence/baseline.json").read_text())
    candidate = json.loads((ROOT / "examples/evidence/candidate.json").read_text())
    candidate["fingerprint"]["hardware"] = "Different GPU"
    result = module.analyze(baseline, candidate)
    assert result["state"] == "FAIL"
    assert result["mismatched_fingerprint_fields"] == ["hardware"]


def test_missing_digest_is_incomplete() -> None:
    module = load_module()
    baseline = json.loads((ROOT / "examples/evidence/baseline.json").read_text())
    candidate = json.loads((ROOT / "examples/evidence/candidate.json").read_text())
    candidate["fingerprint"]["release_digest"] = None
    assert module.analyze(baseline, candidate)["state"] == "INCOMPLETE"


def test_browser_contract_is_local_and_downloadable() -> None:
    html = (ROOT / "site/workbench.html").read_text()
    js = (ROOT / "site/workbench.js").read_text()
    assert "never uploaded" in html
    assert "chromofold.evidence-result.v1" in js
    assert "download-analysis" in js and "download-report" in js
    assert "fetch(" not in js and "XMLHttpRequest" not in js


if __name__ == "__main__":
    test_example_passes_portable_gates()
    test_mismatched_environment_fails_closed()
    test_missing_digest_is_incomplete()
    test_browser_contract_is_local_and_downloadable()
    print("ChromoFold evidence workbench tests: PASS")
