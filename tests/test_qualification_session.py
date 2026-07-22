#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "tools/qualification_session.py"
    spec = importlib.util.spec_from_file_location("qualification_session", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture():
    return json.loads((ROOT / "examples/evidence/session.json").read_text(encoding="utf-8"))


def test_example_passes_and_exports_package() -> None:
    module = load_module()
    session = fixture()
    result = module.analyze(session)
    assert result["state"] == "PASS", result
    assert result["baseline"]["run_count"] == 3
    assert result["candidate"]["run_count"] == 3
    assert result["median_deltas_pct"]["peak_vram_mib"] < 0
    assert result["analysis_digest"].startswith("sha256:")
    with tempfile.TemporaryDirectory() as temp:
        package = Path(temp) / "support.zip"
        module.write_package(package, session, result)
        with zipfile.ZipFile(package) as archive:
            assert set(archive.namelist()) == {"session.json", "analysis.json", "REPORT.md", "manifest.json"}


def test_noise_is_incomplete() -> None:
    module = load_module()
    session = fixture()
    session["candidate_runs"][2]["metrics"]["throughput_tokens_s"] = 20
    result = module.analyze(session)
    assert result["state"] == "INCOMPLETE", result
    assert "throughput_tokens_s" in result["unstable_metrics"]


def test_material_regression_fails() -> None:
    module = load_module()
    session = fixture()
    for run in session["candidate_runs"]:
        run["metrics"]["latency_p95_ms"] = 40
    result = module.analyze(session)
    assert result["state"] == "FAIL", result
    assert "latency_p95_ms" in result["regressions"]


def test_browser_remains_local() -> None:
    source = (ROOT / "site/sessions.js").read_text(encoding="utf-8")
    assert "localStorage" in source
    assert "File.text" not in source or ".text()" in source
    assert "fetch(" not in source
    assert "XMLHttpRequest" not in source
    assert "sendBeacon" not in source


if __name__ == "__main__":
    test_example_passes_and_exports_package()
    test_noise_is_incomplete()
    test_material_regression_fails()
    test_browser_remains_local()
    print("ChromoFold qualification session tests: PASS")
