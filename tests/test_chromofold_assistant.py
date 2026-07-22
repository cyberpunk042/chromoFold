#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_inputs_are_requested() -> None:
    assistant = load("chromofold_assistant", ROOT / "tools/chromofold_assistant.py")
    result = assistant.build_plan({"intent": "longer-context"})
    assert result["state"] == "NEEDS_INPUT"
    assert result["missing"] == ["model", "context", "concurrency"]


def test_explanation_has_evidence_boundary() -> None:
    assistant = load("chromofold_assistant_explain", ROOT / "tools/chromofold_assistant.py")
    result = assistant.build_plan({"intent": "explain"})
    assert result["state"] == "EXPLAINED"
    assert "qualified" in result["evidence_boundary"]


def test_ready_plan_requires_qualification() -> None:
    assistant = load("chromofold_assistant_ready", ROOT / "tools/chromofold_assistant.py")
    result = assistant.build_plan({
        "intent": "longer-context", "model": "model.gguf", "context": 65536, "concurrency": 4,
    })
    assert result["state"] == "READY"
    assert result["qualification_required"] is True
    assert any("qualification" in step.lower() for step in result["plan"])


def test_installers_are_local_only() -> None:
    shell = (ROOT / "install/install.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "install/install.ps1").read_text(encoding="utf-8")
    for source in (shell, powershell):
        assert "curl " not in source.lower()
        assert "wget " not in source.lower()
    assert "CHROMOFOLD_HOME" in shell
    assert "CHROMOFOLD_HOME" in powershell


def test_bundle_contains_assistant_and_installers() -> None:
    builder = load("assistant_bundle", ROOT / "tools/build_product_bundle.py")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "chromofold-assistant.tar.gz"
        builder.build(output, "assistant-test")
        with tarfile.open(output, "r:gz") as archive:
            names = set(archive.getnames())
        prefix = "chromofold-assistant-test/"
        assert prefix + "tools/chromofold_assistant.py" in names
        assert prefix + "install/install.sh" in names
        assert prefix + "install/install.ps1" in names
        assert prefix + "product/assistant-intents.json" in names


def test_intent_catalog_is_versioned() -> None:
    catalog = json.loads((ROOT / "product/assistant-intents.json").read_text(encoding="utf-8"))
    assert catalog["schema"] == "chromofold.assistant-intents.v1"
    assert "fit-model" in catalog["intents"]


if __name__ == "__main__":
    test_missing_inputs_are_requested()
    test_explanation_has_evidence_boundary()
    test_ready_plan_requires_qualification()
    test_installers_are_local_only()
    test_bundle_contains_assistant_and_installers()
    test_intent_catalog_is_versioned()
    print("ChromoFold guided assistant tests: PASS")
