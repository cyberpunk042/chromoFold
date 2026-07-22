#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("chromofold_product", ROOT / "tools/chromofold.py")
assert SPEC and SPEC.loader
product = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = product
SPEC.loader.exec_module(product)


def machine(vram: int = 24 * 1024**3):
    return product.Machine("Linux", "x86_64", "cpu", 64 * 1024**3, "RTX", vram, "driver", "CUDA", False)


def test_recommendation_is_honest() -> None:
    result = product.recommend(machine(), "longer-context", 18 * 1024**3, 65536, 2)
    assert result["profile"] == "maximum-context"
    assert result["qualification_required"] is True
    assert "estimate" in result["estimate_notice"].lower()


def test_low_pressure_prefers_safe() -> None:
    result = product.recommend(machine(80 * 1024**3), "balanced", 4 * 1024**3, 4096, 1)
    assert result["profile"] == "safe"


def test_bundle_disables_silent_fallback() -> None:
    with tempfile.TemporaryDirectory() as temp:
        paths = product.render_config("balanced", "llama.cpp", None, Path(temp))
        config = json.loads(paths[0].read_text(encoding="utf-8"))
        assert config["fallback"] == {"silent": False, "unsupported": "fail"}
        assert config["qualification"]["required"] is True
        assert paths[1].stat().st_mode & 0o111


def test_compare_reports_workload_win() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        baseline = root / "baseline.json"
        candidate = root / "candidate.json"
        baseline.write_text(json.dumps({"max_context": 32768, "peak_vram_bytes": 20}), encoding="utf-8")
        candidate.write_text(json.dumps({"max_context": 65536, "peak_vram_bytes": 15}), encoding="utf-8")
        result = product.compare_results(baseline, candidate)
        assert result["decision"] == "WORKLOAD_WIN"
        assert result["max_context"]["delta_percent"] == 100.0


def test_catalogs_are_versioned() -> None:
    assert product.load_json(product.PROFILES)["schema"] == "chromofold.profiles.v1"
    assert product.load_json(product.COMPATIBILITY)["schema"] == "chromofold.compatibility.v1"


if __name__ == "__main__":
    test_recommendation_is_honest()
    test_low_pressure_prefers_safe()
    test_bundle_disables_silent_fallback()
    test_compare_reports_workload_win()
    test_catalogs_are_versioned()
    print("ChromoFold product tests: PASS")
