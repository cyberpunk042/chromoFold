#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "tools" / "kv_crossover_campaign.py"
    spec = importlib.util.spec_from_file_location("kv_crossover_campaign", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_performance_contract_is_honest() -> None:
    data = json.loads((ROOT / "product/native-kv-performance.json").read_text(encoding="utf-8"))
    assert data["schema"] == "chromofold.native-kv-performance.v1"
    assert data["headline"]["speedup_over_original"] == 122
    assert data["headline"]["latency_win"] is False
    assert data["headline"]["ship_criterion_met"] is False
    assert data["history"][-1]["gap_vs_dense"] > 1.0
    assert "engine-level" in " ".join(data["limitations"]).lower()


def test_campaign_plan_is_deterministic() -> None:
    module = load_module()
    config = json.loads((ROOT / "product/kv-crossover-campaign.json").read_text(encoding="utf-8"))
    cases = module.campaign_cases(config)
    expected = 1
    for values in config["dimensions"].values():
        expected *= len(values)
    assert len(cases) == expected
    assert cases[0] == {"context_tokens": 512, "head_dim": 64, "query_count": 1, "kv_heads": 2, "code_bits": 4}


def sample_case(compressed: list[float], dense: list[float]) -> dict:
    return {
        "context_tokens": 65536,
        "head_dim": 256,
        "query_count": 32,
        "kv_heads": 8,
        "code_bits": 4,
        "compressed_latency_us": compressed,
        "fair_dense_latency_us": dense,
        "compressed_peak_vram_mib": 4000,
        "dense_peak_vram_mib": 8000,
        "correctness_max_error": 0.000001,
    }


def test_evaluator_distinguishes_win_parity_and_no_crossover() -> None:
    module = load_module()
    config = json.loads((ROOT / "product/kv-crossover-campaign.json").read_text(encoding="utf-8"))
    win = module.evaluate_case(sample_case([970, 975, 980, 972, 978], [1000, 1004, 998, 1001, 1002]), 5.0)
    parity = module.evaluate_case(sample_case([1005, 1008, 1002, 1006, 1004], [1000, 1004, 998, 1001, 1002]), 5.0)
    loss = module.evaluate_case(sample_case([1300, 1310, 1295, 1305, 1302], [1000, 1004, 998, 1001, 1002]), 5.0)
    assert win["state"] == "WIN"
    assert parity["state"] == "PARITY"
    assert loss["state"] == "NO_CROSSOVER"
    payload = {"schema": "chromofold.kv-crossover-results.v1", "cases": [sample_case([970, 975, 980], [1000, 1004, 998])]}
    assert module.evaluate(payload, config)["state"] == "WIN"


def test_noisy_case_is_incomplete() -> None:
    module = load_module()
    result = module.evaluate_case(sample_case([700, 1000, 1400], [1000, 1002, 998]), 5.0)
    assert result["state"] == "INCOMPLETE"


if __name__ == "__main__":
    test_native_performance_contract_is_honest()
    test_campaign_plan_is_deterministic()
    test_evaluator_distinguishes_win_parity_and_no_crossover()
    test_noisy_case_is_incomplete()
    print("ChromoFold KV crossover campaign tests: PASS")
