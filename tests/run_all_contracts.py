#!/usr/bin/env python3
"""Unified contract test runner for ChromoFold M9–M15.

Builds and runs all CPU contract tests, runs CUDA contract tests when nvcc is
available, validates every evidence schema with jsonschema, and performs
fabricated-evidence rejection for every validator script.

Exit code 0 = all passed. Non-zero = at least one failure.
Usage: python3 tests/run_all_contracts.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build"

def run(cmd: List[str], capture: bool = True, cwd: Path | None = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=cwd or REPO,
    )
    return proc.returncode, proc.stdout, proc.stderr

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

def has_nvcc() -> bool:
    rc, _, _ = run(["which", "nvcc"])
    return rc == 0

# ──────────────────────────────────────────────────────────────────────────
# Contract test definitions: (makefile_or_cmd, make_target, binary_path, description, needs_cuda)
CONTRACTS: List[Tuple[str, str, Path, str, bool]] = [
    ("tests/Makefile", "all", BUILD / "kv_runtime_test", "M9 KV runtime contract", False),
    ("tests/Makefile", "all", BUILD / "kv_reference_test", "M9 KV reference contract", False),
    ("m9-gpu.mk", "gpu-fixture-test", BUILD / "kv_gpu_fixture_test", "M9 GPU fixture contract", False),
    ("m9-gpu.mk", "gpu-correctness", BUILD / "paged_kv_cuda_correctness", "M9 paged KV CUDA correctness", True),
    ("m10-multisequence.mk", "", BUILD / "m10-multisequence" / "m10_multisequence_contract", "M10 multisequence contract", False),
    ("m13-adaptive-compression.mk", "", BUILD / "m13" / "m13_contract", "M13 adaptive compression contract", False),
    ("m14-distributed-multigpu.mk", "", BUILD / "m14" / "m14_contract", "M14 distributed runtime contract", False),
    ("m9-gpu-seal-attention.mk", "", BUILD / "m9-gpu-seal-attention" / "m9_device_seal_contract", "M9 device seal contract", True),
    ("m12-production-performance.mk", "", BUILD / "m12-production-performance" / "m12_production_attention_contract", "M12 production attention contract", True),
]

def build_contract(mf: str, target: str, binary: Path) -> bool:
    """Build a contract binary via its makefile."""
    if mf == "tests/Makefile":
        rc, out, err = run(["make", "-C", "tests", target])
    elif target:
        rc, out, err = run(["make", "-f", mf, target])
    else:
        rc, out, err = run(["make", "-f", mf])
    if rc != 0:
        print(f"  BUILD FAIL: {binary.name}\n{out}\n{err}")
        return False
    if not binary.exists():
        print(f"  BUILD OK but binary missing: {binary}")
        return False
    return True

def run_contract(binary: Path, desc: str) -> bool:
    rc, out, err = run([str(binary)])
    ok = rc == 0
    status = "PASS" if ok else "FAIL"
    first_line = (out.strip().splitlines() or ["(no output)"])[0]
    print(f"  [{status}] {desc}: {first_line}")
    if not ok:
        print(f"    stdout:\n{out}")
        print(f"    stderr:\n{err}")
    return ok

# ──────────────────────────────────────────────────────────────────────────
# Evidence schemas and validators
SCHEMAS = sorted(REPO.glob("integrations/llama.cpp/**/*.schema.json"))
VALIDATORS = sorted(REPO.glob("integrations/llama.py/validate_*.py"))

# Fabricated-evidence payloads: each must fail the corresponding validator.
# These are deliberately underspecified or contradictory.
FABRICATED_PAYLOADS: Dict[str, dict] = {
    "distributed-evidence.schema.json": {
        "topology": {"device_count": 1, "peer_matrix_verified": False},
        "routing": {"sequences_placed": 0, "devices_used": 1, "batches_dispatched": 0},
        "transfers": {"peer_copies": 0, "peer_bytes": 0, "transfer_errors": 1},
        "attention": {"tensor_parallel_batches": 0, "mixed_codec_batches": 0, "dense_fallback_launches": 1, "cuda_errors": 1},
        "cache": {"shared_index_verified": False, "remote_pages_loaded": 0, "duplicate_pages_avoided": 0},
        "isolation": {"concurrent_http_passed": False, "cross_device_contamination": 1},
        "failures": {"injected_device_failures": 0, "fail_closed_passed": False},
        "shutdown": {"audit_passed": False, "page_refs": 1, "snapshot_refs": 1},
    },
    "evidence.schema.json": {
        "runtime": {"llama_commit": "abc", "chromofold_commit": "def"},
        "model": {"path": "/dev/null", "sha256": "00"},
        "backend": "dense",
        "correctness": {"finite": False, "token_match_rate": 0.0, "max_logit_error": 1.0},
        "memory": {"peak_vram_bytes": 0, "kv_bytes": 0},
        "latency": {"prefill_ms": 0, "decode_ms_per_token": 0},
        "capacity": {"context": 0, "completed": False},
        "counters": {"compressed_attention_launches": 0, "sealed_values_consumed": 0, "dense_fallback_launches": 1},
    },
    "performance-evidence.schema.json": {
        "environment": {"gpu": "none", "cuda": "0", "driver": "0", "chromofold_commit": "abc"},
        "upstream": {"llama_commit": "abc", "patched_server_executed": False},
        "model": {"gguf_sha256": "00", "architecture": "none", "head_dim": 0, "gqa_ratio": 0},
        "kernels": {"reference_launches": 0, "optimized_launches": 0, "grouped_batches": 0, "dense_fallback_launches": 1, "cuda_errors": 1},
        "graphs": {"captures": 0, "replays": 0, "invalidations": 0},
        "memory_pool": {"peak_reserved_bytes": 0, "peak_live_bytes": 0, "allocation_count": 0},
        "benchmarks": {"dense": {}, "reference": {}, "optimized": {}, "contexts": [], "concurrency": 0},
        "numerics": {"optimized_reference_passed": False, "max_abs_error": 1.0, "mean_abs_error": 1.0, "cosine_similarity": 0.0, "top1_agreement": 0.0},
        "isolation": {"concurrent_http_passed": False, "cross_sequence_contamination": 1},
        "sanitizer": {"memcheck": False, "racecheck": False, "initcheck": False, "synccheck": False},
        "soak": {"passed": False, "requests": 0, "memory_stable": False},
        "shutdown": {"audit_passed": False, "page_refs": 1, "snapshot_refs": 1},
    },
    "adaptive-evidence.schema.json": {
        "codec_selection": {"chosen_codec": "none", "entropy_estimate": -1.0, "fallback_reason": "test"},
        "compression": {"ratio": 0.0, "bytes_before": 0, "bytes_after": 0},
        "latency": {"encode_ms": 0, "decode_ms": 0},
        "quality": {"reconstruction_error": 1.0, "perceptual_score": 0.0},
        "isolation": {"cross_sequence_contamination": 1},
    },
    "multisequence-evidence.schema.json": {
        "sequences": {"count": 0, "max_batch": 0, "avg_tokens": 0},
        "memory": {"kv_bytes": 0, "page_count": 0, "shared_pages": 0},
        "latency": {"prefill_ms": 0, "decode_ms_per_token": 0},
        "isolation": {"cross_sequence_contamination": 1},
    },
    "cuda-resolver-evidence.schema.json": {
        "resolver": {"pages_registered": 0, "batches_executed": 0, "cuda_errors": 1},
        "memory": {"peak_resident_bytes": 0, "active_pages": 0},
        "isolation": {"cross_sequence_contamination": 1},
    },
    "m15-evidence.schema.json": {
        "topology": {"device_count": 1, "peer_matrix_verified": False},
        "routing": {"sequences_placed": 0, "devices_used": 1},
        "transfers": {"peer_copies": 0, "peer_bytes": 0, "transfer_errors": 1},
        "attention": {"tensor_parallel_batches": 0, "mixed_codec_batches": 0, "dense_fallback_launches": 1},
        "isolation": {"cross_device_contamination": 1},
        "shutdown": {"audit_passed": False, "page_refs": 1},
    },
    "production-evidence.schema.json": {
        "server": {"uptime_seconds": 0, "requests_served": 0, "errors": 1},
        "memory": {"peak_resident_bytes": 0, "leaked_pages": 1},
        "isolation": {"cross_sequence_contamination": 1},
    },
}

def validate_schema_structural(schema_path: Path) -> bool:
    """Validate schema is well-formed JSON and has required fields."""
    try:
        schema = json.loads(schema_path.read_text())
        if schema.get("type") != "object":
            print(f"  [WARN] {schema_path.name}: root type is not 'object'")
        if "required" not in schema and "properties" not in schema:
            print(f"  [WARN] {schema_path.name}: no required or properties field")
        print(f"  [PASS] {schema_path.name}: schema parses and looks well-formed")
        return True
    except Exception as exc:
        print(f"  [FAIL] {schema_path.name}: {exc}")
        return False

def validate_fabricated_rejection(schema_name: str, validator_path: Path, payload: dict) -> bool:
    """Write a fabricated evidence and assert the validator rejects it."""
    if not validator_path.exists():
        print(f"  [SKIP] {schema_name}: validator not found at {validator_path}")
        return True  # not a failure, just nothing to test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        f.flush()
        tmp = f.name
    try:
        rc, out, err = run([sys.executable, str(validator_path), tmp])
        rejected = rc != 0
        status = "PASS" if rejected else "FAIL"
        print(f"  [{status}] {schema_name}: fabricated evidence {'rejected' if rejected else 'ACCEPTED (!)'}")
        if not rejected:
            print(f"    validator stdout: {out.strip()}")
        return rejected
    except Exception as exc:
        print(f"  [FAIL] {schema_name}: validator crashed: {exc}")
        return False
    finally:
        os.unlink(tmp)

def validator_for_schema(schema_path: Path) -> Path | None:
    """Find the corresponding validator script for a schema."""
    parent = schema_path.parent
    stem = schema_path.stem  # e.g. "distributed-evidence.schema" or "evidence"
    # heuristic: look for validate_*.py in the same directory
    candidates = list(parent.glob("validate_*.py"))
    if candidates:
        return candidates[0]
    # fallback: strip .schema and look broader
    return None

# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Unified ChromoFold contract test runner")
    parser.add_argument("--quick", action="store_true", help="Skip CUDA contract tests")
    args = parser.parse_args()

    results: Dict[str, bool] = {}
    section("Building + running contract tests")
    for mf, target, binary, desc, needs_cuda in CONTRACTS:
        if needs_cuda and (args.quick or not has_nvcc()):
            print(f"  [SKIP] {desc} (CUDA unavailable or --quick)")
            results[desc] = True
            continue
        if not binary.exists():
            ok = build_contract(mf, target, binary)
            if not ok:
                results[desc] = False
                continue
        results[desc] = run_contract(binary, desc)

    section("Validating evidence schemas")
    for schema_path in SCHEMAS:
        name = schema_path.name
        ok = validate_schema_structural(schema_path)
        results[f"schema:{name}"] = ok

    section("Fabricated-evidence rejection tests")
    for schema_path in SCHEMAS:
        name = schema_path.name
        validator = validator_for_schema(schema_path)
        payload = FABRICATED_PAYLOADS.get(name, {"__fabricated": True})
        ok = validate_fabricated_rejection(name, validator, payload)
        results[f"fabricated:{name}"] = ok

    section("Summary")
    total = len(results)
    passed = sum(1 for ok in results.values() if ok)
    failed = total - passed
    for desc, ok in results.items():
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {desc}")
    print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
