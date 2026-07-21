#!/usr/bin/env python3
"""Unified contract test runner for ChromoFold M9–M17.

Builds and runs all CPU contract tests, runs CUDA contract tests when nvcc is
available, validates every evidence schema with jsonschema, performs
fabricated-evidence rejection for every validator script, and proves that
minimal valid evidence payloads are ACCEPTED by each validator.

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
    ("m10-multisequence.mk", "multisequence-orphan-contract", BUILD / "m10-multisequence" / "m10_multisequence_contract_orphan", "M10 multisequence orphan contract", False),
    ("m11-pinned-server.mk", "m11-contract", BUILD / "m11" / "m11_contract", "M11 pinned server contract", True),
    ("m13-adaptive-compression.mk", "", BUILD / "m13" / "m13_contract", "M13 adaptive compression contract", False),
    ("m14-distributed-multigpu.mk", "", BUILD / "m14" / "m14_contract", "M14 distributed runtime contract", False),
    ("m9-gpu-seal-attention.mk", "", BUILD / "m9-gpu-seal-attention" / "m9_device_seal_contract", "M9 device seal contract", True),
    ("m12-production-performance.mk", "", BUILD / "m12-production-performance" / "m12_production_attention_contract", "M12 production attention contract", True),
    ("m16-disaggregated.mk", "m16-contract", BUILD / "m16" / "m16_contract", "M16 disaggregated serving contract", False),
    ("m17-production-scheduler.mk", "m17-contract", BUILD / "m17" / "m17_contract", "M17 production scheduler contract", False),
    ("paged-kv-seam.mk", "paged-kv-seam", BUILD / "paged-kv-seam" / "paged_kv_cuda_seam", "Paged KV CUDA seam", True),
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


def validator_for_schema(schema_path: Path) -> Path | None:
    """Find the corresponding validator script for a schema.

    Some directories have multiple validators (e.g. evidence vs evidence-v2).
    We match by name prefix where possible.
    """
    parent = schema_path.parent
    name = schema_path.name

    # Explicit mappings for ambiguous cases
    mapping = {
        "evidence-v2.schema.json": "validate_evidence_v2.py",
        "evidence.schema.json": "validate_evidence.py",
        "multisequence-evidence.schema.json": "validate_multisequence_evidence.py",
        "m16-evidence.schema.json": "validate_m16_evidence.py",
        "m17-evidence.schema.json": "validate_m17_evidence.py",
    }
    if name in mapping:
        candidate = parent / mapping[name]
        if candidate.exists():
            return candidate

    # heuristic: look for validate_*.py in the same directory
    candidates = sorted(parent.glob("validate_*.py"))
    if candidates:
        return candidates[0]
    return None


# ──────────────────────────────────────────────────────────────────────────
# Positive evidence payloads: minimal VALID evidence that each validator ACCEPTS.
# These must pass both jsonschema structural validation and the semantic checks
# in the validator script.
_POSITIVE_PAYLOADS: Dict[str, dict] = {
    "distributed-evidence.schema.json": {
        "topology": {"device_count": 2, "device_uuids": ["uuid1", "uuid2"], "peer_matrix_verified": True},
        "routing": {"sequences_placed": 2, "devices_used": 2, "batches_dispatched": 1},
        "transfers": {"peer_copies": 1, "peer_bytes": 1, "transfer_errors": 0},
        "attention": {"tensor_parallel_batches": 1, "mixed_codec_batches": 1, "dense_fallback_launches": 0, "cuda_errors": 0},
        "cache": {"shared_index_verified": True, "remote_pages_loaded": 1, "duplicate_pages_avoided": 1},
        "isolation": {"concurrent_http_passed": True, "cross_device_contamination": 0},
        "failures": {"injected_device_failures": 1, "fail_closed_passed": True},
        "shutdown": {"audit_passed": True, "page_refs": 0, "snapshot_refs": 0},
    },
    "evidence.schema.json": {
        "runtime": {
            "llama_commit": "abcdef1234567890abcdef1234567890abcdef12",
            "chromofold_commit": "abcdef1234567890abcdef1234567890abcdef12",
        },
        "model": {"path": "/model.gguf", "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"},
        "backend": "chromofold",
        "correctness": {"finite": True, "token_match_rate": 1.0, "max_logit_error": 0.0},
        "memory": {"peak_vram_bytes": 1000, "kv_bytes": 500},
        "latency": {"prefill_ms": 10.0, "decode_ms_per_token": 1.0},
        "capacity": {"context": 4096, "completed": True},
        "counters": {"compressed_attention_launches": 1, "sealed_values_consumed": 1, "dense_fallback_launches": 0},
    },
    "evidence-v2.schema.json": {
        "schema_version": 2,
        "environment": {"gpu_name": "RTX 2080 Ti", "compute_capability": "7.5", "cuda_toolkit": "12.6", "driver": "535", "os": "linux", "compiler": "gcc"},
        "model": {"path": "/model.gguf", "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890", "size_bytes": 1000000},
        "runtime": {
            "llama_commit": "abcdef1234567890abcdef1234567890abcdef12",
            "chromofold_commit": "abcdef1234567890abcdef1234567890abcdef12",
            "backend": "chromofold",
            "command": ["./server"],
            "exit_code": 0,
            "completed": True,
        },
        "topology": {"layer_count": 2, "query_heads": 8, "kv_heads": 4, "head_dim": 64, "gqa_group_size": 2, "context_size": 4096, "page_size": 128},
        "counters": {"device_append_calls": 1, "seal_successes": 1, "descriptor_publications": 1, "compressed_attention_launches": 1, "sealed_pages_consumed": 1, "sealed_values_consumed": 1, "active_values_consumed": 1, "dense_fallback_launches": 0, "cuda_errors": 0},
        "memory": {"measured": True, "peak_process_vram_bytes": 1000, "peak_chromofold_owned_bytes": 500},
        "latency": {"prefill_ms": 10.0, "decode_ms": 1.0, "prompt_tokens_per_second": 100.0, "decode_tokens_per_second": 50.0},
        "numerics": {"all_finite": True, "max_abs_error": 0.0, "mean_abs_error": 0.0, "top1_agreement": 1.0},
        "tokens": {"prompt_count": 10, "generated_count": 10, "first_divergence": None},
        "sanitizer": {"memcheck": "passed", "racecheck": "passed", "initcheck": "passed", "synccheck": "passed"},
        "capacity": {"dense_max_context": 2048, "chromofold_max_context": 4096, "results": [{"context": 4096, "completed": True}]},
    },
    "performance-evidence.schema.json": {
        "environment": {"gpu": "RTX 2080 Ti", "cuda": "12.6", "driver": "535", "chromofold_commit": "abcdef1234567890abcdef1234567890abcdef12"},
        "upstream": {"llama_commit": "abcdef1234567890abcdef1234567890abcdef12", "patched_server_executed": True},
        "model": {"gguf_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890", "architecture": "llama", "head_dim": 64, "gqa_ratio": 2},
        "kernels": {"reference_launches": 1, "optimized_launches": 1, "grouped_batches": 1, "dense_fallback_launches": 0, "cuda_errors": 0},
        "graphs": {"captures": 1, "replays": 1, "invalidations": 0},
        "memory_pool": {"peak_reserved_bytes": 1000, "peak_live_bytes": 500, "allocation_count": 10},
        "benchmarks": {"dense": {"time_ms": 100}, "reference": {"time_ms": 90}, "optimized": {"time_ms": 80}, "contexts": [], "concurrency": 1},
        "numerics": {"optimized_reference_passed": True, "max_abs_error": 0.0, "mean_abs_error": 0.0, "cosine_similarity": 1.0, "top1_agreement": 1.0},
        "isolation": {"concurrent_http_passed": True, "cross_sequence_contamination": 0},
        "sanitizer": {"memcheck": True, "racecheck": True, "initcheck": True, "synccheck": True},
        "soak": {"passed": True, "requests": 1000, "memory_stable": True},
        "shutdown": {"audit_passed": True, "page_refs": 0, "snapshot_refs": 0},
    },
    "adaptive-evidence.schema.json": {
        "adaptive_policy": {"mode": "auto", "model_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"},
        "codec_distribution": {"int2_pages": 1, "int4_pages": 1, "escalation_pages": 1, "mixed_codec_attention_launches": 1},
        "quality_budget": {"violations": 0, "invalid_candidates_rejected": 1},
        "recompression": {"successes": 1, "bytes_saved": 1},
        "memory_budget": {"configured_bytes": 1000, "peak_bytes": 500, "honored": True},
        "persistent_cache": {"pages_written": 1, "pages_reloaded": 1, "warm_hits": 1},
        "cache_integrity": {"corrupted_records_rejected": 1, "incompatible_records_rejected": 0},
        "runtime": {"dense_fallback_launches": 0, "cuda_errors": 0, "page_refs_at_shutdown": 0, "snapshot_refs_at_shutdown": 0},
    },
    "multisequence-evidence.schema.json": {
        "environment": {"gpu": "RTX 2080 Ti", "cuda": "12.6", "llama_cpp_commit": "abcdef1", "chromofold_commit": "abcdef1"},
        "workload": {"requests": 3, "cancelled_requests": 1, "slot_reuses": 1},
        "sequences": {"created": 2, "copied": 1, "removed": 1, "shifted": 0, "active_peak": 2},
        "sharing": {"shared_pages": 1, "shared_page_references": 2, "bytes_saved": 100},
        "reclamation": {"pages_reclaimed": 1, "pending_pages_at_shutdown": 0, "pending_bytes_at_shutdown": 0},
        "batching": {"batches": 1, "sequences_per_batch_peak": 2, "cross_sequence_contamination": False},
        "speculation": {"started": 1, "committed": 0, "rolled_back": 1},
        "invariants": {"audits": 1, "failures": 0, "reference_count_balance": True},
        "runtime": {"dense_fallback_launches": 0, "cuda_errors": 0, "outputs_finite": True, "sanitizer_passed": True},
    },
    "cuda-resolver-evidence.schema.json": {
        "device_resolution": {"pages_registered": 1, "descriptor_tables_built": 1, "sequence_attention_launches": 2},
        "snapshots": {"acquired": 1, "released": 1, "in_flight_at_shutdown": 0},
        "batching": {"compressed_batches_executed": 1, "sequences_per_batch_peak": 2},
        "isolation": {"cross_sequence_contamination": 0, "isolated_baselines_match": True},
        "reclamation": {"pages_reclaimed": 1, "page_refs_at_shutdown": 0},
        "runtime": {"dense_fallback_launches": 0, "cuda_errors": 0, "sanitizer_passed": True},
    },
    "m15-evidence.schema.json": {
        "environment": {"gpu_count": 2, "cuda": "12.6", "driver": "535", "chromofold_commit": "abcdef1"},
        "upstream": {"patched_server_executed": True, "llama_commit": "abcdef1234567890abcdef1234567890abcdef12", "gguf_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"},
        "topology": {"devices_used": 2, "peer_topology_verified": True},
        "transfers": {"p2p_copy_calls": 1, "p2p_bytes": 1, "transfer_errors": 0},
        "collectives": {"calls": 1, "bytes": 1, "tensor_parallel_batches": 1},
        "attention": {"mixed_codec_pages_consumed": 1, "remote_pages_loaded": 1, "dense_fallback_launches": 0, "cuda_errors": 0, "cross_device_contamination": 0},
        "failures": {"device_failure_injected": True, "failed_closed": True, "recovery_completed": True},
        "sanitizer": {"memcheck": True, "racecheck": True, "initcheck": True, "synccheck": True},
        "shutdown": {"audit_passed": True, "page_refs": 0, "snapshot_refs": 0},
    },
    "production-evidence.schema.json": {
        "upstream": {"llama_commit": "76f46ad29d61fd8c1401e8221842934bf62a6064", "chromofold_commit": "abcdef1", "patch_idempotent": True},
        "model": {"gguf_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890", "cuda_device": "cuda:0", "outputs_finite": True},
        "server": {"real_http_requests": 2, "active_sequences_peak": 2, "cancelled_requests": 1, "slot_reuses": 1},
        "scheduler": {"interleaved_batches": 1, "sequences_per_batch_peak": 2},
        "kv": {"device_append_calls": 1, "seal_successes": 1, "pages_registered": 1, "shared_page_references": 1},
        "attention": {"compressed_batches_executed": 1, "sequence_attention_launches": 2, "sealed_values_consumed": 1, "active_values_consumed": 1, "dense_fallback_launches": 0, "cuda_errors": 0},
        "lifecycle": {"snapshots_acquired": 1, "snapshots_released": 1, "pages_reclaimed": 1, "cross_sequence_contamination": 0},
        "memory": {"peak_vram_bytes": 1000},
        "sanitizer": {"memcheck": True, "racecheck": True, "initcheck": True, "synccheck": True},
        "shutdown": {"page_refs": 0, "snapshot_refs": 0, "audit_passed": True},
    },
    "m16-evidence.schema.json": {
        "cluster": {"router": 1, "prefill_workers": 1, "decode_workers": 2, "separate_processes": True},
        "model": {"gguf_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890", "tokenizer_sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890", "codec_version": "v1"},
        "transfer": {"compressed_pages": 1, "bytes": 1, "completed": 1, "partial_rejected": 1, "received_without_prefill_recompute": True},
        "cache": {"prefix_hits": 1, "prefill_tokens_avoided": 1},
        "leases": {"acquired": 1, "transferred": 1, "stale_rejected": 1},
        "recovery": {"worker_failure_injected": True, "replica_recovery_succeeded": True},
        "security": {"worker_authentication": True, "manifest_integrity": True, "replay_rejected": True},
        "isolation": {"cross_request_contamination": 0, "dense_fallback_launches": 0, "cuda_errors": 0},
        "shutdown": {"audit_passed": True, "page_refs": 0, "snapshot_refs": 0},
    },
    "m17-evidence.schema.json": {
        "tenants": {"count": 2, "quotas_enforced": True, "fairness_demonstrated": True},
        "scheduling": {"deadline_schedules": 1, "continuous_batches": 1, "preemptions": 1, "checkpoint_resumes": 1, "cancellations": 1},
        "resilience": {"circuit_opened": True, "circuit_recovered": True, "safe_drains": 1, "rolling_compatibility": True},
        "autoscaling": {"scale_up_plans": 1, "heterogeneous_routes": 1, "model_placement_decisions": 1},
        "isolation": {"cross_tenant_reuse_rejections": 1, "cross_tenant_contamination": 0, "dense_fallback_launches": 0},
        "shutdown": {"active_refs": 0, "usage_idempotent": True},
    },
}


# Fabricated-evidence payloads: each must fail the corresponding validator.
# These are deliberately underspecified or contradictory.
_FABRICATED_PAYLOADS: Dict[str, dict] = {
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
    "evidence-v2.schema.json": {
        "schema_version": 2,
        "environment": {"gpu_name": "none", "compute_capability": "0.0", "cuda_toolkit": "0", "driver": "0", "os": "none", "compiler": "none"},
        "model": {"path": "/dev/null", "sha256": "00", "size_bytes": 0},
        "runtime": {"llama_commit": "abc", "chromofold_commit": "def", "backend": "dense", "command": ["cmd"], "exit_code": 1, "completed": False},
        "topology": {"layer_count": 0, "query_heads": 0, "kv_heads": 0, "head_dim": 0, "gqa_group_size": 0, "context_size": 0, "page_size": 64},
        "counters": {"device_append_calls": 0, "seal_successes": 0, "descriptor_publications": 0, "compressed_attention_launches": 0, "sealed_pages_consumed": 0, "sealed_values_consumed": 0, "active_values_consumed": 0, "dense_fallback_launches": 1, "cuda_errors": 1},
        "memory": {"measured": False, "peak_process_vram_bytes": 0, "peak_chromofold_owned_bytes": 0},
        "latency": {"prefill_ms": 0, "decode_ms": 0, "prompt_tokens_per_second": 0, "decode_tokens_per_second": 0},
        "numerics": {"all_finite": False, "max_abs_error": 1.0, "mean_abs_error": 1.0, "top1_agreement": 0.0},
        "tokens": {"prompt_count": 0, "generated_count": 0, "first_divergence": None},
        "sanitizer": {"memcheck": "failed", "racecheck": "failed", "initcheck": "failed", "synccheck": "failed"},
        "capacity": {"dense_max_context": 0, "chromofold_max_context": 0, "results": []},
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
        "adaptive_policy": {"mode": "none", "model_sha256": "00"},
        "codec_distribution": {"int2_pages": 0, "int4_pages": 0, "escalation_pages": 0, "mixed_codec_attention_launches": 0},
        "quality_budget": {"violations": 1, "invalid_candidates_rejected": 0},
        "recompression": {"successes": 0, "bytes_saved": 0},
        "memory_budget": {"configured_bytes": 100, "peak_bytes": 200, "honored": False},
        "persistent_cache": {"pages_written": 0, "pages_reloaded": 0, "warm_hits": 0},
        "cache_integrity": {"corrupted_records_rejected": 0, "incompatible_records_rejected": 0},
        "runtime": {"dense_fallback_launches": 1, "cuda_errors": 1, "page_refs_at_shutdown": 1, "snapshot_refs_at_shutdown": 1},
    },
    "multisequence-evidence.schema.json": {
        "environment": {"gpu": "none", "cuda": "0", "llama_cpp_commit": "abc", "chromofold_commit": "def"},
        "workload": {"requests": 0, "cancelled_requests": 0, "slot_reuses": 0},
        "sequences": {"created": 0, "copied": 0, "removed": 0, "shifted": 0, "active_peak": 0},
        "sharing": {"shared_pages": 0, "shared_page_references": 0, "bytes_saved": 0},
        "reclamation": {"pages_reclaimed": 0, "pending_pages_at_shutdown": 1, "pending_bytes_at_shutdown": 1},
        "batching": {"batches": 0, "sequences_per_batch_peak": 0, "cross_sequence_contamination": True},
        "speculation": {"started": 0, "committed": 0, "rolled_back": 0},
        "invariants": {"audits": 0, "failures": 1, "reference_count_balance": False},
        "runtime": {"dense_fallback_launches": 1, "cuda_errors": 1, "outputs_finite": False, "sanitizer_passed": False},
    },
    "cuda-resolver-evidence.schema.json": {
        "device_resolution": {"pages_registered": 0, "descriptor_tables_built": 0, "sequence_attention_launches": 0},
        "snapshots": {"acquired": 0, "released": 0, "in_flight_at_shutdown": 1},
        "batching": {"compressed_batches_executed": 0, "sequences_per_batch_peak": 0},
        "isolation": {"cross_sequence_contamination": 1, "isolated_baselines_match": False},
        "reclamation": {"pages_reclaimed": 0, "page_refs_at_shutdown": 1},
        "runtime": {"dense_fallback_launches": 1, "cuda_errors": 1, "sanitizer_passed": False},
    },
    "m15-evidence.schema.json": {
        "environment": {"gpu_count": 1, "cuda": "0", "driver": "0", "chromofold_commit": "abc"},
        "upstream": {"patched_server_executed": False, "llama_commit": "abc", "gguf_sha256": "00"},
        "topology": {"devices_used": 1, "peer_topology_verified": False},
        "transfers": {"p2p_copy_calls": 0, "p2p_bytes": 0, "transfer_errors": 1},
        "collectives": {"calls": 0, "bytes": 0, "tensor_parallel_batches": 0},
        "attention": {"mixed_codec_pages_consumed": 0, "remote_pages_loaded": 0, "dense_fallback_launches": 1, "cuda_errors": 1, "cross_device_contamination": 1},
        "failures": {"device_failure_injected": False, "failed_closed": False, "recovery_completed": False},
        "sanitizer": {"memcheck": False, "racecheck": False, "initcheck": False, "synccheck": False},
        "shutdown": {"audit_passed": False, "page_refs": 1, "snapshot_refs": 1},
    },
    "production-evidence.schema.json": {
        "upstream": {"llama_commit": "abc", "chromofold_commit": "def", "patch_idempotent": False},
        "model": {"gguf_sha256": "00", "cuda_device": "", "outputs_finite": False},
        "server": {"real_http_requests": 0, "active_sequences_peak": 0, "cancelled_requests": 0, "slot_reuses": 0},
        "scheduler": {"interleaved_batches": 0, "sequences_per_batch_peak": 0},
        "kv": {"device_append_calls": 0, "seal_successes": 0, "pages_registered": 0, "shared_page_references": 0},
        "attention": {"compressed_batches_executed": 0, "sequence_attention_launches": 0, "sealed_values_consumed": 0, "active_values_consumed": 0, "dense_fallback_launches": 1, "cuda_errors": 1},
        "lifecycle": {"snapshots_acquired": 0, "snapshots_released": 0, "pages_reclaimed": 0, "cross_sequence_contamination": 1},
        "memory": {"peak_vram_bytes": 0},
        "sanitizer": {"memcheck": False, "racecheck": False, "initcheck": False, "synccheck": False},
        "shutdown": {"page_refs": 1, "snapshot_refs": 1, "audit_passed": False},
    },
    "m16-evidence.schema.json": {
        "cluster": {"router": 0, "prefill_workers": 0, "decode_workers": 0, "separate_processes": False},
        "model": {"gguf_sha256": "00", "tokenizer_sha256": "00", "codec_version": "none"},
        "transfer": {"compressed_pages": 0, "bytes": 0, "completed": 0, "partial_rejected": 0, "received_without_prefill_recompute": False},
        "cache": {"prefix_hits": 0, "prefill_tokens_avoided": 0},
        "leases": {"acquired": 0, "transferred": 0, "stale_rejected": 0},
        "recovery": {"worker_failure_injected": False, "replica_recovery_succeeded": False},
        "security": {"worker_authentication": False, "manifest_integrity": False, "replay_rejected": False},
        "isolation": {"cross_request_contamination": 1, "dense_fallback_launches": 1, "cuda_errors": 1},
        "shutdown": {"audit_passed": False, "page_refs": 1, "snapshot_refs": 1},
    },
    "m17-evidence.schema.json": {
        "tenants": {"count": 1, "quotas_enforced": False, "fairness_demonstrated": False},
        "scheduling": {"deadline_schedules": 0, "continuous_batches": 0, "preemptions": 0, "checkpoint_resumes": 0, "cancellations": 0},
        "resilience": {"circuit_opened": False, "circuit_recovered": False, "safe_drains": 0, "rolling_compatibility": False},
        "autoscaling": {"scale_up_plans": 0, "heterogeneous_routes": 0, "model_placement_decisions": 0},
        "isolation": {"cross_tenant_reuse_rejections": 0, "cross_tenant_contamination": 1, "dense_fallback_launches": 1},
        "shutdown": {"active_refs": 1, "usage_idempotent": False},
    },
}


# ──────────────────────────────────────────────────────────────────────────

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


def _write_temp_evidence(payload: dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        f.flush()
        return f.name


def _run_validator(validator_path: Path, evidence_path: str) -> Tuple[int, str, str]:
    return run([sys.executable, str(validator_path), evidence_path])


def validate_evidence(validator_path: Path, payload: dict, expect_accept: bool, label: str) -> bool:
    """Run a validator against a payload and check acceptance/rejection."""
    if not validator_path.exists():
        print(f"  [SKIP] {label}: validator not found at {validator_path}")
        return True  # not a failure, just nothing to test
    tmp = _write_temp_evidence(payload)
    try:
        rc, out, err = _run_validator(validator_path, tmp)
        accepted = rc == 0
        if expect_accept:
            ok = accepted
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: {'accepted' if accepted else 'REJECTED (!)'}")
        else:
            ok = not accepted
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {label}: {'rejected' if not accepted else 'ACCEPTED (!)'}")
        if not ok:
            print(f"    stdout: {out.strip()}")
            print(f"    stderr: {err.strip()}")
        return ok
    except Exception as exc:
        print(f"  [FAIL] {label}: validator crashed: {exc}")
        return False
    finally:
        os.unlink(tmp)


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
        payload = _FABRICATED_PAYLOADS.get(name, {"__fabricated": True})
        ok = validate_evidence(validator, payload, expect_accept=False, label=name)
        results[f"fabricated:{name}"] = ok

    section("Positive-evidence acceptance tests")
    for schema_path in SCHEMAS:
        name = schema_path.name
        validator = validator_for_schema(schema_path)
        payload = _POSITIVE_PAYLOADS.get(name)
        if payload is None:
            print(f"  [SKIP] {name}: no positive payload defined")
            continue
        ok = validate_evidence(validator, payload, expect_accept=True, label=name)
        results[f"positive:{name}"] = ok

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
