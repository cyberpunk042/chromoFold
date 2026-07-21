#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"invalid M12 evidence: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_performance_evidence.py evidence.json")
    data = json.loads(Path(sys.argv[1]).read_text())
    required = ["environment", "upstream", "model", "kernels", "graphs", "memory_pool", "benchmarks", "numerics", "isolation", "sanitizer", "soak", "shutdown"]
    for key in required:
        if key not in data:
            fail(f"missing {key}")
    if not data["upstream"].get("patched_server_executed"):
        fail("patched llama-server did not execute")
    if len(data["model"].get("gguf_sha256", "")) != 64:
        fail("GGUF SHA-256 missing")
    kernels = data["kernels"]
    if kernels.get("reference_launches", 0) < 1 or kernels.get("optimized_launches", 0) < 1:
        fail("both reference and optimized kernels must execute")
    if kernels.get("grouped_batches", 0) < 1:
        fail("grouped multi-sequence dispatch was not exercised")
    if kernels.get("dense_fallback_launches") != 0:
        fail("dense fallback occurred")
    if kernels.get("cuda_errors") != 0:
        fail("CUDA errors occurred")
    graphs = data["graphs"]
    if graphs.get("captures", 0) < 1 or graphs.get("replays", 0) < 1:
        fail("CUDA Graph capture/replay missing")
    numerics = data["numerics"]
    if not numerics.get("optimized_reference_passed"):
        fail("optimized/reference comparison failed")
    if numerics.get("cosine_similarity", 0.0) < 0.99:
        fail("cosine similarity below committed threshold")
    if not data["isolation"].get("concurrent_http_passed") or data["isolation"].get("cross_sequence_contamination") != 0:
        fail("concurrent sequence isolation failed")
    if not all(data["sanitizer"].get(tool) for tool in ("memcheck", "racecheck", "initcheck", "synccheck")):
        fail("Compute Sanitizer suite incomplete")
    soak = data["soak"]
    if not soak.get("passed") or soak.get("requests", 0) < 1000 or not soak.get("memory_stable"):
        fail("soak gate failed")
    shutdown = data["shutdown"]
    if not shutdown.get("audit_passed") or shutdown.get("page_refs") != 0 or shutdown.get("snapshot_refs") != 0:
        fail("shutdown references did not reconcile")
    benchmarks = data["benchmarks"]
    for mode in ("dense", "reference", "optimized"):
        if mode not in benchmarks:
            fail(f"missing {mode} benchmark")
    print("M12 production performance evidence valid")


if __name__ == "__main__":
    main()
