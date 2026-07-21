#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_m16_evidence.py evidence.json")
    data = json.loads(Path(sys.argv[1]).read_text())
    cluster = data.get("cluster", {})
    require(cluster.get("router") == 1, "one router required")
    require(cluster.get("prefill_workers", 0) >= 1, "prefill worker required")
    require(cluster.get("decode_workers", 0) >= 2, "two decode workers required")
    require(cluster.get("separate_processes") is True, "roles must execute in separate processes")
    require(len(data.get("model", {}).get("gguf_sha256", "")) == 64, "real GGUF SHA-256 required")
    transfer = data.get("transfer", {})
    require(transfer.get("compressed_pages", 0) > 0 and transfer.get("bytes", 0) > 0, "real compressed transfer required")
    require(transfer.get("received_without_prefill_recompute") is True, "decode must consume transferred KV")
    require(transfer.get("partial_rejected", 0) > 0, "partial transfer rejection must be demonstrated")
    cache = data.get("cache", {})
    require(cache.get("prefix_hits", 0) > 0 and cache.get("prefill_tokens_avoided", 0) > 0, "prefix reuse required")
    leases = data.get("leases", {})
    require(leases.get("acquired", 0) > 0 and leases.get("transferred", 0) > 0 and leases.get("stale_rejected", 0) > 0, "lease lifecycle incomplete")
    recovery = data.get("recovery", {})
    require(recovery.get("worker_failure_injected") is True and recovery.get("replica_recovery_succeeded") is True, "replica recovery required")
    security = data.get("security", {})
    require(all(security.get(key) is True for key in ("worker_authentication", "manifest_integrity", "replay_rejected")), "security gates incomplete")
    isolation = data.get("isolation", {})
    require(isolation.get("cross_request_contamination") == 0, "cross-request contamination detected")
    require(isolation.get("dense_fallback_launches") == 0, "dense fallback prohibited")
    require(isolation.get("cuda_errors") == 0, "CUDA errors detected")
    shutdown = data.get("shutdown", {})
    require(shutdown.get("audit_passed") is True and shutdown.get("page_refs") == 0 and shutdown.get("snapshot_refs") == 0, "shutdown audit failed")
    print("M16 evidence accepted")


if __name__ == "__main__":
    main()
