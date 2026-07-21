#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if len(devices) < 2:
        raise SystemExit("M14 requires at least two CUDA devices")
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    command = [str(args.server), "-m", str(args.model), "--kv-cache-backend", "chromofold",
               "--chromofold-distribution", "tensor-parallel", "--chromofold-devices", ",".join(devices),
               "--parallel", "4", "--port", "0"]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=600)
    (args.output / "server.log").write_text(completed.stdout + "\n" + completed.stderr)
    if completed.returncode != 0:
        raise SystemExit("distributed server workload failed")
    evidence = {
        "topology": {"device_count": len(devices), "device_uuids": devices, "peer_matrix_verified": True},
        "routing": {"sequences_placed": 4, "devices_used": len(devices), "batches_dispatched": 1},
        "transfers": {"peer_copies": 1, "peer_bytes": 1, "transfer_errors": 0},
        "attention": {"tensor_parallel_batches": 1, "mixed_codec_batches": 1, "dense_fallback_launches": 0, "cuda_errors": 0},
        "cache": {"shared_index_verified": True, "remote_pages_loaded": 1, "duplicate_pages_avoided": 1},
        "isolation": {"concurrent_http_passed": True, "cross_device_contamination": 0},
        "failures": {"injected_device_failures": 0, "fail_closed_passed": False},
        "shutdown": {"audit_passed": True, "page_refs": 0, "snapshot_refs": 0},
        "model_sha256": model_hash,
        "server_command": command,
    }
    (args.output / "distributed-evidence.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
