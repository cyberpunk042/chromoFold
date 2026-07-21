#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    evidence = json.loads(Path(sys.argv[1]).read_text())
    require(evidence["topology"]["device_count"] >= 2, "at least two devices required")
    require(evidence["topology"]["peer_matrix_verified"] is True, "peer matrix not verified")
    require(evidence["routing"]["devices_used"] >= 2, "routing did not use multiple devices")
    require(evidence["transfers"]["peer_copies"] > 0 and evidence["transfers"]["peer_bytes"] > 0, "no peer traffic")
    require(evidence["transfers"]["transfer_errors"] == 0, "transfer errors present")
    require(evidence["attention"]["tensor_parallel_batches"] > 0, "tensor-parallel path not executed")
    require(evidence["attention"]["mixed_codec_batches"] > 0, "M13 mixed-codec pages not consumed")
    require(evidence["attention"]["dense_fallback_launches"] == 0, "dense fallback executed")
    require(evidence["attention"]["cuda_errors"] == 0, "CUDA errors present")
    require(evidence["cache"]["shared_index_verified"] is True, "shared cache index not verified")
    require(evidence["cache"]["remote_pages_loaded"] > 0, "no remote persistent pages loaded")
    require(evidence["isolation"]["cross_device_contamination"] == 0, "cross-device contamination detected")
    require(evidence["failures"]["injected_device_failures"] > 0 and evidence["failures"]["fail_closed_passed"] is True, "failure isolation unproven")
    require(evidence["shutdown"]["audit_passed"] is True, "shutdown audit failed")
    require(evidence["shutdown"]["page_refs"] == 0 and evidence["shutdown"]["snapshot_refs"] == 0, "references leaked")
    print("M14 distributed evidence accepted")


if __name__ == "__main__":
    main()
