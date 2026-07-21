#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    require(len(sys.argv) == 2, "usage: validate_m15_evidence.py evidence.json")
    data = json.loads(Path(sys.argv[1]).read_text())
    require(data["upstream"]["patched_server_executed"] is True, "patched server not executed")
    require(data["topology"]["devices_used"] >= 2, "fewer than two devices used")
    require(data["topology"]["peer_topology_verified"] is True, "peer topology not verified")
    require(data["transfers"]["p2p_copy_calls"] > 0 and data["transfers"]["p2p_bytes"] > 0, "no real P2P transfer")
    require(data["transfers"]["transfer_errors"] == 0, "transfer errors present")
    require(data["collectives"]["calls"] > 0 and data["collectives"]["tensor_parallel_batches"] > 0, "collectives not exercised")
    require(data["attention"]["mixed_codec_pages_consumed"] > 0, "M13 pages not consumed")
    require(data["attention"]["remote_pages_loaded"] > 0, "remote pages not loaded")
    require(data["attention"]["dense_fallback_launches"] == 0, "dense fallback observed")
    require(data["attention"]["cuda_errors"] == 0, "CUDA errors present")
    require(data["attention"]["cross_device_contamination"] == 0, "cross-device contamination observed")
    require(data["failures"]["device_failure_injected"] is True, "failure not injected")
    require(data["failures"]["failed_closed"] is True, "failure did not fail closed")
    require(data["failures"]["recovery_completed"] is True, "recovery incomplete")
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        require(data["sanitizer"][tool] is True, f"{tool} did not pass")
    require(data["shutdown"]["audit_passed"] is True, "shutdown audit failed")
    require(data["shutdown"]["page_refs"] == 0 and data["shutdown"]["snapshot_refs"] == 0, "references remain")
    print("M15 evidence accepted")


if __name__ == "__main__":
    main()
