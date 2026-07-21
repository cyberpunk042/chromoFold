#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"invalid M10 CUDA resolver evidence: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_cuda_resolver_evidence.py EVIDENCE.json")
    data = json.loads(Path(sys.argv[1]).read_text())
    device = data["device_resolution"]
    snapshots = data["snapshots"]
    batching = data["batching"]
    isolation = data["isolation"]
    reclamation = data["reclamation"]
    runtime = data["runtime"]

    if device["pages_registered"] <= 0 or device["descriptor_tables_built"] <= 0:
        fail("device pages or descriptor tables were not produced")
    if device["sequence_attention_launches"] < 2:
        fail("multi-sequence attention was not demonstrated")
    if batching["compressed_batches_executed"] <= 0 or batching["sequences_per_batch_peak"] < 2:
        fail("no real concurrent compressed batch")
    if snapshots["acquired"] != snapshots["released"] or snapshots["in_flight_at_shutdown"] != 0:
        fail("snapshot references did not reconcile")
    if isolation["cross_sequence_contamination"] != 0 or isolation["isolated_baselines_match"] is not True:
        fail("sequence isolation failed")
    if reclamation["pages_reclaimed"] <= 0 or reclamation["page_refs_at_shutdown"] != 0:
        fail("device pages were not fully reclaimed")
    if runtime["dense_fallback_launches"] != 0:
        fail("dense fallback occurred")
    if runtime["cuda_errors"] != 0 or runtime["sanitizer_passed"] is not True:
        fail("CUDA or sanitizer validation failed")
    print("M10 CUDA resolver evidence valid")


if __name__ == "__main__":
    main()
