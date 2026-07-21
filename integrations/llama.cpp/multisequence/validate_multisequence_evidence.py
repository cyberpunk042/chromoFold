#!/usr/bin/env python3
import json
import pathlib
import sys

import jsonschema


def fail(message: str) -> None:
    raise SystemExit(f"invalid M10 evidence: {message}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_multisequence_evidence.py EVIDENCE.json")
    root = pathlib.Path(__file__).resolve().parent
    schema = json.loads((root / "multisequence-evidence.schema.json").read_text())
    evidence = json.loads(pathlib.Path(sys.argv[1]).read_text())
    jsonschema.validate(evidence, schema)

    sequences = evidence["sequences"]
    sharing = evidence["sharing"]
    reclamation = evidence["reclamation"]
    batching = evidence["batching"]
    speculation = evidence["speculation"]
    invariants = evidence["invariants"]
    runtime = evidence["runtime"]

    if sequences["copied"] == 0 or sharing["shared_pages"] == 0:
        fail("sequence-copy proof did not produce shared immutable pages")
    if sharing["shared_page_references"] < sharing["shared_pages"] * 2:
        fail("shared-page reference count is inconsistent")
    if reclamation["pending_pages_at_shutdown"] or reclamation["pending_bytes_at_shutdown"]:
        fail("retired pages remain at shutdown")
    if batching["sequences_per_batch_peak"] < 2:
        fail("workload never executed a multi-sequence batch")
    if speculation["rolled_back"] == 0:
        fail("speculative rollback was not exercised")
    if invariants["failures"] or not invariants["reference_count_balance"]:
        fail("invariant auditor did not reconcile page ownership")
    if runtime["dense_fallback_launches"] != 0:
        fail("dense fallback occurred")
    if runtime["cuda_errors"] != 0 or not runtime["outputs_finite"]:
        fail("CUDA execution was not clean and finite")
    if not runtime["sanitizer_passed"]:
        fail("Compute Sanitizer proof is missing")

    print("M10 multi-sequence evidence is valid")


if __name__ == "__main__":
    main()
