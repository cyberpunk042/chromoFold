#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys

import jsonschema


def reject(message: str) -> None:
    raise SystemExit(f"invalid production evidence: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=pathlib.Path)
    parser.add_argument("--schema", type=pathlib.Path, default=pathlib.Path(__file__).with_name("production-evidence.schema.json"))
    args = parser.parse_args()
    document = json.loads(args.evidence.read_text())
    schema = json.loads(args.schema.read_text())
    jsonschema.validate(document, schema)

    lifecycle = document["lifecycle"]
    if lifecycle["snapshots_acquired"] != lifecycle["snapshots_released"]:
        reject("snapshot references did not reconcile")
    kv = document["kv"]
    if kv["pages_registered"] > kv["seal_successes"]:
        reject("more pages were registered than sealed")
    attention = document["attention"]
    if attention["compressed_batches_executed"] > attention["sequence_attention_launches"]:
        reject("batch count exceeds sequence launch count")
    if document["server"]["real_http_requests"] < document["server"]["active_sequences_peak"]:
        reject("active sequence peak exceeds observed HTTP requests")
    if document["shutdown"]["page_refs"] or document["shutdown"]["snapshot_refs"]:
        reject("references remain at shutdown")
    print("M11 production evidence is structurally and semantically valid")


if __name__ == "__main__":
    main()
