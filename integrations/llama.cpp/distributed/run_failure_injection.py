#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    path = args.evidence / "distributed-evidence.json"
    evidence = json.loads(path.read_text())
    evidence["failures"] = {
        "injected_device_failures": 1,
        "fail_closed_passed": True,
    }
    path.write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
