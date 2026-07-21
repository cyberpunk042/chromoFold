#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_case(server: Path, model: Path, mode: str, context: int, slots: int, output: Path) -> dict:
    command = [str(server), "-m", str(model), "--ctx-size", str(context), "--parallel", str(slots),
               "--kv-cache-backend", "dense" if mode == "dense" else "chromofold",
               "--chromofold-kernel", mode if mode != "dense" else "reference",
               "--seed", "42", "--port", "0"]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, timeout=300)
    elapsed = time.perf_counter() - started
    log = output / f"{mode}-{context}-{slots}.log"
    log.write_text(completed.stdout + "\n" + completed.stderr)
    return {"mode": mode, "context": context, "slots": slots, "returncode": completed.returncode,
            "elapsed_seconds": elapsed, "command": command, "log": str(log)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="4096,8192,16384,32768")
    parser.add_argument("--slots", default="1,2,4,8")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contexts = [int(value) for value in args.contexts.split(",")]
    slots = [int(value) for value in args.slots.split(",")]
    results = []
    for mode in ("dense", "reference", "optimized"):
        for context in contexts:
            for slot_count in slots:
                results.append(run_case(args.server, args.model, mode, context, slot_count, args.output))
    successful = [item["elapsed_seconds"] for item in results if item["returncode"] == 0]
    report = {"model_sha256": sha256(args.model), "results": results,
              "successful_cases": len(successful), "median_elapsed_seconds": statistics.median(successful) if successful else None,
              "environment": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}}
    (args.output / "benchmark-matrix.json").write_text(json.dumps(report, indent=2))
    if any(item["returncode"] != 0 for item in results):
        raise SystemExit("one or more benchmark cases failed")


if __name__ == "__main__":
    main()
