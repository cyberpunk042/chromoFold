#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_ready(port: int) -> None:
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not become healthy")


def completion(port: int) -> dict:
    payload = json.dumps({"prompt": "ChromoFold deterministic benchmark", "n_predict": 32, "seed": 42, "temperature": 0}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def run_case(server: Path, model: Path, mode: str, context: int, slots: int, output: Path, port: int) -> dict:
    command = [str(server), "-m", str(model), "--ctx-size", str(context), "--parallel", str(slots),
               "--kv-cache-backend", "dense" if mode == "dense" else "chromofold",
               "--seed", "42", "--port", str(port)]
    if mode != "dense":
        command.extend(["--chromofold-kernel", mode])
    log = output / f"{mode}-{context}-{slots}.log"
    started = time.perf_counter()
    with log.open("w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
        try:
            wait_ready(port)
            request_started = time.perf_counter()
            response = completion(port)
            request_elapsed = time.perf_counter() - request_started
            returncode = 0
        except Exception as exc:
            response = {"error": str(exc)}
            request_elapsed = None
            returncode = 1
        finally:
            process.send_signal(subprocess.signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return {"mode": mode, "context": context, "slots": slots, "returncode": returncode,
            "startup_and_request_seconds": time.perf_counter() - started, "request_seconds": request_elapsed,
            "command": command, "log": str(log), "response": response}


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
    case_index = 0
    for mode in ("dense", "reference", "optimized"):
        for context in contexts:
            for slot_count in slots:
                results.append(run_case(args.server, args.model, mode, context, slot_count, args.output, 18080 + case_index))
                case_index += 1
    successful = [item["request_seconds"] for item in results if item["returncode"] == 0 and item["request_seconds"] is not None]
    report = {"model_sha256": sha256(args.model), "results": results,
              "successful_cases": len(successful), "median_request_seconds": statistics.median(successful) if successful else None,
              "environment": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}}
    (args.output / "benchmark-matrix.json").write_text(json.dumps(report, indent=2))
    if any(item["returncode"] != 0 for item in results):
        raise SystemExit("one or more benchmark cases failed")


if __name__ == "__main__":
    main()
