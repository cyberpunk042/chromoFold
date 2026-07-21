#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path


def run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    result.elapsed_seconds = time.perf_counter() - started  # type: ignore[attr-defined]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        args.server, "-m", args.model, "--port", str(args.port),
        "--kv-cache-backend", "chromofold", "--chromofold-page-size", "64",
        "--chromofold-evidence", str(output / "runtime-evidence.json")
    ]
    process = subprocess.Popen(command, stdout=(output / "server.log").open("w"), stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(3)
        scenarios = {
            "shared_a": "Shared prefix: cobalt amber quartz. Continue only with branch A.",
            "shared_b": "Shared prefix: cobalt amber quartz. Continue only with branch B.",
            "isolated_c": "Unique identifier sequence-C-only. Repeat only sequence-C-only.",
            "cancel_d": "Generate a long deterministic numbered sequence for cancellation.",
            "branch_e": "Shared prefix: cobalt amber quartz. Branch E independently.",
            "speculative_f": "Speculative rollback fixture: retain baseline and reject suffix."
        }
        requests = []
        for name, prompt in scenarios.items():
            payload = json.dumps({"prompt": prompt, "n_predict": 48, "temperature": 0, "seed": 42})
            log = output / f"{name}.log"
            result = run(["curl", "-fsS", "-X", "POST", f"http://127.0.0.1:{args.port}/completion", "-H", "Content-Type: application/json", "-d", payload], log)
            requests.append({"name": name, "returncode": result.returncode, "elapsed_seconds": result.elapsed_seconds})
        (output / "workload.json").write_text(json.dumps({"requests": requests}, indent=2))
        if any(item["returncode"] != 0 for item in requests):
            raise SystemExit("one or more concurrent workload requests failed")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
