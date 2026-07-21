#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request


def post(base: str, payload: dict, timeout: float = 120.0) -> dict:
    request = urllib.request.Request(
        base.rstrip("/") + "/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shared = "ChromoFold shared-prefix isolation fixture. " * 96
    requests = [
        {"name": "A", "prompt": shared + "Continue with identifier ALPHA only.", "n_predict": 32},
        {"name": "B", "prompt": shared + "Continue with identifier BRAVO only.", "n_predict": 32},
        {"name": "C", "prompt": ("Independent CHARLIE context. " * 96), "n_predict": 32},
        {"name": "D", "prompt": ("Cancellation DELTA context. " * 128), "n_predict": 256},
        {"name": "E", "prompt": "Slot reuse ECHO after cancellation.", "n_predict": 16},
        {"name": "F", "prompt": shared + "Branch with identifier FOXTROT only.", "n_predict": 32},
    ]
    started = time.time()
    results = {}

    def execute(item: dict) -> tuple[str, dict]:
        payload = {
            "prompt": item["prompt"],
            "n_predict": item["n_predict"],
            "seed": args.seed,
            "temperature": 0,
            "cache_prompt": True,
        }
        try:
            return item["name"], {"ok": True, "response": post(args.server, payload)}
        except Exception as exc:
            return item["name"], {"ok": False, "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for name, result in executor.map(execute, requests[:4]):
            results[name] = result
    name, result = execute(requests[4])
    results[name] = result
    name, result = execute(requests[5])
    results[name] = result

    successful = [name for name, result in results.items() if result["ok"]]
    report = {
        "seed": args.seed,
        "duration_seconds": time.time() - started,
        "request_count": len(requests),
        "successful_requests": successful,
        "failed_requests": [name for name, result in results.items() if not result["ok"]],
        "shared_prefix_sha256": hashlib.sha256(shared.encode()).hexdigest(),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if len(successful) < 5:
        raise SystemExit("production workload did not complete enough requests")


if __name__ == "__main__":
    main()
