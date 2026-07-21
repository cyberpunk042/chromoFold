#!/usr/bin/env python3
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    samples = []
    failures = 0
    for index in range(args.requests):
        prefix = "shared-prefix " * (8 if index % 3 else 32)
        prompt = f"{prefix} sequence-{index % 17} nonce-{rng.randrange(1_000_000)}"
        started = time.perf_counter()
        try:
            result = post(f"{args.base_url}/completion", {"prompt": prompt, "n_predict": 8, "seed": args.seed + index, "temperature": 0})
            ok = bool(result)
        except Exception as exc:
            result = {"error": str(exc)}
            ok = False
            failures += 1
        samples.append({"index": index, "elapsed_seconds": time.perf_counter() - started, "ok": ok, "result": result})
    report = {"requests": args.requests, "failures": failures, "passed": failures == 0, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"{failures} soak requests failed")


if __name__ == "__main__":
    main()
