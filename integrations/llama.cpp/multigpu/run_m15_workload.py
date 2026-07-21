#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import urllib.request
from pathlib import Path


def request(url: str, prompt: str) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": 32, "temperature": 0, "seed": 42}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080/completion")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    args = parser.parse_args()
    prompts = [f"M15 distributed sequence {index}: explain peer transfer invariant." for index in range(args.requests)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda prompt: request(args.url, prompt), prompts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"requests": len(results), "results": results}, indent=2))
    if len(results) != args.requests:
        raise SystemExit("incomplete workload")


if __name__ == "__main__":
    main()
