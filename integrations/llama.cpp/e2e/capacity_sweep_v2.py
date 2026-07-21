#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dense and ChromoFold context-capacity sweeps")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contexts", default="4096,8192,16384,32768,65536")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--predict", type=int, default=8)
    args = parser.parse_args()

    contexts = [int(value) for value in args.contexts.split(",") if value.strip()]
    prompt = args.prompt.read_text(encoding="utf-8")
    results = []
    maxima = {"dense": 0, "chromofold": 0}

    for backend in ("dense", "chromofold"):
        for context in contexts:
            evidence = args.output.parent / f"capacity-{backend}-{context}.json"
            command = [
                str(args.binary), "-m", str(args.model), "--ctx-size", str(context),
                "--seed", "42", "--temp", "0", "-n", str(args.predict), "-p", prompt,
                "--kv-cache-backend", backend, "--dump-run-json", str(evidence),
            ]
            if backend == "chromofold":
                command.extend(["--chromofold-page-size", str(args.page_size), "--chromofold-evidence", str(evidence)])
            completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            valid = completed.returncode == 0 and evidence.exists()
            counters = {}
            if valid:
                try:
                    counters = json.loads(evidence.read_text(encoding="utf-8")).get("counters", {})
                except Exception:
                    valid = False
            if backend == "chromofold":
                valid = valid and counters.get("compressed_attention_launches", 0) > 0 and counters.get("dense_fallback_launches", 0) == 0
            results.append({"backend": backend, "context": context, "success": valid, "exit_code": completed.returncode, "counters": counters})
            if valid:
                maxima[backend] = context
            else:
                break

    payload = {"dense_max_context": maxima["dense"], "chromofold_max_context": maxima["chromofold"], "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dense_max_context": maxima["dense"], "chromofold_max_context": maxima["chromofold"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
