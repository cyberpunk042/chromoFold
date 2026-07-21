#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_pair.py"))
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contexts", default="8192,16384,32768,49152,65536")
    args = parser.parse_args()
    contexts = [int(value) for value in args.contexts.split(",") if value]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"dense_max_context": 0, "chromofold_max_context": 0, "cases": []}
    for backend in ("dense", "chromofold"):
        for context in contexts:
            output = args.output_dir / f"{backend}-{context}.json"
            command = ["python3", str(args.runner), "--llama-cli", str(args.llama_cli), "--model", str(args.model),
                       "--output", str(output), "--backend", backend, "--context", str(context),
                       "--llama-commit", args.llama_commit]
            completed = subprocess.run(command)
            case = json.loads(output.read_text())
            summary["cases"].append({"backend": backend, "context": context, "completed": completed.returncode == 0})
            if completed.returncode != 0:
                break
            summary[f"{backend}_max_context"] = context
    dense = summary["dense_max_context"]
    compressed = summary["chromofold_max_context"]
    summary["context_multiplier"] = compressed / dense if dense else 0.0
    (args.output_dir / "capacity-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if dense and compressed else 1


if __name__ == "__main__":
    raise SystemExit(main())
