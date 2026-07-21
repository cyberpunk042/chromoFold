#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    return process.returncode, (time.perf_counter() - started) * 1000.0


def nvidia_value(*args: str) -> str:
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired dense and ChromoFold hardware inference")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--page-size", type=int, choices=(64, 128, 256), default=64)
    parser.add_argument("--block-size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--chromofold-commit", required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_hash = sha256(args.model)
    prompt_text = args.prompt.read_text(encoding="utf-8")
    env = os.environ.copy()
    env["CHROMOFOLD_MODEL_SHA256"] = model_hash

    common = [
        str(args.binary), "-m", str(args.model), "--ctx-size", str(args.context),
        "--seed", str(args.seed), "--temp", "0", "-n", str(args.predict), "-p", prompt_text,
        "--dump-run-json", str(args.output_dir / "RUN.json"),
        "--dump-logits-json", str(args.output_dir / "LOGITS.json"),
    ]

    runs: dict[str, Any] = {}
    for backend in ("dense", "chromofold"):
        backend_dir = args.output_dir / backend
        backend_dir.mkdir(exist_ok=True)
        command = list(common)
        command[command.index(str(args.output_dir / "RUN.json"))] = str(backend_dir / "run.json")
        command[command.index(str(args.output_dir / "LOGITS.json"))] = str(backend_dir / "logits.json")
        command.extend(["--kv-cache-backend", backend])
        if backend == "chromofold":
            command.extend([
                "--chromofold-page-size", str(args.page_size),
                "--chromofold-block-size", str(args.block_size),
                "--chromofold-evidence", str(backend_dir / "runtime-evidence.json"),
                "--chromofold-stats",
            ])
        code, elapsed_ms = run(command, backend_dir / "run.log", env)
        runs[backend] = {"command": command, "exit_code": code, "elapsed_ms": elapsed_ms}
        if code != 0:
            (args.output_dir / "pair-summary.json").write_text(json.dumps({"model_sha256": model_hash, "runs": runs}, indent=2) + "\n")
            raise SystemExit(f"{backend} run failed with exit code {code}")

    environment = {
        "gpu_name": nvidia_value("--query-gpu=name", "--format=csv,noheader"),
        "compute_capability": nvidia_value("--query-gpu=compute_cap", "--format=csv,noheader"),
        "driver": nvidia_value("--query-gpu=driver_version", "--format=csv,noheader"),
        "cuda_toolkit": nvidia_value(),
        "os": platform.platform(),
        "compiler": os.environ.get("CXX_VERSION", "unknown"),
    }
    summary = {
        "model_sha256": model_hash,
        "model_size_bytes": args.model.stat().st_size,
        "llama_commit": args.llama_commit,
        "chromofold_commit": args.chromofold_commit,
        "context_size": args.context,
        "page_size": args.page_size,
        "block_size": args.block_size,
        "environment": environment,
        "runs": runs,
    }
    (args.output_dir / "pair-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"model_sha256": model_hash, "dense_ms": runs["dense"]["elapsed_ms"], "chromofold_ms": runs["chromofold"]["elapsed_ms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
