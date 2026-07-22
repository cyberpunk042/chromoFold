#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_used_bytes() -> int:
    """Observed device VRAM in use across all GPUs (MiB -> bytes), via nvidia-smi. 0 if unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return sum(int(x) * 1024 * 1024 for x in out.split() if x.strip().isdigit())
    except Exception:
        return 0


def run(command: list[str], log: Path, env: dict[str, str]) -> tuple[int, float, int]:
    # Sample PEAK device VRAM while the child runs — the binary cannot measure this post-hoc (the model is freed
    # before it exits), so run_pair observes it externally. Real measurement, not a placeholder.
    started = time.perf_counter()
    peak_vram = 0
    with log.open("w") as handle:
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            peak_vram = max(peak_vram, gpu_used_bytes())
            time.sleep(0.1)
        proc.wait()
    return proc.returncode, (time.perf_counter() - started) * 1000.0, peak_vram


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("dense", "chromofold"), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--prompt", default="ChromoFold deterministic validation prompt.")
    parser.add_argument("--llama-commit", required=True)
    parser.add_argument("--chromofold-commit", default=os.environ.get("GITHUB_SHA", "0" * 40))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix(".log")
    command = [str(args.llama_cli), "-m", str(args.model), "-p", args.prompt, "-n", str(args.tokens),
               "-c", str(args.context), "--seed", "42", "--temp", "0", "-no-cnv"]  # non-interactive completion
    env = os.environ.copy()
    env["CHROMOFOLD_KV_BACKEND"] = args.backend
    env["CHROMOFOLD_PAGE_SIZE"] = str(args.page_size)
    env["CHROMOFOLD_EVIDENCE_PATH"] = str(args.output.with_suffix(".runtime.json"))
    code, elapsed_ms, peak_vram = run(command, log, env)
    runtime_path = args.output.with_suffix(".runtime.json")
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    memory = dict(runtime.get("memory", {}))
    memory.setdefault("kv_bytes", 0)
    memory["peak_vram_bytes"] = int(memory.get("peak_vram_bytes") or peak_vram)  # externally-sampled peak
    result = {
        "runtime": {"llama_commit": args.llama_commit, "chromofold_commit": args.chromofold_commit},
        "model": {"path": str(args.model.resolve()), "sha256": sha256(args.model)},
        "backend": args.backend,
        "correctness": runtime.get("correctness", {"finite": code == 0, "token_match_rate": 0.0, "max_logit_error": 0.0}),
        "memory": memory,
        "latency": runtime.get("latency", {"prefill_ms": elapsed_ms, "decode_ms_per_token": elapsed_ms / max(args.tokens, 1)}),
        "capacity": {"context": args.context, "completed": code == 0},
        "counters": runtime.get("counters", {"compressed_attention_launches": 0, "sealed_values_consumed": 0, "dense_fallback_launches": 0}),
        "process": {"exit_code": code, "elapsed_ms": elapsed_ms, "log": str(log)}
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
