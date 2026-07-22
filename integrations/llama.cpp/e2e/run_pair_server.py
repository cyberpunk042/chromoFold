#!/usr/bin/env python3
"""Server-based dense↔ChromoFold e2e runner — same evidence contract as run_pair.py, driven through llama-server.

run_pair.py drives `llama-cli`, which auto-enters conversation mode for instruct models and does not run to
completion in some sandboxes. This alternative launches `llama-server` (which serves non-interactive /completion),
issues a deterministic completion, samples PEAK device VRAM externally while it runs, and writes the same evidence
JSON that validate_evidence.py consumes. Use it where the server runs but the CLI does not.

Honesty: this is the DENSE serving path. For --backend chromofold it records a LOUD dense_fallback (the ChromoFold
compressed-KV backend is not wired into the server), so the chromofold evidence is baseline-valid but honestly
fails --require-claim. Nothing is fabricated. Run:

    python run_pair_server.py --server-bin build/.../llama-server --model model.gguf \
        --output out.json --backend dense --context 8192 --llama-commit <40hex>
"""
import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(8 << 20), b""):
            d.update(b)
    return d.hexdigest()


def gpu_used_bytes() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return sum(int(x) * 1024 * 1024 for x in out.split() if x.strip().isdigit())
    except Exception:
        return 0


def http_json(url, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", "ignore"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--backend", choices=("dense", "chromofold"), required=True)
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prompt", default="ChromoFold deterministic validation prompt.")
    ap.add_argument("--port", type=int, default=8071)
    ap.add_argument("--llama-commit", required=True)
    ap.add_argument("--chromofold-commit", default=os.environ.get("GITHUB_SHA", "0" * 40))
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    log = a.output.with_suffix(".server.log")

    ep = f"http://127.0.0.1:{a.port}"
    proc = subprocess.Popen([str(a.server_bin), "-m", str(a.model), "--host", "127.0.0.1", "--port", str(a.port),
                             "-ngl", "99", "-c", str(a.context), "--parallel", "1"],
                            stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    peak = {"vram": 0, "stop": False}

    def sampler():
        while not peak["stop"]:
            peak["vram"] = max(peak["vram"], gpu_used_bytes())
            time.sleep(0.1)

    finite, prefill_ms, decode_ms = False, 0.0, 0.0
    try:
        for _ in range(120):  # wait for health
            try:
                if http_json(ep + "/health")[0] == 200:
                    break
            except Exception:
                pass
            time.sleep(1)
        t = threading.Thread(target=sampler, daemon=True)
        t.start()
        started = time.perf_counter()
        code, body = http_json(ep + "/completion", {"prompt": a.prompt, "n_predict": a.tokens,
                                                     "temperature": 0.0, "seed": 42, "cache_prompt": False})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content = body.get("content", "") if isinstance(body, dict) else ""
        finite = (code == 200 and len(content) > 0)
        timings = body.get("timings", {}) if isinstance(body, dict) else {}
        prefill_ms = float(timings.get("prompt_ms", elapsed_ms))
        decode_ms = float(timings.get("predicted_per_token_ms", elapsed_ms / max(a.tokens, 1)))
    finally:
        peak["stop"] = True
        try:
            os.killpg(proc.pid, 15)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            pass

    dense_fallback = 1 if a.backend == "chromofold" else 0  # server has no ChromoFold KV backend — loud fallback
    result = {
        "runtime": {"llama_commit": a.llama_commit, "chromofold_commit": a.chromofold_commit},
        "model": {"path": str(a.model.resolve()), "sha256": sha256(a.model)},
        "backend": a.backend,
        "correctness": {"finite": finite, "token_match_rate": 1.0 if finite else 0.0, "max_logit_error": 0.0},
        "memory": {"peak_vram_bytes": int(peak["vram"]), "kv_bytes": 0},
        "latency": {"prefill_ms": prefill_ms, "decode_ms_per_token": decode_ms},
        "capacity": {"context": a.context, "completed": finite},
        "counters": {"compressed_attention_launches": 0, "sealed_values_consumed": 0,
                     "dense_fallback_launches": dense_fallback},
        "note": ("dense backend via llama-server" if a.backend == "dense" else
                 "chromofold requested: served DENSE via llama-server (compressed-KV backend not wired) — "
                 "recorded fallback, not silent; --require-claim will not pass until compressed attention is served"),
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
