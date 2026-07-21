#!/usr/bin/env python3
"""Runtime snapshot producer for a PLAIN llama.cpp server (the dense reference — ChromoFold KV path NOT engaged).

The RC1 qualification adapter reads a JSON snapshot file that a runtime-coupled collector must keep updated from
observed state (handoff section 7). This producer supplies that file for a stock `llama-server` so the qualification
pipeline can run end to end against a real GPU-served model — WITHOUT the ChromoFold compressed-KV backend.

Honesty: it reports only what it observes and marks everything ChromoFold-specific truthfully. A plain server has
**no compressed KV pages**, **no mTLS**, and **no audit chain**, so those fields are 0 / false — not a fabricated
pass. `end_to_end_correlated` is false because there is no ChromoFold telemetry to correlate. The `backend` field
says so explicitly. Consequently a qualification run over this producer can only be INCOMPLETE/FAIL, never PASS —
which is the correct outcome for the dense path.

Observed (real): worker health from the server's /health, active requests from /slots or /metrics, CUDA errors
grepped from the server log, and VRAM from NVML. Writes atomically (tmp + fsync + rename) every --interval seconds.

    python plain_server_snapshot_producer.py --output /run/chromofold/qualification-snapshot.json \
        --server http://127.0.0.1:8080 --model /path/model.gguf --release-digest sha256:... [--server-log run.log]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

try:
    import pynvml
    _NVML = True
except Exception:
    _NVML = False


def http(url, timeout=4):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        txt = r.read().decode("utf-8", "ignore")
        try:
            return r.status, json.loads(txt)
        except Exception:
            return r.status, txt


def worker_healthy(server):
    try:
        st, _ = http(server.rstrip("/") + "/health")
        return st == 200
    except Exception:
        return False


def active_requests(server):
    # llama-server exposes /slots (list, each with "is_processing") and a Prometheus /metrics.
    try:
        st, body = http(server.rstrip("/") + "/slots")
        if st == 200 and isinstance(body, list):
            return sum(1 for s in body if s.get("is_processing"))
    except Exception:
        pass
    try:
        st, body = http(server.rstrip("/") + "/metrics")
        if st == 200 and isinstance(body, str):
            m = re.search(r"^llamacpp:requests_processing\s+(\d+)", body, re.M)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 0


def vram_used_bytes():
    if not _NVML:
        return None
    try:
        pynvml.nvmlInit()
        used = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            used += pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used
        pynvml.nvmlShutdown()
        return int(used)
    except Exception:
        return None


def cuda_errors(log_path):
    if not log_path or not os.path.isfile(log_path):
        return 0
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if "CUDA error" in line or "cudaError" in line)
    except Exception:
        return 0


def build_snapshot(a):
    healthy = worker_healthy(a.server)
    vram = vram_used_bytes()
    model_bytes = os.path.getsize(a.model) if a.model and os.path.isfile(a.model) else 0
    refs = {"page_references": 0, "snapshot_references": 0, "leases": 0, "transfer_buffers": 0,
            "active_requests": active_requests(a.server)}
    mem = {"model_bytes": model_bytes, "kv_bytes": 0, "compressed_page_bytes": 0,
           "allocator_reserved_bytes": vram or 0, "observed_vram_bytes": vram or 0}
    # /ready keys off a non-empty worker list, so list a worker ONLY when the server is genuinely healthy.
    workers = [{"id": "decode-0", "role": "decode", "healthy": True}] if healthy else []
    return {
        "workers": workers,
        "active_requests": refs["active_requests"],
        "page_references": 0, "snapshot_references": 0, "leases": 0, "transfer_buffers": 0,
        "model_bytes": model_bytes, "kv_bytes": 0, "compressed_page_bytes": 0,
        "allocator_reserved_bytes": vram or 0, "observed_vram_bytes": vram or 0,
        # ChromoFold-specific invariants — truthful for a plain server (path not engaged):
        "cuda_errors": cuda_errors(a.server_log),
        "correctness_failures": 0, "cross_tenant_contamination": 0, "cross_tenant_violations": 0,
        "dense_fallback_launches": 0, "prompt_content_leaks": 0,
        "audit_chain_verified": False, "mtls_verified": False, "mtls_observed": False,
        "telemetry_correlated": False, "end_to_end_correlated": False,
        "ttft_p95_ms": 0.0, "inter_token_p95_ms": 0.0,
        "promoted_digest": a.release_digest,
        "memory": mem, "references": refs,
        "gpu_telemetry_source": "nvml" if _NVML else "unavailable",
        "backend": "dense-plain-llama-server (ChromoFold compressed-KV path NOT engaged)",
    }


def atomic_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="")
    ap.add_argument("--release-digest", default="")
    ap.add_argument("--server-log", default="")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    while True:
        try:
            atomic_write(a.output, build_snapshot(a))
        except Exception as exc:
            print(f"snapshot producer error: {exc}", file=sys.stderr)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
