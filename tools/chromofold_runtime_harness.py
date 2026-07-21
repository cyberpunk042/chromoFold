#!/usr/bin/env python3
"""ChromoFold RC1 runtime qualification harness.

The harness never invents unavailable proof. Missing runtime endpoints, GPU telemetry,
or required scenario results are recorded and produce an INCOMPLETE decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MODES = {"smoke": 600, "candidate": 7200, "stable": 86400}
SECRET_PATTERNS = [
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[=:]\s*\S+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


@dataclass
class ScenarioResult:
    name: str
    executed: bool = False
    passed: bool = False
    duration_ms: int = 0
    requests: int = 0
    errors: int = 0
    recovery_ms: int = 0
    reason: str = ""


@dataclass
class HarnessState:
    mode: str
    release_digest: str
    started_at_unix_ms: int
    runtime_endpoint: str
    metrics_endpoint: str
    evidence: dict[str, Any] = field(default_factory=dict)
    scenarios: list[ScenarioResult] = field(default_factory=list)
    processes: list[subprocess.Popen[str]] = field(default_factory=list, repr=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def request_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 15) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw or b"{}")


def request_text(url: str, timeout: int = 15) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def collect_gpu_telemetry() -> tuple[list[dict[str, Any]], str]:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            devices.append({
                "index": index,
                "name": pynvml.nvmlDeviceGetName(handle),
                "uuid": pynvml.nvmlDeviceGetUUID(handle),
                "memory_used_bytes": memory.used,
                "memory_total_bytes": memory.total,
                "gpu_utilization_percent": utilization.gpu,
                "memory_utilization_percent": utilization.memory,
                "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                "power_mw": pynvml.nvmlDeviceGetPowerUsage(handle),
            })
        pynvml.nvmlShutdown()
        return devices, "nvml"
    except Exception as exc:
        return [], f"unavailable: {exc}"


def environment_fingerprint(model: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
    }
    if model and model.is_file():
        result["model"] = {"path": model.name, "size_bytes": model.stat().st_size, "sha256": sha256_file(model)}
    return result


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if re.search(r"(?i)(prompt|completion|password|secret|token|private_key|api_key)", k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        out = value
        for pattern in SECRET_PATTERNS:
            out = pattern.sub("[REDACTED]", out)
        return out
    return value


def start_process(command: str, log_path: Path) -> subprocess.Popen[str]:
    log = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(command, shell=True, text=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_processes(processes: list[subprocess.Popen[str]]) -> None:
    for proc in reversed(processes):
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 15
    for proc in reversed(processes):
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def wait_ready(endpoint: str, timeout: int) -> tuple[bool, str]:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            data = request_json(endpoint.rstrip("/") + "/ready", timeout=5)
            if data.get("ready") is True or data.get("status") in {"ready", "ok"}:
                return True, ""
            last = json.dumps(data)
        except Exception as exc:
            last = str(exc)
        time.sleep(2)
    return False, last


def run_workload(state: HarnessState, name: str, payload: dict[str, Any]) -> ScenarioResult:
    started = time.monotonic()
    try:
        result = request_json(state.runtime_endpoint.rstrip("/") + "/qualification/scenario", "POST", payload, timeout=120)
        return ScenarioResult(
            name=name,
            executed=True,
            passed=bool(result.get("passed")),
            duration_ms=int((time.monotonic() - started) * 1000),
            requests=int(result.get("requests", 0)),
            errors=int(result.get("errors", 0)),
            recovery_ms=int(result.get("recovery_ms", 0)),
            reason=str(result.get("reason", "")),
        )
    except Exception as exc:
        return ScenarioResult(name=name, duration_ms=int((time.monotonic() - started) * 1000), reason=str(exc))


def required_scenarios() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("cache-cold", {"cache": "cold", "concurrency": 1}),
        ("cache-warm", {"cache": "warm", "concurrency": 4}),
        ("multi-tenant-overload", {"tenants": 2, "concurrency": 32, "overload": True}),
        ("cancellation", {"cancel_fraction": 0.25, "concurrency": 8}),
        ("decode-worker-failure", {"inject": "decode-worker-termination"}),
        ("network-fault", {"inject": "network-delay", "delay_ms": 250}),
        ("certificate-rotation", {"inject": "certificate-rotation"}),
        ("rolling-upgrade", {"inject": "rolling-upgrade"}),
        ("rollback", {"inject": "rollback"}),
    ]


def build_evidence(state: HarnessState, gpu: list[dict[str, Any]], gpu_source: str, metrics: str, runtime_snapshot: dict[str, Any]) -> dict[str, Any]:
    scenarios = {s.name: asdict(s) for s in state.scenarios}
    all_executed = all(s.executed for s in state.scenarios)
    all_passed = all(s.passed for s in state.scenarios)
    references = runtime_snapshot.get("references", {})
    memory = runtime_snapshot.get("memory", {})
    return {
        "schema": "chromofold.rc1.runtime-evidence.v1",
        "mode": state.mode,
        "release_digest": state.release_digest,
        "qualified_digest": state.release_digest,
        "promoted_digest": runtime_snapshot.get("promoted_digest", state.release_digest),
        "started_at_unix_ms": state.started_at_unix_ms,
        "finished_at_unix_ms": int(time.time() * 1000),
        "environment": state.evidence["environment"],
        "gpu": {"source": gpu_source, "devices": gpu},
        "telemetry": {
            "end_to_end_correlated": bool(runtime_snapshot.get("end_to_end_correlated", False)),
            "prompt_content_leaks": int(runtime_snapshot.get("prompt_content_leaks", 0)),
            "metrics_scraped": bool(metrics),
            "mtls_observed": bool(runtime_snapshot.get("mtls_observed", False)),
            "audit_chain_verified": bool(runtime_snapshot.get("audit_chain_verified", False)),
        },
        "quality": {
            "correctness_failures": int(runtime_snapshot.get("correctness_failures", 0)),
            "cuda_errors": int(runtime_snapshot.get("cuda_errors", 0)),
            "cross_tenant_contamination": int(runtime_snapshot.get("cross_tenant_contamination", 0)),
            "dense_fallback_launches": int(runtime_snapshot.get("dense_fallback_launches", 0)),
            "ttft_p95_ms": float(runtime_snapshot.get("ttft_p95_ms", 0.0)),
            "inter_token_p95_ms": float(runtime_snapshot.get("inter_token_p95_ms", 0.0)),
        },
        "memory": memory,
        "references": references,
        "scenarios": scenarios,
        "scenario_execution_complete": all_executed,
        "scenario_success": all_passed,
        "incident_bundle_redacted": True,
    }


def create_incident_bundle(output: Path, evidence: dict[str, Any], workdir: Path) -> None:
    bundle_dir = workdir / "incident"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "evidence.json").write_text(json.dumps(redact(evidence), indent=2, sort_keys=True), encoding="utf-8")
    for name in ("runtime.log", "metrics.txt"):
        source = workdir / name
        if source.exists():
            (bundle_dir / name).write_text(str(redact(source.read_text(encoding="utf-8", errors="replace"))), encoding="utf-8")
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle_dir))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--runtime-endpoint", default=os.getenv("CHROMOFOLD_RUNTIME_ENDPOINT", "http://127.0.0.1:8080"))
    parser.add_argument("--metrics-endpoint", default=os.getenv("CHROMOFOLD_METRICS_ENDPOINT", "http://127.0.0.1:8080/metrics"))
    parser.add_argument("--runtime-command", default=os.getenv("CHROMOFOLD_RUNTIME_COMMAND", ""))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--incident-bundle", type=Path)
    parser.add_argument("--release-digest", default=os.getenv("CHROMOFOLD_RELEASE_DIGEST", ""))
    parser.add_argument("--ready-timeout", type=int, default=180)
    args = parser.parse_args()

    if not args.release_digest:
        print("missing --release-digest or CHROMOFOLD_RELEASE_DIGEST", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="chromofold-rc1-") as temp:
        workdir = Path(temp)
        state = HarnessState(args.mode, args.release_digest, int(time.time() * 1000), args.runtime_endpoint, args.metrics_endpoint)
        state.evidence["environment"] = environment_fingerprint(args.model)
        try:
            if args.runtime_command:
                state.processes.append(start_process(args.runtime_command, workdir / "runtime.log"))
            ready, reason = wait_ready(args.runtime_endpoint, args.ready_timeout)
            if not ready:
                incomplete = {
                    "schema": "chromofold.rc1.runtime-evidence.v1",
                    "mode": args.mode,
                    "release_digest": args.release_digest,
                    "decision": "INCOMPLETE",
                    "reason": f"runtime not ready: {reason}",
                    "environment": state.evidence["environment"],
                }
                args.artifact.parent.mkdir(parents=True, exist_ok=True)
                args.artifact.write_text(json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8")
                return 3

            for name, payload in required_scenarios():
                state.scenarios.append(run_workload(state, name, {"name": name, "mode": args.mode, **payload}))

            try:
                metrics = request_text(args.metrics_endpoint)
                (workdir / "metrics.txt").write_text(metrics, encoding="utf-8")
            except Exception:
                metrics = ""
            try:
                snapshot = request_json(args.runtime_endpoint.rstrip("/") + "/qualification/snapshot")
            except Exception as exc:
                snapshot = {"snapshot_error": str(exc)}
            gpu, gpu_source = collect_gpu_telemetry()
            evidence = build_evidence(state, gpu, gpu_source, metrics, snapshot)
            missing_hardware = not gpu or not evidence["telemetry"]["end_to_end_correlated"]
            violations = (
                evidence["telemetry"]["prompt_content_leaks"] != 0
                or evidence["quality"]["correctness_failures"] != 0
                or evidence["quality"]["cuda_errors"] != 0
                or evidence["quality"]["cross_tenant_contamination"] != 0
                or evidence["quality"]["dense_fallback_launches"] != 0
                or evidence["qualified_digest"] != evidence["promoted_digest"]
                or not evidence["scenario_success"]
            )
            evidence["decision"] = "FAIL" if violations else ("INCOMPLETE" if missing_hardware else "PASS")
            args.artifact.parent.mkdir(parents=True, exist_ok=True)
            args.artifact.write_text(json.dumps(redact(evidence), indent=2, sort_keys=True), encoding="utf-8")
            bundle = args.incident_bundle or args.artifact.with_suffix(".incident.tar.gz")
            create_incident_bundle(bundle, evidence, workdir)
            return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[evidence["decision"]]
        finally:
            stop_processes(state.processes)


if __name__ == "__main__":
    raise SystemExit(main())
