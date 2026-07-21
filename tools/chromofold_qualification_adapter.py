#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

SCENARIOS = {
    "cache-cold", "cache-warm", "multi-tenant-overload", "cancellation",
    "decode-worker-failure", "network-fault", "certificate-rotation",
    "rolling-upgrade", "rollback",
}

@dataclass
class State:
    enabled: bool
    admin_token: str
    release_digest: str
    snapshot_path: Path
    hooks_dir: Path | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    scenarios: dict[str, dict[str, Any]] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "qualification_mode": self.enabled,
            "release_digest": self.release_digest,
            "active_requests": 0,
            "page_references": 0,
            "snapshot_references": 0,
            "leases": 0,
            "transfer_buffers": 0,
            "model_bytes": 0,
            "kv_bytes": 0,
            "compressed_page_bytes": 0,
            "allocator_reserved_bytes": 0,
            "observed_vram_bytes": 0,
            "cuda_errors": 0,
            "correctness_failures": 0,
            "cross_tenant_violations": 0,
            "dense_fallback_launches": 0,
            "audit_chain_verified": False,
            "mtls_verified": False,
            "telemetry_correlated": False,
            "workers": [],
        }
        if self.snapshot_path.exists():
            loaded = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("snapshot must be a JSON object")
            base.update(loaded)
        base["qualification_mode"] = self.enabled
        base["release_digest"] = self.release_digest
        return base

    def run_scenario(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name not in SCENARIOS:
            raise ValueError(f"unsupported scenario: {name}")
        started = time.time()
        result: dict[str, Any] = {
            "scenario": name,
            "status": "INCOMPLETE",
            "started_at": started,
            "completed_at": started,
            "requests_started": 0,
            "requests_completed": 0,
            "requests_failed": 0,
            "recovery_ms": 0,
            "references_reconciled": False,
            "observations": [],
        }
        hook = self.hooks_dir / name if self.hooks_dir else None
        if hook and hook.is_file() and os.access(hook, os.X_OK):
            proc = subprocess.run(
                [str(hook)], input=json.dumps(payload), text=True,
                capture_output=True, timeout=int(payload.get("timeout_seconds", 300)), check=False,
            )
            if proc.stdout.strip():
                observed = json.loads(proc.stdout)
                if not isinstance(observed, dict):
                    raise ValueError("scenario hook output must be a JSON object")
                result.update(observed)
            if proc.returncode != 0 and result.get("status") == "PASS":
                result["status"] = "FAIL"
                result.setdefault("observations", []).append({"hook_exit_code": proc.returncode})
        else:
            result["observations"].append({"reason": "runtime hook unavailable"})
        result["completed_at"] = time.time()
        with self.lock:
            self.scenarios[name] = result
        return result

class Handler(BaseHTTPRequestHandler):
    server_version = "ChromoFoldQualificationAdapter/1"

    @property
    def state(self) -> State:
        return self.server.state  # type: ignore[attr-defined]

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        if not self.state.enabled:
            self._json(HTTPStatus.NOT_FOUND, {"error": "qualification mode disabled"})
            return False
        expected = f"Bearer {self.state.admin_token}"
        if not self.state.admin_token or self.headers.get("Authorization") != expected:
            self._json(HTTPStatus.FORBIDDEN, {"error": "administrator authorization required"})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/live":
            self._json(HTTPStatus.OK, {"live": True})
        elif self.path == "/ready":
            try:
                snap = self.state.snapshot()
                ready = bool(snap.get("workers")) and snap.get("release_digest") == self.state.release_digest
                self._json(HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE, {"ready": ready})
            except Exception as exc:
                self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"ready": False, "error": str(exc)})
        elif self.path == "/qualification/snapshot":
            if self._authorized():
                try:
                    self._json(HTTPStatus.OK, self.state.snapshot())
                except Exception as exc:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        elif self.path == "/metrics":
            snap = self.state.snapshot()
            lines = [
                f'chormofold_active_requests {int(snap.get("active_requests", 0))}',
                f'chormofold_page_references {int(snap.get("page_references", 0))}',
                f'chormofold_cuda_errors_total {int(snap.get("cuda_errors", 0))}',
                f'chormofold_dense_fallback_launches_total {int(snap.get("dense_fallback_launches", 0))}',
            ]
            data = ("\n".join(lines) + "\n").encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/qualification/scenario":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"}); return
        if not self._authorized(): return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict): raise ValueError("body must be an object")
            name = str(payload.get("scenario", ""))
            self._json(HTTPStatus.OK, self.state.run_scenario(name, payload))
        except subprocess.TimeoutExpired:
            self._json(HTTPStatus.REQUEST_TIMEOUT, {"status": "FAIL", "error": "scenario timed out"})
        except Exception as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"status": "FAIL", "error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(json.dumps({"event": "qualification_http", "message": fmt % args}))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--release-digest", required=True)
    parser.add_argument("--hooks-dir", type=Path)
    args = parser.parse_args()
    enabled = os.getenv("CHROMOFOLD_QUALIFICATION_MODE") == "1"
    token = os.getenv("CHROMOFOLD_QUALIFICATION_ADMIN_TOKEN", "")
    state = State(enabled, token, args.release_digest, args.snapshot, args.hooks_dir)
    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    server.state = state  # type: ignore[attr-defined]
    server.serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
