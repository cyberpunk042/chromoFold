#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "hub" / "static"
PRODUCT = ROOT / "product"
CLI = ROOT / "tools" / "chromofold.py"
ASSISTANT = ROOT / "tools" / "chromofold_assistant.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_json_program(program: Path, args: list[str], stdin: dict[str, Any] | None = None) -> tuple[int, Any]:
    proc = subprocess.run(
        [sys.executable, str(program), *args], cwd=ROOT, text=True,
        input=None if stdin is None else json.dumps(stdin), capture_output=True,
        check=False, timeout=120,
    )
    try:
        payload: Any = json.loads(proc.stdout or proc.stderr or "{}")
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    return proc.returncode, payload


def run_cli(args: list[str]) -> tuple[int, Any]:
    return run_json_program(CLI, args)


class Handler(BaseHTTPRequestHandler):
    server_version = "ChromoFoldHub/2"

    def _json(self, status: int, body: Any) -> None:
        data = json.dumps(body, indent=2, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("request body exceeds 1 MiB")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/catalog":
            self._json(HTTPStatus.OK, {
                "profiles": read_json(PRODUCT / "profiles.json"),
                "compatibility": read_json(PRODUCT / "compatibility.json"),
                "downloads": read_json(PRODUCT / "downloads.json"),
                "evidence": read_json(PRODUCT / "evidence-registry.json"),
                "assistant": read_json(PRODUCT / "assistant-intents.json"),
            })
            return
        if path == "/api/inspect":
            code, payload = run_cli(["inspect", "--json"])
            self._json(HTTPStatus.OK if code == 0 else HTTPStatus.BAD_GATEWAY, payload)
            return
        self._static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/assistant":
                code, payload = run_json_program(ASSISTANT, [], body)
                self._json(HTTPStatus.OK if code == 0 else HTTPStatus.BAD_REQUEST, payload)
                return
            if path == "/api/recommend":
                args = ["recommend", "--goal", str(body.get("goal", "balanced")), "--json"]
                for key, flag in (("context", "--context"), ("concurrency", "--concurrency"), ("model", "--model")):
                    if body.get(key) not in (None, ""):
                        args += [flag, str(body[key])]
                code, payload = run_cli(args)
                self._json(HTTPStatus.OK if code == 0 else HTTPStatus.BAD_REQUEST, payload)
                return
            if path == "/api/configure":
                output = ROOT / "dist" / "hub-config"
                args = ["configure", "--profile", str(body.get("profile", "balanced")), "--output", str(output), "--json"]
                if body.get("model"):
                    args += ["--model", str(body["model"])]
                code, payload = run_cli(args)
                self._json(HTTPStatus.OK if code == 0 else HTTPStatus.BAD_REQUEST, payload)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (STATIC / relative).resolve()
        if STATIC.resolve() not in target.parents and target != STATIC.resolve():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        if not target.is_file():
            target = STATIC / "index.html"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(json.dumps({"event": "hub_http", "message": fmt % args}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local ChromoFold Hub")
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.listen, args.port), Handler)
    print(f"ChromoFold Hub: http://{args.listen}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
