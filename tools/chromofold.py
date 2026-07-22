#!/usr/bin/env python3
"""ChromoFold product CLI: inspect, recommend, configure, compare and qualify."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "product" / "profiles.json"
COMPATIBILITY = ROOT / "product" / "compatibility.json"


@dataclass
class Machine:
    os: str
    architecture: str
    cpu: str
    ram_bytes: int
    gpu_name: str | None
    gpu_vram_bytes: int | None
    driver: str | None
    cuda: str | None
    nvlink: bool


def _run(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _ram_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError):
        return 0


def inspect_machine() -> Machine:
    query = _run([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    gpu_name: str | None = None
    gpu_vram: int | None = None
    driver: str | None = None
    if query:
        first = query.splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 3:
            gpu_name, driver = parts[0], parts[2]
            try:
                gpu_vram = int(float(parts[1]) * 1024 * 1024)
            except ValueError:
                gpu_vram = None
    cuda = _run(["nvcc", "--version"])
    cuda_line = next((line.strip() for line in cuda.splitlines() if "release" in line), None)
    topo = _run(["nvidia-smi", "topo", "-m"])
    return Machine(
        os=platform.platform(),
        architecture=platform.machine(),
        cpu=platform.processor() or platform.uname().processor,
        ram_bytes=_ram_bytes(),
        gpu_name=gpu_name,
        gpu_vram_bytes=gpu_vram,
        driver=driver,
        cuda=cuda_line,
        nvlink="NV" in topo,
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recommend(machine: Machine, goal: str, model_bytes: int | None, context: int, concurrency: int) -> dict[str, Any]:
    profiles = load_json(PROFILES)["profiles"]
    vram = machine.gpu_vram_bytes or 0
    pressure = "unknown"
    if model_bytes and vram:
        estimated_kv = max(0, context * concurrency * 256 * 1024)
        ratio = (model_bytes + estimated_kv) / vram
        pressure = "critical" if ratio > 0.95 else "high" if ratio > 0.75 else "moderate" if ratio > 0.5 else "low"
    mapping = {
        "longer-context": "maximum-context",
        "more-users": "high-concurrency",
        "shared-prefix": "shared-prefix",
        "lowest-risk": "safe",
        "balanced": "balanced",
    }
    selected = mapping[goal]
    if pressure == "low" and goal not in {"shared-prefix", "more-users"}:
        selected = "safe"
    profile = profiles[selected]
    return {
        "profile": selected,
        "confidence": "medium" if vram else "low",
        "primary_bottleneck": "KV-cache capacity" if goal in {"longer-context", "more-users"} else "workload dependent",
        "memory_pressure": pressure,
        "why": profile["why"],
        "risks": profile["risks"],
        "qualification_required": True,
        "estimate_notice": "Capacity guidance is an estimate until measured on this machine.",
    }


def render_config(profile_name: str, runtime: str, model: Path | None, output: Path) -> list[Path]:
    profile = load_json(PROFILES)["profiles"][profile_name]
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "chromofold.config.v1",
        "profile": profile_name,
        "runtime": runtime,
        "model": str(model) if model else None,
        "kv_cache": profile["kv_cache"],
        "weights": profile["weights"],
        "fallback": {"silent": False, "unsupported": "fail"},
        "qualification": {"required": True, "mode": "smoke"},
    }
    config_path = output / "chromofold.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = ["llama-server", "--kv-cache-backend", "chromofold"]
    if model:
        command += ["-m", str(model)]
    command += ["--ctx-size", str(profile["context_target"])]
    launch = output / "run-chromofold.sh"
    launch.write_text("#!/usr/bin/env sh\nset -eu\nexec " + " ".join(command) + ' "$@"\n', encoding="utf-8")
    launch.chmod(0o755)
    manifest = output / "bundle-manifest.json"
    manifest.write_text(json.dumps({
        "schema": "chromofold.bundle.v1",
        "profile": profile_name,
        "runtime": runtime,
        "files": [config_path.name, launch.name],
        "commands": {
            "inspect": "python3 tools/chromofold.py inspect",
            "run": f"{launch}",
            "qualify": "python3 tools/chromofold.py qualify --mode smoke --artifact qualification/evidence.json",
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [config_path, launch, manifest]


def compare_results(baseline: Path, candidate: Path) -> dict[str, Any]:
    left, right = load_json(baseline), load_json(candidate)
    keys = ["peak_vram_bytes", "ttft_p95_ms", "tokens_per_second", "max_context", "max_concurrency"]
    values: dict[str, Any] = {}
    for key in keys:
        a, b = left.get(key), right.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            values[key] = {"baseline": a, "chromofold": b, "delta_percent": None if a == 0 else ((b - a) / a) * 100}
    capacity_gain = any(
        values.get(key, {}).get("chromofold", 0) > values.get(key, {}).get("baseline", 0)
        for key in ("max_context", "max_concurrency")
    )
    values["decision"] = "WORKLOAD_WIN" if capacity_gain else "NO_PROVEN_WIN"
    values["warning"] = "Results are comparable only when model, hardware, runtime and workload fingerprints match."
    return values


def cmd_inspect(args: argparse.Namespace) -> int:
    data = asdict(inspect_machine())
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    machine = inspect_machine()
    model_bytes = args.model.stat().st_size if args.model and args.model.is_file() else None
    print(json.dumps(recommend(machine, args.goal, model_bytes, args.context, args.concurrency), indent=2, sort_keys=True))
    return 0


def cmd_configure(args: argparse.Namespace) -> int:
    paths = render_config(args.profile, args.runtime, args.model, args.output)
    print(json.dumps({"generated": [str(path) for path in paths]}, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    print(json.dumps(compare_results(args.baseline, args.chromofold), indent=2, sort_keys=True))
    return 0


def cmd_qualify(args: argparse.Namespace) -> int:
    command = [sys.executable, str(ROOT / "tools" / "chromofold_runtime_harness.py"), "--mode", args.mode,
               "--artifact", str(args.artifact), "--release-digest", args.release_digest]
    if args.model:
        command += ["--model", str(args.model)]
    if args.runtime_endpoint:
        command += ["--runtime-endpoint", args.runtime_endpoint]
    return subprocess.call(command)


def cmd_catalog(_: argparse.Namespace) -> int:
    print(json.dumps({"profiles": load_json(PROFILES), "compatibility": load_json(COMPATIBILITY)}, indent=2, sort_keys=True))
    return 0


def _json_flag(command: argparse.ArgumentParser) -> None:
    """Accept the stable product-consumer flag; output is JSON regardless."""
    command.add_argument("--json", action="store_true", help=argparse.SUPPRESS)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chromofold", description="Analyze, configure and qualify ChromoFold workloads")
    sub = p.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Inspect local hardware")
    inspect.add_argument("--output", type=Path)
    _json_flag(inspect)
    inspect.set_defaults(func=cmd_inspect)
    rec = sub.add_parser("recommend", help="Recommend a safe profile")
    rec.add_argument("--goal", choices=["longer-context", "more-users", "shared-prefix", "lowest-risk", "balanced"], default="balanced")
    rec.add_argument("--model", type=Path)
    rec.add_argument("--context", type=int, default=32768)
    rec.add_argument("--concurrency", type=int, default=1)
    _json_flag(rec)
    rec.set_defaults(func=cmd_recommend)
    config = sub.add_parser("configure", help="Generate a runtime bundle")
    config.add_argument("--profile", choices=["safe", "balanced", "maximum-context", "high-concurrency", "shared-prefix"], default="balanced")
    config.add_argument("--runtime", choices=["llama.cpp"], default="llama.cpp")
    config.add_argument("--model", type=Path)
    config.add_argument("--output", type=Path, default=Path("chromofold-bundle"))
    _json_flag(config)
    config.set_defaults(func=cmd_configure)
    compare = sub.add_parser("compare", help="Compare baseline and ChromoFold result JSON")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--chromofold", type=Path, required=True)
    _json_flag(compare)
    compare.set_defaults(func=cmd_compare)
    qualify = sub.add_parser("qualify", help="Run the repository qualification harness")
    qualify.add_argument("--mode", choices=["smoke", "candidate", "stable"], default="smoke")
    qualify.add_argument("--artifact", type=Path, required=True)
    qualify.add_argument("--release-digest", required=True)
    qualify.add_argument("--model", type=Path)
    qualify.add_argument("--runtime-endpoint")
    qualify.set_defaults(func=cmd_qualify)
    catalog = sub.add_parser("catalog", help="Show profiles and compatibility")
    _json_flag(catalog)
    catalog.set_defaults(func=cmd_catalog)
    return p


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
