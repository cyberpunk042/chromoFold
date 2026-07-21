#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "patch_manifest.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--require-applied", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    actual = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    if actual != MANIFEST["llama_commit"]:
        raise SystemExit("pinned commit mismatch")
    for path in MANIFEST["required_upstream_files"]:
        if not (checkout / path).is_file():
            raise SystemExit(f"missing upstream path: {path}")
    if not args.require_applied:
        print(json.dumps({"commit": actual, "applicable": True}))
        return 0
    state_path = checkout / ".chromofold-runtime-state.json"
    if not state_path.is_file():
        raise SystemExit("runtime patch state is missing")
    state = json.loads(state_path.read_text())
    if state.get("commit") != actual:
        raise SystemExit("runtime patch state commit mismatch")
    for path, expected in state.get("generated", {}).items():
        target = checkout / path
        if not target.is_file() or sha(target) != expected:
            raise SystemExit(f"generated runtime file drifted: {path}")
    cmake = (checkout / "CMakeLists.txt").read_text()
    if cmake.count("# BEGIN CHROMOFOLD RUNTIME") != 1 or cmake.count("# END CHROMOFOLD RUNTIME") != 1:
        raise SystemExit("top-level CMake hook is missing or duplicated")
    if not (checkout / "chromofold-overlay/PINNED_COMMIT").is_file():
        raise SystemExit("overlay pin marker is missing")
    print(json.dumps({"commit": actual, "applied": True, "generated_files": len(state["generated"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
