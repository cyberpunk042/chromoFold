#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "graph_patch_manifest.json").read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    state_path = checkout / ".chromofold-graph-state.json"
    if not state_path.is_file():
        raise SystemExit("graph wiring state is missing")
    state = json.loads(state_path.read_text())
    if state.get("commit") != MANIFEST["llama_commit"]:
        raise SystemExit("graph wiring was generated for a different llama.cpp revision")
    for relative, expected in state.get("generated", {}).items():
        path = checkout / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"generated graph file drift: {relative}")
    combined = "\n".join((checkout / path).read_text(errors="replace") for path in (
        "common/arg.cpp", "src/llama-context.cpp", "src/llama-graph.cpp", "src/llama-kv-cache.cpp"))
    missing = [symbol for symbol in MANIFEST["required_symbols"] if symbol not in combined and symbol not in "\n".join(state.get("required_symbols", []))]
    if missing:
        raise SystemExit("missing graph symbols: " + ", ".join(missing))
    if (checkout / "CMakeLists.txt").read_text().count("include(cmake/chromofold-graph.cmake)") != 1:
        raise SystemExit("graph CMake hook must appear exactly once")
    print(json.dumps({"commit": state["commit"], "verified": True, "generated_files": len(state["generated"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
