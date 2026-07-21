#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def run(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--pin", type=Path, default=Path(__file__).with_name("llama-pin.json"))
    args = parser.parse_args()
    pin = json.loads(args.pin.read_text())
    checkout = args.checkout.resolve()
    if not (checkout / ".git").exists():
        raise SystemExit(f"not a git checkout: {checkout}")
    actual = run(checkout, "rev-parse", "HEAD")
    if actual != pin["commit"]:
        raise SystemExit(f"llama.cpp revision mismatch: expected {pin['commit']}, got {actual}")
    dirty = run(checkout, "status", "--porcelain")
    if dirty:
        raise SystemExit("llama.cpp checkout must be clean before patching")
    missing = [path for path in pin["required_paths"] if not (checkout / path).is_file()]
    if missing:
        raise SystemExit("missing pinned paths: " + ", ".join(missing))
    kv_source = (checkout / "src/llama-kv-cache.cpp").read_text(errors="replace")
    for token in ("seq_rm", "seq_cp", "seq_keep", "get_size"):
        if token not in kv_source:
            raise SystemExit(f"required KV capability anchor missing: {token}")
    print(json.dumps({"commit": actual, "clean": True, "required_paths": len(pin["required_paths"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
