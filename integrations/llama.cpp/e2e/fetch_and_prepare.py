#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIN = json.loads((ROOT / "llama-pin.json").read_text())


def call(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--chromofold-root", type=Path, default=ROOT.parents[2])
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    if args.reset and checkout.exists():
        shutil.rmtree(checkout)
    if not checkout.exists():
        call("git", "clone", "--filter=blob:none", PIN["repository"], str(checkout))
    call("git", "fetch", "origin", PIN["commit"], "--depth=1", cwd=checkout)
    call("git", "checkout", "--detach", PIN["commit"], cwd=checkout)
    call("python3", str(ROOT / "verify_upstream.py"), str(checkout))

    overlay = checkout / "chromofold-overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    overlay.mkdir()
    source_root = args.chromofold_root.resolve()
    for relative in (
        "include/chromofold",
        "src/runtime",
        "src/cuda",
        "integrations/llama.cpp/chromofold_kv_adapter.h",
        "integrations/llama.cpp/chromofold_kv_adapter.cpp",
    ):
        source = source_root / relative
        destination = overlay / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    (overlay / "PINNED_COMMIT").write_text(PIN["commit"] + "\n")
    print(f"prepared {checkout} at {PIN['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
