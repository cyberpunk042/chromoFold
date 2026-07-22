#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = archive.gettarinfo(str(source), arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def build(output: Path, version: str) -> dict[str, object]:
    staging = output.parent / f"chromofold-{version}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    selected = [
        "tools/chromofold.py",
        "tools/chromofold_runtime_harness.py",
        "tools/chromofold_qualification_adapter.py",
        "hub/server.py",
        "hub/static/index.html",
        "hub/static/app.js",
        "hub/static/styles.css",
        "product/profiles.json",
        "product/compatibility.json",
        "product/downloads.json",
        "product/evidence-registry.json",
        "docs/PRODUCT.md",
        "docs/HUB.md",
    ]
    files: list[dict[str, object]] = []
    for relative in selected:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.suffix == ".py":
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
        files.append({"path": relative, "sha256": sha256(target), "size": target.stat().st_size})
    launcher = staging / "chromofold-hub"
    launcher.write_text("#!/bin/sh\nexec python3 \"$(dirname \"$0\")/hub/server.py\" \"$@\"\n", encoding="utf-8")
    launcher.chmod(0o755)
    files.append({"path": "chromofold-hub", "sha256": sha256(launcher), "size": launcher.stat().st_size})
    manifest = {"schema": "chromofold.product-bundle.v1", "version": version, "files": files}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                add_file(archive, path, str(Path(staging.name) / path.relative_to(staging)))
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    return {"bundle": str(output), "sha256": sha256(output), "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.version), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
