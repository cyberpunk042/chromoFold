#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_manifest(manifest_path: pathlib.Path, root: pathlib.Path) -> None:
    manifest = load_json(manifest_path)
    required = ["release_version", "git_commit", "protocol_version", "codec_version", "artifacts"]
    missing = [name for name in required if name not in manifest]
    if missing:
        raise ValueError(f"missing release fields: {', '.join(missing)}")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifacts must be a non-empty list")
    seen: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("artifact entries must be objects")
        path_text = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(path_text, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("artifact path and 64-character sha256 are required")
        if path_text in seen:
            raise ValueError(f"duplicate artifact: {path_text}")
        seen.add(path_text)
        path = (root / path_text).resolve()
        if root.resolve() not in path.parents and path != root.resolve():
            raise ValueError(f"artifact escapes release root: {path_text}")
        if not path.is_file():
            raise ValueError(f"artifact missing: {path_text}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"hash mismatch for {path_text}: expected {expected}, got {actual}")
    if not manifest.get("sbom") or not manifest.get("provenance"):
        raise ValueError("release manifest must reference SBOM and provenance")


def generate_sbom(root: pathlib.Path, output: pathlib.Path) -> None:
    components = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        components.append({
            "type": "file",
            "name": path.relative_to(root).as_posix(),
            "hashes": [{"alg": "SHA-256", "content": sha256_file(path)}],
        })
    document = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": components,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compatibility(manifest_path: pathlib.Path, protocol: int, codec: int, checkpoint: int) -> None:
    manifest = load_json(manifest_path)
    actual = (
        int(manifest.get("protocol_version", -1)),
        int(manifest.get("codec_version", -1)),
        int(manifest.get("checkpoint_version", -1)),
    )
    requested = (protocol, codec, checkpoint)
    if actual != requested:
        raise ValueError(f"incompatible release: release={actual}, requested={requested}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="chromofold-release")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("manifest", type=pathlib.Path)
    verify.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))

    sbom = subparsers.add_parser("sbom")
    sbom.add_argument("root", type=pathlib.Path)
    sbom.add_argument("output", type=pathlib.Path)

    compat = subparsers.add_parser("compatibility")
    compat.add_argument("manifest", type=pathlib.Path)
    compat.add_argument("--protocol", type=int, required=True)
    compat.add_argument("--codec", type=int, required=True)
    compat.add_argument("--checkpoint", type=int, required=True)

    args = parser.parse_args()
    try:
        if args.command == "verify":
            verify_manifest(args.manifest, args.root)
        elif args.command == "sbom":
            generate_sbom(args.root, args.output)
        else:
            compatibility(args.manifest, args.protocol, args.codec, args.checkpoint)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
