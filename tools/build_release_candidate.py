#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_BUILDER = ROOT / "tools" / "build_product_bundle.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("chromofold_bundle", BUNDLE_BUILDER)
    if not spec or not spec.loader:
        raise RuntimeError("unable to load product bundle builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build(version: str, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"chromofold-{version}.tar.gz"
    result = load_builder().build(archive, version)
    manifest = result["manifest"]
    archive_digest = sha256(archive)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"ChromoFold {version}",
        "documentNamespace": f"https://github.com/cyberpunk042/chromoFold/releases/{version}/{archive_digest}",
        "creationInfo": {"created": created, "creators": ["Tool: tools/build_release_candidate.py"]},
        "packages": [{
            "name": "ChromoFold",
            "SPDXID": "SPDXRef-Package-ChromoFold",
            "versionInfo": version,
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
        }],
        "files": [
            {
                "fileName": item["path"],
                "SPDXID": f"SPDXRef-File-{index}",
                "checksums": [{"algorithm": "SHA256", "checksumValue": item["sha256"]}],
            }
            for index, item in enumerate(manifest["files"], start=1)
        ],
    }
    sbom_path = output / f"chromofold-{version}.sbom.spdx.json"
    sbom_path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    provenance = {
        "schema": "chromofold.provenance.v1",
        "version": version,
        "created": created,
        "source_repository": "https://github.com/cyberpunk042/chromoFold",
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_ref": os.environ.get("GITHUB_REF", git_value("branch", "--show-current")),
        "builder": os.environ.get("GITHUB_WORKFLOW", "tools/build_release_candidate.py"),
        "archive": {"name": archive.name, "sha256": archive_digest},
        "qualification": {
            "state": "UNQUALIFIED_CANDIDATE",
            "required_for_production_claim": True,
            "artifact_digest": f"sha256:{archive_digest}",
        },
    }
    provenance_path = output / f"chromofold-{version}.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksums = output / "SHA256SUMS"
    assets = [archive, archive.with_suffix(archive.suffix + ".sha256"), sbom_path, provenance_path]
    checksums.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in assets), encoding="utf-8")
    return {"version": version, "assets": [str(path) for path in [*assets, checksums]], "archive_sha256": archive_digest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a verifiable ChromoFold release candidate")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, default=Path("dist/release"))
    args = parser.parse_args()
    print(json.dumps(build(args.version, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
