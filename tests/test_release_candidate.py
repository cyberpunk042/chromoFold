#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_release_channel_contract() -> None:
    channel = json.loads((ROOT / "product/release-channel.json").read_text(encoding="utf-8"))
    assert channel["schema"] == "chromofold.release-channel.v1"
    assert channel["qualification"]["digest_bound"] is True
    for key in ("archive", "checksum", "sbom", "provenance"):
        assert key in channel["artifact_contract"]


def test_candidate_contains_sbom_provenance_and_checksums() -> None:
    builder = load("release_candidate", ROOT / "tools/build_release_candidate.py")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        result = builder.build("v0.0.0-test", output)
        names = {Path(path).name for path in result["assets"]}
        assert "chromofold-v0.0.0-test.tar.gz" in names
        assert "chromofold-v0.0.0-test.sbom.spdx.json" in names
        assert "chromofold-v0.0.0-test.provenance.json" in names
        assert "SHA256SUMS" in names
        provenance = json.loads((output / "chromofold-v0.0.0-test.provenance.json").read_text(encoding="utf-8"))
        assert provenance["qualification"]["state"] == "UNQUALIFIED_CANDIDATE"
        assert provenance["qualification"]["artifact_digest"].startswith("sha256:")
        sbom = json.loads((output / "chromofold-v0.0.0-test.sbom.spdx.json").read_text(encoding="utf-8"))
        assert sbom["spdxVersion"] == "SPDX-2.3"
        with tarfile.open(output / "chromofold-v0.0.0-test.tar.gz", "r:gz") as archive:
            assert any(name.endswith("manifest.json") for name in archive.getnames())


def test_workflow_never_calls_candidate_qualified() -> None:
    workflow = (ROOT / ".github/workflows/release-candidate.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "prerelease: true" in workflow
    assert "sha256sum --check SHA256SUMS" in workflow
    assert "not production-qualified" in workflow


if __name__ == "__main__":
    test_release_channel_contract()
    test_candidate_contains_sbom_provenance_and_checksums()
    test_workflow_never_calls_candidate_qualified()
    print("ChromoFold release candidate tests: PASS")
