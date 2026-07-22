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


def test_static_assets_exist() -> None:
    for name in ("index.html", "app.js", "styles.css"):
        path = ROOT / "hub" / "static" / name
        assert path.is_file() and path.stat().st_size > 100


def test_catalogs_are_json() -> None:
    for name in ("profiles.json", "compatibility.json", "downloads.json", "evidence-registry.json"):
        value = json.loads((ROOT / "product" / name).read_text(encoding="utf-8"))
        assert isinstance(value, dict)


def test_bundle_has_manifest_and_launcher() -> None:
    builder = load("build_product_bundle", ROOT / "tools" / "build_product_bundle.py")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "chromofold-test.tar.gz"
        result = builder.build(output, "test")
        assert output.is_file()
        assert output.with_suffix(".gz.sha256").is_file()
        assert result["sha256"]
        with tarfile.open(output, "r:gz") as archive:
            names = archive.getnames()
        assert "chromofold-test/manifest.json" in names
        assert "chromofold-test/chromofold-hub" in names
        assert "chromofold-test/hub/static/index.html" in names


def test_hub_binds_loopback_by_default() -> None:
    source = (ROOT / "hub" / "server.py").read_text(encoding="utf-8")
    assert 'default="127.0.0.1"' in source
    assert '"/api/catalog"' in source
    assert '"/api/recommend"' in source


if __name__ == "__main__":
    test_static_assets_exist()
    test_catalogs_are_json()
    test_bundle_has_manifest_and_launcher()
    test_hub_binds_loopback_by_default()
    print("ChromoFold Hub tests: PASS")
