#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("index.html", "technology.html", "evidence.html", "compatibility.html", "start.html", "releases.html", "contribute.html")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_public_claims_are_explicitly_bounded() -> None:
    catalog = json.loads((ROOT / "product/public-claims.json").read_text(encoding="utf-8"))
    assert catalog["schema"] == "chromofold.public-claims.v1"
    assert "Production claims require" in catalog["boundary"]
    for claim in catalog["claims"]:
        assert claim["evidence_level"] in {"estimate", "measured", "qualified", "independently-reproduced"}
        assert claim["hardware"] and claim["scope"]


def test_portal_sources_are_versioned() -> None:
    compatibility = json.loads((ROOT / "product/compatibility.json").read_text(encoding="utf-8"))
    profiles = json.loads((ROOT / "product/profiles.json").read_text(encoding="utf-8"))
    portal = json.loads((ROOT / "product/portal.json").read_text(encoding="utf-8"))
    assert compatibility["schema"] == "chromofold.compatibility.v1"
    assert profiles["schema"] == "chromofold.profiles.v1"
    assert portal["schema"] == "chromofold.portal.v1"
    assert portal["reproduction_fields"] and portal["contribution_paths"]


def test_pages_preserve_evidence_boundaries() -> None:
    text = "\n".join((ROOT / "site" / page).read_text(encoding="utf-8") for page in PAGES)
    assert "exact-digest" in text.lower() or "exact artifact" in text.lower()
    assert "guaranteed" not in text.lower()
    assert "works on every" not in text.lower()
    assert "not a benchmark" in text.lower()


def test_builder_generates_complete_portal() -> None:
    builder = load("build_public_site", ROOT / "tools/build_public_site.py")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "site"
        result = builder.build(output)
        assert result["pages"] == list(PAGES)
        for name in (*PAGES, "404.html", "styles.css", "app.js", "data.js", "robots.txt", "sitemap.xml", ".nojekyll"):
            assert (output / name).is_file(), name
        for page in PAGES:
            html = (output / page).read_text(encoding="utf-8")
            assert '<script src="data.js"></script>' in html
            assert '<script src="app.js" defer></script>' in html
        data = (output / "data.js").read_text(encoding="utf-8")
        assert "chromofold.public-site-data.v3" in data
        for key in ("release_channel", "compatibility", "profiles", "portal"):
            assert key in data
        sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
        for page in PAGES[1:]:
            assert page in sitemap


def test_client_runtime_is_safe_and_route_aware() -> None:
    source = (ROOT / "site/app.js").read_text(encoding="utf-8")
    assert "const esc" in source
    assert "qualification_required:true" in source
    assert "renderCompatibility" in source
    assert "renderEvidenceLadder" in source
    assert "navigator.clipboard" in source


def test_manual_dispatch_deploys_pages() -> None:
    workflow = (ROOT / ".github/workflows/public-site.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert workflow.count("if: github.event_name != 'pull_request'") == 2
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "actions/deploy-pages@v4" in workflow
    for path in ("product/compatibility.json", "product/profiles.json", "product/portal.json"):
        assert workflow.count(path) == 2


if __name__ == "__main__":
    test_public_claims_are_explicitly_bounded()
    test_portal_sources_are_versioned()
    test_pages_preserve_evidence_boundaries()
    test_builder_generates_complete_portal()
    test_client_runtime_is_safe_and_route_aware()
    test_manual_dispatch_deploys_pages()
    print("ChromoFold product portal tests: PASS")