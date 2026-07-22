#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
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


def test_public_claims_are_explicitly_bounded() -> None:
    catalog = json.loads((ROOT / "product/public-claims.json").read_text(encoding="utf-8"))
    assert catalog["schema"] == "chromofold.public-claims.v1"
    assert "Production claims require" in catalog["boundary"]
    assert catalog["claims"]
    for claim in catalog["claims"]:
        assert claim["evidence_level"] in {"estimate", "measured", "qualified", "independently-reproduced"}
        assert claim["hardware"]
        assert claim["scope"]


def test_site_avoids_unqualified_universal_claims() -> None:
    html = (ROOT / "site/index.html").read_text(encoding="utf-8")
    assert "Measured results with visible boundaries" in html
    assert "exact release digest" in html
    assert "guaranteed" not in html.lower()
    assert "works on every" not in html.lower()
    assert "No values leave the browser" in html


def test_builder_generates_complete_static_site() -> None:
    builder = load("build_public_site", ROOT / "tools/build_public_site.py")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "site"
        result = builder.build(output)
        assert result["output"] == str(output)
        for name in ("index.html", "404.html", "styles.css", "app.js", "data.js", "robots.txt", "sitemap.xml", ".nojekyll"):
            assert (output / name).is_file()
        html = (output / "index.html").read_text(encoding="utf-8")
        assert '<script src="data.js"></script>' in html
        data = (output / "data.js").read_text(encoding="utf-8")
        assert "chromofold.public-site-data.v2" in data
        assert "release_channel" in data
        assert "public-claims" not in data


def test_estimator_is_labeled_advisory_and_escapes_output() -> None:
    source = (ROOT / "site/app.js").read_text(encoding="utf-8")
    assert 'evidence_level: "estimate"' in source
    assert "not a benchmark" in source
    assert "qualification_required: true" in source
    assert "function esc" in source


def test_manual_dispatch_deploys_pages() -> None:
    workflow = (ROOT / ".github/workflows/public-site.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert workflow.count("if: github.event_name != 'pull_request'") == 2
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "actions/deploy-pages@v4" in workflow


if __name__ == "__main__":
    test_public_claims_are_explicitly_bounded()
    test_site_avoids_unqualified_universal_claims()
    test_builder_generates_complete_static_site()
    test_estimator_is_labeled_advisory_and_escapes_output()
    test_manual_dispatch_deploys_pages()
    print("ChromoFold public site tests: PASS")
