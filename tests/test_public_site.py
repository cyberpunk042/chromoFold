#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("index.html", "technology.html", "evidence.html", "compatibility.html", "start.html", "workbench.html", "sessions.html", "releases.html", "contribute.html")


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
    scaling = json.loads((ROOT / "product/kv-scaling.json").read_text(encoding="utf-8"))
    native = json.loads((ROOT / "product/native-kv-performance.json").read_text(encoding="utf-8"))
    campaign = json.loads((ROOT / "product/kv-crossover-campaign.json").read_text(encoding="utf-8"))
    evidence_schema = json.loads((ROOT / "product/evidence-result.schema.json").read_text(encoding="utf-8"))
    session_schema = json.loads((ROOT / "product/qualification-session.schema.json").read_text(encoding="utf-8"))
    assert compatibility["schema"] == "chromofold.compatibility.v1"
    assert profiles["schema"] == "chromofold.profiles.v1"
    assert portal["schema"] == "chromofold.portal.v1"
    assert scaling["schema"] == "chromofold.kv-scaling.v1"
    assert native["schema"] == "chromofold.native-kv-performance.v1"
    assert campaign["schema"] == "chromofold.kv-crossover-campaign.v1"
    assert native["headline"]["latency_win"] is True
    assert native["headline"]["universal_latency_win"] is False
    assert native["headline"]["ship_criterion_met_in_regime"] is True
    assert native["history"][-1]["gap_vs_dense"] > 1
    crossover = native["crossover_sweep"]
    assert crossover["status"] == "measured"
    assert any(row["head_dim"] == 64 and max(row["dense_over_compressed"]) >= 1.02 for row in crossover["rows"])
    assert any(row["head_dim"] == 128 and max(row["dense_over_compressed"]) < 1 for row in crossover["rows"])
    assert campaign["status"] == "initial-sweep-measured"
    assert "NO_CROSSOVER" in campaign["publication_states"]
    assert evidence_schema["properties"]["schema"]["const"] == "chromofold.evidence-result.v1"
    assert session_schema["properties"]["schema"]["const"] == "chromofold.qualification-session.v1"
    assert session_schema["properties"]["baseline_runs"]["minItems"] == 3
    assert len(scaling["context_tokens"]) == len(scaling["latency_ms"]["decode_all"])
    assert "prototype" in scaling["boundary"].lower()
    assert portal["reproduction_fields"] and portal["contribution_paths"]


def test_pages_preserve_evidence_boundaries() -> None:
    text = "\n".join((ROOT / "site" / page).read_text(encoding="utf-8") for page in PAGES)
    assert "exact-digest" in text.lower() or "exact artifact" in text.lower()
    assert "guaranteed" not in text.lower()
    assert "works on every" not in text.lower()
    evidence = (ROOT / "site/evidence.html").read_text(encoding="utf-8")
    for marker in ('id="kv-chart"', 'id="m11-history-chart"', 'id="fair-performance-table"', 'id="campaign-summary"'):
        assert marker in evidence
    assert "does not mean faster than fair dense" in evidence
    assert '<script src="performance.js" defer></script>' in evidence
    workbench = (ROOT / "site/workbench.html").read_text(encoding="utf-8")
    assert "not maintainer qualification" in workbench
    sessions = (ROOT / "site/sessions.html").read_text(encoding="utf-8")
    assert "at least three" in sessions.lower()
    assert "Session PASS" in sessions


def test_builder_generates_complete_portal() -> None:
    builder = load("build_public_site", ROOT / "tools/build_public_site.py")
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "site"
        result = builder.build(output)
        assert result["pages"] == list(PAGES)
        assets = ("styles.css", "app.js", "performance.js", "workbench.js", "workbench.css", "sessions.js", "sessions.css", "data.js", "robots.txt", "sitemap.xml", ".nojekyll")
        for name in (*PAGES, "404.html", *assets):
            assert (output / name).is_file(), name
        data = (output / "data.js").read_text(encoding="utf-8")
        assert "chromofold.public-site-data.v7" in data
        for key in ("kv_scaling", "native_kv_performance", "kv_crossover_campaign", "evidence_result_schema", "qualification_session_schema"):
            assert key in data
        sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
        for page in PAGES[1:]:
            assert page in sitemap


def test_client_runtime_is_safe_and_route_aware() -> None:
    source = (ROOT / "site/app.js").read_text(encoding="utf-8")
    performance = (ROOT / "site/performance.js").read_text(encoding="utf-8")
    assert "const esc" in source
    assert "renderKvScaling" in source
    assert "historyChart" in performance
    assert "current_gap_vs_fair_dense" in performance
    assert "crossover_sweep" in performance
    assert "fetch(" not in performance and "XMLHttpRequest" not in performance


def test_manual_dispatch_deploys_pages() -> None:
    workflow = (ROOT / ".github/workflows/public-site.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert workflow.count("if: github.event_name != 'pull_request'") == 2
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "actions/deploy-pages@v4" in workflow
    for path in ("product/native-kv-performance.json", "product/kv-crossover-campaign.json"):
        assert workflow.count(path) == 2
    assert "tools/kv_crossover_campaign.py" in workflow
    assert "dist/site/performance.js" in workflow
    assert "chromofold.public-site-data.v7" in workflow


if __name__ == "__main__":
    test_public_claims_are_explicitly_bounded()
    test_portal_sources_are_versioned()
    test_pages_preserve_evidence_boundaries()
    test_builder_generates_complete_portal()
    test_client_runtime_is_safe_and_route_aware()
    test_manual_dispatch_deploys_pages()
    print("ChromoFold product portal tests: PASS")
