#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PRODUCT = ROOT / "product"
PUBLIC_URL = "https://cyberpunk042.github.io/chromoFold/"
PAGES = (
    "index.html", "technology.html", "evidence.html", "compatibility.html",
    "start.html", "workbench.html", "sessions.html", "releases.html", "contribute.html",
)
ASSETS = ("styles.css", "app.js", "workbench.js", "workbench.css", "sessions.js", "sessions.css")


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def normalized_downloads(catalog: dict[str, Any]) -> list[dict[str, str]]:
    raw = catalog.get("downloads") or catalog.get("bundles") or catalog
    if not isinstance(raw, dict):
        return []
    result: list[dict[str, str]] = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        result.append({
            "name": str(value.get("name") or key),
            "description": str(value.get("description") or value.get("purpose") or "ChromoFold release asset"),
            "status": str(value.get("status") or "candidate"),
            "href": "https://github.com/cyberpunk042/chromoFold/releases",
            "action": "Open releases",
        })
    return result or [{
        "name": "ChromoFold product bundle",
        "description": "Local Hub, Assistant, CLI, installers, qualification tools and product registries.",
        "status": "release asset",
        "href": "https://github.com/cyberpunk042/chromoFold/releases",
        "action": "Open releases",
    }]


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name in (*PAGES, *ASSETS):
        source = SITE / name
        if not source.is_file():
            raise FileNotFoundError(name)
        shutil.copy2(source, output / name)

    claims = load_object(PRODUCT / "public-claims.json")
    payload = {
        "schema": "chromofold.public-site-data.v6",
        "claims": claims["claims"],
        "claim_boundary": claims["boundary"],
        "downloads": normalized_downloads(load_object(PRODUCT / "downloads.json")),
        "evidence": load_object(PRODUCT / "evidence-registry.json"),
        "release_channel": load_object(PRODUCT / "release-channel.json"),
        "compatibility": load_object(PRODUCT / "compatibility.json"),
        "profiles": load_object(PRODUCT / "profiles.json"),
        "portal": load_object(PRODUCT / "portal.json"),
        "kv_scaling": load_object(PRODUCT / "kv-scaling.json"),
        "evidence_result_schema": load_object(PRODUCT / "evidence-result.schema.json"),
        "qualification_session_schema": load_object(PRODUCT / "qualification-session.schema.json"),
        "repository": "https://github.com/cyberpunk042/chromoFold",
    }
    (output / "data.js").write_text(
        "window.CHROMOFOLD_SITE_DATA = " + json.dumps(payload, indent=2, sort_keys=True) + ";\n",
        encoding="utf-8",
    )

    for page in PAGES:
        path = output / page
        html = path.read_text(encoding="utf-8")
        if '<script src="data.js"></script>' not in html:
            html = html.replace('<script src="app.js" defer></script>', '<script src="data.js"></script>\n<script src="app.js" defer></script>')
        path.write_text(html, encoding="utf-8")

    index = (output / "index.html").read_text(encoding="utf-8")
    (output / "404.html").write_text(index, encoding="utf-8")
    (output / "robots.txt").write_text(f"User-agent: *\nAllow: /\nSitemap: {PUBLIC_URL}sitemap.xml\n", encoding="utf-8")
    urls = "\n".join(f"  <url><loc>{PUBLIC_URL}{'' if page == 'index.html' else page}</loc></url>" for page in PAGES)
    (output / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n</urlset>\n",
        encoding="utf-8",
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return {"output": str(output), "pages": list(PAGES), "files": sorted(path.name for path in output.iterdir())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ChromoFold public product portal")
    parser.add_argument("--output", type=Path, default=Path("dist/site"))
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
