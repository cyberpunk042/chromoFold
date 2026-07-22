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
            "href": "https://github.com/cyberpunk042/chromoFold/releases/latest",
            "action": "Open latest release",
        })
    if not result:
        result.append({
            "name": "ChromoFold product bundle",
            "description": "Local Hub, Assistant, CLI, installers, qualification tools and product registries.",
            "status": "release asset",
            "href": "https://github.com/cyberpunk042/chromoFold/releases/latest",
            "action": "Open latest release",
        })
    return result


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(SITE / name, output / name)

    claims = load_object(PRODUCT / "public-claims.json")
    downloads = load_object(PRODUCT / "downloads.json")
    evidence = load_object(PRODUCT / "evidence-registry.json")
    payload = {
        "schema": "chromofold.public-site-data.v1",
        "claims": claims["claims"],
        "claim_boundary": claims["boundary"],
        "downloads": normalized_downloads(downloads),
        "evidence": evidence,
        "repository": "https://github.com/cyberpunk042/chromoFold",
    }
    data = "window.CHROMOFOLD_SITE_DATA = " + json.dumps(payload, indent=2, sort_keys=True) + ";\n"
    (output / "data.js").write_text(data, encoding="utf-8")

    html = (output / "index.html").read_text(encoding="utf-8")
    html = html.replace('<script src="app.js" defer></script>', '<script src="data.js"></script>\n<script src="app.js" defer></script>')
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return {"output": str(output), "files": sorted(path.name for path in output.iterdir())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ChromoFold public static site")
    parser.add_argument("--output", type=Path, default=Path("dist/site"))
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
