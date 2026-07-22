# ChromoFold product portal

The public site is a static, registry-driven product portal published through GitHub Pages.

## Routes

- `index.html` — product overview and current status;
- `technology.html` — compressed-compute mechanism and loss regions;
- `evidence.html` — filterable public claims and evidence ladder;
- `compatibility.html` — runtime, model, GPU and operating-system status;
- `start.html` — advisory profile selection and local handoff;
- `releases.html` — release verification and promotion gates;
- `contribute.html` — reproductions, integrations and negative results.

## Sources of truth

The pages do not maintain separate copies of product facts. `tools/build_public_site.py` assembles `data.js` from:

- `product/public-claims.json`;
- `product/evidence-registry.json`;
- `product/compatibility.json`;
- `product/profiles.json`;
- `product/downloads.json`;
- `product/release-channel.json`;
- `product/portal.json`.

Changes to any of these registries trigger a portal rebuild and Pages deployment.

## Security and privacy

The portal is fully static. User planner inputs remain in the browser. Registry values are escaped before insertion into HTML. The browser planner is advisory and cannot establish hardware compatibility or performance.

## Evidence governance

Public claims must preserve their hardware, scope and evidence level. A candidate release is not production-qualified. Qualification requires PASS evidence for the exact artifact digest, runtime, hardware and workload fingerprint. Conflicting and negative results remain visible.

## Contribution contracts

Structured issue forms collect:

- benchmark reproductions;
- runtime integration proposals;
- negative or conflicting results.

A reproduction requires exact environment information, immutable revisions, complete commands and raw result artifacts. Screenshot-only submissions do not qualify as independent reproduction.

## Local build

```bash
make -f site.mk site-all
python3 -m http.server 8080 --directory dist/site
```

All routes, the sitemap, shared registry payload and deployment contract are validated by `tests/test_public_site.py`.