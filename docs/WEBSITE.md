# ChromoFold public website

The public website explains ChromoFold, presents bounded evidence, links release assets and gives users a safe first-use path.

## Build locally

```bash
make -f site.mk site-all
python3 -m http.server 8080 --directory dist/site
```

Open `http://127.0.0.1:8080`.

## Source of truth

The website does not own independent copies of product facts:

- `product/public-claims.json` contains public benchmark claims and their evidence scope;
- `product/downloads.json` defines downloadable product bundles;
- `product/evidence-registry.json` defines evidence maturity;
- `tools/build_public_site.py` combines those registries into `dist/site/data.js`.

The browser estimator is deliberately advisory. It selects a starter profile from user-supplied goals but cannot inspect hardware, establish compatibility or predict performance. The downloadable local Assistant and CLI perform machine inspection and generate reviewable configuration files.

## Claim rules

A public claim must include:

1. a stable identifier;
2. the result;
3. plain-language meaning;
4. hardware or benchmark environment;
5. scope and caveats;
6. evidence level.

Allowed evidence levels are:

- `estimate`;
- `measured`;
- `qualified`;
- `independently-reproduced`.

Do not change `measured` to `qualified` because a CI build passed. Qualification requires the repository hardware harness to return `PASS` for the exact runtime and artifact digest. Independent reproduction requires an external result with enough environment and command information to audit.

## Deployment

`.github/workflows/public-site.yml` builds a preview artifact on pull requests. Pushes to `main` also publish the static output through GitHub Pages.

The repository owner must enable GitHub Pages with **GitHub Actions** as the deployment source. The workflow does not modify repository settings.

## Downloads

The site links to the latest GitHub release rather than embedding mutable binary URLs. Release archives retain their manifest and SHA-256 sidecar produced by the Hub distribution workflow.

## Privacy

The public estimator runs entirely in the browser and sends no workload inputs to a server. Real machine inspection remains part of the local ChromoFold Hub and Assistant.
