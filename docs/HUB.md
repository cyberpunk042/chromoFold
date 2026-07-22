# ChromoFold Hub

ChromoFold Hub is the local, user-facing layer over the deterministic product contracts in `tools/chromofold.py` and `product/*.json`.

## Run locally

```bash
python3 hub/server.py
```

Open `http://127.0.0.1:8090`.

The Hub can:

- inspect local NVIDIA, CUDA, CPU and memory signals;
- collect workload goals;
- request conservative profile recommendations;
- generate local runtime bundles;
- display compatibility and evidence maturity;
- keep estimates visibly separate from measurements and qualification.

The server listens on loopback by default. Hardware information is not uploaded by the application.

## API

- `GET /api/catalog`
- `GET /api/inspect`
- `POST /api/recommend`
- `POST /api/configure`

The API delegates to the repository-owned CLI instead of reimplementing recommendation logic in JavaScript.

## Build a downloadable bundle

```bash
python3 tools/build_product_bundle.py \
  --version dev \
  --output dist/chromofold-dev.tar.gz
```

This produces:

- a versioned archive;
- `manifest.json` containing per-file SHA-256 hashes;
- a sidecar archive checksum;
- a `chromofold-hub` launcher.

Set `SOURCE_DATE_EPOCH` for reproducible archive metadata.

## Evidence boundary

The Hub may recommend and configure, but it must not claim a measured improvement. A result becomes qualified only when the hardware harness produces validated PASS evidence for the exact runtime and artifact digest.
