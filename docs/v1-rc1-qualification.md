# ChromoFold v1.0.0-rc1 qualification

This scope converts the M13–M19 contracts into an executable release decision path. It does not claim that hosted CPU CI proves GPU performance, multi-node recovery, mTLS custody, long-duration leak freedom, rolling upgrades, rollback, or stable promotion.

## Decision path

```text
runtime events
-> correlated latency, correctness, memory, and reference accounting
-> redacted evidence collection
-> security and operational proof
-> exact artifact-digest comparison
-> PASS, FAIL, or INCOMPLETE
```

## Hosted pull-request gate

```bash
make -f rc1-qualification.mk rc1-all
```

This compiles the runtime contract, exercises pass and failure decisions, validates the evidence schema, compiles the CLI, and rejects fabricated passing evidence.

## Manual hardware qualification

Run `RC1 hardware qualification` explicitly on a registered `[self-hosted, gpu]` runner. The workflow requires `CHROMOFOLD_RUNTIME_HARNESS` and refuses to generate evidence itself.

Modes:

- `smoke`: orchestration and instrumentation check;
- `candidate`: release-candidate soak and recovery qualification;
- `stable`: 24-hour qualification appropriate for a stable-release decision.

## Required runtime integrations

The patched serving harness must emit evidence derived from real operations across admission, scheduling, prefill, compressed KV lifecycle, transport, decode, authentication, audit, failure recovery, upgrade, rollback, and artifact promotion.

## Release rule

A PASS is eligible for `v1.0.0-rc1`, not automatic publication. The promoted digest must exactly equal the qualified digest. Missing hardware or operational evidence yields INCOMPLETE rather than a synthetic pass.
