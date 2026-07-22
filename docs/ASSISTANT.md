# ChromoFold Guided Assistant

The guided assistant is a deterministic local planner. It does not call a hosted language model and does not make unmeasured performance predictions.

## What it does

- identifies the user's goal;
- collects model, context and concurrency inputs;
- delegates profile selection to `tools/chromofold.py`;
- explains why hardware qualification is required;
- produces a reviewable sequence of inspect, configure, compare and qualify actions;
- preserves the distinction between estimates, measurements, qualified evidence and independent reproduction.

## CLI

```bash
printf '%s' '{"intent":"longer-context","model":"model.gguf","context":65536,"concurrency":4}' \
  | python3 tools/chromofold_assistant.py
```

Supported intents:

- `fit-model`
- `longer-context`
- `more-users`
- `shared-prompts`
- `explain`

When required fields are missing, the response state is `NEEDS_INPUT`. A complete workload returns `READY` with a recommendation, plan and next command.

## Hub API

```http
POST /api/assistant
Content-Type: application/json
```

The request and response use versioned JSON schemas in their `schema` fields. The Hub remains bound to loopback by default.

## Installation

From an extracted release bundle:

```bash
./install/install.sh
chromofold-hub
```

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File install\install.ps1
chromofold-hub
```

The installers copy only the extracted, checksum-verifiable bundle contents. They do not download executable code and do not modify global system directories by default.

## Claim boundary

A generated plan is an estimate. A public workload claim requires RC1 hardware evidence with `decision: PASS` for the exact runtime and artifact digest.
