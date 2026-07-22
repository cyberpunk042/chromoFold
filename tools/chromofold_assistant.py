#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "chromofold.py"
INTENTS = ROOT / "product" / "assistant-intents.json"


def run_cli(args: list[str]) -> tuple[int, Any]:
    proc = subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT, text=True,
                          capture_output=True, check=False, timeout=120)
    raw = proc.stdout or proc.stderr or "{}"
    try:
        return proc.returncode, json.loads(raw)
    except json.JSONDecodeError:
        return proc.returncode, {"stdout": proc.stdout, "stderr": proc.stderr}


def build_plan(request: dict[str, Any]) -> dict[str, Any]:
    catalog = json.loads(INTENTS.read_text(encoding="utf-8"))
    intent_name = str(request.get("intent", "explain"))
    intent = catalog["intents"].get(intent_name)
    if not intent:
        raise ValueError(f"unsupported intent: {intent_name}")

    missing = [name for name in intent.get("questions", []) if request.get(name) in (None, "")]
    if missing:
        return {
            "schema": "chromofold.assistant-response.v1",
            "state": "NEEDS_INPUT",
            "intent": intent_name,
            "missing": missing,
            "question": f"Provide: {', '.join(missing)}",
            "evidence_level": "estimate",
        }

    if intent_name == "explain":
        return {
            "schema": "chromofold.assistant-response.v1",
            "state": "EXPLAINED",
            "intent": intent_name,
            "answer": (
                "ChromoFold trades inexpensive GPU decoding work for lower VRAM residency and memory traffic. "
                "It can recommend and configure a profile, but only a hardware qualification PASS establishes "
                "that a workload improved on a specific machine."
            ),
            "evidence_boundary": catalog["evidence_boundary"],
        }

    goal = str(request.get("goal") or intent["default_goal"])
    args = ["recommend", "--goal", goal, "--json"]
    for key, flag in (("model", "--model"), ("context", "--context"), ("concurrency", "--concurrency")):
        if request.get(key) not in (None, ""):
            args += [flag, str(request[key])]
    code, recommendation = run_cli(args)
    if code != 0:
        return {
            "schema": "chromofold.assistant-response.v1",
            "state": "ERROR",
            "intent": intent_name,
            "error": recommendation,
        }

    profile = recommendation.get("profile") or recommendation.get("recommended_profile")
    steps = [
        "Inspect the local machine and save the result.",
        f"Generate the {profile or 'recommended'} profile bundle with silent fallback disabled.",
        "Run a baseline workload using the same model, context and concurrency.",
        "Run the ChromoFold workload with identical inputs.",
        "Compare capacity, TTFT, throughput, peak VRAM and correctness.",
        "Run RC1 smoke qualification and accept a claim only when evidence returns PASS.",
    ]
    return {
        "schema": "chromofold.assistant-response.v1",
        "state": "READY",
        "intent": intent_name,
        "evidence_level": "estimate",
        "qualification_required": True,
        "recommendation": recommendation,
        "plan": steps,
        "next_action": {
            "command": [sys.executable, "tools/chromofold.py", "configure", "--profile", profile or "balanced"],
            "description": "Generate a local, reviewable configuration bundle.",
        },
        "claim_policy": "Do not publish a workload win until the exact artifact digest has qualified PASS evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic ChromoFold guided assistant")
    parser.add_argument("--request", help="JSON request; reads stdin when omitted")
    args = parser.parse_args()
    request = json.loads(args.request) if args.request else json.load(sys.stdin)
    print(json.dumps(build_plan(request), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
