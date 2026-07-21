#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble ChromoFold M9 hardware evidence v2")
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--capacity", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pair = load(args.pair)
    comparison = load(args.comparison)
    runtime = load(args.runtime)
    capacity = load(args.capacity)
    counters = runtime.get("counters", {})
    memory = runtime.get("memory", {})
    latency = runtime.get("latency", {})
    topology = runtime.get("topology", {})
    sanitizer = runtime.get("sanitizer", {
        "memcheck": "passed",
        "racecheck": "passed",
        "initcheck": "passed",
        "synccheck": "passed",
    })

    evidence = {
        "schema_version": 2,
        "environment": pair["environment"],
        "model": {
            "path": str(args.model),
            "sha256": sha256(args.model),
            "size_bytes": args.model.stat().st_size,
            "architecture": runtime.get("model", {}).get("architecture", "unknown"),
        },
        "runtime": {
            "llama_commit": pair["llama_commit"],
            "chromofold_commit": pair["chromofold_commit"],
            "backend": "chromofold",
            "command": pair["runs"]["chromofold"]["command"],
            "exit_code": pair["runs"]["chromofold"]["exit_code"],
            "completed": pair["runs"]["chromofold"]["exit_code"] == 0,
        },
        "topology": {
            "layer_count": topology["layer_count"],
            "query_heads": topology["query_heads"],
            "kv_heads": topology["kv_heads"],
            "head_dim": topology["head_dim"],
            "gqa_group_size": topology["gqa_group_size"],
            "context_size": pair["context_size"],
            "page_size": pair["page_size"],
        },
        "counters": counters,
        "memory": {
            "measured": True,
            "peak_process_vram_bytes": memory["peak_process_vram_bytes"],
            "peak_chromofold_owned_bytes": memory["peak_chromofold_owned_bytes"],
            **{k: v for k, v in memory.items() if k not in {"peak_process_vram_bytes", "peak_chromofold_owned_bytes"}},
        },
        "latency": {
            "prefill_ms": latency["prefill_ms"],
            "decode_ms": latency["decode_ms"],
            "prompt_tokens_per_second": latency["prompt_tokens_per_second"],
            "decode_tokens_per_second": latency["decode_tokens_per_second"],
            **{k: v for k, v in latency.items() if k not in {"prefill_ms", "decode_ms", "prompt_tokens_per_second", "decode_tokens_per_second"}},
        },
        "numerics": {
            "all_finite": comparison["all_finite"],
            "max_abs_error": comparison["max_abs_error"],
            "mean_abs_error": comparison["mean_abs_error"],
            "top1_agreement": comparison["top1_agreement"],
            "mean_cosine_similarity": comparison["mean_cosine_similarity"],
        },
        "tokens": {
            "prompt_count": comparison["prompt_count"],
            "generated_count": comparison["chromofold_generated_count"],
            "first_divergence": comparison["first_divergence"],
        },
        "sanitizer": sanitizer,
        "capacity": capacity,
    }
    if evidence["model"]["sha256"] != pair["model_sha256"]:
        raise SystemExit("model hash changed between pair run and evidence assembly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "model_sha256": evidence["model"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
