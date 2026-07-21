#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--require-claim", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    for key in ("runtime", "model", "backend", "correctness", "memory", "latency", "capacity", "counters"):
        require(key in data, f"missing evidence section: {key}")
    require(len(data["runtime"].get("llama_commit", "")) == 40, "invalid llama.cpp commit")
    require(len(data["model"].get("sha256", "")) == 64, "invalid model digest")
    require(data["correctness"].get("finite") is True, "non-finite model output")
    require(data["memory"].get("peak_vram_bytes", 0) > 0, "peak VRAM not recorded")
    require(data["capacity"].get("completed") is True, "capacity case did not complete")
    if args.require_claim and data["backend"] == "chromofold":
        counters = data["counters"]
        require(counters.get("compressed_attention_launches", 0) > 0, "compressed attention was not exercised")
        require(counters.get("sealed_values_consumed", 0) > 0, "sealed pages were not consumed")
        require(counters.get("dense_fallback_launches") == 0, "dense fallback was used")
        require(data["correctness"].get("token_match_rate", 0.0) >= 0.99, "token agreement below 99%")
    print(json.dumps({"valid": True, "backend": data["backend"], "context": data["capacity"]["context"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
