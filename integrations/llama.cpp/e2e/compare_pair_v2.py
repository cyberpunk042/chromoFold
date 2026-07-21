#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dense and ChromoFold llama.cpp run artifacts")
    parser.add_argument("dense", type=Path)
    parser.add_argument("chromofold", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dense = load(args.dense)
    compressed = load(args.chromofold)
    for key in ("model_sha256", "prompt_tokens", "generated_tokens", "logits"):
        if key not in dense or key not in compressed:
            raise SystemExit(f"missing required pair field: {key}")
    if dense["model_sha256"] != compressed["model_sha256"]:
        raise SystemExit("dense and ChromoFold model hashes differ")
    if dense["prompt_tokens"] != compressed["prompt_tokens"]:
        raise SystemExit("prompt tokenization differs")

    dense_tokens = dense["generated_tokens"]
    compressed_tokens = compressed["generated_tokens"]
    count = min(len(dense_tokens), len(compressed_tokens))
    first_divergence = next((i for i in range(count) if dense_tokens[i] != compressed_tokens[i]), None)
    if first_divergence is None and len(dense_tokens) != len(compressed_tokens):
        first_divergence = count
    top1_agreement = sum(dense_tokens[i] == compressed_tokens[i] for i in range(count)) / count if count else 0.0

    rows = []
    max_abs = 0.0
    abs_sum = 0.0
    abs_count = 0
    cosine_sum = 0.0
    cosine_count = 0
    all_finite = True
    for position, (drow, crow) in enumerate(zip(dense["logits"], compressed["logits"])):
        dv = [float(v) for v in drow["values"]]
        cv = [float(v) for v in crow["values"]]
        if len(dv) != len(cv):
            raise SystemExit(f"logit width differs at position {position}")
        errors = [abs(x - y) for x, y in zip(dv, cv)]
        all_finite = all_finite and all(math.isfinite(v) for v in dv + cv)
        row_max = max(errors, default=0.0)
        row_mean = sum(errors) / len(errors) if errors else 0.0
        row_cosine = cosine(dv, cv)
        max_abs = max(max_abs, row_max)
        abs_sum += sum(errors)
        abs_count += len(errors)
        cosine_sum += row_cosine
        cosine_count += 1
        rows.append({
            "position": position,
            "dense_top1": drow.get("top1"),
            "chromofold_top1": crow.get("top1"),
            "max_abs_logit_error": row_max,
            "mean_abs_logit_error": row_mean,
            "cosine_similarity": row_cosine,
        })

    result = {
        "model_sha256": dense["model_sha256"],
        "prompt_count": len(dense["prompt_tokens"]),
        "dense_generated_count": len(dense_tokens),
        "chromofold_generated_count": len(compressed_tokens),
        "first_divergence": first_divergence,
        "top1_agreement": top1_agreement,
        "all_finite": all_finite,
        "max_abs_error": max_abs,
        "mean_abs_error": abs_sum / abs_count if abs_count else 0.0,
        "mean_cosine_similarity": cosine_sum / cosine_count if cosine_count else 0.0,
        "positions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("top1_agreement", "first_divergence", "max_abs_error", "all_finite")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
