#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import aiohttp


async def completion(session: aiohttp.ClientSession, endpoint: str, prompt: str, request_id: int) -> dict:
    started = time.perf_counter()
    async with session.post(f"{endpoint}/completion", json={"prompt": prompt, "n_predict": 32, "seed": request_id}) as response:
        body = await response.json()
        return {"request_id": request_id, "status": response.status, "elapsed": time.perf_counter() - started, "body": body}


async def main_async(args: argparse.Namespace) -> None:
    prompts = ["cluster prefix shared: " + ("alpha " * 64) + f"request {index}" for index in range(args.requests)]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        results = await asyncio.gather(*(completion(session, args.router, prompt, index) for index, prompt in enumerate(prompts)))
    if any(item["status"] != 200 for item in results):
        raise SystemExit("one or more cluster requests failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"requests": results, "count": len(results)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="http://127.0.0.1:8090")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
